#!/usr/bin/env python3
"""Run reset-window recovery for corpus_v2.

Two recovery modes are intentionally separated:

1. Missing-fill resumes the original run_id against the original trial plan.
   The runner skips already-written planned_trial_id rows and only fills
   planned trials that never produced rows.
2. Invalid-row recovery writes a new run_id and a new recovery trial plan,
   preserving the original raw evidence while re-running rows that were
   quarantined as infrastructure invalid.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN_RUN_ID = "corpus_v2_20260505_064053"
MAIN_PLAN = ROOT / "results/designs/D-corpus-v2-20260505/trial_plan.jsonl"
MAIN_QUARANTINE = ROOT / "results/quarantine/corpus_v2_20260505_064053"
TASK_BY_CHANNEL = {
    "file_content": "readme_summarize",
    "code_comment": "refactor_app",
    "data_row": "csv_summarize",
    "tool_output": "triage_log",
}
RESET_AGENT_PREFIXES = ("claude_cli", "codex_cli", "hermes_minimax")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def provider(agent_id: str) -> str:
    if agent_id.startswith("claude_cli"):
        return "claude"
    if agent_id.startswith("codex_cli"):
        return "codex"
    if agent_id.startswith("hermes_minimax"):
        return "minimax"
    return "other"


def is_reset_agent(agent_id: str) -> bool:
    return agent_id.startswith(RESET_AGENT_PREFIXES)


def build_invalid_recovery_plan(design_id: str, out_dir: Path) -> Path:
    original_plan = {r["planned_trial_id"]: r for r in load_jsonl(MAIN_PLAN)}
    invalid_rows = load_jsonl(MAIN_QUARANTINE / "invalid_infrastructure_rows.jsonl")
    original_ids: list[str] = []
    seen: set[str] = set()
    reasons: dict[str, str] = {}
    for row in invalid_rows:
        agent_id = row.get("agent_id") or ""
        planned_id = row.get("planned_trial_id")
        if not planned_id or not is_reset_agent(agent_id) or planned_id in seen:
            continue
        if planned_id not in original_plan:
            continue
        seen.add(planned_id)
        original_ids.append(planned_id)
        reasons[planned_id] = row.get("quarantine_reason") or row.get("invalid_reason") or "invalid_infrastructure"

    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "trial_plan.jsonl"
    summary_path = out_dir / "trial_plan_summary.json"
    with plan_path.open("w") as f:
        for seq, original_id in enumerate(original_ids):
            row = dict(original_plan[original_id])
            row["recovery_source_run_id"] = MAIN_RUN_ID
            row["recovery_original_design_id"] = row.get("design_id")
            row["recovery_original_planned_trial_id"] = original_id
            row["recovery_reason"] = reasons[original_id]
            row["design_id"] = design_id
            row["planned_trial_id"] = f"{design_id}:{seq:09d}"
            row["run_id"] = None
            row["assignment_order"] = seq
            row["wave_id"] = f"reset_recovery_{seq // 8:04d}"
            f.write(json.dumps(row, sort_keys=True) + "\n")

    by_agent = Counter(original_plan[tid]["agent_id"] for tid in original_ids)
    by_channel = Counter(original_plan[tid]["channel"] for tid in original_ids)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "design_id": design_id,
                "source_run_id": MAIN_RUN_ID,
                "source_plan": str(MAIN_PLAN.relative_to(ROOT)),
                "planned_trials": len(original_ids),
                "by_agent": dict(sorted(by_agent.items())),
                "by_channel": dict(sorted(by_channel.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return plan_path


def grouped_jobs_from_plan(plan_path: Path, run_id: str, mode: str) -> list[dict]:
    groups: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in load_jsonl(plan_path):
        groups[(row["agent_id"], int(row["shard"]), row["channel"])] += 1
    jobs = []
    for (agent_id, shard, channel), selected in sorted(groups.items()):
        jobs.append(
            {
                "mode": mode,
                "run_id": run_id,
                "plan": str(plan_path.relative_to(ROOT)),
                "agent_id": agent_id,
                "shard": shard,
                "channel": channel,
                "task": TASK_BY_CHANNEL[channel],
                "max_attacks": selected,
            }
        )
    return jobs


def missing_fill_plan(out_dir: Path) -> Path:
    missing_rows = [
        row
        for row in load_jsonl(MAIN_QUARANTINE / "missing_planned_trials.jsonl")
        if is_reset_agent(row.get("agent_id") or "")
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "missing_fill_plan.jsonl"
    with path.open("w") as f:
        for row in missing_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def command_for(job: dict) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "run_agent_shard.py"),
        "--shard-id",
        str(job["shard"]),
        "--agent-id",
        job["agent_id"],
        "--channel",
        job["channel"],
        "--max-attacks",
        str(job["max_attacks"]),
        "--run-id",
        job["run_id"],
        "--split",
        "all",
        "--n-repeats",
        "1",
        "--task",
        job["task"],
        "--trial-plan",
        job["plan"],
    ]


def run_jobs(jobs: list[dict], log_dir: Path, caps: dict[str, int], global_cap: int) -> list[dict]:
    pending = deque(jobs)
    running: list[dict] = []
    results: list[dict] = []
    active_by_provider: Counter = Counter()

    def can_launch(job: dict) -> bool:
        p = provider(job["agent_id"])
        return len(running) < global_cap and active_by_provider[p] < caps.get(p, 1)

    def launch(job: dict) -> None:
        p = provider(job["agent_id"])
        tag = f"{job['mode']}__{job['agent_id']}__s{job['shard']}__{job['channel']}__{job['task']}"
        log_path = log_dir / f"{tag}.log"
        cmd = command_for(job)
        fh = log_path.open("w")
        fh.write(f"cmd: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        job.update({"proc": proc, "fh": fh, "log": str(log_path), "started": time.time(), "provider": p, "tag": tag})
        running.append(job)
        active_by_provider[p] += 1
        log(f"launched {tag} pid={proc.pid}")

    while pending or running:
        launched = True
        while pending and launched:
            launched = False
            for _ in range(len(pending)):
                job = pending.popleft()
                if can_launch(job):
                    launch(job)
                    launched = True
                    break
                pending.append(job)

        still = []
        for job in running:
            rc = job["proc"].poll()
            if rc is None:
                still.append(job)
                continue
            job["fh"].close()
            active_by_provider[job["provider"]] -= 1
            elapsed = round(time.time() - job["started"], 1)
            result = {
                "tag": job["tag"],
                "mode": job["mode"],
                "agent_id": job["agent_id"],
                "channel": job["channel"],
                "shard": job["shard"],
                "rc": rc,
                "elapsed_sec": elapsed,
                "log": job["log"],
            }
            results.append(result)
            log(f"finished {job['tag']} rc={rc} elapsed={elapsed}s")
        running = still
        time.sleep(2)
    return results


def run_post(recovery_run_id: str, recovery_plan: Path, log_dir: Path) -> list[dict]:
    commands = [
        (
            "analysis_main",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/analyze_asr_methodology.py",
                "--glob",
                f"results/agent_shard_runs/*{MAIN_RUN_ID}*.jsonl",
                "--out",
                "results/analysis/corpus_v2.json",
                "--audit-csv",
                "results/analysis/corpus_v2_audit_sample.csv",
            ],
        ),
        (
            "quarantine_main",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/quarantine_invalid_rows.py",
                "--run-id",
                MAIN_RUN_ID,
                "--plan",
                str(MAIN_PLAN.relative_to(ROOT)),
                "--runs",
                f"results/agent_shard_runs/*{MAIN_RUN_ID}*.jsonl",
                "--out-dir",
                "results/quarantine/corpus_v2_20260505_064053",
            ],
        ),
        (
            "analysis_recovery",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/analyze_asr_methodology.py",
                "--glob",
                f"results/agent_shard_runs/*{recovery_run_id}*.jsonl",
                "--out",
                "results/analysis/corpus_v2_reset_recovery.json",
                "--audit-csv",
                "results/analysis/corpus_v2_reset_recovery_audit_sample.csv",
            ],
        ),
        (
            "quarantine_recovery",
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/quarantine_invalid_rows.py",
                "--run-id",
                recovery_run_id,
                "--plan",
                str(recovery_plan.relative_to(ROOT)),
                "--runs",
                f"results/agent_shard_runs/*{recovery_run_id}*.jsonl",
                "--out-dir",
                "results/quarantine/corpus_v2_reset_recovery",
            ],
        ),
    ]
    results = []
    for name, cmd in commands:
        log_path = log_dir / f"post_{name}.log"
        with log_path.open("w") as f:
            f.write(f"cmd: {' '.join(cmd)}\n")
            f.flush()
            started = time.time()
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT)
        results.append(
            {"name": name, "rc": proc.returncode, "elapsed_sec": round(time.time() - started, 1), "log": str(log_path)}
        )
        log(f"post {name} rc={proc.returncode}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-run-id", default="corpus_v2_reset_recovery_20260505_1108")
    parser.add_argument("--recovery-design-id", default="D-corpus-v2-reset-recovery-20260505")
    parser.add_argument("--log-dir", type=Path, default=Path("/tmp/fleet/corpus_v2_reset_recovery_20260505_1108"))
    parser.add_argument("--global-cap", type=int, default=8)
    parser.add_argument("--claude-cap", type=int, default=2)
    parser.add_argument("--codex-cap", type=int, default=3)
    parser.add_argument("--minimax-cap", type=int, default=3)
    args = parser.parse_args()

    log_dir = args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    recovery_dir = ROOT / "results/designs" / args.recovery_design_id
    recovery_plan = build_invalid_recovery_plan(args.recovery_design_id, recovery_dir)
    missing_plan = missing_fill_plan(recovery_dir)

    missing_jobs = grouped_jobs_from_plan(missing_plan, MAIN_RUN_ID, "missing_fill")
    recovery_jobs = grouped_jobs_from_plan(recovery_plan, args.recovery_run_id, "invalid_recovery")
    jobs = missing_jobs + recovery_jobs
    (log_dir / "jobs.json").write_text(json.dumps(jobs, indent=2, sort_keys=True))
    log(f"reset recovery starting jobs={len(jobs)} missing={len(missing_jobs)} invalid_recovery={len(recovery_jobs)}")

    results = run_jobs(
        jobs,
        log_dir,
        caps={"claude": args.claude_cap, "codex": args.codex_cap, "minimax": args.minimax_cap, "other": 1},
        global_cap=args.global_cap,
    )
    (log_dir / "job_results.json").write_text(json.dumps(results, indent=2, sort_keys=True))

    post = run_post(args.recovery_run_id, recovery_plan, log_dir)
    (log_dir / "post_results.json").write_text(json.dumps(post, indent=2, sort_keys=True))
    log("reset recovery complete")
    return 0 if all(item["rc"] == 0 for item in post) else 1


if __name__ == "__main__":
    raise SystemExit(main())
