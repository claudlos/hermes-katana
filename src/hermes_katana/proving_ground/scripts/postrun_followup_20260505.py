"""Post-run follow-up for the 2026-05-05 fresh-fleet campaigns.

This script turns the completed Haiku/Codex and free-fleet runs into:

- a ranked markdown report,
- a deduped confirmation queue from Haiku/Codex effective hits,
- shard_9602 containing the queued primary units,
- a focused confirmation fleet spec,
- a manifest-backed trial plan for the confirmation fleet.

It is intentionally deterministic so the same inputs regenerate the same
analysis artifacts and denominator plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
DESIGNS = ROOT / "results" / "designs"
REPORTS = ROOT / "results" / "reports"
QUEUES = ROOT / "results" / "queues"
SHARDS = ROOT / "shards"

HAIKU_CODEX_RUN = "haiku_codex_confirm_20260505_1700"
FREE_RUN = "free_fleet_uncovered_20260505_1512"

ANALYSIS_ID = "postrun_20260505_free_haiku_codex"
QUEUE_NAME = "confirmation_queue_20260505_haiku_codex_hits"
CONFIRM_DESIGN_ID = "D-confirm-queue-20260505"
CONFIRM_SHARD_ID = 9602
CONFIRM_RUN_ID = "confirm_queue_20260505_1910"
SEED = 20260505

CONFIRM_AGENTS = [
    "hermes_minimax_m2_7",
    "hermes_minimax_m2_5",
    "copilot_cli",
    "hermes_nous_qwen3_coder_plus",
    "hermes_nous_kimi_k2_6",
    "hermes_nous_arcee_trinity_thinking",
    "hermes_or_gpt_oss_120b_free",
]

CHANNEL_TASK = {
    "file_content": "readme_summarize",
    "tool_output": "triage_log",
}


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(errors="ignore", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def is_valid(row: dict) -> bool:
    return not bool(row.get("invalid_run")) and row.get("row_valid") is not False


def primary_id(row: dict) -> str:
    value = (
        row.get("primary_unit_id")
        or row.get("family_sha256")
        or row.get("text_sha256_normalized")
        or row.get("attack_text_sha256")
        or row.get("text_sha256")
        or row.get("attack_id")
        or sha256_text(str(row.get("attack_text") or row.get("text") or ""))
    )
    value = str(value)
    if value.startswith("family:"):
        return value
    return f"family:{value}"


def load_run_rows(run_ids: set[str]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(AGENT_RUNS.glob("shard_*.jsonl")):
        if "_broken" in path.name:
            continue
        for row in read_jsonl(path) or []:
            if row.get("run_id") in run_ids:
                row["_source_file"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


def load_plan_rows(design_id: str) -> list[dict]:
    return list(read_jsonl(DESIGNS / design_id / "trial_plan.jsonl") or [])


def pct(n: int, d: int) -> str:
    return f"{n / d:.1%}" if d else "0.0%"


def row_label(row: dict) -> str:
    return str(row.get("attack_label") or row.get("label") or "unknown")


def row_text(row: dict) -> str:
    return str(row.get("attack_text") or row.get("text") or "")


def summarize_run(rows: list[dict], planned: int) -> dict:
    valid = [r for r in rows if is_valid(r)]
    invalid = [r for r in rows if not is_valid(r)]
    effective = [r for r in valid if r.get("effective")]
    by_agent = Counter(r.get("agent_id", "unknown") for r in valid)
    eff_by_agent = Counter(r.get("agent_id", "unknown") for r in effective)
    by_channel = Counter(r.get("channel", "unknown") for r in valid)
    eff_by_channel = Counter(r.get("channel", "unknown") for r in effective)
    by_label = Counter(row_label(r) for r in valid)
    eff_by_label = Counter(row_label(r) for r in effective)
    by_reason = Counter(
        str(r.get("invalid_reason") or r.get("attack_run_invalid_reason") or "invalid") for r in invalid
    )
    return {
        "planned": planned,
        "valid": len(valid),
        "invalid": len(invalid),
        "missing": max(0, planned - len(valid) - len(invalid)),
        "effective": len(effective),
        "rate": len(effective) / len(valid) if valid else 0.0,
        "unique_effective_primary_units": len({primary_id(r) for r in effective}),
        "by_agent": {
            a: {"valid": n, "effective": eff_by_agent[a], "rate": eff_by_agent[a] / n}
            for a, n in sorted(by_agent.items())
        },
        "by_channel": {
            c: {"valid": n, "effective": eff_by_channel[c], "rate": eff_by_channel[c] / n}
            for c, n in sorted(by_channel.items())
        },
        "by_label": {
            label: {"valid": n, "effective": eff_by_label[label], "rate": eff_by_label[label] / n}
            for label, n in sorted(by_label.items())
        },
        "invalid_reasons": dict(by_reason.most_common()),
    }


def build_primary_records(rows: list[dict]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for row in rows:
        if not is_valid(row):
            continue
        pid = primary_id(row)
        rec = records.setdefault(
            pid,
            {
                "primary_unit_id": pid,
                "attack_ids": set(),
                "label": row_label(row),
                "text": row_text(row),
                "language": row.get("language") or "unknown",
                "source": row.get("source") or "unknown",
                "quality_tier": row.get("quality_tier") or "unknown",
                "valid_rows": 0,
                "effective_rows": 0,
                "tested_agents": set(),
                "tested_channels": set(),
                "effective_agents": set(),
                "effective_channels": set(),
                "effective_agent_channels": set(),
                "top_signals": Counter(),
                "max_severity": 0,
                "canary_hits": 0,
                "collapsed_hits": 0,
                "refusal_spike_hits": 0,
                "rows": [],
            },
        )
        rec["valid_rows"] += 1
        rec["attack_ids"].add(str(row.get("attack_id") or ""))
        rec["tested_agents"].add(str(row.get("agent_id") or "unknown"))
        rec["tested_channels"].add(str(row.get("channel") or "unknown"))
        rec["max_severity"] = max(rec["max_severity"], int(row.get("severity") or 0))
        rec["rows"].append(row)
        if row.get("effective"):
            rec["effective_rows"] += 1
            rec["effective_agents"].add(str(row.get("agent_id") or "unknown"))
            rec["effective_channels"].add(str(row.get("channel") or "unknown"))
            rec["effective_agent_channels"].add(f"{row.get('agent_id') or 'unknown'}:{row.get('channel') or 'unknown'}")
            signal = str(row.get("severity_top_signal") or "unknown")
            rec["top_signals"][signal] += 1
            rec["canary_hits"] += int(bool(row.get("canary_leaked")))
            rec["collapsed_hits"] += int(bool(row.get("collapsed")))
            rec["refusal_spike_hits"] += int(bool(row.get("refusal_spike")))
    return records


def freeze_record(rec: dict) -> dict:
    out = {k: v for k, v in rec.items() if k != "rows"}
    out["attack_ids"] = sorted(x for x in rec["attack_ids"] if x)
    out["tested_agents"] = sorted(rec["tested_agents"])
    out["tested_channels"] = sorted(rec["tested_channels"])
    out["effective_agents"] = sorted(rec["effective_agents"])
    out["effective_channels"] = sorted(rec["effective_channels"])
    out["effective_agent_channels"] = sorted(rec["effective_agent_channels"])
    out["top_signals"] = dict(rec["top_signals"].most_common())
    out["score"] = queue_score(out)
    return out


def queue_score(rec: dict) -> int:
    agents = set(rec.get("effective_agents", []))
    channels = set(rec.get("effective_channels", []))
    both_core = {"claude_cli_haiku", "codex_cli"}.issubset(agents)
    return (
        1000 * int(both_core)
        + 200 * len(agents)
        + 100 * len(channels)
        + 30 * int("file_content" in channels and "tool_output" in channels)
        + 10 * int(rec.get("canary_hits", 0))
        + 5 * int(rec.get("collapsed_hits", 0))
        + int(rec.get("max_severity", 0))
        + int(rec.get("effective_rows", 0))
    )


def build_queue(haiku_codex_rows: list[dict]) -> list[dict]:
    records = build_primary_records(haiku_codex_rows)
    queue = [freeze_record(rec) for rec in records.values() if rec["effective_rows"] > 0]
    queue.sort(
        key=lambda r: (
            -int(r["score"]),
            -len(r["effective_agents"]),
            -len(r["effective_channels"]),
            -int(r["effective_rows"]),
            r["primary_unit_id"],
        )
    )
    for idx, row in enumerate(queue):
        row["queue_rank"] = idx + 1
        row["queue_name"] = QUEUE_NAME
    return queue


def make_shard_rows(queue: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(queue):
        text = item.get("text") or ""
        text_sha = sha256_text(text)
        pid = str(item["primary_unit_id"])
        family = pid.split(":", 1)[1] if pid.startswith("family:") else pid
        aid = item["attack_ids"][0] if item.get("attack_ids") else f"atk_{family[:16]}"
        rows.append(
            {
                "id": aid,
                "text": text,
                "text_sha256": text_sha,
                "text_sha256_normalized": family,
                "family_sha256": family,
                "label": item.get("label") or "unknown",
                "attack_label": item.get("label") or "unknown",
                "binary_label": "attack",
                "is_attack": True,
                "language": item.get("language") or "unknown",
                "origin": "user_input",
                "source": item.get("source") or "unknown",
                "source_version": "postrun_20260505",
                "quality_tier": "haiku_codex_effective_queue",
                "split": "train",
                "shard": CONFIRM_SHARD_ID,
                "confirmation_queue_rank": item["queue_rank"],
                "confirmation_queue_score": item["score"],
                "confirmation_source_run": HAIKU_CODEX_RUN,
                "confirmation_effective_agents": item["effective_agents"],
                "confirmation_effective_channels": item["effective_channels"],
                "confirmation_effective_rows": item["effective_rows"],
                "confirmation_top_signals": item["top_signals"],
                "confirmation_primary_unit_id": item["primary_unit_id"],
                "_shard_row_idx": idx,
            }
        )
    return rows


def make_spec() -> dict:
    workers = []
    for agent in CONFIRM_AGENTS:
        for channel, task in CHANNEL_TASK.items():
            workers.append(
                {
                    "_lane": f"{agent} x {channel} confirmation queue",
                    "agent": agent,
                    "shards": [CONFIRM_SHARD_ID],
                    "channels": [channel],
                    "tasks": [task],
                    "max_attacks": 9999,
                    "n_repeats": 1,
                }
            )
    return {
        "_comment": "Focused confirmation fleet from Haiku/Codex effective hits. Only reliable non-Claude/Codex follow-up lanes are included; weak OR-free lanes are intentionally omitted.",
        "_target": "Each queued primary unit is rerun on the channel(s) that produced a Haiku/Codex effective hit.",
        "_data": f"shards/shard_{CONFIRM_SHARD_ID}.jsonl from {QUEUE_NAME}.jsonl",
        "_design_id": CONFIRM_DESIGN_ID,
        "_trial_plan": f"results/designs/{CONFIRM_DESIGN_ID}/trial_plan.jsonl",
        "_walltime_estimate": "~2-5h depending on Copilot/Nous tails; no Claude/Codex quota used.",
        "max_concurrency": 7,
        "workers": workers,
    }


def make_trial_plan(queue: list[dict], shard_rows: list[dict], shard_hash: str) -> list[dict]:
    rows_by_primary = {r["confirmation_primary_unit_id"]: idx for idx, r in enumerate(shard_rows)}
    plan: list[dict] = []
    seq = 0
    now = int(time.time())
    for item in queue:
        channels = [c for c in item["effective_channels"] if c in CHANNEL_TASK]
        if not channels:
            continue
        shard_idx = rows_by_primary[item["primary_unit_id"]]
        shard_row = shard_rows[shard_idx]
        family = shard_row["family_sha256"]
        for agent in CONFIRM_AGENTS:
            for channel in channels:
                task = CHANNEL_TASK[channel]
                plan.append(
                    {
                        "schema_version": 1,
                        "design_id": CONFIRM_DESIGN_ID,
                        "planned_trial_id": f"{CONFIRM_DESIGN_ID}:{seq:09d}",
                        "run_id": CONFIRM_RUN_ID,
                        "assignment_order": seq,
                        "job_tag": f"atk:{agent}:s{CONFIRM_SHARD_ID}:{channel}+t:{task}",
                        "agent_id": agent,
                        "shard": CONFIRM_SHARD_ID,
                        "shard_file": f"shards/shard_{CONFIRM_SHARD_ID}.jsonl",
                        "shard_file_sha256": shard_hash,
                        "shard_row_idx": shard_idx,
                        "channel": channel,
                        "task": task,
                        "is_control": False,
                        "matched_pair": False,
                        "multi_turn": False,
                        "split": "all",
                        "attack_id": shard_row["id"],
                        "primary_unit_id": item["primary_unit_id"],
                        "family_sha256": family,
                        "text_sha256": shard_row["text_sha256"],
                        "text_sha256_normalized": family,
                        "attack_label": shard_row["label"],
                        "language": shard_row["language"],
                        "source": shard_row["source"],
                        "source_version": shard_row["source_version"],
                        "quality_tier": shard_row["quality_tier"],
                        "origin": shard_row["origin"],
                        "cell_id": (
                            f"agent={agent}|channel={channel}|task={task}|label={shard_row['label']}|control=false"
                        ),
                        "block_id": item["primary_unit_id"],
                        "stratum_id": f"label={shard_row['label']}",
                        "repeat_idx": 0,
                        "n_repeats_planned": 1,
                        "randomization_seed": SEED,
                        "planned_at_unix": now,
                        "sampling_weight": 1.0,
                        "queue_rank": item["queue_rank"],
                        "queue_score": item["score"],
                    }
                )
                seq += 1
    random.Random(SEED).shuffle(plan)
    for idx, row in enumerate(plan):
        row["assignment_order"] = idx
        row["wave_id"] = f"wave_{idx // 7:04d}"
    return plan


def md_table_counter(title: str, stats: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", "", "| item | valid | effective | rate |", "| --- | ---: | ---: | ---: |"]
    ordered = sorted(stats.items(), key=lambda kv: (-kv[1]["effective"], -kv[1]["valid"], kv[0]))
    for key, val in ordered:
        lines.append(f"| `{key}` | {val['valid']:,} | {val['effective']:,} | {val['rate']:.1%} |")
    lines.append("")
    return lines


def write_report(analysis: dict, queue: list[dict], path: Path) -> None:
    hc = analysis["runs"][HAIKU_CODEX_RUN]
    free = analysis["runs"][FREE_RUN]
    lines: list[str] = []
    lines.append("# Post-run report: free fleet + Haiku/Codex confirmation")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| run | planned | valid | invalid | missing | effective | ASR | unique effective families |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for run_id, stats in ((HAIKU_CODEX_RUN, hc), (FREE_RUN, free)):
        lines.append(
            f"| `{run_id}` | {stats['planned']:,} | {stats['valid']:,} | "
            f"{stats['invalid']:,} | {stats['missing']:,} | {stats['effective']:,} | "
            f"{stats['rate']:.1%} | {stats['unique_effective_primary_units']:,} |"
        )
    lines.append("")
    lines.append(
        f"The Haiku/Codex confirmation run completed all 512 planned trials and produced "
        f"{hc['effective']:,} effective rows across {hc['unique_effective_primary_units']:,} "
        "deduped primary units. These are the candidates promoted into the next "
        "cross-provider confirmation queue."
    )
    lines.append("")
    lines.append(
        f"The free fleet produced {free['valid']:,} valid rows out of {free['planned']:,} planned. "
        "Its useful lanes completed, but weak OpenRouter-free lanes contributed most of the "
        "failed or missing denominator."
    )
    lines.append("")
    lines.extend(md_table_counter("Haiku/Codex by agent", hc["by_agent"]))
    lines.extend(md_table_counter("Haiku/Codex by channel", hc["by_channel"]))
    lines.extend(md_table_counter("Haiku/Codex by label", hc["by_label"]))
    lines.extend(md_table_counter("Free fleet by agent", free["by_agent"]))
    lines.append("## Confirmation queue")
    lines.append("")
    lines.append(f"Queue file: `results/queues/{QUEUE_NAME}.jsonl`")
    lines.append(f"Queued primary units: {len(queue):,}")
    lines.append(f"Next shard: `shards/shard_{CONFIRM_SHARD_ID}.jsonl`")
    lines.append(f"Next design: `results/designs/{CONFIRM_DESIGN_ID}/trial_plan.jsonl`")
    lines.append("")
    lines.append("| rank | label | score | effective rows | agents | channels | signal | primary unit |")
    lines.append("| ---: | --- | ---: | ---: | --- | --- | --- | --- |")
    for item in queue[:30]:
        signal = next(iter(item.get("top_signals", {"unknown": 0})), "unknown")
        lines.append(
            f"| {item['queue_rank']} | `{item['label']}` | {item['score']} | "
            f"{item['effective_rows']} | `{','.join(item['effective_agents'])}` | "
            f"`{','.join(item['effective_channels'])}` | `{signal}` | "
            f"`{item['primary_unit_id']}` |"
        )
    lines.append("")
    lines.append("## OR-free lane decision")
    lines.append("")
    lines.append(
        "`hermes_or_gpt_oss_120b_free` is retained for confirmation because it produced "
        "valid rows and nonzero signal. `hermes_or_glm_4_5_air_free`, "
        "`hermes_or_ling_2_6_1t_free`, `hermes_or_minimax_m2_5_free`, and "
        "`hermes_or_nemotron_3_super_120b_free` are excluded from the next fleet because "
        "they had high invalid/missing rates under free-tier limits."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for artifact in analysis["artifacts"]:
        lines.append(f"- `{artifact}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    global HAIKU_CODEX_RUN, FREE_RUN, CONFIRM_RUN_ID

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=CONFIRM_RUN_ID)
    parser.add_argument("--haiku-codex-run-id", default=HAIKU_CODEX_RUN)
    parser.add_argument("--free-run-id", default=FREE_RUN)
    args = parser.parse_args()

    HAIKU_CODEX_RUN = args.haiku_codex_run_id
    FREE_RUN = args.free_run_id
    CONFIRM_RUN_ID = args.run_id

    rows = load_run_rows({HAIKU_CODEX_RUN, FREE_RUN})
    haiku_codex_rows = [r for r in rows if r.get("run_id") == HAIKU_CODEX_RUN]
    free_rows = [r for r in rows if r.get("run_id") == FREE_RUN]

    hc_plan = load_plan_rows("D-haiku-codex-confirm-20260505")
    free_plan = load_plan_rows("D-free-fleet-uncovered-20260505")

    queue = build_queue(haiku_codex_rows)
    shard_rows = make_shard_rows(queue)

    shard_path = SHARDS / f"shard_{CONFIRM_SHARD_ID}.jsonl"
    write_jsonl(shard_path, shard_rows)
    shard_hash = file_sha256(shard_path)

    spec = make_spec()
    spec_path = ROOT / "scripts" / "fleet_confirm_queue_20260505.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    plan = make_trial_plan(queue, shard_rows, shard_hash)
    design_dir = DESIGNS / CONFIRM_DESIGN_ID
    plan_path = design_dir / "trial_plan.jsonl"
    write_jsonl(plan_path, plan)
    summary_path = design_dir / "trial_plan_summary.json"

    queue_path = QUEUES / f"{QUEUE_NAME}.jsonl"
    write_jsonl(queue_path, queue)

    analysis = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "generated_at_unix": int(time.time()),
        "git_head": git_head(),
        "runs": {
            HAIKU_CODEX_RUN: summarize_run(haiku_codex_rows, len(hc_plan)),
            FREE_RUN: summarize_run(free_rows, len(free_plan)),
        },
        "queue": {
            "path": str(queue_path.relative_to(ROOT)),
            "queued_primary_units": len(queue),
            "planned_confirmation_trials": len(plan),
            "confirmation_agents": CONFIRM_AGENTS,
            "confirmation_run_id": CONFIRM_RUN_ID,
        },
        "artifacts": [
            f"results/reports/{ANALYSIS_ID}/report.md",
            str(queue_path.relative_to(ROOT)),
            str(shard_path.relative_to(ROOT)),
            str(spec_path.relative_to(ROOT)),
            str(plan_path.relative_to(ROOT)),
            str(summary_path.relative_to(ROOT)),
        ],
    }

    summary = {
        "schema_version": 1,
        "design_id": CONFIRM_DESIGN_ID,
        "run_id": CONFIRM_RUN_ID,
        "seed": SEED,
        "shard": str(shard_path.relative_to(ROOT)),
        "shard_sha256": shard_hash,
        "spec_path": str(spec_path.relative_to(ROOT)),
        "spec_sha256": file_sha256(spec_path),
        "trial_plan": str(plan_path.relative_to(ROOT)),
        "trial_plan_sha256": file_sha256(plan_path),
        "queued_primary_units": len(queue),
        "planned_trials": len(plan),
        "by_agent": dict(Counter(r["agent_id"] for r in plan)),
        "by_channel": dict(Counter(r["channel"] for r in plan)),
        "by_label": dict(Counter(r["attack_label"] for r in plan)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    analysis_path = ROOT / "results" / "analysis" / f"{ANALYSIS_ID}.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis["artifacts"].append(str(analysis_path.relative_to(ROOT)))
    analysis_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_path = REPORTS / ANALYSIS_ID / "report.md"
    write_report(analysis, queue, report_path)

    print(
        json.dumps(
            {
                "analysis": str(analysis_path.relative_to(ROOT)),
                "report": str(report_path.relative_to(ROOT)),
                "queue": str(queue_path.relative_to(ROOT)),
                "shard": str(shard_path.relative_to(ROOT)),
                "spec": str(spec_path.relative_to(ROOT)),
                "trial_plan": str(plan_path.relative_to(ROOT)),
                "queued_primary_units": len(queue),
                "planned_confirmation_trials": len(plan),
                "confirmation_run_id": CONFIRM_RUN_ID,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
