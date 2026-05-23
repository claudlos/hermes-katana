"""Build a confirmation queue from the quota-mix fresh fleet.

Inputs:
  - results/agent_shard_runs/*quota_mix_fresh_20260505_2158*.jsonl

Outputs:
  - results/queues/confirmation_queue_20260506_quota_mix_hits.jsonl
  - shards/shard_9605.jsonl
  - scripts/fleet_confirm_quota_mix_20260506.json
  - results/designs/D-confirm-quota-mix-20260506/trial_plan.jsonl
  - results/designs/D-confirm-quota-mix-20260506/trial_plan_summary.json

The first queue block is reserved for candidates that reproduced on at least
two agents in the source run.
"""

from __future__ import annotations

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
QUEUES = ROOT / "results" / "queues"
SHARDS = ROOT / "shards"

SOURCE_RUN_ID = "quota_mix_fresh_20260505_2158"
SOURCE_DESIGN_ID = "D-quota-mix-fresh-20260505"

QUEUE_NAME = "confirmation_queue_20260506_quota_mix_hits"
CONFIRM_DESIGN_ID = "D-confirm-quota-mix-20260506"
CONFIRM_RUN_ID = "confirm_quota_mix_20260506"
CONFIRM_SHARD_ID = 9605
SEED = 2026050601

CHANNEL_TASK = {
    "file_content": "readme_summarize",
    "tool_output": "triage_log",
}

# Keep the confirmation plan broad enough to test portability, but avoid
# spending a large denominator on lanes that produced no signal and had
# obvious infra churn in the source run.
CONFIRM_AGENTS = [
    "codex_cli",
    "claude_cli_haiku",
    "hermes_minimax_m2_7",
    "hermes_minimax_m2_5",
    "hermes_nous_kimi_k2_6",
    "hermes_nous_qwen3_coder_plus",
]


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="ignore") as f:
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
    return value if value.startswith("family:") else f"family:{value}"


def row_label(row: dict) -> str:
    return str(row.get("attack_label") or row.get("label") or "unknown")


def row_text(row: dict) -> str:
    return str(row.get("attack_text") or row.get("text") or "")


def load_source_rows() -> list[dict]:
    rows: list[dict] = []
    for path in sorted(AGENT_RUNS.glob("shard_9604_*__run_quota_mix_fresh_20260505_2158.jsonl")):
        for row in read_jsonl(path) or []:
            if row.get("run_id") == SOURCE_RUN_ID:
                row["_source_file"] = str(path.relative_to(ROOT))
                rows.append(row)
    return rows


def new_record(row: dict, pid: str) -> dict:
    return {
        "primary_unit_id": pid,
        "attack_ids": set(),
        "label": row_label(row),
        "text": row_text(row),
        "language": row.get("language") or "unknown",
        "source": row.get("source") or "unknown",
        "quality_tier": row.get("quality_tier") or "unknown",
        "valid_rows": 0,
        "invalid_rows": 0,
        "effective_rows": 0,
        "tested_agents": set(),
        "tested_channels": set(),
        "effective_agents": set(),
        "effective_channels": set(),
        "effective_agent_channels": set(),
        "top_signals": Counter(),
        "reason_counts": Counter(),
        "max_severity": 0,
        "canary_hits": 0,
        "collapsed_hits": 0,
        "refusal_spike_hits": 0,
        "evidence_files": set(),
    }


def queue_score(row: dict) -> int:
    agent_count = int(row["effective_agent_count"])
    channel_count = int(row["effective_channel_count"])
    reproduced = int(bool(row["multi_agent_reproduced"]))
    return (
        10_000 * reproduced
        + 700 * agent_count
        + 250 * channel_count
        + 100 * int(row["both_channels_effective"])
        + 25 * int(row["canary_hits"])
        + 20 * int(row["collapsed_hits"])
        + int(row["max_severity"])
        + int(row["effective_rows"])
    )


def build_queue(rows: list[dict]) -> list[dict]:
    records: dict[str, dict] = {}
    for row in rows:
        pid = primary_id(row)
        rec = records.setdefault(pid, new_record(row, pid))
        rec["attack_ids"].add(str(row.get("attack_id") or ""))
        rec["tested_agents"].add(str(row.get("agent_id") or "unknown"))
        rec["tested_channels"].add(str(row.get("channel") or "unknown"))
        rec["evidence_files"].add(str(row.get("_source_file") or ""))
        if not is_valid(row):
            rec["invalid_rows"] += 1
            continue
        rec["valid_rows"] += 1
        rec["max_severity"] = max(rec["max_severity"], int(row.get("severity") or 0))
        if not row.get("effective"):
            continue
        rec["effective_rows"] += 1
        rec["effective_agents"].add(str(row.get("agent_id") or "unknown"))
        rec["effective_channels"].add(str(row.get("channel") or "unknown"))
        rec["effective_agent_channels"].add(f"{row.get('agent_id') or 'unknown'}:{row.get('channel') or 'unknown'}")
        rec["canary_hits"] += int(bool(row.get("canary_leaked")))
        rec["collapsed_hits"] += int(bool(row.get("collapsed")))
        rec["refusal_spike_hits"] += int(bool(row.get("refusal_spike")))
        signal = str(row.get("severity_top_signal") or "unknown")
        rec["top_signals"][signal] += 1
        for reason in row.get("reasons") or []:
            rec["reason_counts"][str(reason)] += 1

    queue: list[dict] = []
    for rec in records.values():
        if rec["effective_rows"] <= 0:
            continue
        out = {
            "queue_name": QUEUE_NAME,
            "source_run_id": SOURCE_RUN_ID,
            "source_design_id": SOURCE_DESIGN_ID,
            "primary_unit_id": rec["primary_unit_id"],
            "attack_ids": sorted(x for x in rec["attack_ids"] if x),
            "label": rec["label"],
            "text": rec["text"],
            "language": rec["language"],
            "source": rec["source"],
            "quality_tier": rec["quality_tier"],
            "valid_rows": rec["valid_rows"],
            "invalid_rows": rec["invalid_rows"],
            "effective_rows": rec["effective_rows"],
            "tested_agents": sorted(rec["tested_agents"]),
            "tested_channels": sorted(rec["tested_channels"]),
            "effective_agents": sorted(rec["effective_agents"]),
            "effective_channels": sorted(rec["effective_channels"]),
            "effective_agent_channels": sorted(rec["effective_agent_channels"]),
            "effective_agent_count": len(rec["effective_agents"]),
            "effective_channel_count": len(rec["effective_channels"]),
            "multi_agent_reproduced": len(rec["effective_agents"]) >= 2,
            "both_channels_effective": {"file_content", "tool_output"}.issubset(rec["effective_channels"]),
            "top_signals": dict(rec["top_signals"].most_common()),
            "reason_counts": dict(rec["reason_counts"].most_common()),
            "max_severity": rec["max_severity"],
            "canary_hits": rec["canary_hits"],
            "collapsed_hits": rec["collapsed_hits"],
            "refusal_spike_hits": rec["refusal_spike_hits"],
            "evidence_files": sorted(x for x in rec["evidence_files"] if x),
        }
        out["score"] = queue_score(out)
        queue.append(out)

    queue.sort(
        key=lambda r: (
            not r["multi_agent_reproduced"],
            -int(r["score"]),
            -int(r["effective_agent_count"]),
            -int(r["effective_channel_count"]),
            -int(r["effective_rows"]),
            r["primary_unit_id"],
        )
    )
    for idx, row in enumerate(queue, start=1):
        row["queue_rank"] = idx
        row["priority_block"] = "multi_agent_reproduced" if row["multi_agent_reproduced"] else "single_agent_hit"
    return queue


def make_shard_rows(queue: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for idx, item in enumerate(queue):
        text = item.get("text") or ""
        text_sha = sha256_text(text)
        pid = str(item["primary_unit_id"])
        family = pid.split(":", 1)[1] if pid.startswith("family:") else pid
        attack_id = item["attack_ids"][0] if item.get("attack_ids") else f"atk_{family[:16]}"
        rows.append(
            {
                "id": attack_id,
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
                "source_version": SOURCE_RUN_ID,
                "quality_tier": "quota_mix_effective_queue",
                "split": "train",
                "shard": CONFIRM_SHARD_ID,
                "confirmation_queue_rank": item["queue_rank"],
                "confirmation_queue_score": item["score"],
                "confirmation_priority_block": item["priority_block"],
                "confirmation_source_run": SOURCE_RUN_ID,
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
                    "_lane": f"{agent} x {channel} quota-mix confirmation queue",
                    "agent": agent,
                    "shards": [CONFIRM_SHARD_ID],
                    "channels": [channel],
                    "tasks": [task],
                    "max_attacks": 9999,
                    "n_repeats": 1,
                }
            )
    return {
        "_comment": "Focused confirmation fleet from quota_mix_fresh_20260505_2158 effective hits. Queue ranks 1-9 are the units that reproduced on >=2 agents in the source run.",
        "_target": "Confirm 74 unique effective primary units, prioritizing multi-agent reproductions first.",
        "_data": f"shards/shard_{CONFIRM_SHARD_ID}.jsonl from results/queues/{QUEUE_NAME}.jsonl",
        "_design_id": CONFIRM_DESIGN_ID,
        "_trial_plan": f"results/designs/{CONFIRM_DESIGN_ID}/trial_plan.jsonl",
        "_walltime_estimate": "~1.5-4h depending on Codex/Claude/MiniMax/Kimi tails; Copilot and OR-free omitted from the confirmation denominator.",
        "max_concurrency": 6,
        "workers": workers,
    }


def make_trial_plan(queue: list[dict], shard_rows: list[dict], shard_hash: str) -> list[dict]:
    rows_by_primary = {row["confirmation_primary_unit_id"]: idx for idx, row in enumerate(shard_rows)}
    plan: list[dict] = []
    now = int(time.time())
    seq = 0
    for item in queue:
        # Confirm on the channel(s) where the candidate actually hit. This keeps
        # the denominator tight and respects the queue ranking.
        channels = [c for c in item["effective_channels"] if c in CHANNEL_TASK]
        if not channels:
            continue
        shard_idx = rows_by_primary[item["primary_unit_id"]]
        shard_row = shard_rows[shard_idx]
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
                        "family_sha256": shard_row["family_sha256"],
                        "text_sha256": shard_row["text_sha256"],
                        "text_sha256_normalized": shard_row["family_sha256"],
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
                        "priority_block": item["priority_block"],
                    }
                )
                seq += 1

    multi = [row for row in plan if row["priority_block"] == "multi_agent_reproduced"]
    single = [row for row in plan if row["priority_block"] != "multi_agent_reproduced"]
    rng = random.Random(SEED)
    rng.shuffle(multi)
    rng.shuffle(single)
    plan = multi + single
    for idx, row in enumerate(plan):
        row["assignment_order"] = idx
        row["wave_id"] = f"wave_{idx // 6:04d}"
    return plan


def main() -> int:
    rows = load_source_rows()
    if not rows:
        raise SystemExit(f"no rows found for {SOURCE_RUN_ID}")

    source_plan = list(read_jsonl(DESIGNS / SOURCE_DESIGN_ID / "trial_plan.jsonl") or [])
    queue = build_queue(rows)
    shard_rows = make_shard_rows(queue)

    queue_path = QUEUES / f"{QUEUE_NAME}.jsonl"
    shard_path = SHARDS / f"shard_{CONFIRM_SHARD_ID}.jsonl"
    spec_path = ROOT / "scripts" / "fleet_confirm_quota_mix_20260506.json"
    design_dir = DESIGNS / CONFIRM_DESIGN_ID
    plan_path = design_dir / "trial_plan.jsonl"
    summary_path = design_dir / "trial_plan_summary.json"

    write_jsonl(queue_path, queue)
    write_jsonl(shard_path, shard_rows)
    shard_hash = file_sha256(shard_path)
    spec_path.write_text(json.dumps(make_spec(), indent=2, sort_keys=False) + "\n")
    plan = make_trial_plan(queue, shard_rows, shard_hash)
    write_jsonl(plan_path, plan)

    effective_rows = [r for r in rows if is_valid(r) and r.get("effective")]
    invalid_rows = [r for r in rows if not is_valid(r)]
    summary = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "git_head": git_head(),
        "source_run_id": SOURCE_RUN_ID,
        "source_design_id": SOURCE_DESIGN_ID,
        "source_rows": len(rows),
        "source_planned_trials": len(source_plan),
        "source_effective_rows": len(effective_rows),
        "source_invalid_rows": len(invalid_rows),
        "queued_primary_units": len(queue),
        "multi_agent_reproduced_units": sum(1 for r in queue if r["multi_agent_reproduced"]),
        "single_agent_hit_units": sum(1 for r in queue if not r["multi_agent_reproduced"]),
        "design_id": CONFIRM_DESIGN_ID,
        "run_id": CONFIRM_RUN_ID,
        "seed": SEED,
        "queue": str(queue_path.relative_to(ROOT)),
        "queue_sha256": file_sha256(queue_path),
        "shard": str(shard_path.relative_to(ROOT)),
        "shard_sha256": shard_hash,
        "spec_path": str(spec_path.relative_to(ROOT)),
        "spec_sha256": file_sha256(spec_path),
        "trial_plan": str(plan_path.relative_to(ROOT)),
        "trial_plan_sha256": file_sha256(plan_path),
        "planned_trials": len(plan),
        "by_agent": dict(Counter(r["agent_id"] for r in plan)),
        "by_channel": dict(Counter(r["channel"] for r in plan)),
        "by_label": dict(Counter(r["attack_label"] for r in plan)),
        "priority_block_trials": dict(Counter(r["priority_block"] for r in plan)),
        "confirm_agents": CONFIRM_AGENTS,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
