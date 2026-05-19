"""Build a confirmation queue from the 2026-05-06 free discovery runs.

Inputs:
  - free_reliable_discovery_20260506
  - free_booster_mix_20260506
  - copilot_gpt5mini_tail_20260506

Outputs:
  - results/queues/confirmation_queue_20260506_free_discovery_hits.jsonl
  - shards/shard_9609.jsonl
  - scripts/fleet_confirm_free_discovery_20260506.json
  - results/designs/D-confirm-free-discovery-20260506/trial_plan.jsonl
  - results/designs/D-confirm-free-discovery-20260506/trial_plan_summary.json
  - results/reports/free_discovery_20260506/report.md
  - results/reports/free_discovery_20260506/summary.json

The report intentionally avoids printing attack text. The queue/shard artifacts
retain text because the fleet runner needs executable inputs.
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
REPORT_DIR = ROOT / "results" / "reports" / "free_discovery_20260506"
SHARDS = ROOT / "shards"

SOURCE_RUNS = [
    "free_reliable_discovery_20260506",
    "free_booster_mix_20260506",
    "copilot_gpt5mini_tail_20260506",
]
SOURCE_DESIGNS = {
    "free_reliable_discovery_20260506": "D-free-reliable-discovery-20260506",
    "free_booster_mix_20260506": "D-free-booster-mix-20260506",
    "copilot_gpt5mini_tail_20260506": "D-copilot-gpt5mini-tail-20260506",
}

QUEUE_NAME = "confirmation_queue_20260506_free_discovery_hits"
CONFIRM_DESIGN_ID = "D-confirm-free-discovery-20260506"
CONFIRM_RUN_ID = "confirm_free_discovery_20260506"
CONFIRM_SHARD_ID = 9609
SEED = 2026050610

CHANNEL_TASK = {
    "file_content": "readme_summarize",
    "tool_output": "triage_log",
}

# Keep failed Nous and broken/rate-limited OR probes out. OR Nemotron nano is
# retained only on file_content because that is the lane that completed cleanly.
CONFIRM_AGENT_CHANNELS = {
    "hermes_minimax_m2_7": {"file_content", "tool_output"},
    "hermes_minimax_m2_5": {"file_content", "tool_output"},
    "copilot_cli": {"file_content", "tool_output"},
    "hermes_or_nemotron_3_nano_30b_free": {"file_content"},
}


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
        or sha256_text(row_text(row))
    )
    value = str(value)
    return value if value.startswith("family:") else f"family:{value}"


def row_label(row: dict) -> str:
    return str(row.get("attack_label") or row.get("label") or "unknown")


def row_text(row: dict) -> str:
    text = str(row.get("attack_text") or row.get("text") or "")
    if text:
        return text
    shard = row.get("shard")
    idx = row.get("shard_row_idx")
    if shard is None or idx is None:
        return ""
    shard_path = SHARDS / f"shard_{int(shard):04d}.jsonl"
    for i, shard_row in enumerate(read_jsonl(shard_path) or []):
        if i == int(idx):
            return str(shard_row.get("text") or shard_row.get("attack_text") or "")
    return ""


def load_run_rows(run_id: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(AGENT_RUNS.glob(f"shard_*__run_{run_id}.jsonl")):
        for row in read_jsonl(path) or []:
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
        "tested_rows": 0,
        "valid_rows": 0,
        "invalid_rows": 0,
        "effective_rows": 0,
        "source_runs": set(),
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
    return (
        10_000 * int(row["multi_agent_reproduced"])
        + 700 * agent_count
        + 250 * channel_count
        + 100 * int(row["both_channels_effective"])
        + 75 * int(row["canary_hits"])
        + 30 * int(row["collapsed_hits"])
        + int(row["max_severity"])
        + int(row["effective_rows"])
    )


def build_queue(rows: list[dict]) -> list[dict]:
    records: dict[str, dict] = {}
    for row in rows:
        pid = primary_id(row)
        rec = records.setdefault(pid, new_record(row, pid))
        rec["tested_rows"] += 1
        rec["attack_ids"].add(str(row.get("attack_id") or ""))
        rec["source_runs"].add(str(row.get("run_id") or "unknown"))
        rec["tested_agents"].add(str(row.get("agent_id") or "unknown"))
        rec["tested_channels"].add(str(row.get("channel") or "unknown"))
        rec["evidence_files"].add(str(row.get("_source_file") or ""))
        if not rec["text"]:
            rec["text"] = row_text(row)
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
        rec["top_signals"][str(row.get("severity_top_signal") or "unknown")] += 1
        for reason in row.get("reasons") or []:
            rec["reason_counts"][str(reason)] += 1

    queue: list[dict] = []
    for rec in records.values():
        if rec["effective_rows"] <= 0:
            continue
        out = {
            "queue_name": QUEUE_NAME,
            "source_run_ids": sorted(rec["source_runs"]),
            "source_design_ids": sorted(SOURCE_DESIGNS[r] for r in rec["source_runs"] if r in SOURCE_DESIGNS),
            "primary_unit_id": rec["primary_unit_id"],
            "attack_ids": sorted(x for x in rec["attack_ids"] if x),
            "label": rec["label"],
            "text": rec["text"],
            "language": rec["language"],
            "source": rec["source"],
            "quality_tier": rec["quality_tier"],
            "tested_rows": rec["tested_rows"],
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
                "source_version": "+".join(item.get("source_run_ids") or SOURCE_RUNS),
                "quality_tier": "free_discovery_effective_queue",
                "split": "train",
                "shard": CONFIRM_SHARD_ID,
                "confirmation_queue_rank": item["queue_rank"],
                "confirmation_queue_score": item["score"],
                "confirmation_priority_block": item["priority_block"],
                "confirmation_source_runs": item["source_run_ids"],
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
    for agent, channels in CONFIRM_AGENT_CHANNELS.items():
        for channel in sorted(channels):
            workers.append(
                {
                    "_lane": f"{agent} x {channel} free-discovery confirmation",
                    "agent": agent,
                    "shards": [CONFIRM_SHARD_ID],
                    "channels": [channel],
                    "tasks": [CHANNEL_TASK[channel]],
                    "max_attacks": 9999,
                    "n_repeats": 1,
                }
            )
    return {
        "_comment": "Focused confirmation fleet from 2026-05-06 free discovery effective hits. Failed Nous and broken/rate-limited OR probes are omitted.",
        "_target": "Confirm deduped effective primary units from free/Copilot/MiniMax discovery on the channels where they hit.",
        "_data": f"shards/shard_{CONFIRM_SHARD_ID}.jsonl from results/queues/{QUEUE_NAME}.jsonl",
        "_design_id": CONFIRM_DESIGN_ID,
        "_trial_plan": f"results/designs/{CONFIRM_DESIGN_ID}/trial_plan.jsonl",
        "_walltime_estimate": "~45-120m if launched now; Copilot is the slow lane.",
        "max_concurrency": 4,
        "workers": workers,
    }


def make_trial_plan(queue: list[dict], shard_rows: list[dict], shard_hash: str) -> list[dict]:
    rows_by_primary = {row["confirmation_primary_unit_id"]: idx for idx, row in enumerate(shard_rows)}
    plan: list[dict] = []
    now = int(time.time())
    seq = 0
    for item in queue:
        channels = [c for c in item["effective_channels"] if c in CHANNEL_TASK]
        if not channels:
            continue
        shard_idx = rows_by_primary[item["primary_unit_id"]]
        shard_row = shard_rows[shard_idx]
        for agent, allowed_channels in CONFIRM_AGENT_CHANNELS.items():
            for channel in channels:
                if channel not in allowed_channels:
                    continue
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
        row["wave_id"] = f"wave_{idx // 4:04d}"
    return plan


def summarize_run(rows: list[dict]) -> dict:
    valid = [r for r in rows if is_valid(r)]
    invalid = [r for r in rows if not is_valid(r)]
    effective = [r for r in valid if r.get("effective")]
    by_agent = Counter(str(r.get("agent_id") or "unknown") for r in rows)
    eff_by_agent = Counter(str(r.get("agent_id") or "unknown") for r in effective)
    invalid_by_agent = Counter(str(r.get("agent_id") or "unknown") for r in invalid)
    by_label = Counter(row_label(r) for r in rows)
    eff_by_label = Counter(row_label(r) for r in effective)
    return {
        "rows": len(rows),
        "valid": len(valid),
        "invalid": len(invalid),
        "effective_rows": len(effective),
        "effective_unique_units": len({primary_id(r) for r in effective}),
        "by_agent": {
            agent: {
                "rows": by_agent[agent],
                "effective": eff_by_agent[agent],
                "invalid": invalid_by_agent[agent],
            }
            for agent in sorted(by_agent)
        },
        "by_label": {lab: {"rows": by_label[lab], "effective": eff_by_label[lab]} for lab in sorted(by_label)},
    }


def write_report(summary: dict, queue: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Free discovery hit report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Summary",
        "",
        f"- Source rows: {summary['source_rows']:,}",
        f"- Valid rows: {summary['source_valid_rows']:,}",
        f"- Invalid rows: {summary['source_invalid_rows']:,}",
        f"- Effective rows: {summary['source_effective_rows']:,}",
        f"- Deduped effective units: {summary['queued_primary_units']:,}",
        f"- Multi-agent reproduced units: {summary['multi_agent_reproduced_units']:,}",
        f"- Confirmation plan rows: {summary['planned_trials']:,}",
        "",
        "## Runs",
        "",
        "| run | rows | valid | invalid | effective rows | effective units |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id in SOURCE_RUNS:
        s = summary["runs"][run_id]
        lines.append(
            f"| `{run_id}` | {s['rows']:,} | {s['valid']:,} | {s['invalid']:,} | "
            f"{s['effective_rows']:,} | {s['effective_unique_units']:,} |"
        )
    lines.extend(
        [
            "",
            "## Queue Leaders",
            "",
            "| rank | label | agents | channels | rows | signal | primary unit |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for item in queue[:30]:
        signal = ",".join(item["top_signals"].keys()) or "unknown"
        lines.append(
            f"| {item['queue_rank']} | `{item['label']}` | "
            f"`{','.join(item['effective_agents'])}` | "
            f"`{','.join(item['effective_channels'])}` | "
            f"{item['effective_rows']} | `{signal}` | `{item['primary_unit_id']}` |"
        )

    lines.extend(["", "## By Agent", ""])
    for run_id in SOURCE_RUNS:
        lines.extend(
            [
                f"### `{run_id}`",
                "",
                "| agent | rows | effective | invalid |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for agent, stats in sorted(
            summary["runs"][run_id]["by_agent"].items(),
            key=lambda kv: (-kv[1]["effective"], kv[0]),
        ):
            lines.append(f"| `{agent}` | {stats['rows']:,} | {stats['effective']:,} | {stats['invalid']:,} |")
        lines.append("")

    lines.extend(
        [
            "## Confirmation Artifacts",
            "",
            f"- Queue: `{summary['queue']}`",
            f"- Shard: `{summary['shard']}`",
            f"- Spec: `{summary['spec_path']}`",
            f"- Trial plan: `{summary['trial_plan']}`",
            "",
            "Failed Nous lanes and failed/rate-limited OR probes are intentionally omitted from the confirmation denominator. OR Nemotron nano is kept only for `file_content`, the lane that completed cleanly.",
            "",
        ]
    )
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows_by_run = {run_id: load_run_rows(run_id) for run_id in SOURCE_RUNS}
    source_rows = [row for rows in rows_by_run.values() for row in rows]
    if not source_rows:
        raise SystemExit("no source rows found")

    queue = build_queue(source_rows)
    shard_rows = make_shard_rows(queue)

    queue_path = QUEUES / f"{QUEUE_NAME}.jsonl"
    shard_path = SHARDS / f"shard_{CONFIRM_SHARD_ID}.jsonl"
    spec_path = ROOT / "scripts" / "fleet_confirm_free_discovery_20260506.json"
    design_dir = DESIGNS / CONFIRM_DESIGN_ID
    plan_path = design_dir / "trial_plan.jsonl"
    plan_summary_path = design_dir / "trial_plan_summary.json"
    report_summary_path = REPORT_DIR / "summary.json"

    write_jsonl(queue_path, queue)
    write_jsonl(shard_path, shard_rows)
    shard_hash = file_sha256(shard_path)
    spec_path.write_text(json.dumps(make_spec(), indent=2, sort_keys=False) + "\n")
    plan = make_trial_plan(queue, shard_rows, shard_hash)
    write_jsonl(plan_path, plan)

    valid_rows = [r for r in source_rows if is_valid(r)]
    invalid_rows = [r for r in source_rows if not is_valid(r)]
    effective_rows = [r for r in valid_rows if r.get("effective")]
    summary = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "git_head": git_head(),
        "source_run_ids": SOURCE_RUNS,
        "source_design_ids": SOURCE_DESIGNS,
        "source_rows": len(source_rows),
        "source_valid_rows": len(valid_rows),
        "source_invalid_rows": len(invalid_rows),
        "source_effective_rows": len(effective_rows),
        "source_effective_unique_units": len({primary_id(r) for r in effective_rows}),
        "queued_primary_units": len(queue),
        "multi_agent_reproduced_units": sum(1 for r in queue if r["multi_agent_reproduced"]),
        "single_agent_hit_units": sum(1 for r in queue if not r["multi_agent_reproduced"]),
        "runs": {run_id: summarize_run(rows) for run_id, rows in rows_by_run.items()},
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
        "confirm_agent_channels": {agent: sorted(channels) for agent, channels in CONFIRM_AGENT_CHANNELS.items()},
        "report": str((REPORT_DIR / "report.md").relative_to(ROOT)),
        "report_summary": str(report_summary_path.relative_to(ROOT)),
    }
    plan_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(summary, queue)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
