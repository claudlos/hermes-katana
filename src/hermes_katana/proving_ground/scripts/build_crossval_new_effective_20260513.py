#!/usr/bin/env python3
"""Build a focused cross-validation trial plan from newly effective v8 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = (
    ROOT.parent
    / "hermes-katana"
    / "results"
    / "derived_20260513_stop_snapshot"
    / "worker_valid_rows_v8_main_20260511.jsonl"
)
DEFAULT_SOURCE_RUN_ID = "v8_main_20260511"
DEFAULT_RUN_ID = "v8_crossval_new_effective_20260513"
DEFAULT_DESIGN_ID = "D-crossval-new-effective-20260513"
DEFAULT_AGENTS = [
    "codex_cli",
    "codex_cli_spark",
    "hermes_nous_step_flash",
    "hermes_or_nemotron_3_nano_30b_free",
]


def read_jsonl(path: Path):
    with path.open(errors="ignore") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def invalid_reason(row: dict) -> str | None:
    if row.get("row_valid") is False:
        return str(row.get("invalid_reason") or "row_valid_false")
    if row.get("invalid_run") is True:
        return str(row.get("invalid_reason") or "invalid_run")
    if row.get("baseline_valid") is False:
        return str(row.get("baseline_invalid_reason") or "baseline_invalid")
    if row.get("attack_run_valid") is False:
        return str(row.get("attack_run_invalid_reason") or "attack_run_invalid")
    return None


def effective_key(row: dict) -> tuple:
    return (
        row.get("agent_id"),
        int(row.get("shard", -1)),
        row.get("channel"),
        row.get("task"),
        row.get("attack_id"),
        int(row.get("repeat_idx", 0)),
    )


def load_snapshot_effective_keys(path: Path) -> set[tuple]:
    keys = set()
    for row in read_jsonl(path):
        if row.get("effective"):
            keys.add(effective_key(row))
    return keys


def load_current_new_effective(source_run_id: str, snapshot_keys: set[tuple]) -> list[dict]:
    rows = []
    run_dir = ROOT / "results" / "agent_shard_runs"
    for path in sorted(run_dir.glob(f"*run_{source_run_id}.jsonl")):
        for row in read_jsonl(path):
            if row.get("run_id") != source_run_id:
                continue
            if invalid_reason(row) or not row.get("effective"):
                continue
            if effective_key(row) in snapshot_keys:
                continue
            row = dict(row)
            row["_source_file"] = str(path)
            rows.append(row)
    return rows


def load_shard_index(shard: int) -> dict[str, tuple[int, dict]]:
    path = ROOT / "shards" / f"shard_{shard:03d}.jsonl"
    out = {}
    with path.open() as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("id") or row.get("attack_id"))] = (idx, row)
    return out


def unique_attack_channel_pairs(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("attack_id")),
                int(row.get("shard")),
                str(row.get("channel")),
                str(row.get("task") or "secrets_audit"),
            )
        ].append(row)
    out = []
    for (_, _, _, _), items in grouped.items():
        first = items[0]
        first = dict(first)
        first["_effective_source_agents"] = sorted({str(r.get("agent_id")) for r in items})
        first["_effective_source_rows"] = len(items)
        first["_canary_source_rows"] = sum(1 for r in items if r.get("canary_leaked"))
        out.append(first)
    return out


def balanced_select(rows: list[dict], max_pairs: int | None) -> list[dict]:
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row.get("attack_label") or "unknown", row.get("channel") or "unknown")].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: (int(r.get("shard", 0)), str(r.get("attack_id"))), reverse=True)

    selected = []
    keys = sorted(buckets, key=lambda k: (-len(buckets[k]), k))
    while keys and (max_pairs is None or len(selected) < max_pairs):
        next_keys = []
        for key in keys:
            if max_pairs is not None and len(selected) >= max_pairs:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop(0))
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def make_plan(selected: list[dict], agents: list[str], run_id: str, design_id: str) -> list[dict]:
    shard_indexes: dict[int, dict[str, tuple[int, dict]]] = {}
    shard_hashes: dict[int, str] = {}
    plan = []
    now = int(time.time())
    seq = 0
    for source_rank, item in enumerate(selected):
        shard = int(item["shard"])
        if shard not in shard_indexes:
            shard_path = ROOT / "shards" / f"shard_{shard:03d}.jsonl"
            shard_indexes[shard] = load_shard_index(shard)
            shard_hashes[shard] = file_sha256(shard_path)
        attack_id = str(item["attack_id"])
        if attack_id not in shard_indexes[shard]:
            continue
        shard_idx, shard_row = shard_indexes[shard][attack_id]
        channel = str(item.get("channel") or "file_content")
        task = str(item.get("task") or "secrets_audit")
        for agent in agents:
            plan.append(
                {
                    "schema_version": 1,
                    "design_id": design_id,
                    "planned_trial_id": f"{design_id}:{seq:09d}",
                    "run_id": run_id,
                    "assignment_order": seq,
                    "source_rank": source_rank,
                    "source_run_id": DEFAULT_SOURCE_RUN_ID,
                    "source_effective_agents": item.get("_effective_source_agents", []),
                    "source_effective_rows": item.get("_effective_source_rows", 1),
                    "source_canary_rows": item.get("_canary_source_rows", 0),
                    "job_tag": f"atk:{agent}:s{shard:03d}:{channel}+t:{task}",
                    "agent_id": agent,
                    "shard": shard,
                    "shard_file": f"shards/shard_{shard:03d}.jsonl",
                    "shard_file_sha256": shard_hashes[shard],
                    "shard_row_idx": shard_idx,
                    "channel": channel,
                    "task": task,
                    "is_control": False,
                    "matched_pair": False,
                    "multi_turn": False,
                    "split": "all",
                    "attack_id": attack_id,
                    "family_sha256": shard_row.get("family_sha256"),
                    "text_sha256": shard_row.get("text_sha256"),
                    "text_sha256_normalized": shard_row.get("text_sha256_normalized") or shard_row.get("family_sha256"),
                    "attack_label": shard_row.get("label") or item.get("attack_label"),
                    "language": shard_row.get("language"),
                    "source": shard_row.get("source"),
                    "source_version": shard_row.get("source_version"),
                    "quality_tier": shard_row.get("quality_tier"),
                    "origin": shard_row.get("origin"),
                    "cell_id": (
                        f"agent={agent}|channel={channel}|task={task}|"
                        f"label={shard_row.get('label') or item.get('attack_label')}|control=false"
                    ),
                    "block_id": f"{attack_id}|{channel}|{task}",
                    "stratum_id": f"label={shard_row.get('label') or item.get('attack_label')}",
                    "repeat_idx": 0,
                    "n_repeats_planned": 1,
                    "planned_at_unix": now,
                    "sampling_weight": 1.0,
                }
            )
            seq += 1
    return plan


def make_spec(
    plan: list[dict],
    agents: list[str],
    run_id: str,
    design_id: str,
    plan_path: Path,
    max_concurrency: int,
) -> dict:
    combos = sorted(
        {
            (row["agent_id"], int(row["shard"]), row["channel"], row["task"])
            for row in plan
            if row["agent_id"] in agents
        },
        key=lambda item: (item[1], item[2], item[3], agents.index(item[0])),
    )
    workers = []
    for agent, shard, channel, task in combos:
        workers.append(
            {
                "_lane": f"{agent} shard {shard} {channel} {task}",
                "agent": agent,
                "shards": [shard],
                "channels": [channel],
                "tasks": [task],
                "max_attacks": 9999,
                "n_repeats": 1,
            }
        )
    return {
        "_comment": "First-wave cross-validation of newly effective v8 MiniMax rows on Codex/Spark/free-router agents.",
        "_design_id": design_id,
        "_trial_plan": str(plan_path.relative_to(ROOT)),
        "_source_run_id": DEFAULT_SOURCE_RUN_ID,
        "_run_id": run_id,
        "max_concurrency": max_concurrency,
        "workers": workers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=DEFAULT_SOURCE_RUN_ID)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--design-id", default=DEFAULT_DESIGN_ID)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--agents", nargs="+", default=DEFAULT_AGENTS)
    args = parser.parse_args()

    snapshot_keys = load_snapshot_effective_keys(args.snapshot)
    new_rows = load_current_new_effective(args.source_run_id, snapshot_keys)
    pairs = unique_attack_channel_pairs(new_rows)
    selected = balanced_select(pairs, args.max_pairs)
    plan = make_plan(selected, args.agents, args.run_id, args.design_id)

    design_dir = ROOT / "results" / "designs" / args.design_id
    queue_path = ROOT / "results" / "queues" / f"{args.design_id}.source_pairs.jsonl"
    plan_path = design_dir / "trial_plan.jsonl"
    summary_path = design_dir / "trial_plan_summary.json"
    spec_path = ROOT / "scripts" / f"fleet_{args.design_id}.json"

    write_jsonl(queue_path, selected)
    write_jsonl(plan_path, plan)
    spec = make_spec(
        plan,
        args.agents,
        args.run_id,
        args.design_id,
        plan_path,
        args.max_concurrency,
    )
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")

    summary = {
        "source_run_id": args.source_run_id,
        "run_id": args.run_id,
        "design_id": args.design_id,
        "snapshot": str(args.snapshot),
        "new_effective_rows": len(new_rows),
        "new_unique_pairs": len(pairs),
        "selected_pairs": len(selected),
        "planned_trials": len(plan),
        "agents": args.agents,
        "labels_selected": dict(Counter(row.get("attack_label") for row in selected)),
        "channels_selected": dict(Counter(row.get("channel") for row in selected)),
        "shards_selected": dict(Counter(str(row.get("shard")) for row in selected)),
        "queue": str(queue_path.relative_to(ROOT)),
        "trial_plan": str(plan_path.relative_to(ROOT)),
        "spec": str(spec_path.relative_to(ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
