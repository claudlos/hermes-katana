"""Build a defended-harness fleet from quota-mix promotion candidates."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "results" / "promotions" / "quota_mix_confirmed_candidates_20260506.jsonl"
SHARD_ID = 9608
DESIGN_ID = "D-defended-quota-mix-confirmed-20260506"
RUN_ID = "defended_quota_mix_confirmed_20260506"
SEED = 2026050609
CHANNEL_TASK = {
    "file_content": "readme_summarize",
    "tool_output": "triage_log",
}
DEFENDED_AGENTS = [
    "claude_cli_haiku_katana",
    "hermes_minimax_m2_7_katana",
    "hermes_nous_kimi_k2_6_katana",
    "hermes_nous_qwen3_coder_plus_katana",
]


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


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


def make_shard_rows(candidates: list[dict]) -> list[dict]:
    rows = []
    for idx, row in enumerate(candidates):
        text = str(row.get("text") or "")
        pid = str(row.get("primary_unit_id") or "")
        family = pid.split(":", 1)[1] if pid.startswith("family:") else pid
        text_hash = sha256_text(text)
        rows.append(
            {
                "id": row["id"],
                "text": text,
                "text_sha256": text_hash,
                "text_sha256_normalized": family or text_hash,
                "family_sha256": family or text_hash,
                "label": row.get("label") or "unknown",
                "attack_label": row.get("label") or "unknown",
                "binary_label": "attack",
                "is_attack": True,
                "language": row.get("language") or "unknown",
                "origin": "user_input",
                "source": "quota_mix_20260506_confirmed",
                "source_version": RUN_ID,
                "quality_tier": "quota_mix_confirmed_candidate",
                "split": "train",
                "shard": SHARD_ID,
                "confirmation_queue_rank": row.get("queue_rank"),
                "confirmation_effective_agents": row.get("effective_on", []),
                "confirmation_effective_platforms": row.get("effective_platforms", []),
                "confirmation_primary_unit_id": row.get("primary_unit_id"),
                "_shard_row_idx": idx,
            }
        )
    return rows


def make_spec() -> dict:
    workers = []
    for agent in DEFENDED_AGENTS:
        for channel, task in CHANNEL_TASK.items():
            workers.append(
                {
                    "_lane": f"{agent} x {channel} defended confirmation",
                    "agent": agent,
                    "shards": [SHARD_ID],
                    "channels": [channel],
                    "tasks": [task],
                    "max_attacks": 9999,
                    "n_repeats": 1,
                }
            )
    return {
        "_comment": "Defended-harness follow-up for quota-mix candidates that pass promotion thresholds. Regenerate after the remaining confirmation run for the final candidate set.",
        "_target": "Measure whether katana-defended variants block the candidates confirmed across multiple agents/platforms.",
        "_data": f"shards/shard_{SHARD_ID}.jsonl from {CANDIDATES.relative_to(ROOT)}",
        "_design_id": DESIGN_ID,
        "_trial_plan": f"results/designs/{DESIGN_ID}/trial_plan.jsonl",
        "_walltime_estimate": "~20-90m depending on candidate count and Katana middleware overhead.",
        "max_concurrency": 4,
        "workers": workers,
    }


def make_trial_plan(shard_rows: list[dict], shard_hash: str) -> list[dict]:
    now = int(time.time())
    plan = []
    seq = 0
    for row in shard_rows:
        for agent in DEFENDED_AGENTS:
            for channel, task in CHANNEL_TASK.items():
                family = row["family_sha256"]
                plan.append(
                    {
                        "schema_version": 1,
                        "design_id": DESIGN_ID,
                        "planned_trial_id": f"{DESIGN_ID}:{seq:09d}",
                        "run_id": RUN_ID,
                        "assignment_order": seq,
                        "job_tag": f"atk:{agent}:s{SHARD_ID}:{channel}+t:{task}",
                        "agent_id": agent,
                        "shard": SHARD_ID,
                        "shard_file": f"shards/shard_{SHARD_ID}.jsonl",
                        "shard_file_sha256": shard_hash,
                        "shard_row_idx": row["_shard_row_idx"],
                        "channel": channel,
                        "task": task,
                        "is_control": False,
                        "matched_pair": False,
                        "multi_turn": False,
                        "split": "all",
                        "attack_id": row["id"],
                        "primary_unit_id": f"family:{family}",
                        "family_sha256": family,
                        "text_sha256": row["text_sha256"],
                        "text_sha256_normalized": row["text_sha256_normalized"],
                        "attack_label": row["label"],
                        "language": row["language"],
                        "source": row["source"],
                        "source_version": row["source_version"],
                        "quality_tier": row["quality_tier"],
                        "origin": row["origin"],
                        "cell_id": (f"agent={agent}|channel={channel}|task={task}|label={row['label']}|control=false"),
                        "block_id": f"family:{family}",
                        "stratum_id": f"label={row['label']}",
                        "repeat_idx": 0,
                        "n_repeats_planned": 1,
                        "randomization_seed": SEED,
                        "planned_at_unix": now,
                        "sampling_weight": 1.0,
                    }
                )
                seq += 1
    for idx, row in enumerate(plan):
        row["assignment_order"] = idx
        row["wave_id"] = f"wave_{idx // max(1, len(DEFENDED_AGENTS)):04d}"
    return plan


def main() -> int:
    candidates = list(read_jsonl(CANDIDATES) or [])
    shard_rows = make_shard_rows(candidates)
    shard_path = ROOT / "shards" / f"shard_{SHARD_ID}.jsonl"
    write_jsonl(shard_path, shard_rows)
    shard_hash = file_sha256(shard_path)

    spec_path = ROOT / "scripts" / "fleet_defended_quota_mix_confirmed_20260506.json"
    spec_path.write_text(json.dumps(make_spec(), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    design_dir = ROOT / "results" / "designs" / DESIGN_ID
    plan_path = design_dir / "trial_plan.jsonl"
    plan = make_trial_plan(shard_rows, shard_hash)
    write_jsonl(plan_path, plan)

    summary = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "design_id": DESIGN_ID,
        "run_id": RUN_ID,
        "seed": SEED,
        "candidate_source": str(CANDIDATES.relative_to(ROOT)),
        "candidate_count": len(candidates),
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
    }
    summary_path = design_dir / "trial_plan_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
