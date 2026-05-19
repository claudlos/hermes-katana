"""Artifact/contamination audit for the quota-mix confirmation queue."""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUEUE_PATH = ROOT / "results" / "queues" / "confirmation_queue_20260506_quota_mix_hits.jsonl"
DESIGNS = ROOT / "results" / "designs"
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
OUT_DIR = ROOT / "results" / "audits" / "quota_mix_20260506"
OWN_DESIGN_PREFIXES = (
    "D-quota-mix-fresh-20260505",
    "D-confirm-quota-mix",
)
CONFIRMED = [
    ROOT / "results" / "confirmed_attacks.jsonl",
    ROOT / "results" / "confirmed_attacks.v1_legacy.jsonl",
    ROOT / "results" / "synth_smoke_confirmed.jsonl",
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


def text(row: dict) -> str:
    return str(row.get("text") or row.get("attack_text") or "")


def primary_id(row: dict) -> str:
    value = (
        row.get("primary_unit_id")
        or row.get("family_sha256")
        or row.get("text_sha256_normalized")
        or row.get("attack_text_sha256")
        or row.get("text_sha256")
        or row.get("id")
        or row.get("attack_id")
        or ""
    )
    value = str(value)
    return value if value.startswith("family:") else f"family:{value}"


def tokens(value: str) -> list[str]:
    return re.findall(r"[\w']+", value.lower())


def shingles(value: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = tokens(value)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_reference_keys() -> dict:
    prior: dict[str, set[str]] = {
        "confirmed_primary": set(),
        "confirmed_ids": set(),
        "plan_primary": set(),
        "observed_primary": set(),
    }
    for path in CONFIRMED:
        for row in read_jsonl(path) or []:
            prior["confirmed_primary"].add(primary_id(row))
            if row.get("id"):
                prior["confirmed_ids"].add(str(row["id"]))
    for path in DESIGNS.glob("*/trial_plan.jsonl"):
        design_name = path.parent.name
        if any(design_name.startswith(prefix) for prefix in OWN_DESIGN_PREFIXES):
            continue
        for row in read_jsonl(path) or []:
            prior["plan_primary"].add(primary_id(row))
    for path in AGENT_RUNS.glob("shard_*.jsonl"):
        if "quota_mix_fresh_20260505_2158" in path.name:
            continue
        if "confirm_quota_mix" in path.name:
            continue
        for row in read_jsonl(path) or []:
            prior["observed_primary"].add(primary_id(row))
    return prior


def main() -> int:
    queue = list(read_jsonl(QUEUE_PATH) or [])
    refs = load_reference_keys()
    by_primary = Counter(primary_id(row) for row in queue)
    by_attack_id = Counter(aid for row in queue for aid in row.get("attack_ids", []))
    text_shingles = [(row, shingles(text(row))) for row in queue]

    near_pairs = []
    for i, (left, left_sh) in enumerate(text_shingles):
        for right, right_sh in text_shingles[i + 1 :]:
            score = jaccard(left_sh, right_sh)
            if score >= 0.8:
                near_pairs.append(
                    {
                        "left_rank": left.get("queue_rank"),
                        "right_rank": right.get("queue_rank"),
                        "left_primary_unit_id": primary_id(left),
                        "right_primary_unit_id": primary_id(right),
                        "jaccard_5gram": round(score, 4),
                    }
                )

    overlaps = {
        "confirmed_primary": sorted(primary_id(row) for row in queue if primary_id(row) in refs["confirmed_primary"]),
        "confirmed_ids": sorted(
            aid for row in queue for aid in row.get("attack_ids", []) if aid in refs["confirmed_ids"]
        ),
        "prior_plan_primary": sorted(primary_id(row) for row in queue if primary_id(row) in refs["plan_primary"]),
        "prior_observed_primary": sorted(
            primary_id(row) for row in queue if primary_id(row) in refs["observed_primary"]
        ),
    }
    canary_only = []
    for row in queue:
        reasons = set(row.get("reason_counts") or row.get("top_signals") or {})
        top = set(row.get("top_signals") or {})
        if row.get("effective_rows", 0) > 0 and (top == {"canary"} or reasons == {"CANARY"}):
            canary_only.append(
                {
                    "queue_rank": row.get("queue_rank"),
                    "primary_unit_id": primary_id(row),
                    "label": row.get("label"),
                    "effective_rows": row.get("effective_rows"),
                    "effective_agents": row.get("effective_agents"),
                }
            )

    report = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "queue": str(QUEUE_PATH.relative_to(ROOT)),
        "queue_rows": len(queue),
        "duplicate_primary_units": {k: v for k, v in by_primary.items() if v > 1},
        "duplicate_attack_ids": {k: v for k, v in by_attack_id.items() if v > 1},
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "overlaps": overlaps,
        "near_duplicate_pairs_threshold_0_8": near_pairs,
        "canary_only_evidence_rows": canary_only,
        "by_label": dict(Counter(row.get("label", "unknown") for row in queue)),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "audit.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md = [
        "# Quota-mix queue audit",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        f"Queue rows: {len(queue)}",
        "",
        "| check | count |",
        "| --- | ---: |",
        f"| duplicate primary units | {len(report['duplicate_primary_units'])} |",
        f"| duplicate attack ids | {len(report['duplicate_attack_ids'])} |",
        f"| confirmed primary overlap | {report['overlap_counts']['confirmed_primary']} |",
        f"| confirmed id overlap | {report['overlap_counts']['confirmed_ids']} |",
        f"| prior plan primary overlap | {report['overlap_counts']['prior_plan_primary']} |",
        f"| prior observed primary overlap | {report['overlap_counts']['prior_observed_primary']} |",
        f"| near-duplicate queue pairs >=0.8 | {len(near_pairs)} |",
        f"| canary-only evidence rows | {len(canary_only)} |",
        "",
        "## Canary-only Evidence",
        "",
        "| rank | label | effective rows | agents | primary unit |",
        "| ---: | --- | ---: | --- | --- |",
    ]
    for row in canary_only[:40]:
        md.append(
            f"| {row['queue_rank']} | `{row['label']}` | {row['effective_rows']} | "
            f"`{','.join(row['effective_agents'])}` | `{row['primary_unit_id']}` |"
        )
    md.append("")
    (OUT_DIR / "audit.md").write_text("\n".join(md), encoding="utf-8")
    print(
        json.dumps(
            {
                "audit": str(json_path.relative_to(ROOT)),
                **report["overlap_counts"],
                "near_pairs": len(near_pairs),
                "canary_only": len(canary_only),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
