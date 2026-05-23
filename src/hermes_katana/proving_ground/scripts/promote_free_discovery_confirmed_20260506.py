"""Promote free-discovery confirmations into results/confirmed_attacks.jsonl.

Policy for this follow-up:
  - unit is present in confirmation_queue_20260506_free_discovery_hits.jsonl
  - at least one valid/effective row in confirm_free_discovery_20260506
  - not already present in results/confirmed_attacks.jsonl by id or family

The script also updates results/reports/free_discovery_20260506/report.md and
summary.json with the confirmation and promotion results.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
CONFIRMED_PATH = ROOT / "results" / "confirmed_attacks.jsonl"
QUEUE_PATH = ROOT / "results" / "queues" / "confirmation_queue_20260506_free_discovery_hits.jsonl"
REPORT_DIR = ROOT / "results" / "reports" / "free_discovery_20260506"
SUMMARY_PATH = REPORT_DIR / "summary.json"
REPORT_PATH = REPORT_DIR / "report.md"
OUT_DIR = ROOT / "results" / "promotions"

SOURCE_RUNS = [
    "free_reliable_discovery_20260506",
    "free_booster_mix_20260506",
    "copilot_gpt5mini_tail_20260506",
]
CONFIRM_RUN_ID = "confirm_free_discovery_20260506"
ALL_RUNS = SOURCE_RUNS + [CONFIRM_RUN_ID]


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
        or "unknown"
    )
    value = str(value)
    return value if value.startswith("family:") else f"family:{value}"


def label(row: dict) -> str:
    return str(row.get("attack_label") or row.get("label") or "unknown")


def agent_platform(agent_id: str) -> str:
    if agent_id.startswith("claude_cli"):
        return "claude_code_cli"
    if agent_id.startswith("codex_cli"):
        return "codex_cli"
    if agent_id == "copilot_cli":
        return "copilot_cli"
    if agent_id.startswith("hermes_or_") or ":free" in agent_id:
        return "openrouter_api_free"
    if agent_id.startswith("hermes_"):
        return "hermes_agent_cli"
    return "unknown"


def load_run_rows(run_id: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(AGENT_RUNS.glob(f"shard_*__run_{run_id}.jsonl")):
        for row in read_jsonl(path) or []:
            row["_source_file"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def existing_keys() -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    families: set[str] = set()
    for row in read_jsonl(CONFIRMED_PATH) or []:
        if row.get("id"):
            ids.add(str(row["id"]))
        family = (
            row.get("primary_unit_id")
            or row.get("family_sha256")
            or row.get("text_sha256_normalized")
            or row.get("text_sha256")
        )
        if family:
            family = str(family)
            families.add(family if family.startswith("family:") else f"family:{family}")
    return ids, families


def summarize_run(rows: list[dict]) -> dict:
    valid = [r for r in rows if is_valid(r)]
    invalid = [r for r in rows if not is_valid(r)]
    effective = [r for r in valid if r.get("effective")]
    by_agent = Counter(str(r.get("agent_id") or "unknown") for r in rows)
    eff_by_agent = Counter(str(r.get("agent_id") or "unknown") for r in effective)
    invalid_by_agent = Counter(str(r.get("agent_id") or "unknown") for r in invalid)
    by_label = Counter(label(r) for r in rows)
    eff_by_label = Counter(label(r) for r in effective)
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


def build_records(rows_by_run: dict[str, list[dict]], queue: list[dict]) -> dict[str, dict]:
    queue_by_pid = {str(q["primary_unit_id"]): q for q in queue}
    records: dict[str, dict] = {}
    for run_id, rows in rows_by_run.items():
        for row in rows:
            pid = primary_id(row)
            if pid not in queue_by_pid:
                continue
            q = queue_by_pid[pid]
            rec = records.setdefault(
                pid,
                {
                    "primary_unit_id": pid,
                    "id": (q.get("attack_ids") or [str(row.get("attack_id") or "")])[0],
                    "text": q.get("text") or row.get("attack_text") or row.get("text") or "",
                    "label": q.get("label") or label(row),
                    "queue_rank": q.get("queue_rank"),
                    "tested_by": set(),
                    "effective_on": set(),
                    "tested_platforms": set(),
                    "effective_platforms": set(),
                    "tested_runs": set(),
                    "effective_runs": set(),
                    "confirm_effective_on": set(),
                    "confirm_effective_platforms": set(),
                    "valid_rows": 0,
                    "invalid_rows": 0,
                    "effective_rows": 0,
                    "confirmation_effective_rows": 0,
                    "canary_hits": 0,
                    "collapsed_hits": 0,
                    "reason_counts": Counter(),
                    "top_signals": Counter(),
                    "evidence_files": set(),
                },
            )
            agent = str(row.get("agent_id") or "unknown")
            platform = agent_platform(agent)
            rec["tested_runs"].add(run_id)
            rec["tested_by"].add(agent)
            rec["tested_platforms"].add(platform)
            rec["evidence_files"].add(str(row.get("_source_file") or ""))
            if not is_valid(row):
                rec["invalid_rows"] += 1
                continue
            rec["valid_rows"] += 1
            if not row.get("effective"):
                continue
            rec["effective_rows"] += 1
            rec["effective_on"].add(agent)
            rec["effective_platforms"].add(platform)
            rec["effective_runs"].add(run_id)
            rec["canary_hits"] += int(bool(row.get("canary_leaked")))
            rec["collapsed_hits"] += int(bool(row.get("collapsed")))
            rec["top_signals"][str(row.get("severity_top_signal") or "unknown")] += 1
            for reason in row.get("reasons") or []:
                rec["reason_counts"][str(reason)] += 1
            if run_id == CONFIRM_RUN_ID:
                rec["confirmation_effective_rows"] += 1
                rec["confirm_effective_on"].add(agent)
                rec["confirm_effective_platforms"].add(platform)
    return records


def candidate_rows(records: dict[str, dict]) -> list[dict]:
    existing_ids, existing_families = existing_keys()
    candidates: list[dict] = []
    for rec in records.values():
        if rec["confirmation_effective_rows"] <= 0:
            continue
        if rec["id"] in existing_ids or rec["primary_unit_id"] in existing_families:
            continue
        candidates.append(
            {
                "id": rec["id"],
                "text": rec["text"],
                "label": rec["label"],
                "primary_unit_id": rec["primary_unit_id"],
                "provenance": "free_discovery_20260506_confirmed",
                "queue_rank": rec["queue_rank"],
                "n_models_tested": len(rec["tested_by"]),
                "n_models_effective": len(rec["effective_on"]),
                "n_platforms_tested": len(rec["tested_platforms"]),
                "n_platforms_effective": len(rec["effective_platforms"]),
                "confirmation_models_effective": len(rec["confirm_effective_on"]),
                "confirmation_platforms_effective": len(rec["confirm_effective_platforms"]),
                "tested_by": sorted(rec["tested_by"]),
                "effective_on": sorted(rec["effective_on"]),
                "tested_platforms": sorted(rec["tested_platforms"]),
                "effective_platforms": sorted(rec["effective_platforms"]),
                "tested_runs": sorted(rec["tested_runs"]),
                "effective_runs": sorted(rec["effective_runs"]),
                "confirmation_run_id": CONFIRM_RUN_ID,
                "confirmation_effective_on": sorted(rec["confirm_effective_on"]),
                "confirmation_effective_platforms": sorted(rec["confirm_effective_platforms"]),
                "valid_rows": rec["valid_rows"],
                "invalid_rows": rec["invalid_rows"],
                "effective_rows": rec["effective_rows"],
                "confirmation_effective_rows": rec["confirmation_effective_rows"],
                "canary_hits": rec["canary_hits"],
                "collapsed_hits": rec["collapsed_hits"],
                "reason_counts": dict(rec["reason_counts"].most_common()),
                "top_signals": dict(rec["top_signals"].most_common()),
                "evidence_files": sorted(x for x in rec["evidence_files"] if x),
            }
        )
    candidates.sort(
        key=lambda r: (
            -(r["confirmation_models_effective"]),
            -(r["n_models_effective"]),
            -(r["effective_rows"]),
            r["queue_rank"] or 999999,
            r["id"],
        )
    )
    return candidates


def append_confirmed(rows: list[dict]) -> Path | None:
    if not rows:
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = CONFIRMED_PATH.with_suffix(f".jsonl.bak_{ts}")
    shutil.copy2(CONFIRMED_PATH, backup)
    with CONFIRMED_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return backup


def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def update_report(summary: dict, promoted: list[dict], candidates: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_run = summary["runs"]
    lines = [
        "# Free discovery hit report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Summary",
        "",
        f"- Source rows: {summary['source_rows']:,}",
        f"- Source effective rows: {summary['source_effective_rows']:,}",
        f"- Deduped source effective units: {summary['queued_primary_units']:,}",
        f"- Confirmation rows: {summary['confirmation']['rows']:,}",
        f"- Confirmation valid rows: {summary['confirmation']['valid']:,}",
        f"- Confirmation effective rows: {summary['confirmation']['effective_rows']:,}",
        f"- Confirmation effective units: {summary['confirmation']['effective_unique_units']:,}",
        f"- Newly promoted units: {summary['promotion']['promoted_count']:,}",
        "",
        "## Runs",
        "",
        "| run | rows | valid | invalid | effective rows | effective units |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id in ALL_RUNS:
        s = rows_by_run[run_id]
        lines.append(
            f"| `{run_id}` | {s['rows']:,} | {s['valid']:,} | {s['invalid']:,} | "
            f"{s['effective_rows']:,} | {s['effective_unique_units']:,} |"
        )

    lines.extend(
        [
            "",
            "## Promoted Confirmations",
            "",
            "| label | confirmation agent | signal | effective runs | primary unit |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in promoted or candidates:
        signal = ",".join(row.get("top_signals", {}).keys()) or "unknown"
        lines.append(
            f"| `{row['label']}` | `{','.join(row['confirmation_effective_on'])}` | "
            f"`{signal}` | `{','.join(row['effective_runs'])}` | "
            f"`{row['primary_unit_id']}` |"
        )

    lines.extend(["", "## Confirmation By Agent", ""])
    confirm_stats = rows_by_run[CONFIRM_RUN_ID]["by_agent"]
    lines.extend(["| agent | rows | effective | invalid |", "| --- | ---: | ---: | ---: |"])
    for agent, stats in sorted(confirm_stats.items(), key=lambda kv: (-kv[1]["effective"], kv[0])):
        lines.append(f"| `{agent}` | {stats['rows']:,} | {stats['effective']:,} | {stats['invalid']:,} |")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Queue: `{summary['queue']}`",
            f"- Confirmation shard: `{summary['shard']}`",
            f"- Confirmation spec: `{summary['spec_path']}`",
            f"- Confirmation trial plan: `{summary['trial_plan']}`",
            f"- Promotion candidates: `{summary['promotion']['candidate_path']}`",
        ]
    )
    if summary["promotion"].get("backup"):
        lines.append(f"- Confirmed backup: `{summary['promotion']['backup']}`")
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    queue = list(read_jsonl(QUEUE_PATH) or [])
    rows_by_run = {run_id: load_run_rows(run_id) for run_id in ALL_RUNS}
    records = build_records(rows_by_run, queue)
    candidates = candidate_rows(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = OUT_DIR / "free_discovery_confirmed_candidates_20260506.jsonl"
    write_jsonl(candidate_path, candidates)

    promoted = candidates
    backup = append_confirmed(promoted) if args.apply else None

    summary = load_summary()
    summary.update(
        {
            "generated_at_unix": int(time.time()),
            "runs": {run_id: summarize_run(rows) for run_id, rows in rows_by_run.items()},
            "confirmation": summarize_run(rows_by_run[CONFIRM_RUN_ID]),
            "promotion": {
                "applied": bool(args.apply),
                "candidate_count": len(candidates),
                "promoted_count": len(promoted) if args.apply else 0,
                "candidate_path": str(candidate_path.relative_to(ROOT)),
                "backup": str(backup.relative_to(ROOT)) if backup else None,
                "confirmed_path": str(CONFIRMED_PATH.relative_to(ROOT)),
                "policy": "valid/effective in confirm_free_discovery_20260506 and not previously confirmed by id/family",
            },
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_report(summary, promoted if args.apply else [], candidates)

    out = {
        "applied": bool(args.apply),
        "candidate_count": len(candidates),
        "promoted_count": len(promoted) if args.apply else 0,
        "candidate_path": str(candidate_path.relative_to(ROOT)),
        "backup": str(backup.relative_to(ROOT)) if backup else None,
        "report": str(REPORT_PATH.relative_to(ROOT)),
        "summary": str(SUMMARY_PATH.relative_to(ROOT)),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
