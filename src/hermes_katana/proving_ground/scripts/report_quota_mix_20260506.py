"""Generate the quota-mix follow-up report.

The report is safe to rerun while confirmation fleets are active. It reads the
available JSONL/status files and writes:

  results/reports/quota_mix_20260506/report.md
  results/reports/quota_mix_20260506/summary.json
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
REPORT_DIR = ROOT / "results" / "reports" / "quota_mix_20260506"
QUEUE_PATH = ROOT / "results" / "queues" / "confirmation_queue_20260506_quota_mix_hits.jsonl"

SOURCE_RUN = "quota_mix_fresh_20260505_2158"
PRIORITY_RUN = "confirm_quota_mix_priority_20260506"
REMAINING_RUN = "confirm_quota_mix_remaining_20260506"


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


def load_run_rows(run_id: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(AGENT_RUNS.glob(f"shard_*__run_{run_id}.jsonl")):
        for row in read_jsonl(path) or []:
            row["_source_file"] = str(path.relative_to(ROOT))
            rows.append(row)
    return rows


def load_queue() -> list[dict]:
    return list(read_jsonl(QUEUE_PATH) or [])


def summarize(rows: list[dict]) -> dict:
    valid = [r for r in rows if is_valid(r)]
    invalid = [r for r in rows if not is_valid(r)]
    effective = [r for r in valid if r.get("effective")]
    by_agent = Counter(r.get("agent_id", "unknown") for r in rows)
    eff_by_agent = Counter(r.get("agent_id", "unknown") for r in effective)
    invalid_by_agent = Counter(r.get("agent_id", "unknown") for r in invalid)
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
        "by_label": {item: {"rows": by_label[item], "effective": eff_by_label[item]} for item in sorted(by_label)},
    }


def build_unit_records(queue: list[dict], run_rows: dict[str, list[dict]]) -> list[dict]:
    rank_by_unit = {q["primary_unit_id"]: q.get("queue_rank") for q in queue}
    label_by_unit = {q["primary_unit_id"]: q.get("label", "unknown") for q in queue}
    records: dict[str, dict] = {}
    for run_id, rows in run_rows.items():
        for row in rows:
            pid = primary_id(row)
            rec = records.setdefault(
                pid,
                {
                    "primary_unit_id": pid,
                    "queue_rank": rank_by_unit.get(pid),
                    "label": label_by_unit.get(pid, label(row)),
                    "tested_rows": 0,
                    "valid_rows": 0,
                    "effective_rows": 0,
                    "tested_agents": set(),
                    "effective_agents": set(),
                    "effective_channels": set(),
                    "effective_runs": set(),
                    "canary_hits": 0,
                    "collapsed_hits": 0,
                    "top_signals": Counter(),
                },
            )
            rec["tested_rows"] += 1
            rec["tested_agents"].add(str(row.get("agent_id") or "unknown"))
            if not is_valid(row):
                continue
            rec["valid_rows"] += 1
            if not row.get("effective"):
                continue
            rec["effective_rows"] += 1
            rec["effective_agents"].add(str(row.get("agent_id") or "unknown"))
            rec["effective_channels"].add(str(row.get("channel") or "unknown"))
            rec["effective_runs"].add(run_id)
            rec["canary_hits"] += int(bool(row.get("canary_leaked")))
            rec["collapsed_hits"] += int(bool(row.get("collapsed")))
            rec["top_signals"][str(row.get("severity_top_signal") or "unknown")] += 1
    out = []
    for rec in records.values():
        item = dict(rec)
        item["tested_agents"] = sorted(rec["tested_agents"])
        item["effective_agents"] = sorted(rec["effective_agents"])
        item["effective_channels"] = sorted(rec["effective_channels"])
        item["effective_runs"] = sorted(rec["effective_runs"])
        item["top_signals"] = dict(rec["top_signals"].most_common())
        item["agent_count"] = len(item["effective_agents"])
        item["channel_count"] = len(item["effective_channels"])
        out.append(item)
    out.sort(
        key=lambda r: (
            -(r["agent_count"]),
            -(r["channel_count"]),
            -(r["effective_rows"]),
            r["queue_rank"] or 999999,
            r["primary_unit_id"],
        )
    )
    return out


def table_run(run_id: str, stats: dict) -> list[str]:
    lines = [
        f"### `{run_id}`",
        "",
        "| agent | rows | effective | invalid |",
        "| --- | ---: | ---: | ---: |",
    ]
    for agent, s in sorted(stats["by_agent"].items(), key=lambda kv: (-kv[1]["effective"], kv[0])):
        lines.append(f"| `{agent}` | {s['rows']:,} | {s['effective']:,} | {s['invalid']:,} |")
    lines.append("")
    return lines


def write_report(summary: dict, records: list[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Quota-mix follow-up report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Runs",
        "",
        "| run | rows | valid | invalid | effective rows | effective units |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_id in [SOURCE_RUN, PRIORITY_RUN, REMAINING_RUN]:
        s = summary["runs"][run_id]
        lines.append(
            f"| `{run_id}` | {s['rows']:,} | {s['valid']:,} | {s['invalid']:,} | "
            f"{s['effective_rows']:,} | {s['effective_unique_units']:,} |"
        )
    lines.extend(
        [
            "",
            "## Queue",
            "",
            f"Queue file: `{QUEUE_PATH.relative_to(ROOT)}`",
            f"Queued units: {summary['queue']['queued_units']:,}",
            f"Priority reproduced units: {summary['queue']['multi_agent_reproduced_units']:,}",
            "",
            "The priority run covered ranks 1-9. The remaining run covers ranks 10-74, so the combined confirmation denominator still covers all 74 queued units without retesting the priority block.",
            "",
            "## Confirmation Leaders",
            "",
            "| rank | label | effective agents | channels | effective rows | runs | primary unit |",
            "| ---: | --- | --- | --- | ---: | --- | --- |",
        ]
    )
    for rec in records[:30]:
        if rec["effective_rows"] <= 0:
            continue
        lines.append(
            f"| {rec['queue_rank'] or ''} | `{rec['label']}` | "
            f"`{','.join(rec['effective_agents'])}` | "
            f"`{','.join(rec['effective_channels'])}` | "
            f"{rec['effective_rows']} | `{','.join(rec['effective_runs'])}` | "
            f"`{rec['primary_unit_id']}` |"
        )
    lines.append("")
    lines.append("## By Agent")
    lines.append("")
    for run_id in [SOURCE_RUN, PRIORITY_RUN, REMAINING_RUN]:
        lines.extend(table_run(run_id, summary["runs"][run_id]))
    lines.append("## Artifacts")
    lines.append("")
    for path in summary["artifacts"]:
        lines.append(f"- `{path}`")
    lines.append("")
    (REPORT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    queue = load_queue()
    rows = {
        SOURCE_RUN: load_run_rows(SOURCE_RUN),
        PRIORITY_RUN: load_run_rows(PRIORITY_RUN),
        REMAINING_RUN: load_run_rows(REMAINING_RUN),
    }
    summary = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "runs": {run_id: summarize(run_rows) for run_id, run_rows in rows.items()},
        "queue": {
            "path": str(QUEUE_PATH.relative_to(ROOT)),
            "queued_units": len(queue),
            "multi_agent_reproduced_units": sum(1 for q in queue if q.get("multi_agent_reproduced")),
            "single_agent_hit_units": sum(1 for q in queue if not q.get("multi_agent_reproduced")),
        },
        "artifacts": [
            "results/reports/quota_mix_20260506/report.md",
            "results/reports/quota_mix_20260506/summary.json",
            "results/queues/confirmation_queue_20260506_quota_mix_hits.jsonl",
            "results/queues/confirmation_queue_20260506_quota_mix_priority9.jsonl",
            "results/designs/D-confirm-quota-mix-priority-20260506/trial_plan.jsonl",
            "results/designs/D-confirm-quota-mix-remaining-20260506/trial_plan.jsonl",
        ],
    }
    records = build_unit_records(queue, rows)
    summary["unit_records"] = records
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary, records)
    print(
        json.dumps(
            {
                "report": str((REPORT_DIR / "report.md").relative_to(ROOT)),
                "summary": str((REPORT_DIR / "summary.json").relative_to(ROOT)),
                "runs": {
                    run_id: {
                        "rows": s["rows"],
                        "effective_rows": s["effective_rows"],
                        "effective_unique_units": s["effective_unique_units"],
                    }
                    for run_id, s in summary["runs"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
