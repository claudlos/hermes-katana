"""Write a compact quota-mix status dashboard."""

from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
FLEET_RUNS = ROOT / "results" / "fleet_runs"
OUT_DIR = ROOT / "results" / "reports" / "quota_mix_20260506"

RUN_IDS = [
    "quota_mix_fresh_20260505_2158",
    "confirm_quota_mix_priority_20260506",
    "confirm_quota_mix_remaining_20260506",
    "free_reliable_discovery_20260506",
    "targeted_labels_fresh_20260506",
    "defended_quota_mix_confirmed_20260506",
]


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_status(run_id: str) -> dict:
    statuses = []
    for path in AGENT_RUNS.glob(f"*__run_{run_id}.status.json"):
        data = load_json(path)
        if data:
            statuses.append(data)
    done = sum(int(s.get("done") or 0) for s in statuses)
    total = sum(int(s.get("total") or 0) for s in statuses)
    effective = sum(int(s.get("effective") or 0) for s in statuses)
    invalid = sum(int(s.get("invalid_runs") or 0) for s in statuses)

    sup = FLEET_RUNS / run_id / "supervisor.log"
    last = ""
    if sup.exists() and sup.stat().st_size:
        try:
            last = sup.read_text(errors="ignore", encoding="utf-8").splitlines()[-1]
        except Exception:
            last = ""
    active = bool(last and "fleet exit" not in last)
    return {
        "run_id": run_id,
        "done": done,
        "total_visible": total,
        "effective": effective,
        "invalid": invalid,
        "active": active,
        "last_supervisor_line": last,
    }


def main() -> int:
    rows = [run_status(run_id) for run_id in RUN_IDS]
    payload = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "runs": rows,
        "artifacts": {
            "report": "results/reports/quota_mix_20260506/report.md",
            "summary": "results/reports/quota_mix_20260506/summary.json",
            "audit": "results/audits/quota_mix_20260506/audit.md",
            "promotion_candidates": "results/promotions/quota_mix_confirmed_candidates_20260506.jsonl",
            "free_reliable_summary": "results/designs/D-free-reliable-discovery-20260506/trial_plan_summary.json",
            "targeted_summary": "results/designs/D-targeted-labels-fresh-20260506/trial_plan_summary.json",
            "defended_summary": "results/designs/D-defended-quota-mix-confirmed-20260506/trial_plan_summary.json",
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "status.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Quota-mix status",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "| run | active | done | visible total | effective | invalid | last supervisor line |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        last = row["last_supervisor_line"].replace("|", "\\|")[:140]
        lines.append(
            f"| `{row['run_id']}` | {str(row['active']).lower()} | "
            f"{row['done']:,} | {row['total_visible']:,} | {row['effective']:,} | "
            f"{row['invalid']:,} | `{last}` |"
        )
    lines.extend(["", "## Artifacts", ""])
    for name, path in payload["artifacts"].items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    md_path = OUT_DIR / "status.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": str(md_path.relative_to(ROOT)), "json": str(json_path.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
