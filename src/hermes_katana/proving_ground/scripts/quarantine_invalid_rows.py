#!/usr/bin/env python3
"""Validate observed rows against a planned trial manifest and quarantine bad data.

This script is intentionally conservative: invalid infrastructure rows,
duplicate planned_trial_id rows, malformed JSON, and orphan observations are
kept as artifacts but excluded from valid_rows.jsonl. It makes ASR denominators
explicit before report/dashboard analysis.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


INVALID_FIELDS = (
    "invalid_run",
    "baseline_valid",
    "attack_run_valid",
    "row_valid",
)


def legacy_trial_key(row: dict) -> tuple:
    return (
        row.get("run_id"),
        row.get("shard"),
        row.get("agent_id"),
        row.get("channel"),
        row.get("task"),
        row.get("attack_id"),
        row.get("repeat_idx", 0),
        bool(row.get("is_control", False)),
    )


def row_invalid_reason(row: dict) -> str | None:
    if row.get("row_valid") is False:
        return str(row.get("invalid_reason") or "row_valid_false")
    if row.get("invalid_run") is True:
        return str(row.get("invalid_reason") or "invalid_run")
    if row.get("baseline_valid") is False:
        return str(row.get("baseline_invalid_reason") or "baseline_invalid")
    if row.get("attack_run_valid") is False:
        return str(row.get("attack_run_invalid_reason") or "attack_run_invalid")
    return None


def _load_plan(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    plan: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            tid = row.get("planned_trial_id")
            if not tid:
                raise ValueError(f"plan row {lineno} missing planned_trial_id")
            if tid in plan:
                raise ValueError(f"duplicate planned_trial_id in plan: {tid}")
            plan[tid] = row
    return plan


def _iter_jsonl(paths: Iterable[str]) -> Iterable[tuple[Path, int, dict | None, str | None]]:
    for pat in paths:
        for name in sorted(glob.glob(pat)):
            p = Path(name)
            with p.open(errors="ignore", encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        yield p, lineno, json.loads(line), None
                    except Exception as exc:  # malformed rows are evidence too
                        yield p, lineno, None, str(exc)


def _row_matches_exclude_rule(row: dict, rule: dict) -> bool:
    for key, expected in rule.items():
        if key == "reason":
            continue
        if row.get(key) != expected:
            return False
    return True


def quarantine_rows(
    *,
    plan_rows: dict[str, dict],
    observed_rows: Iterable[dict | None],
    run_id: str | None = None,
    exclude_rules: list[dict] | None = None,
) -> dict[str, list[dict]]:
    buckets = {
        "valid_rows": [],
        "excluded_rows": [],
        "invalid_infrastructure_rows": [],
        "duplicate_trial_rows": [],
        "orphan_observed_rows": [],
        "malformed_rows": [],
        "missing_planned_trials": [],
    }
    seen: set[str | tuple] = set()
    seen_plan_ids: set[str] = set()
    exclude_rules = exclude_rules or []

    for row in observed_rows:
        if row is None:
            buckets["malformed_rows"].append({"quarantine_status": "malformed"})
            continue
        if run_id and row.get("run_id") != run_id:
            continue
        tid = row.get("planned_trial_id")
        if tid:
            if plan_rows and tid not in plan_rows:
                row = {**row, "quarantine_status": "orphan", "quarantine_reason": "planned_trial_id_not_in_plan"}
                buckets["orphan_observed_rows"].append(row)
                continue
            key: str | tuple = tid
            seen_plan_ids.add(tid)
        else:
            key = legacy_trial_key(row)

        matched_rule = next((rule for rule in exclude_rules if _row_matches_exclude_rule(row, rule)), None)
        if matched_rule is not None:
            row = {
                **row,
                "quarantine_status": "excluded",
                "quarantine_reason": str(matched_rule.get("reason") or "explicit_exclusion"),
            }
            buckets["excluded_rows"].append(row)
            seen.add(key)
            continue

        if key in seen:
            row = {**row, "quarantine_status": "duplicate", "quarantine_reason": "duplicate_trial_key"}
            buckets["duplicate_trial_rows"].append(row)
            continue
        seen.add(key)

        reason = row_invalid_reason(row)
        if reason:
            row = {**row, "quarantine_status": "invalid_infrastructure", "quarantine_reason": reason}
            buckets["invalid_infrastructure_rows"].append(row)
        else:
            row = {**row, "quarantine_status": "valid"}
            buckets["valid_rows"].append(row)

    for tid, plan in plan_rows.items():
        if tid not in seen_plan_ids:
            buckets["missing_planned_trials"].append({**plan, "quarantine_status": "missing"})
    return buckets


def denominator_summary(buckets: dict[str, list[dict]], *, planned_n: int) -> dict:
    valid = buckets["valid_rows"]
    attack_valid = [r for r in valid if not r.get("is_control")]
    control_valid = [r for r in valid if r.get("is_control")]
    by_cell: Counter = Counter()
    for r in attack_valid:
        cell = r.get("cell_id") or "|".join(str(x) for x in legacy_trial_key(r)[2:5])
        by_cell[cell] += 1
    return {
        "schema_version": 1,
        "planned_trials": planned_n,
        "valid_rows": len(valid),
        "valid_attack_rows": len(attack_valid),
        "valid_control_rows": len(control_valid),
        "excluded_rows": len(buckets["excluded_rows"]),
        "invalid_infrastructure_rows": len(buckets["invalid_infrastructure_rows"]),
        "duplicate_trial_rows": len(buckets["duplicate_trial_rows"]),
        "orphan_observed_rows": len(buckets["orphan_observed_rows"]),
        "malformed_rows": len(buckets["malformed_rows"]),
        "missing_planned_trials": len(buckets["missing_planned_trials"]),
        "valid_attack_rows_by_cell": dict(sorted(by_cell.items())),
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_exclude_rules(path: Path | None) -> list[dict]:
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("exclude rules file must contain a JSON list")
    out: list[dict] = []
    for idx, row in enumerate(data, 1):
        if not isinstance(row, dict):
            raise ValueError(f"exclude rule #{idx} must be an object")
        if len([k for k in row if k != "reason"]) == 0:
            raise ValueError(f"exclude rule #{idx} must match at least one field")
        out.append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--plan", type=Path, default=None)
    ap.add_argument("--runs", nargs="+", default=["results/agent_shard_runs/*.jsonl"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--exclude-rules", type=Path, default=None)
    args = ap.parse_args(argv)

    plan = _load_plan(args.plan)
    exclude_rules = _load_exclude_rules(args.exclude_rules)
    observed = []
    malformed = []
    for path, lineno, row, err in _iter_jsonl(args.runs):
        if err:
            malformed.append({"path": str(path), "line": lineno, "error": err})
        else:
            observed.append(row)
    buckets = quarantine_rows(
        plan_rows=plan,
        observed_rows=observed,
        run_id=args.run_id,
        exclude_rules=exclude_rules,
    )
    buckets["malformed_rows"].extend(malformed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in buckets.items():
        _write_jsonl(args.out_dir / f"{name}.jsonl", rows)
    summary = denominator_summary(buckets, planned_n=len(plan))
    (args.out_dir / "denominators.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
