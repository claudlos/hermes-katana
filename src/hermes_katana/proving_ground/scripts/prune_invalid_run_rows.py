#!/usr/bin/env python3
"""Remove infrastructure-invalid rows from run JSONLs after backing them up."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "results" / "agent_shard_runs"
DEFAULT_BACKUP_ROOT = ROOT / "results" / "agent_shard_runs_backups"


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


def read_rows(path: Path) -> tuple[list[str], list[dict | None]]:
    raw_lines = []
    parsed = []
    with path.open(errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            raw_lines.append(line)
            try:
                parsed.append(json.loads(line))
            except Exception:
                parsed.append(None)
    return raw_lines, parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--tag", default="prune_invalid")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = args.backup_root / f"{args.run_id}_{args.tag}_{timestamp}"
    files = sorted(args.run_dir.glob(f"*run_{args.run_id}.jsonl"))

    summary = {
        "run_id": args.run_id,
        "dry_run": args.dry_run,
        "backup_dir": str(backup_dir),
        "files_scanned": len(files),
        "files_touched": 0,
        "rows_removed": 0,
        "reasons": Counter(),
        "touched": [],
    }

    for path in files:
        raw_lines, parsed = read_rows(path)
        keep_lines = []
        removed = []
        for raw, row in zip(raw_lines, parsed):
            if row is None:
                keep_lines.append(raw)
                continue
            if row.get("run_id") != args.run_id:
                keep_lines.append(raw)
                continue
            reason = invalid_reason(row)
            if reason:
                removed.append((raw, reason))
            else:
                keep_lines.append(raw)

        if not removed:
            continue

        summary["files_touched"] += 1
        summary["rows_removed"] += len(removed)
        for _, reason in removed:
            summary["reasons"][reason] += 1
        summary["touched"].append({"path": str(path), "removed": len(removed), "remaining": len(keep_lines)})

        if not args.dry_run:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_dir / path.name)
            with path.open("w") as f:
                f.writelines(keep_lines)

    summary["reasons"] = dict(summary["reasons"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
