"""Union per-worker shard DBs into one battery.db.

Every run_shard.py process writes to its own SQLite
(`results/shard_runs/shard_NNN_<model>.db`) so parallel workers don't
collide on sqlite locks. This script glues them all into one DB matching
the existing schema, suitable for the analyzer pipeline.

Also concatenates the per-shard sessions.jsonl and signatures.jsonl into
merged top-level files for easy grep / counts.

Usage:
    python scripts/merge_shard_runs.py
    python scripts/merge_shard_runs.py --input results/shard_runs --output results/battery.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def _copy_table(src_conn, dst_conn, table: str, cols: list[str]):
    """INSERT OR IGNORE rows from src.table into dst.table, skipping duplicates
    based on the dst table's PRIMARY KEY / UNIQUE constraints."""
    placeholders = ",".join(["?"] * len(cols))
    col_list = ",".join(cols)
    src_rows = src_conn.execute(f"SELECT {col_list} FROM {table}").fetchall()
    if not src_rows:
        return 0
    dst_conn.executemany(
        f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})",
        src_rows,
    )
    return len(src_rows)


def _init_dst(path: Path):
    from hermes_katana.proving_ground.sandbox.behavioral_tracker import BehavioralTracker

    # Reuse the canonical schema so downstream analyzers just work.
    tr = BehavioralTracker(str(path))
    tr.close()


def merge(input_dir: str, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    _init_dst(out)

    dst = sqlite3.connect(out)
    dst.row_factory = sqlite3.Row

    session_cols = [
        "session_id",
        "attack_id",
        "attack_text",
        "attack_label",
        "attack_strategy",
        "task",
        "model",
        "provider",
        "start_time",
        "end_time",
        "total_turns",
        "total_tool_calls",
        "phase",
        "outcome",
    ]
    event_cols = ["session_id", "kind", "phase", "turn", "timestamp", "data"]
    tool_cols = [
        "session_id",
        "turn",
        "phase",
        "tool",
        "args",
        "success",
        "output",
        "error",
        "latency_ms",
        "timestamp",
    ]

    dbs = sorted(Path(input_dir).glob("shard_*.db"))
    print(f"Merging {len(dbs)} shard DBs → {out}")
    totals = {"sessions": 0, "events": 0, "tool_calls": 0}
    skipped = []
    for db in dbs:
        try:
            src = sqlite3.connect(db)
            src.row_factory = sqlite3.Row
            s = _copy_table(src, dst, "sessions", session_cols)
            e = _copy_table(src, dst, "events", event_cols)
            t = _copy_table(src, dst, "tool_calls", tool_cols)
            totals["sessions"] += s
            totals["events"] += e
            totals["tool_calls"] += t
            src.close()
        except sqlite3.DatabaseError as ex:
            skipped.append((db.name, str(ex)))
    dst.commit()

    # Concatenate the JSONL side-channels too.
    sessions_jsonl = out.parent / "battery_sessions.jsonl"
    sigs_jsonl = out.parent / "battery_signatures.jsonl"
    for dst_file, glob in [
        (sessions_jsonl, "shard_*.sessions.jsonl"),
        (sigs_jsonl, "shard_*.signatures.jsonl"),
    ]:
        n_lines = 0
        with dst_file.open("w") as f:
            for src in sorted(Path(input_dir).glob(glob)):
                with src.open() as g:
                    for line in g:
                        line = line.strip()
                        if not line:
                            continue
                        f.write(line + "\n")
                        n_lines += 1
        print(f"  {dst_file.name}: {n_lines:,} lines")

    print(f"\nMerged into {out}:")
    print(f"  sessions:   {totals['sessions']:,}")
    print(f"  events:     {totals['events']:,}")
    print(f"  tool_calls: {totals['tool_calls']:,}")
    if skipped:
        print(f"\n  Skipped {len(skipped)} DBs with errors:")
        for name, err in skipped[:5]:
            print(f"    {name}: {err[:80]}")

    # Quick per-model summary.
    print("\nSessions by model:")
    for r in dst.execute("SELECT model, COUNT(*) n FROM sessions GROUP BY model ORDER BY n DESC"):
        print(f"  {r['model']:<40} {r['n']}")
    dst.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/shard_runs")
    p.add_argument("--output", default="results/battery.db")
    args = p.parse_args()

    # Make the sandbox package importable.

    merge(args.input, args.output)
