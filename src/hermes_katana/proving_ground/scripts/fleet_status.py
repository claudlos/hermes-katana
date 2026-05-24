#!/usr/bin/env python3
"""Snapshot of fleet progress — walks results/agent_shard_runs/*.status.json
and prints a tight summary by run_id and agent. Intended for tmux watching:

    watch -n 30 'python3 scripts/fleet_status.py'

Or one-shot:

    python3 scripts/fleet_status.py
    python3 scripts/fleet_status.py --run-id wave_a_main_20260501_1134
    python3 scripts/fleet_status.py --json

Reads only — safe to run while a fleet is live. No effect on trial data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "agent_shard_runs"
FLEET_DIR = ROOT / "results" / "fleet_runs"


def _load_status_files(run_id_filter: str | None):
    """Yield (status_dict, mtime) for every shard status that matches."""
    for sp in RUNS_DIR.glob("*.status.json"):
        try:
            d = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if run_id_filter and d.get("run_id") != run_id_filter:
            # status.json doesn't always carry run_id — fall back to mtime
            # within the last hour for the latest-running view.
            if "run_id" not in d:
                continue
            continue
        try:
            mtime = sp.stat().st_mtime
        except Exception:
            mtime = 0
        yield d, mtime


def _active_runs() -> list[str]:
    if not FLEET_DIR.exists():
        return []
    return sorted([d.name for d in FLEET_DIR.iterdir() if d.is_dir()])[-5:]


def _format_eta(done: int, total: int, mtime: float) -> str:
    if done <= 0 or total <= 0 or done >= total:
        return ""
    age = max(time.time() - mtime, 1.0)
    rate = done / age  # rows per second from start of shard
    remaining = (total - done) / max(rate, 1e-9)
    if remaining > 86400:
        return f"~{remaining / 86400:.1f}d"
    if remaining > 3600:
        return f"~{remaining / 3600:.1f}h"
    return f"~{remaining / 60:.0f}m"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-id", default=None, help="filter by run_id (most useful for active runs)")
    p.add_argument(
        "--top",
        type=int,
        default=20,
        help="how many active shards to show (default 20)",
    )
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    by_agent = defaultdict(
        lambda: {
            "done": 0,
            "total": 0,
            "effective": 0,
            "shards_open": 0,
            "shards_done": 0,
            "last_update": 0.0,
        }
    )
    active_shards = []  # (mtime, status_dict)
    total_done = total_total = total_eff = 0

    for d, mtime in _load_status_files(args.run_id):
        a = d.get("agent_id", "")
        done = d.get("done", 0)
        total = d.get("total", 0)
        eff = d.get("effective", 0)
        by_agent[a]["done"] += done
        by_agent[a]["total"] += total
        by_agent[a]["effective"] += eff
        by_agent[a]["last_update"] = max(by_agent[a]["last_update"], mtime)
        if total > 0:
            if done >= total:
                by_agent[a]["shards_done"] += 1
            else:
                by_agent[a]["shards_open"] += 1
                active_shards.append((mtime, d))
        total_done += done
        total_total += total
        total_eff += eff

    if args.json:
        print(
            json.dumps(
                {
                    "by_agent": dict(by_agent),
                    "active_shards": [d for _, d in active_shards[-args.top :]],
                    "totals": {
                        "done": total_done,
                        "total": total_total,
                        "effective": total_eff,
                    },
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"=== Fleet snapshot @ {time.strftime('%F %T')} ===")
    if args.run_id:
        print(f"run_id filter: {args.run_id}")
    print()
    print(
        f"{'agent_id':<40}{'done':>10}{'total':>10}{'eff':>6}{'eff%':>6}{'shards (open/done)':>22}{'  last update':>20}"
    )
    print("-" * 114)
    for a in sorted(by_agent, key=lambda k: -by_agent[k]["done"]):
        s = by_agent[a]
        rate = (100 * s["effective"] / s["done"]) if s["done"] else 0
        last = time.strftime("%m-%d %H:%M", time.localtime(s["last_update"])) if s["last_update"] else ""
        shards = f"{s['shards_open']:>3}/{s['shards_done']:>4}"
        print(f"{a:<40}{s['done']:>10}{s['total']:>10}{s['effective']:>6}{rate:>5.1f}%{shards:>22}{last:>20}")

    print()
    print(
        f"TOTALS: {total_done:,} / {total_total:,} trials   "
        f"effective: {total_eff:,} ({100 * total_eff / max(total_done, 1):.1f}%)"
    )
    print()

    # Recently-updated active shards (fleet's "what's running right now" view)
    if active_shards:
        print(f"Most recently active shards (top {args.top}):")
        active_shards.sort(key=lambda kv: -kv[0])
        print(f"  {'shard':>6} {'agent':<40}{'done/total':>14}{'eff':>6}{'eta':>10}{'  last':>15}")
        for mtime, d in active_shards[: args.top]:
            shard = d.get("shard", "?")
            a = d.get("agent_id", "")
            done, total = d.get("done", 0), d.get("total", 0)
            eff = d.get("effective", 0)
            eta = _format_eta(done, total, mtime)
            ago = (time.time() - mtime) / 60
            ago_str = f"{ago:.1f}m" if ago < 60 else f"{ago / 60:.1f}h"
            print(f"  {str(shard):>6} {a:<40}{done:>5}/{total:<6}{eff:>6}{eta:>10}{ago_str:>15}")

    # Active fleet runs (last 5)
    runs = _active_runs()
    if runs:
        print()
        print("Recent fleet runs:")
        for r in runs:
            sup = FLEET_DIR / r / "supervisor.log"
            if not sup.exists():
                continue
            tail = sup.read_text(encoding="utf-8").splitlines()[-1:] if sup.stat().st_size else []
            tail_str = (tail[0] if tail else "")[:120]
            print(f"  {r:<32} {tail_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
