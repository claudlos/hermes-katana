#!/usr/bin/env python3
"""Estimate $ spent on the codex_cli_gpt5_4_mini lane.

Counts attacks completed by the gpt-5.4-mini worker(s), multiplies by an
average $/attack estimate, and reports remaining budget. Strictly a
back-of-envelope check — actual cost is what OpenAI's dashboard says.

Usage:
    python scripts/check_openai_burn.py
    python scripts/check_openai_burn.py --start-budget 3.62
    python scripts/check_openai_burn.py --cost-per-attack 0.0018
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-budget", type=float, default=3.62)
    ap.add_argument(
        "--cost-per-attack",
        type=float,
        default=0.0018,
        help="Estimated $/attack at agent-loop token sizes for gpt-5.4-mini "
        "real-time API (8K input + 1K output ≈ $0.0018 at $0.15/M in + "
        "$0.60/M out). Override with the rate OpenAI's dashboard shows.",
    )
    ap.add_argument(
        "--run-id",
        default="v8_main_20260511",
        help="Run ID to scan status files for.",
    )
    args = ap.parse_args()

    runs_dir = ROOT / "results" / "agent_shard_runs"
    pattern = f"*codex_cli_gpt5_4_mini*__run_{args.run_id}.status.json"

    files = sorted(runs_dir.glob(pattern))
    if not files:
        print(f"no gpt-5.4-mini status files yet (looked for {pattern})")
        return 0

    total_done = 0
    total_effective = 0
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        total_done += m.get("done", 0)
        total_effective += m.get("effective", 0)

    spent = total_done * args.cost_per_attack
    remaining = args.start_budget - spent
    burn_rate = "n/a"
    if files:
        latest = max(f.stat().st_mtime for f in files)
        # crude per-hour rate from first status file mtime to latest
        oldest = min(f.stat().st_mtime for f in files)
        if latest > oldest:
            atks_per_hr = total_done / max((latest - oldest) / 3600, 0.01)
            spend_per_hr = atks_per_hr * args.cost_per_attack
            burn_rate = f"~{atks_per_hr:.0f} atk/hr ≈ ${spend_per_hr:.2f}/hr"

    print(f"gpt-5.4-mini lane burn (run_id={args.run_id})")
    print(f"  attacks done   : {total_done}")
    print(f"  effective hits : {total_effective}")
    print(f"  est $ spent    : ${spent:.2f}   (${args.cost_per_attack:.4f}/atk)")
    print(f"  est $ remaining: ${remaining:.2f}  of ${args.start_budget:.2f}")
    print(f"  burn rate      : {burn_rate}")
    if remaining < 0.50:
        print("  ⚠  remaining < $0.50 — consider stopping the lane.")


if __name__ == "__main__":
    raise SystemExit(main())
