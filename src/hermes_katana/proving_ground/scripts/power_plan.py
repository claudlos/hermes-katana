#!/usr/bin/env python3
"""Power/precision planning for proving-ground campaigns.

The key unit is the independent attack family / primary unit, not the raw
physical output row. Repeats improve stochastic stability but do not linearly
increase independent evidence.
"""

from __future__ import annotations

import argparse
import json

from hermes_katana.proving_ground.research.statistics import sample_size_paired_delta, sample_size_single_proportion


def build_plan(args: argparse.Namespace) -> dict:
    single_n = sample_size_single_proportion(
        half_width=args.single_half_width,
        conf=args.conf,
        p=args.assumed_p,
    )
    paired_n = sample_size_paired_delta(
        delta=args.paired_delta,
        discordance=args.discordance,
        power=args.power,
        alpha=args.alpha,
    )
    physical_trials_per_cell = single_n * args.n_repeats * args.n_judge_repeats
    paired_physical_per_comparison = paired_n * args.n_repeats * args.n_judge_repeats
    return {
        "schema_version": 1,
        "analysis_unit": "attack_family",
        "single_cell_precision": {
            "conf": args.conf,
            "assumed_p": args.assumed_p,
            "half_width": args.single_half_width,
            "required_independent_families_per_cell": single_n,
            "n_repeats": args.n_repeats,
            "n_judge_repeats": args.n_judge_repeats,
            "physical_trials_per_cell_including_judges": physical_trials_per_cell,
        },
        "paired_delta_power": {
            "alpha": args.alpha,
            "power": args.power,
            "delta": args.paired_delta,
            "discordance": args.discordance,
            "required_paired_families": paired_n,
            "n_repeats": args.n_repeats,
            "n_judge_repeats": args.n_judge_repeats,
            "physical_trials_per_comparison_including_judges": paired_physical_per_comparison,
        },
        "budget_hint": {
            "n_cells": args.n_cells,
            "estimated_physical_trials_for_cells": physical_trials_per_cell * args.n_cells,
            "note": "Use this for planning. Final CIs should use cluster bootstrap over family/primary_unit_id.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--single-half-width", type=float, default=0.05)
    ap.add_argument("--assumed-p", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.95)
    ap.add_argument("--paired-delta", type=float, default=0.10)
    ap.add_argument("--discordance", type=float, default=0.30)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-repeats", type=int, default=2)
    ap.add_argument("--n-judge-repeats", type=int, default=1)
    ap.add_argument("--n-cells", type=int, default=1)
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args(argv)

    plan = build_plan(args)
    if args.json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    print("Power / precision plan")
    print("analysis unit: attack_family")
    sp = plan["single_cell_precision"]
    print(
        f"single-cell CI ±{sp['half_width']:.1%} at {sp['conf']:.0%}: "
        f"{sp['required_independent_families_per_cell']} independent families/cell"
    )
    pp = plan["paired_delta_power"]
    print(
        f"paired delta {pp['delta']:.1%}, discordance {pp['discordance']:.1%}, "
        f"power {pp['power']:.0%}: {pp['required_paired_families']} paired families"
    )
    print(f"physical trials/cell including repeats+judge repeats: {sp['physical_trials_per_cell_including_judges']}")
    print(
        f"estimated physical trials for {args.n_cells} cells: "
        f"{plan['budget_hint']['estimated_physical_trials_for_cells']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
