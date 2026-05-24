#!/usr/bin/env python3
"""Publication-grade ASR analysis over quarantined valid rows.

Primary ASR collapses repeated physical trials inside each primary_unit_id
(attack family) before bootstrapping across families. Row-counted ASR is kept
only as a sanity metric.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hermes_katana.proving_ground.research.statistics import (
    cluster_bootstrap_mean_ci,
    paired_cluster_bootstrap_ci,
    wilson_ci,
)


def _is_valid_attack(row: dict) -> bool:
    return (
        row.get("row_valid", True) is not False
        and not row.get("invalid_run", False)
        and row.get("baseline_valid", True) is not False
        and row.get("attack_run_valid", True) is not False
        and not row.get("is_control", False)
    )


def _cell_id(row: dict) -> str:
    if row.get("cell_id"):
        return str(row["cell_id"])
    return "|".join(str(row.get(k, "")) for k in ("agent_id", "channel", "task", "attack_label"))


def _primary_unit_id(row: dict) -> str:
    return str(
        row.get("primary_unit_id")
        or row.get("family_sha256")
        or row.get("text_sha256_normalized")
        or row.get("attack_text_sha256")
        or row.get("attack_id")
    )


def _cluster_by_cell(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not _is_valid_attack(row):
            continue
        cells[_cell_id(row)][_primary_unit_id(row)].append(1.0 if row.get("effective") else 0.0)
    return cells


def _row_sanity(rows: list[dict], cell: str) -> dict:
    valid = [r for r in rows if _is_valid_attack(r) and _cell_id(r) == cell]
    n = len(valid)
    k = sum(1 for r in valid if r.get("effective"))
    lo, hi = wilson_ci(k, n) if n else (0.0, 0.0)
    return {
        "row_counted_effective": k,
        "row_counted_n": n,
        "row_counted_asr": 0.0 if n == 0 else k / n,
        "row_counted_ci_low": lo,
        "row_counted_ci_high": hi,
        "ci_method": "wilson_row_counted_sanity",
    }


def analyze_rows(
    rows: list[dict],
    *,
    comparisons: list[tuple[str, str]] | None = None,
    bootstrap_iters: int = 5000,
    seed: int = 42,
) -> dict:
    clusters = _cluster_by_cell(rows)
    cells = {}
    for cell, by_family in sorted(clusters.items()):
        point, lo, hi = cluster_bootstrap_mean_ci(by_family, iters=bootstrap_iters, seed=seed)
        repeat_counts = [len(v) for v in by_family.values()]
        cells[cell] = {
            "analysis_unit": "primary_unit_id",
            "valid_families": len(by_family),
            "valid_physical_trials": sum(repeat_counts),
            "valid_repeats_mean": 0.0 if not repeat_counts else sum(repeat_counts) / len(repeat_counts),
            "valid_repeats_min": min(repeat_counts) if repeat_counts else 0,
            "valid_repeats_max": max(repeat_counts) if repeat_counts else 0,
            "primary_asr": point,
            "ci_low": lo,
            "ci_high": hi,
            "ci_method": "cluster_bootstrap_primary_unit",
            **_row_sanity(rows, cell),
        }

    comp_out = {}
    for a, b in comparisons or []:
        point, lo, hi, n = paired_cluster_bootstrap_ci(
            clusters.get(a, {}),
            clusters.get(b, {}),
            iters=bootstrap_iters,
            seed=seed,
        )
        comp_out[f"{a}::{b}"] = {
            "cell_a": a,
            "cell_b": b,
            "delta": point,
            "ci_low": lo,
            "ci_high": hi,
            "paired_families": n,
            "ci_method": "paired_cluster_bootstrap_primary_unit",
        }

    return {
        "schema_version": 1,
        "analysis_unit": "primary_unit_id",
        "bootstrap_iters": bootstrap_iters,
        "seed": seed,
        "input_rows": len(rows),
        "valid_attack_rows": sum(1 for r in rows if _is_valid_attack(r)),
        "cells": cells,
        "comparisons": comp_out,
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(errors="ignore", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_markdown(result: dict, denominators: dict | None = None) -> str:
    lines = ["# Scientific ASR analysis", ""]
    lines.append(f"Analysis unit: `{result['analysis_unit']}`")
    lines.append(f"Bootstrap iterations: {result['bootstrap_iters']}")
    lines.append("")
    if denominators:
        lines.extend(["## Denominators", ""])
        for key in (
            "planned_trials",
            "valid_attack_rows",
            "valid_control_rows",
            "invalid_infrastructure_rows",
            "duplicate_trial_rows",
            "orphan_observed_rows",
            "missing_planned_trials",
        ):
            if key in denominators:
                lines.append(f"- {key}: {denominators[key]}")
        lines.append("")
    lines.extend(["## Primary ASR by cell", ""])
    lines.append("| cell | families | physical trials | primary ASR | 95% CI | row-counted sanity |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cell, c in result["cells"].items():
        lines.append(
            f"| `{cell}` | {c['valid_families']} | {c['valid_physical_trials']} | "
            f"{c['primary_asr']:.3f} | [{c['ci_low']:.3f}, {c['ci_high']:.3f}] | "
            f"{c['row_counted_asr']:.3f} |"
        )
    if result["comparisons"]:
        lines.extend(["", "## Paired deltas", ""])
        lines.append("| comparison | paired families | delta A-B | 95% CI |")
        lines.append("|---|---:|---:|---:|")
        for name, c in result["comparisons"].items():
            lines.append(
                f"| `{name}` | {c['paired_families']} | {c['delta']:.3f} | [{c['ci_low']:.3f}, {c['ci_high']:.3f}] |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_comparisons(values: list[str]) -> list[tuple[str, str]]:
    out = []
    for value in values:
        if "::" not in value:
            raise ValueError(f"comparison must be CELL_A::CELL_B, got {value!r}")
        a, b = value.split("::", 1)
        out.append((a, b))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--valid-rows", type=Path, required=True)
    ap.add_argument("--denominators", type=Path, default=None)
    ap.add_argument("--compare", action="append", default=[], help="CELL_A::CELL_B paired delta")
    ap.add_argument("--bootstrap-iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.valid_rows)
    denominators = json.loads(args.denominators.read_text(encoding="utf-8")) if args.denominators else None
    result = analyze_rows(
        rows,
        comparisons=_parse_comparisons(args.compare),
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    md = render_markdown(result, denominators=denominators)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
