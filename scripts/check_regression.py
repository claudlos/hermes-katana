#!/usr/bin/env python3
"""Compare the latest run against the previous best/last for the same
(model_version, benchmark, kind) tuple, and emit a regression verdict.

The metrics DB is at ``evals/_history/metrics.jsonl`` (one JSON per line,
append-only). For each comparison key, we pick:

  * ``current``  — the most recently appended row (by ts), and
  * ``baseline`` — either the previous run for that key (default) or
                   the best run on macro F1 (with --vs-best).

We compute deltas on the headline metrics and apply per-metric thresholds.
Exit code 0 means no regression; 1 means at least one threshold tripped.

Default thresholds (override with CLI):

  --max-macro-f1-drop      0.01    (1% absolute)
  --max-binary-f1-drop     0.005   (0.5% absolute)
  --max-fpr-increase       0.005   (0.5% absolute)
  --max-per-class-f1-drop  0.02    (2% absolute, applied to each class)
  --max-ece-increase       0.02    (2% absolute, only for kind=calibration)

Use --strict to fail the run if any per-class F1 drops by more than the
threshold, even when macro F1 holds; without --strict we report but don't
gate on per-class drops.

Usage:

  # Default: compare every (model, benchmark, eval) pair to its prior entry.
  python scripts/check_regression.py

  # Single pair:
  python scripts/check_regression.py --model katana_v14 --benchmark confirmed_only_v1

  # Best-of-history baseline instead of last-prior:
  python scripts/check_regression.py --vs-best
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "evals" / "_history" / "metrics.jsonl"


def load_db() -> list[dict]:
    if not DB_PATH.is_file():
        return []
    rows = []
    with DB_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key(row: dict) -> tuple[str, str, str]:
    return (row.get("model_version", ""), row.get("benchmark", ""), row.get("kind", ""))


def fmt_delta(d: float | None, good_is_positive: bool, threshold: float) -> str:
    """Render a delta with a verdict marker. `good_is_positive` is True for F1
    (up is good) and False for FPR/ECE (down is good)."""
    if d is None:
        return "—"
    sign = "+" if d >= 0 else ""
    s = f"{sign}{d:.4f}"
    if good_is_positive:
        # Regression if d < 0 and abs(d) > threshold
        if d < -threshold:
            return f"{s} ⚠"
        return f"{s} ✓"
    else:
        # Regression if d > threshold
        if d > threshold:
            return f"{s} ⚠"
        return f"{s} ✓"


def compare_eval(
    curr: dict, base: dict, t_macro: float, t_bin_f1: float, t_fpr: float, t_per_class: float, strict: bool
) -> tuple[bool, list[str]]:
    """Return (regressed, lines)."""
    regressed = False
    out = []
    out.append(f"### {curr['model_version']} on {curr['benchmark']}  (kind=eval)")
    out.append("")
    out.append(f"- current  : {curr.get('ts')} (git {curr.get('git_sha', '?')})")
    out.append(f"- baseline : {base.get('ts')} (git {base.get('git_sha', '?')})")
    out.append("")
    out.append("| metric | baseline | current | delta | gate |")
    out.append("| --- | ---: | ---: | --- | --- |")
    metrics = [
        ("macro_f1", True, t_macro, "F1↓"),
        ("binary_f1", True, t_bin_f1, "F1↓"),
        ("binary_precision", True, t_bin_f1, "P↓"),
        ("binary_recall", True, t_bin_f1, "R↓"),
        ("binary_fpr", False, t_fpr, "FPR↑"),
    ]
    for m, good_pos, thr, _gate in metrics:
        c = curr.get(m)
        b = base.get(m)
        if c is None or b is None:
            out.append(f"| {m} | {b} | {c} | — | n/a |")
            continue
        d = c - b
        verdict = fmt_delta(d, good_pos, thr)
        if (good_pos and d < -thr) or (not good_pos and d > thr):
            regressed = True
        out.append(f"| {m} | {b:.4f} | {c:.4f} | {verdict} | thr {thr:+.4f} |")

    # Per-class
    cur_pc = curr.get("per_class_f1") or {}
    bas_pc = base.get("per_class_f1") or {}
    if cur_pc and bas_pc:
        out.append("")
        out.append("Per-class F1 drops (only classes that fell):")
        any_dropped = False
        for cls in sorted(set(cur_pc) | set(bas_pc)):
            c = cur_pc.get(cls)
            b = bas_pc.get(cls)
            if c is None or b is None:
                continue
            d = c - b
            if d < -t_per_class:
                any_dropped = True
                out.append(f"  - {cls}: {b:.4f} -> {c:.4f}  ({d:+.4f}) ⚠")
                if strict:
                    regressed = True
        if not any_dropped:
            out.append("  (none beyond -{:.2%})".format(t_per_class))
    out.append("")
    return regressed, out


def compare_calibration(curr: dict, base: dict, t_ece: float) -> tuple[bool, list[str]]:
    regressed = False
    out = [
        f"### {curr['model_version']} on {curr['benchmark']}  (kind=calibration)",
        "",
        f"- current  : {curr.get('ts')} (git {curr.get('git_sha', '?')})",
        f"- baseline : {base.get('ts')} (git {base.get('git_sha', '?')})",
        "",
        "| metric | baseline | current | delta | gate |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for m, good_pos, thr in [("ece", False, t_ece), ("brier_macro", False, t_ece)]:
        c = curr.get(m)
        b = base.get(m)
        if c is None or b is None:
            out.append(f"| {m} | {b} | {c} | — | n/a |")
            continue
        d = c - b
        verdict = fmt_delta(d, good_pos, thr)
        if d > thr:
            regressed = True
        out.append(f"| {m} | {b:.4f} | {c:.4f} | {verdict} | thr {thr:+.4f} |")
    out.append("")
    return regressed, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="", help="Filter to one model (e.g. katana_v14).")
    ap.add_argument("--benchmark", default="", help="Filter to one benchmark.")
    ap.add_argument("--kind", default="", help="Filter to one kind (eval|calibration|baselines).")
    ap.add_argument(
        "--vs-best",
        action="store_true",
        help="Compare against the best historical run for the key (by macro_f1) instead of the immediately-prior one.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any per-class F1 drops past threshold (default only fails on macro/binary/fpr).",
    )
    ap.add_argument("--max-macro-f1-drop", type=float, default=0.01)
    ap.add_argument("--max-binary-f1-drop", type=float, default=0.005)
    ap.add_argument("--max-fpr-increase", type=float, default=0.005)
    ap.add_argument("--max-per-class-f1-drop", type=float, default=0.02)
    ap.add_argument("--max-ece-increase", type=float, default=0.02)
    args = ap.parse_args()

    db = load_db()
    if not db:
        print(f"[regress] empty DB at {DB_PATH.relative_to(ROOT)}; nothing to check.")
        return 0

    if args.model:
        db = [r for r in db if r.get("model_version") == args.model]
    if args.benchmark:
        db = [r for r in db if r.get("benchmark") == args.benchmark]
    if args.kind:
        db = [r for r in db if r.get("kind") == args.kind]
    if not db:
        print("[regress] no rows match filters.")
        return 0

    # Group by key (model, benchmark, kind), order by ts.
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in db:
        grouped[key(row)].append(row)
    for k in grouped:
        grouped[k].sort(key=lambda r: r.get("ts", ""))

    any_regressed = False
    sections: list[str] = []
    for k, runs in sorted(grouped.items()):
        if len(runs) < 2:
            continue  # need at least 2 to compare
        curr = runs[-1]
        if args.vs_best and k[2] == "eval":
            # pick the best run by macro_f1 (excluding the current)
            candidates = [r for r in runs[:-1] if r.get("macro_f1") is not None]
            if not candidates:
                continue
            base = max(candidates, key=lambda r: r["macro_f1"])
        else:
            base = runs[-2]

        if k[2] == "eval":
            reg, lines = compare_eval(
                curr,
                base,
                args.max_macro_f1_drop,
                args.max_binary_f1_drop,
                args.max_fpr_increase,
                args.max_per_class_f1_drop,
                args.strict,
            )
        elif k[2] == "calibration":
            reg, lines = compare_calibration(curr, base, args.max_ece_increase)
        else:
            # baselines: no historical comparison enforced (external numbers don't move on our schedule)
            continue
        sections.extend(lines)
        any_regressed = any_regressed or reg

    print("# Regression check")
    print("")
    if not sections:
        print("(No (model, benchmark, kind) tuple has at least 2 history rows yet.)")
        return 0
    print("\n".join(sections))

    print("---")
    print(f"Verdict: **{'REGRESSED' if any_regressed else 'OK'}**")
    return 1 if any_regressed else 0


if __name__ == "__main__":
    raise SystemExit(main())
