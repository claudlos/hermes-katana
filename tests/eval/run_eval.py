#!/usr/bin/env python3
"""Standalone evaluation script for HermesKatana scanner coverage.

Usage:
    python tests/eval/run_eval.py                    # print summary
    python tests/eval/run_eval.py --max-records 200  # quick capped smoke run
    python tests/eval/run_eval.py --update-baseline  # save new baseline.json
    python tests/eval/run_eval.py --compare          # compare against saved baseline

Exit codes:
    0 = all floors met (or no corpus to check)
    1 = regression detected
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from tests.eval._control import (  # noqa: E402
    configured_eval_corpus_path,
    EVAL_MAX_RECORDS_ENV,
    baseline_label_scanner_reference,
    configured_max_records,
    load_jsonl_records,
)
from tests.eval.scanner_runner import make_scanner_suite, run_scanners_detailed  # noqa: E402
from tests.eval.test_coverage import INJECTION_COVERAGE_FLOOR  # noqa: E402
from tests.eval.test_coverage_by_category import CATEGORY_FLOORS  # noqa: E402
from hermes_katana.cli._support import collect_ml_runtime_status  # noqa: E402

try:
    from hermes_katana.scanner.semantic_recall import semantic_backend_status  # noqa: E402
except Exception:
    semantic_backend_status = None

CORPUS_PATH = configured_eval_corpus_path(Path(_root))
BASELINE_PATH = Path(__file__).parent / "baseline.json"
BASELINE_SCHEMA_VERSION = 2


def load_corpus(path: Path, *, max_records: int | None = None) -> list[dict]:
    return load_jsonl_records(path, max_records=max_records)


LABEL_NAMES = ("injection", "content_harm", "system_prompt_leak", "meta_discussion")


def print_semantic_backend_info() -> None:
    """Print semantic backend readiness for eval/debug visibility."""
    if semantic_backend_status is None:
        print("  Semantic backend: unavailable (semantic_recall import failed)")
        return
    status = semantic_backend_status()
    print(f"  Semantic backend: {status['backend']}")
    print(f"  Reason:           {status['reason']}")
    print(f"  Active index:     {status['active_index_dir']}")


def print_ml_runtime_readiness() -> None:
    status = collect_ml_runtime_status()
    eval_status = status["eval"]
    print(f"  Eval readiness:   {'ready' if eval_status['ready'] else 'partial'}")
    if eval_status["blockers"]:
        print(f"  Blockers:         {'; '.join(eval_status['blockers'][:3])}")
    if eval_status["warnings"]:
        print(f"  Warnings:         {'; '.join(eval_status['warnings'][:2])}")


def print_summary(details: dict, label: str = "injection", floor: float | None = None) -> None:
    """Print a formatted coverage summary table."""
    if floor is None:
        floor = INJECTION_COVERAGE_FLOOR
    print()
    print(f"{'=' * 70}")
    print(f"  HermesKatana Coverage Report — {label}")
    print(f"{'=' * 70}")
    print()
    print(f"  Total samples:  {details['total']}")
    print(f"  Caught:         {details['caught']}")
    print(f"  Coverage:       {details['coverage']:.1%}")
    print(f"  Floor:          {floor:.1%}")
    status = "PASS" if details["coverage"] >= floor else "FAIL"
    print(f"  Status:         {status}")
    print()

    # Per-scanner table
    print(f"  {'Scanner':<15} {'Detections':>12}")
    print(f"  {'-' * 15} {'-' * 12}")
    for sname, sdata in details["per_scanner"].items():
        print(f"  {sname:<15} {sdata['deny']:>12}")
    print()

    # Per-category table
    print(f"  {'Category':<25} {'Caught':>8} {'Total':>8} {'Coverage':>10} {'Floor':>8} {'Status':>8}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 8} {'-' * 8}")
    floor_map = dict(CATEGORY_FLOORS)
    for cat, cdata in details["per_category"].items():
        floor = floor_map.get(cat, None)
        floor_str = f"{floor:.0%}" if floor is not None else "—"
        if floor is not None:
            cat_status = "PASS" if cdata["coverage"] >= floor else "FAIL"
        else:
            cat_status = "—"
        print(
            f"  {cat:<25} {cdata['caught']:>8} {cdata['total']:>8}"
            f" {cdata['coverage']:>9.1%} {floor_str:>8} {cat_status:>8}"
        )
    print()

    if details.get("missed_samples"):
        print("  First missed samples (up to 20):")
        for i, sample in enumerate(details["missed_samples"][:5], 1):
            print(f"    {i}. {sample!r}")
        if len(details["missed_samples"]) > 5:
            print(f"    ... and {len(details['missed_samples']) - 5} more")
        print()


def build_baseline(details: dict, label_details: dict[str, dict], *, corpus_path: Path) -> dict:
    """Build a baseline dict from detailed results."""
    try:
        import subprocess

        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=_root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    per_label = {
        lbl: {
            "total": lbl_details["total"],
            "caught": lbl_details["caught"],
            "coverage": round(lbl_details["coverage"], 4),
            "per_scanner": lbl_details["per_scanner"],
            "per_category": lbl_details["per_category"],
        }
        for lbl, lbl_details in label_details.items()
    }
    injection_details = per_label.get("injection")
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "corpus_path": str(corpus_path),
        "total_records": details["total"],
        "coverage": round(details["coverage"], 4),
        "per_scanner": details["per_scanner"],
        "per_category": details["per_category"],
        "per_label": per_label,
        "ml_runtime": collect_ml_runtime_status(),
    }
    if injection_details is not None:
        baseline["total_injection"] = injection_details["total"]
        baseline["caught_injection"] = injection_details["caught"]
        baseline["coverage_injection"] = injection_details["coverage"]
    if semantic_backend_status is not None:
        baseline["semantic_backend"] = semantic_backend_status()
    return baseline


def compare_baseline(current: dict, saved: dict, *, label: str = "injection") -> tuple[list[str], list[str]]:
    """Compare current results against a saved baseline.

    Returns (regressions, notes). The comparison uses label-scoped baseline
    slices when available so injection-only evals are compared like-for-like.
    """
    regressions = []
    notes: list[str] = []
    baseline_label = saved.get("per_label", {}).get(label, {})

    # Overall coverage
    baseline_coverage = baseline_label.get("coverage", saved.get("coverage_injection", 0))
    if current["coverage"] < baseline_coverage - 0.005:
        regressions.append(f"Overall coverage: {current['coverage']:.1%} < baseline {baseline_coverage:.1%}")

    # Per-scanner
    baseline_scanners, skip_reason = baseline_label_scanner_reference(saved, label=label)
    if baseline_scanners is None:
        notes.append(skip_reason)
    else:
        for sname, sdata in current.get("per_scanner", {}).items():
            baseline_deny = baseline_scanners.get(sname, {}).get("deny", 0)
            if sdata["deny"] < baseline_deny:
                regressions.append(f"Scanner {sname}: {sdata['deny']} detections < baseline {baseline_deny}")

    # Per-category
    baseline_categories = baseline_label.get("per_category")
    if not isinstance(baseline_categories, dict):
        notes.append(f"Baseline lacks per-category data for label {label!r}; skipping category regression checks.")
    else:
        for cat, cdata in current.get("per_category", {}).items():
            baseline_cat = baseline_categories.get(cat, {})
            baseline_cov = baseline_cat.get("coverage")
            if baseline_cov is None:
                continue
            if cdata["coverage"] < baseline_cov - 0.02:
                regressions.append(f"Category {cat}: {cdata['coverage']:.1%} < baseline {baseline_cov:.1%}")

    return regressions, notes


def main() -> int:
    parser = argparse.ArgumentParser(description="HermesKatana coverage evaluator")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Save current results as new baseline",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against saved baseline",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help=(f"Cap the number of corpus records for a faster smoke run. Defaults to {EVAL_MAX_RECORDS_ENV} when set."),
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CORPUS_PATH,
        help="Path to the eval corpus JSONL. Defaults to the strict held-out corpus when present.",
    )
    args = parser.parse_args()
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be a positive integer")

    corpus_path = args.corpus
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        print("Run phase 0.1 first to build the corpus.")
        return 0  # Not a failure — corpus is optional

    print("Loading corpus...")
    max_records = args.max_records if args.max_records is not None else configured_max_records()
    corpus = load_corpus(corpus_path, max_records=max_records)
    if max_records is not None:
        print(f"Corpus capped to first {len(corpus)} records")

    # Split by label
    by_label: dict[str, list[dict]] = {}
    for rec in corpus:
        lbl = rec.get("clean_label", "unknown")
        by_label.setdefault(lbl, []).append(rec)

    print(f"Loaded {len(corpus)} records total:")
    for lbl in LABEL_NAMES:
        print(f"  {lbl}: {len(by_label.get(lbl, []))}")

    print("\nBuilding scanner suite...")
    scanners = make_scanner_suite()
    print(f"  Active scanners: {', '.join(scanners.keys())}")
    print_semantic_backend_info()
    print_ml_runtime_readiness()

    exit_code = 0

    # --- Overall (all 5091) ---
    print("\n--- Overall Evaluation (all records) ---")
    t0 = time.monotonic()
    all_details = run_scanners_detailed(corpus, scanners)
    elapsed = time.monotonic() - t0
    print(f"Evaluation completed in {elapsed:.1f}s")
    print_summary(all_details, label="ALL LABELS", floor=0.0)

    # --- Per-label breakdown ---
    label_details: dict[str, dict] = {}
    for lbl in LABEL_NAMES:
        subset = by_label.get(lbl, [])
        if not subset:
            continue
        print(f"--- {lbl} ({len(subset)} records) ---")
        lbl_details = run_scanners_detailed(subset, scanners)
        label_details[lbl] = lbl_details
        lbl_floor = INJECTION_COVERAGE_FLOOR if lbl == "injection" else 0.0
        print_summary(lbl_details, label=lbl, floor=lbl_floor)

    # --- Per-scanner contributions (overall) ---
    print(f"  {'Scanner':<20} {'Detections':>12} {'% of Total':>12}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12}")
    for sname, sdata in all_details["per_scanner"].items():
        pct = sdata["deny"] / all_details["total"] * 100 if all_details["total"] else 0
        print(f"  {sname:<20} {sdata['deny']:>12} {pct:>11.1f}%")
    print()

    # Use injection details for floor / baseline checks (backwards compat)
    details = label_details.get("injection", all_details)

    # Check injection floor
    if details["coverage"] < INJECTION_COVERAGE_FLOOR:
        print(f"FAIL: Injection coverage {details['coverage']:.1%} below floor {INJECTION_COVERAGE_FLOOR:.1%}")
        exit_code = 1

    floor_map = dict(CATEGORY_FLOORS)
    for cat, cdata in details["per_category"].items():
        floor = floor_map.get(cat)
        if floor is not None and cdata["coverage"] < floor:
            print(f"FAIL: Category {cat!r} coverage {cdata['coverage']:.1%} below floor {floor:.1%}")
            exit_code = 1

    if args.update_baseline:
        baseline = build_baseline(all_details, label_details, corpus_path=corpus_path)
        with open(BASELINE_PATH, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Baseline saved to {BASELINE_PATH}")

    if args.compare:
        if not BASELINE_PATH.exists():
            print("No baseline.json found — run with --update-baseline first")
            return 1
        with open(BASELINE_PATH) as f:
            saved = json.load(f)
        regressions, notes = compare_baseline(details, saved, label="injection")
        saved_corpus_path = saved.get("corpus_path")
        if isinstance(saved_corpus_path, str) and saved_corpus_path != str(corpus_path):
            print(f"NOTE: corpus path changed since baseline: current={corpus_path}, baseline={saved_corpus_path}")
        if semantic_backend_status is not None and saved.get("semantic_backend"):
            current_backend = semantic_backend_status().get("backend")
            baseline_backend = saved.get("semantic_backend", {}).get("backend")
            if current_backend != baseline_backend:
                print(
                    f"NOTE: semantic backend changed since baseline: current={current_backend}, baseline={baseline_backend}"
                )
        for note in notes:
            print(f"NOTE: {note}")
        if regressions:
            print("REGRESSIONS detected vs baseline:")
            for r in regressions:
                print(f"  - {r}")
            exit_code = 1
        else:
            print("No regressions vs baseline")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
