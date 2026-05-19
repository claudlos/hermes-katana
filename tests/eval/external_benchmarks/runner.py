"""Runner for JailbreakBench evaluations against HermesKatana.

Measures detection rates across JBB attack methods, models, and categories.
Uses the shared scanner_runner infrastructure from the eval harness.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from tests.eval.scanner_runner import (  # noqa: E402
    make_scanner_suite,
    run_scanners_detailed,
    run_scanners_on_benign,
)
from tests.eval.external_benchmarks.loader import (  # noqa: E402
    METHODS,
    MODELS,
    load_full_jbb_corpus,
    load_jbb_benign_prompts,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


def run_jbb_evaluation(
    methods: tuple[str, ...] = METHODS,
    models: tuple[str, ...] = MODELS,
    include_behaviors: bool = True,
) -> dict[str, Any]:
    """Run full JailbreakBench evaluation and return detailed results.

    Returns a dict with:
      - overall: run_scanners_detailed output for the full corpus
      - by_method: {method: detailed_results}
      - by_jbb_category: {jbb_cat: detailed_results}
      - benign: {false_positives, total, fpr}
      - timing: {elapsed_s, samples_per_sec}
    """
    scanners = make_scanner_suite()

    # Load corpus
    corpus = load_full_jbb_corpus(methods=methods, models=models, include_behaviors=include_behaviors)
    if not corpus:
        return {"error": "No corpus loaded — is jailbreakbench installed?"}

    t0 = time.monotonic()

    # Overall evaluation
    overall = run_scanners_detailed(corpus, scanners)

    # By method
    by_method: dict[str, dict] = {}
    for method in methods:
        subset = [r for r in corpus if r.get("method") == method]
        if subset:
            by_method[method] = run_scanners_detailed(subset, scanners)

    # Behaviors subset (no method field)
    behavior_subset = [r for r in corpus if r.get("source") == "jailbreakbench_behaviors"]
    if behavior_subset:
        by_method["behaviors"] = run_scanners_detailed(behavior_subset, scanners)

    # By JBB category
    by_jbb_category: dict[str, dict] = {}
    jbb_cats = {r.get("jbb_category", "unknown") for r in corpus}
    for cat in sorted(jbb_cats):
        subset = [r for r in corpus if r.get("jbb_category") == cat]
        if subset:
            by_jbb_category[cat] = run_scanners_detailed(subset, scanners)

    # Benign FP check
    benign = load_jbb_benign_prompts()
    fp_count, benign_total = run_scanners_on_benign(benign, scanners)

    elapsed = time.monotonic() - t0
    total_samples = len(corpus) + len(benign)

    return {
        "overall": overall,
        "by_method": by_method,
        "by_jbb_category": by_jbb_category,
        "benign": {
            "false_positives": fp_count,
            "total": benign_total,
            "fpr": fp_count / benign_total if benign_total else 0.0,
        },
        "timing": {
            "elapsed_s": round(elapsed, 2),
            "samples_per_sec": round(total_samples / elapsed, 1) if elapsed > 0 else 0,
        },
        "corpus_size": len(corpus),
        "scanners_active": list(scanners.keys()),
    }


def print_jbb_report(results: dict[str, Any]) -> None:
    """Print a formatted report of JBB evaluation results."""
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    overall = results["overall"]
    print()
    print("=" * 70)
    print("  JailbreakBench Evaluation Report")
    print("=" * 70)
    print()
    print(f"  Corpus size:    {results['corpus_size']}")
    print(f"  Active scanners: {len(results['scanners_active'])}")
    print(f"  Time:           {results['timing']['elapsed_s']}s ({results['timing']['samples_per_sec']} samples/s)")
    print()

    # Overall
    print(f"  Overall Detection Rate: {overall['coverage']:.1%} ({overall['caught']}/{overall['total']})")
    print()

    # By method
    print(f"  {'Method':<15} {'Caught':>8} {'Total':>8} {'Detection':>10}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 8} {'-' * 10}")
    for method, data in results["by_method"].items():
        print(f"  {method:<15} {data['caught']:>8} {data['total']:>8} {data['coverage']:>9.1%}")
    print()

    # By JBB category
    print(f"  {'JBB Category':<30} {'Caught':>8} {'Total':>8} {'Detection':>10}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 10}")
    for cat, data in results["by_jbb_category"].items():
        print(f"  {cat:<30} {data['caught']:>8} {data['total']:>8} {data['coverage']:>9.1%}")
    print()

    # Benign FP
    benign = results["benign"]
    print(f"  Benign FP Rate: {benign['fpr']:.1%} ({benign['false_positives']}/{benign['total']})")
    print()

    # Per-scanner contributions
    print(f"  {'Scanner':<20} {'Detections':>12}")
    print(f"  {'-' * 20} {'-' * 12}")
    for sname, sdata in overall["per_scanner"].items():
        if sdata["deny"] > 0:
            print(f"  {sname:<20} {sdata['deny']:>12}")
    print()


def save_results(results: dict[str, Any], path: Path | None = None) -> Path:
    """Save results to JSON file."""
    if path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "jailbreakbench_results.json"

    # Make results JSON-serializable
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    return path


def main() -> int:
    """CLI entry point for standalone JBB evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="Run JailbreakBench evaluation")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--no-behaviors", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    results = run_jbb_evaluation(
        methods=tuple(args.methods),
        models=tuple(args.models),
        include_behaviors=not args.no_behaviors,
    )
    print_jbb_report(results)

    if args.save:
        path = save_results(results)
        print(f"Results saved to {path}")

    # Return 0 unless error
    return 1 if "error" in results else 0


if __name__ == "__main__":
    sys.exit(main())
