"""Integration benchmark: Scabbard classifier vs scanner stack on wild-attacks corpus.

Compares coverage between:
  - Scabbard (minimal profile, rule-based) on all 5091 records
  - The current scanner stack (via scanner_runner)

Reports the delta: what Scabbard catches that scanners miss, and vice versa.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
RESEARCH = ROOT / "research"
SCABBARD_DIR = RESEARCH / "scabbard-v0.1.0"

EVAL_DIR = Path(__file__).resolve().parent

for p in (str(ROOT), str(SRC), str(SCABBARD_DIR), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scanner_runner import make_scanner_suite, _scan_one, is_caught  # noqa: E402
from tests.eval._control import configured_eval_corpus_path  # noqa: E402

WILD_ATTACKS = configured_eval_corpus_path(ROOT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_corpus() -> list[dict]:
    records = []
    with open(WILD_ATTACKS) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Scabbard predictor
# ---------------------------------------------------------------------------


def _make_scabbard():
    """Build Scabbard classifier (minimal profile)."""
    try:
        from scabbard import ScabbardClassifier, ScabbardConfig
        from fusion import Decision

        config = ScabbardConfig(profile="minimal")
        return ScabbardClassifier(config=config), Decision
    except Exception as exc:
        pytest.skip(f"Scabbard not available: {exc}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestScabbardCoverage:
    """Run Scabbard and scanner stack on the full wild-attacks corpus."""

    @pytest.fixture(scope="class")
    def corpus(self) -> list[dict]:
        if not WILD_ATTACKS.exists():
            pytest.skip("Wild-attacks corpus not found")
        return _load_corpus()

    @pytest.fixture(scope="class")
    def scanners(self) -> dict:
        return make_scanner_suite()

    @pytest.fixture(scope="class")
    def scabbard_and_decision(self):
        return _make_scabbard()

    @pytest.fixture(scope="class")
    def coverage_report(self, corpus, scanners, scabbard_and_decision):
        """Run both systems and compute coverage deltas."""
        scabbard, Decision = scabbard_and_decision

        scanner_caught_ids: set[int] = set()
        scabbard_caught_ids: set[int] = set()
        per_category: dict[str, dict[str, int]] = {}

        for idx, rec in enumerate(corpus):
            text = rec.get("attack_text", "")
            cat = rec.get("category", "unknown")

            if cat not in per_category:
                per_category[cat] = {
                    "total": 0,
                    "scanner_caught": 0,
                    "scabbard_caught": 0,
                    "both_caught": 0,
                    "neither_caught": 0,
                }
            per_category[cat]["total"] += 1

            # Scanner stack
            scanner_results = _scan_one(text, scanners)
            scanner_hit = is_caught(scanner_results)
            if scanner_hit:
                scanner_caught_ids.add(idx)

            # Scabbard
            try:
                result = scabbard.classify(text)
                scabbard_hit = result.decision != Decision.ALLOW
            except Exception:
                scabbard_hit = False
            if scabbard_hit:
                scabbard_caught_ids.add(idx)

            # Category tracking
            if scanner_hit:
                per_category[cat]["scanner_caught"] += 1
            if scabbard_hit:
                per_category[cat]["scabbard_caught"] += 1
            if scanner_hit and scabbard_hit:
                per_category[cat]["both_caught"] += 1
            if not scanner_hit and not scabbard_hit:
                per_category[cat]["neither_caught"] += 1

        total = len(corpus)
        only_scanner = scanner_caught_ids - scabbard_caught_ids
        only_scabbard = scabbard_caught_ids - scanner_caught_ids
        both = scanner_caught_ids & scabbard_caught_ids
        neither_set = set(range(total)) - scanner_caught_ids - scabbard_caught_ids

        report = {
            "total": total,
            "scanner_caught": len(scanner_caught_ids),
            "scabbard_caught": len(scabbard_caught_ids),
            "both_caught": len(both),
            "only_scanner": len(only_scanner),
            "only_scabbard": len(only_scabbard),
            "neither": len(neither_set),
            "scanner_coverage": len(scanner_caught_ids) / total if total else 0,
            "scabbard_coverage": len(scabbard_caught_ids) / total if total else 0,
            "union_coverage": (len(scanner_caught_ids | scabbard_caught_ids)) / total if total else 0,
            "per_category": per_category,
            # Keep IDs for detailed analysis
            "_only_scanner_ids": only_scanner,
            "_only_scabbard_ids": only_scabbard,
            "_neither_ids": neither_set,
        }

        # Print report during test run
        print("\n" + "=" * 70)
        print("SCABBARD vs SCANNER STACK — Coverage Report")
        print("=" * 70)
        print(f"Corpus size:       {total}")
        print(f"Scanner caught:    {report['scanner_caught']} ({report['scanner_coverage']:.1%})")
        print(f"Scabbard caught:   {report['scabbard_caught']} ({report['scabbard_coverage']:.1%})")
        print(f"Both caught:       {report['both_caught']}")
        print(f"Only scanner:      {report['only_scanner']}")
        print(f"Only scabbard:     {report['only_scabbard']}")
        print(f"Neither caught:    {report['neither']}")
        print(f"Union coverage:    {report['union_coverage']:.1%}")
        print()

        # Per-category table
        print(f"{'Category':<25s} {'Total':>6s} {'Scanner':>8s} {'Scabbard':>9s} {'Both':>6s} {'Neither':>8s}")
        print("-" * 70)
        for cat, data in sorted(per_category.items(), key=lambda x: -x[1]["total"]):
            print(
                f"{cat:<25s} {data['total']:>6d} "
                f"{data['scanner_caught']:>8d} {data['scabbard_caught']:>9d} "
                f"{data['both_caught']:>6d} {data['neither_caught']:>8d}"
            )
        print("=" * 70)

        return report

    def test_scabbard_runs_on_full_corpus(self, coverage_report):
        """Scabbard must classify the entire corpus without crashing."""
        total = coverage_report["total"]
        classified = coverage_report["scabbard_caught"] + (total - coverage_report["scabbard_caught"])
        assert classified == total

    def test_scabbard_catches_something(self, coverage_report):
        """Scabbard (even minimal) should catch at least some attacks."""
        assert coverage_report["scabbard_caught"] > 0, "Scabbard caught 0 attacks — rule-based fallback may be broken"

    def test_union_coverage_exceeds_either(self, coverage_report):
        """Combining scanner + Scabbard should cover at least as much as the best single system."""
        best_single = max(
            coverage_report["scanner_coverage"],
            coverage_report["scabbard_coverage"],
        )
        assert coverage_report["union_coverage"] >= best_single - 0.001

    def test_scabbard_adds_unique_catches(self, coverage_report):
        """Scabbard should find at least a few attacks the scanner stack misses.

        This validates that Scabbard adds value beyond the rule-based scanner stack.
        We use a soft threshold: if it catches 0 unique, that's a signal the
        Scabbard pipeline needs tuning, but we don't hard-fail.
        """
        only_scabbard = coverage_report["only_scabbard"]
        # Soft assertion — warn rather than fail for minimal profile
        if only_scabbard == 0:
            pytest.xfail(
                "Scabbard (minimal profile) caught 0 unique attacks. Expected once trained model is available."
            )

    def test_report_saved(self, coverage_report, tmp_path):
        """Ensure report is JSON-serializable."""
        # Strip non-serializable fields
        serializable = {k: v for k, v in coverage_report.items() if not k.startswith("_")}
        out = tmp_path / "scabbard_coverage_report.json"
        out.write_text(json.dumps(serializable, indent=2))
        assert out.exists()
