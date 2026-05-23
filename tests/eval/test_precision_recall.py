"""Precision, recall, and F1 tests for the evaluation harness.

Measures attack detection (recall) and false positive rate on benign corpus
to compute precision, recall, and F1 score. These are the primary metrics
for CI regression detection.
"""

from __future__ import annotations

from tests.eval._control import eval_false_positive_ceiling
from tests.eval.scanner_runner import run_scanners, run_scanners_on_benign


# ---------------------------------------------------------------------------
# Precision / Recall / F1
# ---------------------------------------------------------------------------

# Minimum acceptable F1 score (ratchet up with improvements)
F1_FLOOR = 0.30


def test_recall_on_injection_corpus(injection_corpus, scanner_suite):
    """Recall (detection rate) on injection attacks must exceed floor."""
    caught, total = run_scanners(injection_corpus, scanner_suite)
    recall = caught / total if total > 0 else 0.0
    assert recall >= 0.42, f"Recall {recall:.1%} ({caught}/{total}) below 42% floor"


def test_fp_rate_on_extended_benign_corpus(benign_corpus, scanner_suite):
    """False positive rate on extended benign corpus (currently 1,163 samples) must stay below ceiling."""
    fps, total = run_scanners_on_benign(benign_corpus, scanner_suite)
    fp_rate = fps / total if total > 0 else 0.0
    fp_rate_ceiling = eval_false_positive_ceiling()
    assert fp_rate <= fp_rate_ceiling, f"FP rate {fp_rate:.1%} ({fps}/{total}) above ceiling {fp_rate_ceiling:.1%}"


def test_f1_score(injection_corpus, benign_corpus, scanner_suite):
    """F1 score combining recall on attacks and precision from benign corpus.

    F1 = 2 * precision * recall / (precision + recall)
    where precision = TP / (TP + FP) = (total_benign - FP) / total_benign
    """
    # Recall on attacks
    caught, attack_total = run_scanners(injection_corpus, scanner_suite)
    recall = caught / attack_total if attack_total > 0 else 0.0

    # Precision from benign corpus
    fps, benign_total = run_scanners_on_benign(benign_corpus, scanner_suite)
    precision = (benign_total - fps) / benign_total if benign_total > 0 else 1.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    assert f1 >= F1_FLOOR, (
        f"F1 {f1:.3f} below floor {F1_FLOOR:.3f} "
        f"(precision={precision:.3f}, recall={recall:.3f}, "
        f"FP={fps}/{benign_total}, caught={caught}/{attack_total})"
    )


# ---------------------------------------------------------------------------
# Per-scanner precision (which scanners cause the most FPs?)
# ---------------------------------------------------------------------------


def test_per_scanner_false_positives(benign_corpus, scanner_suite):
    """Report per-scanner false positive counts. Fail if any scanner has >25% FP rate."""
    from tests.eval.scanner_runner import _scan_one

    per_scanner_fps = {name: 0 for name in scanner_suite}
    total = len(benign_corpus)

    for text in benign_corpus:
        results = _scan_one(text, scanner_suite)
        for name, findings in results.items():
            if findings:
                per_scanner_fps[name] += 1

    failing = []
    for name, fps in per_scanner_fps.items():
        rate = fps / total if total > 0 else 0.0
        if rate > 0.25:  # >25% FP rate is unacceptable
            failing.append(f"  {name}: {fps}/{total} ({rate:.1%})")

    assert not failing, "Scanners with >25% FP rate on benign corpus:\n" + "\n".join(failing)
