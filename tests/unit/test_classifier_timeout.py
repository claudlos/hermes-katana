"""Tests for KatanaScabbardMiddleware._classify_with_timeout.

When the underlying classifier exceeds the configured timeout, the middleware
synthesizes a fallback ClassificationResult according to
``classifier_timeout_decision`` (allow|deny). This protects the agent loop
from a hung or slow classifier without dropping the whole call.
"""

from __future__ import annotations

import time


from hermes_katana.middleware.integration import KatanaScabbardMiddleware
from hermes_katana.scabbard.fusion import ClassificationResult, Decision


class _SlowClassifier:
    """Mimics ScabbardClassifier minimally, with a configurable sleep."""

    class _Cfg:
        def __init__(self, timeout: float, decision: str) -> None:
            self.classifier_timeout_seconds = timeout
            self.classifier_timeout_decision = decision
            self.model_version = "test-slow-v1"

    def __init__(self, sleep_seconds: float, timeout: float, decision: str) -> None:
        self._sleep = sleep_seconds
        self.config = self._Cfg(timeout, decision)

    def classify(self, text, origin=None):
        time.sleep(self._sleep)
        return ClassificationResult(
            scores={"clean": 0.9},
            decision=Decision.ALLOW,
            top_category="clean",
            confidence=0.9,
        )


def _mw_with(classifier) -> KatanaScabbardMiddleware:
    mw = KatanaScabbardMiddleware()
    mw._classifier = classifier  # bypass lazy load
    return mw


def test_within_timeout_passes_through():
    """Classifier finishes faster than budget -> normal result returned."""
    clf = _SlowClassifier(sleep_seconds=0.01, timeout=1.0, decision="allow")
    mw = _mw_with(clf)
    result = mw._classify_with_timeout("hi", "user_input")
    assert result.decision == Decision.ALLOW
    assert result.top_category == "clean"


def test_timeout_with_fallback_allow():
    """Classifier hangs past budget -> fallback ALLOW."""
    clf = _SlowClassifier(sleep_seconds=0.5, timeout=0.05, decision="allow")
    mw = _mw_with(clf)
    t0 = time.perf_counter()
    result = mw._classify_with_timeout("hi", "user_input")
    elapsed = time.perf_counter() - t0
    assert result.decision == Decision.ALLOW
    assert result.top_category == "timeout_fallback"
    # Should return promptly after the timeout fires.
    assert elapsed < 0.3, f"timeout enforcement too slow: {elapsed:.3f}s"


def test_timeout_with_fallback_deny():
    """Classifier hangs past budget -> fallback DENY (fail-closed)."""
    clf = _SlowClassifier(sleep_seconds=0.5, timeout=0.05, decision="deny")
    mw = _mw_with(clf)
    result = mw._classify_with_timeout("hi", "user_input")
    assert result.decision == Decision.BLOCK
    assert result.top_category == "timeout_fallback"
    assert result.confidence == 1.0


def test_timeout_zero_disables_wrapper():
    """timeout=0 means no wrapping — slow calls block normally."""
    clf = _SlowClassifier(sleep_seconds=0.05, timeout=0.0, decision="allow")
    mw = _mw_with(clf)
    t0 = time.perf_counter()
    result = mw._classify_with_timeout("hi", None)
    elapsed = time.perf_counter() - t0
    assert result.decision == Decision.ALLOW
    assert result.top_category == "clean"  # original result, not fallback
    assert elapsed >= 0.04, f"unexpectedly fast: {elapsed:.3f}s (timeout disabled)"
