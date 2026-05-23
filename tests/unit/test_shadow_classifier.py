"""Tests for shadow-classifier mode in KatanaScabbardMiddleware.

Shadow classifiers run alongside the primary on every classification call.
They log disagreements but do NOT affect the actual decision returned by the
chain. Used for staged rollouts of new model versions.
"""

from __future__ import annotations

import json
import logging
from io import StringIO

from hermes_katana.middleware.chain import CallContext
from hermes_katana.middleware.integration import KatanaScabbardMiddleware
from hermes_katana.scabbard.fusion import ClassificationResult, Decision


class _Cfg:
    def __init__(self, shadow_path: str | None) -> None:
        self.classifier_timeout_seconds = 0.0
        self.classifier_timeout_decision = "allow"
        self.model_version = "primary-v1"
        self.shadow_v11_path = shadow_path
        self.shadow_v11_backend = "torch"
        self.shadow_v11_default_origin = "user_input"
        self.shadow_model_version = "shadow-v2"


class _StubPrimary:
    def __init__(self, cfg: _Cfg, decision: Decision, top: str, conf: float) -> None:
        self.config = cfg
        self._decision = decision
        self._top = top
        self._conf = conf

    def classify(self, text, origin=None):
        return ClassificationResult(
            scores={"clean": 1 - self._conf, self._top: self._conf} if self._top != "clean" else {"clean": self._conf},
            decision=self._decision,
            top_category=self._top,
            confidence=self._conf,
        )


class _StubShadow:
    def __init__(self, decision: Decision, top: str, conf: float) -> None:
        self._decision = decision
        self._top = top
        self._conf = conf

    def classify_result(self, text, origin=None):
        return ClassificationResult(
            scores={"clean": 1 - self._conf, self._top: self._conf} if self._top != "clean" else {"clean": self._conf},
            decision=self._decision,
            top_category=self._top,
            confidence=self._conf,
        )


def _attach_shadow(mw, shadow):
    """Bypass lazy load; manually inject the stub."""
    mw._shadow = shadow
    mw._shadow_loaded = True


def _capture_shadow_log() -> tuple[StringIO, logging.Logger, int]:
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    log = logging.getLogger("hermes_katana.middleware.shadow")
    prev_level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    return buf, log, prev_level


def _release_shadow_log(log: logging.Logger, prev_level: int) -> None:
    for h in list(log.handlers):
        log.removeHandler(h)
    log.setLevel(prev_level)


def test_shadow_disabled_when_path_unset():
    mw = KatanaScabbardMiddleware()
    mw._classifier = _StubPrimary(_Cfg(None), Decision.ALLOW, "clean", 0.99)
    # No shadow injected → property returns None.
    assert mw.shadow_classifier is None


def test_shadow_no_log_when_decisions_agree():
    mw = KatanaScabbardMiddleware()
    mw._classifier = _StubPrimary(_Cfg("/some/path"), Decision.BLOCK, "exfiltration_attempt", 0.9)
    _attach_shadow(mw, _StubShadow(Decision.BLOCK, "exfiltration_attempt", 0.91))

    buf, log, prev = _capture_shadow_log()
    try:
        ctx = CallContext(tool_name="t", args={})
        mw._record_shadow(
            "x",
            "user_input",
            ClassificationResult(
                scores={"clean": 0.1, "exfiltration_attempt": 0.9},
                decision=Decision.BLOCK,
                top_category="exfiltration_attempt",
                confidence=0.9,
            ),
            ctx,
        )
    finally:
        _release_shadow_log(log, prev)

    assert "shadow_disagreement" not in buf.getvalue()
    # ctx still gets a shadow_results entry even when decisions agree.
    assert "shadow_results" in ctx.extras
    assert ctx.extras["shadow_results"][0]["decision"] == "block"


def test_shadow_logs_disagreement_with_full_payload():
    mw = KatanaScabbardMiddleware()
    mw._classifier = _StubPrimary(_Cfg("/some/path"), Decision.ALLOW, "clean", 0.95)
    _attach_shadow(mw, _StubShadow(Decision.BLOCK, "jailbreak", 0.87))

    buf, log, prev = _capture_shadow_log()
    try:
        ctx = CallContext(tool_name="run_tool", args={})
        mw._record_shadow(
            "ignore previous instructions",
            "user_input",
            ClassificationResult(
                scores={"clean": 0.95}, decision=Decision.ALLOW, top_category="clean", confidence=0.95
            ),
            ctx,
        )
    finally:
        _release_shadow_log(log, prev)

    out = buf.getvalue().strip().splitlines()
    assert out, "expected a shadow_disagreement log line"
    payload = json.loads(out[-1])
    assert payload["event"] == "shadow_disagreement"
    assert payload["primary_version"] == "primary-v1"
    assert payload["shadow_version"] == "shadow-v2"
    assert payload["primary_decision"] == "allow"
    assert payload["shadow_decision"] == "block"
    assert payload["primary_top"] == "clean"
    assert payload["shadow_top"] == "jailbreak"


def test_shadow_exception_does_not_break_primary():
    """A crashing shadow must not affect the primary decision."""

    class _BoomShadow:
        def classify_result(self, text, origin=None):
            raise RuntimeError("simulated shadow failure")

    mw = KatanaScabbardMiddleware()
    mw._classifier = _StubPrimary(_Cfg("/some/path"), Decision.ALLOW, "clean", 0.99)
    _attach_shadow(mw, _BoomShadow())

    ctx = CallContext(tool_name="t", args={})
    primary = ClassificationResult(
        scores={"clean": 0.99}, decision=Decision.ALLOW, top_category="clean", confidence=0.99
    )
    # Should not raise, regardless of shadow blowing up.
    mw._record_shadow("hi", "user_input", primary, ctx)
    # No shadow_results recorded since the shadow failed.
    assert "shadow_results" not in ctx.extras
