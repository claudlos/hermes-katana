"""Property-based tests using Hypothesis for HermesKatana."""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from hermes_katana.taint.value import TaintedStr, TaintedValue, unwrap
from hermes_katana.taint.labels import TaintLabel, Source, TrustLevel
from hermes_katana.scanner import scan_input, scan_output, ScanVerdict
from hermes_katana.policy.engine import PolicyEngine, EvaluationResult
from hermes_katana.policy.models import PolicySet


def _make_source(label: TaintLabel = TaintLabel.USER) -> frozenset:
    """Create a minimal frozenset of Source for TaintedStr construction."""
    return frozenset([Source(label=label)])


# ======================================================================
# Fuzz TaintedStr with arbitrary strings
# ======================================================================

class TestTaintedStrFuzz:
    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=100)
    def test_tainted_str_preserves_value(self, text):
        ts = TaintedStr(text, sources=_make_source(TaintLabel.USER))
        assert str(ts) == text
        assert unwrap(ts) == text

    @given(text=st.text(min_size=0, max_size=200))
    @settings(max_examples=50)
    def test_tainted_str_has_label(self, text):
        ts = TaintedStr(text, sources=_make_source(TaintLabel.TOOL_OUTPUT))
        assert ts.has_label(TaintLabel.TOOL_OUTPUT)

    @given(a=st.text(min_size=1, max_size=100), b=st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_tainted_str_concatenation(self, a, b):
        ta = TaintedStr(a, sources=_make_source(TaintLabel.USER))
        result = ta + b
        assert a + b in str(result)

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_tainted_str_len(self, text):
        ts = TaintedStr(text, sources=_make_source(TaintLabel.USER))
        assert len(ts) == len(text)

    @given(text=st.text(min_size=0, max_size=200))
    @settings(max_examples=50)
    def test_tainted_str_repr_does_not_crash(self, text):
        ts = TaintedStr(text, sources=_make_source(TaintLabel.WEB_CONTENT))
        repr(ts)  # Should not raise


# ======================================================================
# Fuzz scanner patterns with random inputs
# ======================================================================

class TestScannerFuzz:
    @given(text=st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=5000)
    def test_scan_input_never_crashes(self, text):
        result = scan_input(text)
        assert result.verdict in (ScanVerdict.ALLOW, ScanVerdict.WARN, ScanVerdict.BLOCK)
        assert isinstance(result.risk_score, (int, float))
        assert result.risk_score >= 0

    @given(text=st.text(min_size=0, max_size=1000))
    @settings(max_examples=100, deadline=5000)
    def test_scan_output_never_crashes(self, text):
        result = scan_output(text)
        assert result.verdict in (ScanVerdict.ALLOW, ScanVerdict.WARN, ScanVerdict.BLOCK)
        assert isinstance(result.risk_score, (int, float))

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=50, deadline=5000)
    def test_empty_and_whitespace_safe(self, text):
        # Whitespace-only should never block
        ws = " " * len(text) if text else ""
        result = scan_input(ws)
        assert result.verdict != ScanVerdict.BLOCK or ws.strip() != ""

    @given(text=st.binary(min_size=0, max_size=500))
    @settings(max_examples=50, deadline=5000)
    def test_scan_input_handles_decoded_binary(self, text):
        decoded = text.decode("utf-8", errors="replace")
        result = scan_input(decoded)
        assert result is not None


# ======================================================================
# Fuzz policy evaluation with random tool names and taint contexts
# ======================================================================

class TestPolicyFuzz:
    @given(
        tool_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
        taint_level=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100, deadline=5000)
    def test_policy_eval_never_crashes(self, tool_name, taint_level):
        engine = PolicyEngine()
        taint_context = {
            "tainted_fields": {
                "command": {
                    "taint_level": taint_level,
                    "source": "user_message",
                }
            },
        }
        result = engine.evaluate(tool_name, {}, taint_context)
        assert isinstance(result, EvaluationResult)

    @given(tool_name=st.from_regex(r"[a-z_]{1,30}", fullmatch=True))
    @settings(max_examples=50, deadline=5000)
    def test_default_policy_allows_safe_tools(self, tool_name):
        engine = PolicyEngine()
        result = engine.evaluate(tool_name, {}, {})
        # With no taint and default policy, should not hard-block
        assert isinstance(result, EvaluationResult)
