"""Tests for hermes_katana.exceptions."""

from __future__ import annotations

import pytest

from hermes_katana.exceptions import (
    EscalationRequired,
    KatanaSecurityError,
    PolicyDenied,
    ScanBlocked,
    TaintFlowDenied,
)


class TestKatanaSecurityError:
    """Base exception carries tool_name, reasons, call_id, scan_score."""

    def test_basic(self):
        err = KatanaSecurityError("blocked")
        assert str(err) == "blocked"
        assert err.tool_name == ""
        assert err.reasons == []
        assert err.call_id == ""
        assert err.scan_score == 0.0

    def test_with_kwargs(self):
        err = KatanaSecurityError(
            "bad call",
            tool_name="terminal",
            reasons=["tainted", "policy"],
            call_id="abc123",
            scan_score=0.85,
        )
        assert err.tool_name == "terminal"
        assert err.reasons == ["tainted", "policy"]
        assert err.call_id == "abc123"
        assert err.scan_score == 0.85

    def test_is_exception(self):
        assert issubclass(KatanaSecurityError, Exception)


class TestEscalationRequired:
    """EscalationRequired adds escalation_context."""

    def test_basic(self):
        err = EscalationRequired("needs approval")
        assert isinstance(err, KatanaSecurityError)
        assert err.escalation_context == {}

    def test_with_context(self):
        ctx = {"tool_name": "terminal", "reasons": ["tainted data"]}
        err = EscalationRequired(
            "needs approval",
            tool_name="terminal",
            escalation_context=ctx,
        )
        assert err.escalation_context == ctx
        assert err.tool_name == "terminal"


class TestTaintFlowDenied:
    """TaintFlowDenied adds source_labels and target_tool."""

    def test_basic(self):
        err = TaintFlowDenied(
            "flow blocked",
            source_labels=["WEB_CONTENT", "MCP"],
            target_tool="terminal",
        )
        assert isinstance(err, KatanaSecurityError)
        assert err.source_labels == ["WEB_CONTENT", "MCP"]
        assert err.target_tool == "terminal"
        assert err.tool_name == "terminal"  # inherited from super


class TestScanBlocked:
    """ScanBlocked adds findings_summary."""

    def test_basic(self):
        err = ScanBlocked(
            "scan failed",
            findings_summary="2 injection(s), 1 secret(s)",
            scan_score=0.92,
        )
        assert isinstance(err, KatanaSecurityError)
        assert err.findings_summary == "2 injection(s), 1 secret(s)"
        assert err.scan_score == 0.92


class TestPolicyDenied:
    """PolicyDenied adds policy_name and policy_action."""

    def test_basic(self):
        err = PolicyDenied(
            "policy says no",
            policy_name="block_tainted_terminal",
            policy_action="deny",
            tool_name="terminal",
        )
        assert isinstance(err, KatanaSecurityError)
        assert err.policy_name == "block_tainted_terminal"
        assert err.policy_action == "deny"
        assert err.tool_name == "terminal"

    def test_defaults(self):
        err = PolicyDenied("nope")
        assert err.policy_name == ""
        assert err.policy_action == "deny"


class TestExceptionHierarchy:
    """All exceptions are catchable as KatanaSecurityError."""

    def test_catch_all(self):
        exceptions = [
            KatanaSecurityError("a"),
            EscalationRequired("b"),
            TaintFlowDenied("c"),
            ScanBlocked("d"),
            PolicyDenied("e"),
        ]
        for exc in exceptions:
            with pytest.raises(KatanaSecurityError):
                raise exc
