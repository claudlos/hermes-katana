"""Tests for unknown tool handling across all policy presets.

Verifies that unknown/new tools are handled securely:
- MAX: deny unknown tools (both clean and tainted)
- BALANCED: escalate tainted unknown, allow clean (catch-all)
- PERMISSIVE: log_only for unknown tools
- Flow analyzer: defaults to ASK_USER for unknown flows
- Engine: preset-appropriate default_action when no policy matches
"""

from __future__ import annotations

import pytest

from hermes_katana.policy.engine import PolicyEngine
from hermes_katana.policy.models import PolicyResult


# ── Helpers ──────────────────────────────────────────────────────────────────

FAKE_TOOL = "super_duper_nonexistent_tool_xyz_9999"

NO_TAINT: dict = {}

TAINTED_CTX: dict = {
    "tainted_fields": {
        "content": {
            "is_tainted": True,
            "source": "web_content",
            "labels": ["untrusted"],
            "readers": [],
            "level": 5,
        }
    }
}

HIGH_TAINT_CTX: dict = {
    "tainted_fields": {
        "content": {
            "is_tainted": True,
            "source": "web_content",
            "labels": ["untrusted"],
            "readers": [],
            "level": 9,
        }
    }
}


# ══════════════════════════════════════════════════════════════════════════════
# Task 1: Unknown tool defaults per preset
# ══════════════════════════════════════════════════════════════════════════════


class TestMaxUnknownTools:
    """MAX preset must deny all unknown tools."""

    def setup_method(self):
        self.engine = PolicyEngine.with_defaults("max")

    def test_unknown_tool_clean_is_denied(self):
        result = self.engine.evaluate(FAKE_TOOL, {"arg": "hello"}, NO_TAINT)
        assert result.action == PolicyResult.DENY

    def test_unknown_tool_tainted_is_denied(self):
        result = self.engine.evaluate(FAKE_TOOL, {"content": "x"}, TAINTED_CTX)
        assert result.action == PolicyResult.DENY

    def test_unknown_tool_high_taint_is_denied(self):
        result = self.engine.evaluate(FAKE_TOOL, {"content": "x"}, HIGH_TAINT_CTX)
        assert result.action == PolicyResult.DENY

    def test_default_action_is_deny(self):
        assert self.engine.default_action == PolicyResult.DENY


class TestBalancedUnknownTools:
    """BALANCED preset: escalate tainted unknown, allow clean unknown (catch-all)."""

    def setup_method(self):
        self.engine = PolicyEngine.with_defaults("balanced")

    def test_unknown_tool_clean_is_allowed(self):
        """Clean call to unknown tool → ESCALATE via balanced catch-all (GAP 4.2)."""
        result = self.engine.evaluate(FAKE_TOOL, {"arg": "hello"}, NO_TAINT)
        assert result.action == PolicyResult.ESCALATE

    def test_unknown_tool_tainted_is_escalated(self):
        """Tainted call to unknown tool → ESCALATE in balanced mode."""
        result = self.engine.evaluate(FAKE_TOOL, {"content": "x"}, TAINTED_CTX)
        assert result.action == PolicyResult.ESCALATE

    def test_unknown_tool_high_taint_is_escalated(self):
        """High-taint call to unknown tool → ESCALATE in balanced mode."""
        result = self.engine.evaluate(FAKE_TOOL, {"content": "x"}, HIGH_TAINT_CTX)
        assert result.action == PolicyResult.ESCALATE

    def test_default_action_is_escalate(self):
        """Engine default_action should be ESCALATE for balanced preset."""
        assert self.engine.default_action == PolicyResult.ESCALATE


class TestPermissiveUnknownTools:
    """PERMISSIVE preset must log_only for unknown tools."""

    def setup_method(self):
        self.engine = PolicyEngine.with_defaults("permissive")

    def test_unknown_tool_clean_is_log_only(self):
        result = self.engine.evaluate(FAKE_TOOL, {"arg": "hello"}, NO_TAINT)
        assert result.action == PolicyResult.LOG_ONLY

    def test_unknown_tool_tainted_is_log_only(self):
        result = self.engine.evaluate(FAKE_TOOL, {"content": "x"}, TAINTED_CTX)
        assert result.action == PolicyResult.LOG_ONLY

    def test_default_action_is_log_only(self):
        assert self.engine.default_action == PolicyResult.LOG_ONLY


# ══════════════════════════════════════════════════════════════════════════════
# Known tools still work correctly
# ══════════════════════════════════════════════════════════════════════════════


class TestKnownToolsUnchanged:
    """Ensure known tools still get their expected policy results."""

    def test_balanced_clean_terminal_allowed(self):
        engine = PolicyEngine.with_defaults("balanced")
        result = engine.evaluate("terminal", {"command": "ls"}, NO_TAINT)
        assert result.action == PolicyResult.ALLOW

    def test_balanced_clean_read_file_allowed(self):
        engine = PolicyEngine.with_defaults("balanced")
        result = engine.evaluate("read_file", {"path": "/tmp/x"}, NO_TAINT)
        assert result.action == PolicyResult.ALLOW

    def test_max_tainted_terminal_denied(self):
        engine = PolicyEngine.with_defaults("max")
        result = engine.evaluate("terminal", {"command": "ls"}, TAINTED_CTX)
        assert result.action == PolicyResult.DENY

    def test_permissive_clean_terminal_log_only(self):
        """Clean terminal in permissive hits catchall → log_only."""
        engine = PolicyEngine.with_defaults("permissive")
        result = engine.evaluate("terminal", {"command": "ls"}, NO_TAINT)
        assert result.action == PolicyResult.LOG_ONLY

    def test_balanced_clean_notes_allowed(self):
        """Clean notes call → ALLOW via explicit balanced_notes_clean policy."""
        engine = PolicyEngine.with_defaults("balanced")
        result = engine.evaluate("notes", {"text": "Meeting notes"}, NO_TAINT)
        assert result.action == PolicyResult.ALLOW

    def test_balanced_tainted_notes_denied(self):
        """Tainted notes in balanced should be denied (explicit policy)."""
        engine = PolicyEngine.with_defaults("balanced")
        result = engine.evaluate("notes", {"text": "x"}, TAINTED_CTX)
        assert result.action == PolicyResult.DENY


# ══════════════════════════════════════════════════════════════════════════════
# Task 2: Flow analyzer defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestFlowAnalyzerDefaults:
    """Flow analyzer should default to ASK_USER (fail-closed)."""

    def test_default_decision_is_ask_user(self):
        from hermes_katana.taint.flow import FlowAnalyzer, FlowDecision

        analyzer = FlowAnalyzer(rules=[])
        assert analyzer._default == FlowDecision.ASK_USER

    def test_default_decision_with_default_rules(self):
        from hermes_katana.taint.flow import FlowAnalyzer, FlowDecision

        analyzer = FlowAnalyzer()
        assert analyzer._default == FlowDecision.ASK_USER

    def test_strict_mode_is_ask_user(self):
        from hermes_katana.taint.flow import FlowAnalyzer, FlowDecision

        analyzer = FlowAnalyzer(strict_mode=True)
        assert analyzer._default == FlowDecision.ASK_USER

    def test_explicit_allow_override_works(self):
        from hermes_katana.taint.flow import FlowAnalyzer, FlowDecision

        analyzer = FlowAnalyzer(default_decision=FlowDecision.ALLOW)
        assert analyzer._default == FlowDecision.ALLOW


# ══════════════════════════════════════════════════════════════════════════════
# Engine preset defaults
# ══════════════════════════════════════════════════════════════════════════════


class TestEnginePresetDefaults:
    """Engine.with_defaults sets appropriate default_action per preset."""

    def test_max_default_deny(self):
        engine = PolicyEngine.with_defaults("max")
        assert engine.default_action == PolicyResult.DENY

    def test_balanced_default_escalate(self):
        engine = PolicyEngine.with_defaults("balanced")
        assert engine.default_action == PolicyResult.ESCALATE

    def test_permissive_default_log_only(self):
        engine = PolicyEngine.with_defaults("permissive")
        assert engine.default_action == PolicyResult.LOG_ONLY

    def test_custom_engine_default_deny(self):
        engine = PolicyEngine()
        assert engine.default_action == PolicyResult.DENY

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            PolicyEngine.with_defaults("nonexistent")
