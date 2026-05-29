"""Unit tests for the centralized ESCALATE resolver."""

from __future__ import annotations

import sys
import types

import pytest

from hermes_katana import escalation


@pytest.fixture(autouse=True)
def _clear_auto_approve_env(monkeypatch):
    monkeypatch.delenv("KATANA_AUTO_APPROVE_ESCALATIONS", raising=False)


class TestNormalizeEscalateAction:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("block", "block"),
            ("ACP_PROMPT", "acp_prompt"),
            ("  auto_approve  ", "auto_approve"),
            ("nonsense", "block"),
            ("", "block"),
            (None, "block"),
        ],
    )
    def test_normalize(self, raw, expected):
        assert escalation.normalize_escalate_action(raw) == expected


class TestResolveEscalation:
    def test_block_is_default_and_fails_closed(self):
        assert escalation.resolve_escalation("block", tool_name="terminal", reasons=["r"]) is False
        # Unknown action normalizes to block.
        assert escalation.resolve_escalation("bogus", tool_name="terminal", reasons=["r"]) is False

    def test_auto_approve_allows(self):
        assert escalation.resolve_escalation("auto_approve", tool_name="terminal", reasons=["r"]) is True

    def test_env_var_forces_auto_approve(self, monkeypatch):
        monkeypatch.setenv("KATANA_AUTO_APPROVE_ESCALATIONS", "1")
        # Even though configured to block, the env override wins (back-compat).
        assert escalation.resolve_escalation("block", tool_name="terminal", reasons=["r"]) is True

    def test_acp_prompt_blocks_without_approver(self, monkeypatch):
        monkeypatch.setattr(escalation, "_get_interactive_approver", lambda: None)
        assert escalation.resolve_escalation("acp_prompt", tool_name="terminal", reasons=["r"]) is False

    def test_acp_prompt_allows_when_human_approves(self, monkeypatch):
        captured = {}

        def fake_cb(command, description, allow_permanent=True):
            captured["command"] = command
            captured["description"] = description
            return "once"  # Hermes "allow once" outcome

        monkeypatch.setattr(escalation, "_get_interactive_approver", lambda: fake_cb)
        result = escalation.resolve_escalation(
            "acp_prompt",
            tool_name="terminal",
            reasons=["dangerous command"],
            args={"command": "rm -rf /tmp/x"},
        )
        assert result is True
        assert captured["command"] == "rm -rf /tmp/x"
        assert "terminal" in captured["description"]

    def test_acp_prompt_blocks_on_deny(self, monkeypatch):
        monkeypatch.setattr(escalation, "_get_interactive_approver", lambda: lambda **k: "deny")
        assert escalation.resolve_escalation("acp_prompt", tool_name="terminal", reasons=["r"]) is False

    def test_acp_prompt_blocks_when_approver_raises(self, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("approver exploded")

        monkeypatch.setattr(escalation, "_get_interactive_approver", lambda: boom)
        assert escalation.resolve_escalation("acp_prompt", tool_name="terminal", reasons=["r"]) is False

    def test_acp_prompt_supports_positional_approver(self, monkeypatch):
        def positional_only(command, description):
            return "session"

        monkeypatch.setattr(escalation, "_get_interactive_approver", lambda: positional_only)
        assert escalation.resolve_escalation("acp_prompt", tool_name="terminal", reasons=["r"]) is True


class TestInteractiveApproverDiscovery:
    def test_reads_hermes_terminal_tool_callback(self, monkeypatch):
        def sentinel(**kwargs):
            return "deny"

        fake_tools = types.ModuleType("tools")
        fake_terminal = types.ModuleType("tools.terminal_tool")
        fake_terminal._get_approval_callback = lambda: sentinel
        fake_tools.terminal_tool = fake_terminal
        monkeypatch.setitem(sys.modules, "tools", fake_tools)
        monkeypatch.setitem(sys.modules, "tools.terminal_tool", fake_terminal)

        assert escalation._get_interactive_approver() is sentinel

    def test_returns_none_when_hermes_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "tools", None)
        assert escalation._get_interactive_approver() is None


class TestOutcomeInterpretation:
    @pytest.mark.parametrize("value", ["once", "session", "always", "allow", True, "YES"])
    def test_allow_values(self, value):
        assert escalation._outcome_is_allow(value) is True

    @pytest.mark.parametrize("value", ["deny", "deny_always", "", None, False, "maybe"])
    def test_block_values(self, value):
        assert escalation._outcome_is_allow(value) is False
