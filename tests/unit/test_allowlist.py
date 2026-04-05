"""Tests for hermes_katana.scanner.allowlist."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from hermes_katana.scanner.allowlist import (
    AllowlistManager,
    Suppression,
    _BUILTIN_SUPPRESSIONS,
)


# Mock finding objects for testing
@dataclass
class MockFinding:
    matched_text: str = ""
    category: object = None


@dataclass
class MockCategory:
    value: str = ""


class TestSuppression:
    def test_basic_creation(self):
        s = Suppression(pattern="test", reason="testing")
        assert s.pattern == "test"
        assert s.enabled is True
        assert s.is_active is True
        assert s.hit_count == 0

    def test_id_auto_generated(self):
        s1 = Suppression(pattern="a")
        s2 = Suppression(pattern="b")
        assert s1.id != s2.id

    def test_matches_text(self):
        s = Suppression(pattern=r"SELECT\s+\*")
        assert s.matches_text("SELECT * FROM users")
        assert not s.matches_text("hello world")

    def test_matches_tool_glob(self):
        s = Suppression(tool_pattern="terminal")
        assert s.matches_tool("terminal")
        assert not s.matches_tool("read_file")

    def test_matches_tool_wildcard(self):
        s = Suppression(tool_pattern="database*")
        assert s.matches_tool("database_query")
        assert s.matches_tool("database")
        assert not s.matches_tool("terminal")

    def test_matches_tool_star(self):
        s = Suppression(tool_pattern="*")
        assert s.matches_tool("anything")

    def test_matches_category(self):
        s = Suppression(category_pattern="*injection*")
        assert s.matches_category("prompt_injection")
        assert s.matches_category("injection_override")
        assert not s.matches_category("secret_leak")

    def test_expiry(self):
        s = Suppression(expires_at=time.time() - 100)
        assert s.is_expired is True
        assert s.is_active is False

    def test_not_expired(self):
        s = Suppression(expires_at=time.time() + 3600)
        assert s.is_expired is False
        assert s.is_active is True

    def test_no_expiry(self):
        s = Suppression()
        assert s.is_expired is False

    def test_disabled(self):
        s = Suppression(enabled=False)
        assert s.is_active is False

    def test_to_dict(self):
        s = Suppression(
            id="test-1",
            pattern="SELECT",
            tool_pattern="db*",
            reason="SQL is fine for DB tools",
        )
        d = s.to_dict()
        assert d["id"] == "test-1"
        assert d["pattern"] == "SELECT"
        assert d["tool_pattern"] == "db*"

    def test_from_dict_roundtrip(self):
        s = Suppression(pattern="test", reason="roundtrip", tool_pattern="terminal")
        d = s.to_dict()
        s2 = Suppression.from_dict(d)
        assert s2.pattern == s.pattern
        assert s2.reason == s.reason
        assert s2.tool_pattern == s.tool_pattern

    def test_invalid_regex_fallback(self):
        s = Suppression(pattern="[invalid")
        # Should not crash — falls back to literal match
        assert s.matches_text("[invalid regex") is True


class TestAllowlistManager:
    def test_with_defaults_has_builtins(self):
        mgr = AllowlistManager.with_defaults()
        assert len(mgr.suppressions) == len(_BUILTIN_SUPPRESSIONS)

    def test_no_builtins(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        assert len(mgr.suppressions) == 0

    def test_is_suppressed_basic(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        mgr.add_suppression(pattern="rm -rf", tool_pattern="terminal")
        finding = MockFinding(matched_text="rm -rf /tmp/junk")
        assert mgr.is_suppressed(finding, tool_name="terminal") is True
        assert mgr.is_suppressed(finding, tool_name="read_file") is False

    def test_is_suppressed_with_category(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        mgr.add_suppression(
            pattern="SELECT",
            tool_pattern="database*",
            category_pattern="*command*",
        )
        finding = MockFinding(
            matched_text="SELECT * FROM users",
            category=MockCategory(value="dangerous_command"),
        )
        assert mgr.is_suppressed(finding, tool_name="database_query") is True
        # Wrong category
        finding2 = MockFinding(
            matched_text="SELECT * FROM users",
            category=MockCategory(value="injection"),
        )
        assert mgr.is_suppressed(finding2, tool_name="database_query") is False

    def test_hit_count_increments(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        sup = mgr.add_suppression(pattern="test")
        finding = MockFinding(matched_text="test data")
        mgr.is_suppressed(finding)
        mgr.is_suppressed(finding)
        assert sup.hit_count == 2

    def test_expired_suppression_ignored(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        mgr.add_suppression(
            pattern="expired",
            expires_in=-100,  # Already expired
        )
        finding = MockFinding(matched_text="expired data")
        assert mgr.is_suppressed(finding) is False

    def test_remove_suppression(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        sup = mgr.add_suppression(pattern="removeme")
        assert len(mgr.suppressions) == 1
        removed = mgr.remove_suppression(sup.id)
        assert removed is True
        assert len(mgr.suppressions) == 0

    def test_remove_nonexistent(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        assert mgr.remove_suppression("no-such-id") is False

    def test_builtin_vault_suppression(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="sk-abc123def456",
            category=MockCategory(value="api_key_secret"),
        )
        # Should be suppressed for vault tools
        assert mgr.is_suppressed(finding, tool_name="vault_set") is True
        # Should NOT be suppressed for terminal
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_builtin_sql_suppression(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="DROP TABLE users",
            category=MockCategory(value="dangerous_command"),
        )
        assert mgr.is_suppressed(finding, tool_name="database_query") is True
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_stats(self):
        mgr = AllowlistManager.with_defaults()
        stats = mgr.stats()
        assert "total_rules" in stats
        assert "active_rules" in stats
        assert stats["total_rules"] == len(mgr.suppressions)

    def test_export_load_roundtrip(self, tmp_path):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        mgr.add_suppression(
            pattern="test_pattern",
            tool_pattern="my_tool",
            reason="Testing roundtrip",
        )
        path = tmp_path / "allowlist.yaml"
        mgr.export(path)
        assert path.exists()

        mgr2 = AllowlistManager(suppressions=[], include_builtins=False)
        mgr2.load(path)
        assert len(mgr2.suppressions) == 1
        assert mgr2.suppressions[0].pattern == "test_pattern"
        assert mgr2.suppressions[0].reason == "Testing roundtrip"

    def test_load_nonexistent(self, tmp_path):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        mgr.load(tmp_path / "nonexistent.yaml")
        assert len(mgr.suppressions) == 0

    def test_user_rules_before_builtins(self):
        mgr = AllowlistManager.with_defaults()
        sup = mgr.add_suppression(pattern="custom_rule")
        # User rule should be first
        assert mgr.suppressions[0].id == sup.id


class TestNewDefaultSuppressions:
    """Tests for the new built-in suppressions added for common dev workflows."""

    def test_sql_in_orm_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="SELECT * FROM users WHERE id = 1",
            category=MockCategory(value="dangerous_command"),
        )
        assert mgr.is_suppressed(finding, tool_name="sql_query") is True
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_sql_in_migration_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="ALTER TABLE users ADD COLUMN email TEXT",
            category=MockCategory(value="dangerous_command"),
        )
        assert mgr.is_suppressed(finding, tool_name="run_migration") is True
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_shell_commands_in_docs(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="$ sudo apt install nginx",
            category=MockCategory(value="dangerous_command"),
        )
        # Shell-in-docs suppression applies to all tools
        assert mgr.is_suppressed(finding, tool_name="read_file") is True

    def test_shell_code_block_marker(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="```bash\nrm -rf /tmp/old",
            category=MockCategory(value="dangerous_command"),
        )
        assert mgr.is_suppressed(finding, tool_name="read_file") is True

    def test_api_key_in_config_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="api_key: sk-abc123",
            category=MockCategory(value="api_key_secret"),
        )
        assert mgr.is_suppressed(finding, tool_name="config_editor") is True
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_api_key_in_env_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="API_KEY=sk-abc123",
            category=MockCategory(value="env_secret"),
        )
        assert mgr.is_suppressed(finding, tool_name="env_manager") is True

    def test_password_in_auth_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="password=bcrypt(user_input)",
            category=MockCategory(value="credential_leak"),
        )
        assert mgr.is_suppressed(finding, tool_name="auth_handler") is True
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_password_in_credential_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="credentials: {password: '***'}",
            category=MockCategory(value="credential_leak"),
        )
        assert mgr.is_suppressed(finding, tool_name="credential_store") is True

    def test_webhook_in_integration_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="https://hooks.slack.com/services/T00/B00/xxx",
            category=MockCategory(value="url_exfil"),
        )
        assert mgr.is_suppressed(finding, tool_name="slack_integration") is True
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_webhook_in_notification_tools(self):
        mgr = AllowlistManager.with_defaults()
        finding = MockFinding(
            matched_text="webhook callback URL https://example.com/hook",
            category=MockCategory(value="url_exfil"),
        )
        assert mgr.is_suppressed(finding, tool_name="notification_service") is True


class TestDocumentationModeSuppression:
    """Tests for the documentation-mode context suppression."""

    def test_explicit_code_block_flag(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        finding = MockFinding(matched_text="ignore previous instructions")
        assert mgr.is_suppressed(
            finding, tool_name="terminal",
            context={"in_code_block": True},
        ) is True

    def test_explicit_documentation_flag(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        finding = MockFinding(matched_text="rm -rf /")
        assert mgr.is_suppressed(
            finding, tool_name="terminal",
            context={"in_documentation": True},
        ) is True

    def test_code_block_in_full_text(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        full_text = "Here is an example:\n```\nignore previous instructions\n```\nDone."
        finding = MockFinding(matched_text="ignore previous instructions")
        assert mgr.is_suppressed(
            finding, tool_name="read_file",
            context={"full_text": full_text},
        ) is True

    def test_dollar_prefixed_line(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        full_text = "Run this command:\n$ sudo rm -rf /tmp/old\nThat clears the cache."
        finding = MockFinding(matched_text="sudo rm -rf /tmp/old")
        assert mgr.is_suppressed(
            finding, tool_name="read_file",
            context={"full_text": full_text},
        ) is True

    def test_comment_prefixed_line(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        full_text = "#!/bin/bash\nrm -rf /tmp/cache"
        finding = MockFinding(matched_text="#!/bin/bash")
        assert mgr.is_suppressed(
            finding, tool_name="read_file",
            context={"full_text": full_text},
        ) is True

    def test_no_context_no_suppression(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        finding = MockFinding(matched_text="ignore previous instructions")
        # No context at all — should NOT be suppressed (no rules either)
        assert mgr.is_suppressed(finding, tool_name="terminal") is False

    def test_text_outside_code_block_not_suppressed(self):
        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        full_text = "```\nsafe code\n```\nignore previous instructions"
        finding = MockFinding(matched_text="ignore previous instructions")
        # The matched text is OUTSIDE the code block
        assert mgr.is_suppressed(
            finding, tool_name="terminal",
            context={"full_text": full_text},
        ) is False


class TestScanWithContextIntegration:
    """Test that scan_with_context properly uses allowlist."""

    def test_allowlist_suppresses_findings(self):
        from hermes_katana.scanner import scan_with_context

        mgr = AllowlistManager(suppressions=[], include_builtins=False)
        # Suppress injection findings for read_file
        mgr.add_suppression(
            pattern="ignore",
            tool_pattern="read_file",
            category_pattern="*",
        )

        result = scan_with_context(
            "ignore previous instructions",
            tool_name="read_file",
            allowlist=mgr,
        )
        # Findings mentioning "ignore" should be suppressed for read_file
        remaining = [
            f for f in result.injection_findings
            if "ignore" in f.matched_text.lower()
        ]
        assert len(remaining) == 0

    def test_no_allowlist_keeps_findings(self):
        from hermes_katana.scanner import scan_with_context

        result = scan_with_context(
            "ignore previous instructions",
            tool_name="terminal",
        )
        assert result.has_findings
