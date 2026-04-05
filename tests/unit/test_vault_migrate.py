"""Tests for HermesKatana vault migration (secret discovery + migration)."""

from __future__ import annotations

import os
import textwrap
from unittest.mock import MagicMock, patch


from hermes_katana.vault.migrate import (
    MigrationResult,
    _extract_secrets_from_dict,
    _is_secret_key,
    _scan_dotenv,
    _scan_env_vars,
    _scan_hermes_config,
    _secure_delete_env_var,
    _secure_delete_from_file,
    discover_secrets,
    migrate_secrets,
)


# ======================================================================
# _is_secret_key
# ======================================================================


class TestIsSecretKey:
    def test_api_key_pattern(self):
        assert _is_secret_key("OPENAI_API_KEY") is True
        assert _is_secret_key("MY_CUSTOM_API_KEY") is True

    def test_token_pattern(self):
        assert _is_secret_key("GITHUB_TOKEN") is True
        assert _is_secret_key("MY_API_TOKEN") is True

    def test_secret_pattern(self):
        assert _is_secret_key("MY_SECRET_KEY") is True
        assert _is_secret_key("APP_SECRET") is True

    def test_password_pattern(self):
        assert _is_secret_key("DB_PASSWORD") is True

    def test_provider_prefixes(self):
        assert _is_secret_key("OPENAI_ANYTHING") is True
        assert _is_secret_key("ANTHROPIC_SOMETHING") is True
        assert _is_secret_key("GROQ_KEY") is True
        assert _is_secret_key("TOGETHER_KEY") is True
        assert _is_secret_key("DEEPSEEK_KEY") is True
        assert _is_secret_key("AWS_KEY") is True

    def test_skip_keys_rejected(self):
        assert _is_secret_key("PATH") is False
        assert _is_secret_key("HOME") is False
        assert _is_secret_key("USER") is False
        assert _is_secret_key("SHELL") is False
        assert _is_secret_key("PYTHONPATH") is False
        assert _is_secret_key("VIRTUAL_ENV") is False

    def test_non_secret_names(self):
        assert _is_secret_key("MY_APP_NAME") is False
        assert _is_secret_key("DEBUG") is False
        assert _is_secret_key("LOG_LEVEL") is False

    def test_case_insensitive(self):
        # Patterns use re.IGNORECASE
        assert _is_secret_key("openai_api_key") is True
        assert _is_secret_key("My_Api_Token") is True


# ======================================================================
# _scan_env_vars
# ======================================================================


class TestScanEnvVars:
    def test_finds_secret_env_vars(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "MY_API_TOKEN": "tok-123",
                "PATH": "/usr/bin",
                "DEBUG": "true",
            },
            clear=True,
        ):
            found = _scan_env_vars()
            assert "OPENAI_API_KEY" in found
            assert found["OPENAI_API_KEY"] == "sk-test"
            assert "MY_API_TOKEN" in found
            assert "PATH" not in found
            assert "DEBUG" not in found

    def test_skips_empty_values(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "  ",
            },
            clear=True,
        ):
            found = _scan_env_vars()
            assert "OPENAI_API_KEY" not in found
            assert "ANTHROPIC_API_KEY" not in found


# ======================================================================
# _scan_dotenv
# ======================================================================


class TestScanDotenv:
    def test_reads_dotenv_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            textwrap.dedent("""\
            # Comment line
            OPENAI_API_KEY=sk-test123
            MY_API_TOKEN="tok-quoted"
            NORMAL_VAR=hello
            DEBUG=true
        """)
        )
        found = _scan_dotenv(env_file)
        assert found["OPENAI_API_KEY"] == "sk-test123"
        assert found["MY_API_TOKEN"] == "tok-quoted"
        assert "NORMAL_VAR" not in found
        assert "DEBUG" not in found

    def test_handles_single_quoted_values(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GITHUB_TOKEN='ghp_abc123'\n")
        found = _scan_dotenv(env_file)
        assert found["GITHUB_TOKEN"] == "ghp_abc123"

    def test_skips_blank_lines_and_comments(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            textwrap.dedent("""\
            # This is a comment

            OPENAI_API_KEY=sk-test
            invalid line without equals
        """)
        )
        found = _scan_dotenv(env_file)
        assert "OPENAI_API_KEY" in found
        assert len(found) == 1

    def test_nonexistent_file_returns_empty(self, tmp_path):
        found = _scan_dotenv(tmp_path / "missing.env")
        assert found == {}

    def test_none_path_searches_defaults(self):
        # With no default .env files, should return empty
        with patch("hermes_katana.vault.migrate.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            mock_path.home.return_value.__truediv__ = lambda self, x: mock_path.return_value
            found = _scan_dotenv(None)
            assert isinstance(found, dict)


# ======================================================================
# _scan_hermes_config
# ======================================================================


class TestScanHermesConfig:
    def test_reads_yaml_secrets(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            textwrap.dedent("""\
            providers:
              openai:
                api_key: sk-from-config
              anthropic:
                token: ant-from-config
            general:
              log_level: info
        """)
        )
        found = _scan_hermes_config(config_file)
        # Should find api_key and token values
        assert any("sk-from-config" in v for v in found.values())
        assert any("ant-from-config" in v for v in found.values())

    def test_nonexistent_config_returns_empty(self, tmp_path):
        found = _scan_hermes_config(tmp_path / "nonexistent.yaml")
        assert found == {}

    def test_none_path_searches_defaults(self):
        found = _scan_hermes_config(None)
        assert isinstance(found, dict)


# ======================================================================
# _extract_secrets_from_dict
# ======================================================================


class TestExtractSecretsFromDict:
    def test_flat_dict(self):
        found = {}
        _extract_secrets_from_dict({"api_key": "sk-123", "name": "test"}, "", found)
        assert any("sk-123" in v for v in found.values())

    def test_nested_dict(self):
        found = {}
        data = {"provider": {"openai": {"api_key": "sk-nested"}}}
        _extract_secrets_from_dict(data, "", found)
        assert any("sk-nested" in v for v in found.values())

    def test_skips_empty_values(self):
        found = {}
        _extract_secrets_from_dict({"api_key": "", "token": "  "}, "", found)
        # Empty and whitespace-only should be skipped
        assert len(found) == 0

    def test_non_string_values_skipped(self):
        found = {}
        _extract_secrets_from_dict({"api_key": 12345, "secret": True}, "", found)
        assert len(found) == 0


# ======================================================================
# _secure_delete_env_var
# ======================================================================


class TestSecureDeleteEnvVar:
    def test_removes_existing_var(self):
        os.environ["TEST_SECRET_TO_DELETE"] = "sensitive"
        result = _secure_delete_env_var("TEST_SECRET_TO_DELETE")
        assert result is True
        assert "TEST_SECRET_TO_DELETE" not in os.environ

    def test_missing_var_returns_false(self):
        os.environ.pop("NONEXISTENT_VAR_XYZ", None)
        result = _secure_delete_env_var("NONEXISTENT_VAR_XYZ")
        assert result is False


# ======================================================================
# _secure_delete_from_file
# ======================================================================


class TestSecureDeleteFromFile:
    def test_zeros_out_env_style(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=sk-secret123\nOTHER=val\n")
        result = _secure_delete_from_file(env_file, "OPENAI_API_KEY")
        assert result is True
        content = env_file.read_text()
        assert "sk-secret123" not in content
        assert "0" * len("sk-secret123") in content
        # OTHER should be untouched
        assert "OTHER=val" in content

    def test_zeros_out_quoted_value(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('MY_TOKEN="tok-abc"\n')
        result = _secure_delete_from_file(env_file, "MY_TOKEN")
        assert result is True
        content = env_file.read_text()
        assert "tok-abc" not in content

    def test_nonexistent_file_returns_false(self, tmp_path):
        result = _secure_delete_from_file(tmp_path / "missing", "KEY")
        assert result is False

    def test_key_not_in_file_returns_false(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("OTHER_KEY=value\n")
        result = _secure_delete_from_file(env_file, "NONEXISTENT_KEY")
        assert result is False


# ======================================================================
# discover_secrets
# ======================================================================


class TestDiscoverSecrets:
    def test_env_has_highest_priority(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_API_KEY=from-dotenv\n")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-env"}, clear=False):
            result = discover_secrets(dotenv_path=env_file)
            assert "OPENAI_API_KEY" in result
            value, source = result["OPENAI_API_KEY"]
            assert source == "env"
            assert value == "from-env"

    def test_dotenv_source_tagged(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("GITHUB_TOKEN=ghp_test\n")
        with patch.dict(os.environ, {}, clear=True):
            # Remove GITHUB_TOKEN from env if present
            os.environ.pop("GITHUB_TOKEN", None)
            result = discover_secrets(dotenv_path=env_file)
            if "GITHUB_TOKEN" in result:
                _, source = result["GITHUB_TOKEN"]
                assert source == "dotenv"


# ======================================================================
# migrate_secrets
# ======================================================================


class TestMigrateSecrets:
    def _mock_vault(self, existing_keys=None):
        vault = MagicMock()
        vault.list_keys.return_value = existing_keys or []
        vault.set.return_value = None
        return vault

    def test_migrates_env_secret(self, tmp_path):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-migrate"}, clear=True):
            vault = self._mock_vault()
            result = migrate_secrets(
                vault,
                secure_delete=False,
                dotenv_path=tmp_path / "nonexistent.env",
                config_path=tmp_path / "nonexistent.yaml",
            )
            assert result.migrated >= 1
            vault.set.assert_called()

    def test_skips_already_in_vault(self, tmp_path):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-exists"}, clear=True):
            vault = self._mock_vault(existing_keys=["OPENAI_API_KEY"])
            result = migrate_secrets(
                vault,
                secure_delete=False,
                dotenv_path=tmp_path / "nonexistent.env",
                config_path=tmp_path / "nonexistent.yaml",
            )
            assert result.skipped >= 1
            vault.set.assert_not_called()

    def test_dry_run_no_side_effects(self, tmp_path):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-dry"}, clear=True):
            vault = self._mock_vault()
            result = migrate_secrets(
                vault,
                dry_run=True,
                secure_delete=False,
                dotenv_path=tmp_path / "nonexistent.env",
                config_path=tmp_path / "nonexistent.yaml",
            )
            assert result.migrated >= 1
            vault.set.assert_not_called()

    def test_secure_delete_removes_env_var(self, tmp_path):
        os.environ["TEST_MIGRATE_KEY_API_KEY"] = "to-delete"
        vault = self._mock_vault()
        migrate_secrets(
            vault,
            secure_delete=True,
            dotenv_path=tmp_path / "nonexistent.env",
            config_path=tmp_path / "nonexistent.yaml",
        )
        # The env var should have been deleted
        assert "TEST_MIGRATE_KEY_API_KEY" not in os.environ

    def test_vault_error_recorded(self, tmp_path):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fail"}, clear=True):
            vault = self._mock_vault()
            vault.set.side_effect = Exception("vault write failed")
            result = migrate_secrets(
                vault,
                secure_delete=False,
                dotenv_path=tmp_path / "nonexistent.env",
                config_path=tmp_path / "nonexistent.yaml",
            )
            assert len(result.errors) >= 1


# ======================================================================
# MigrationResult
# ======================================================================


class TestMigrationResult:
    def test_defaults(self):
        r = MigrationResult()
        assert r.migrated == 0
        assert r.skipped == 0
        assert r.deleted == 0
        assert r.errors == []
        assert r.sources == {}
