"""Tests for hermes_katana.vault.migrate — secret discovery and migration."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


from hermes_katana.vault.migrate import (
    MigrationResult,
    _is_secret_key,
    _scan_dotenv,
    _scan_env_vars,
    _scan_hermes_config,
    _extract_secrets_from_dict,
    _secure_delete_env_var,
    _secure_delete_from_file,
    discover_secrets,
    migrate_secrets,
)


# ---------------------------------------------------------------------------
# _is_secret_key
# ---------------------------------------------------------------------------


class TestIsSecretKey:
    def test_api_key_suffix(self):
        assert _is_secret_key("OPENAI_API_KEY")
        assert _is_secret_key("MY_API_KEY")

    def test_token_suffix(self):
        assert _is_secret_key("GITHUB_TOKEN")
        assert _is_secret_key("REPLICATE_API_TOKEN")

    def test_password_suffix(self):
        assert _is_secret_key("DB_PASSWORD")

    def test_secret_suffix(self):
        assert _is_secret_key("APP_SECRET")
        assert _is_secret_key("MY_SECRET_KEY")

    def test_provider_prefixes(self):
        assert _is_secret_key("OPENAI_ORG")
        assert _is_secret_key("ANTHROPIC_MODEL")
        assert _is_secret_key("AWS_ACCESS_KEY")

    def test_skip_keys(self):
        assert not _is_secret_key("PATH")
        assert not _is_secret_key("HOME")
        assert not _is_secret_key("USER")
        assert not _is_secret_key("SHELL")
        assert not _is_secret_key("PYTHONPATH")

    def test_non_secret_key(self):
        assert not _is_secret_key("MY_APP_NAME")
        assert not _is_secret_key("LOG_LEVEL")

    def test_credential_suffix(self):
        assert _is_secret_key("AWS_CREDENTIALS")
        assert _is_secret_key("MY_CREDENTIAL")

    def test_case_insensitive(self):
        assert _is_secret_key("openai_api_key")


# ---------------------------------------------------------------------------
# _scan_dotenv
# ---------------------------------------------------------------------------


class TestScanDotenv:
    def test_scan_dotenv_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_API_KEY=sk-test123\nLOG_LEVEL=debug\nANTHROPIC_API_KEY='sk-ant-abc'\n# comment line\n\nEMPTY_KEY=\n"
        )
        found = _scan_dotenv(env_file)
        assert "OPENAI_API_KEY" in found
        assert found["OPENAI_API_KEY"] == "sk-test123"
        assert "ANTHROPIC_API_KEY" in found
        assert "LOG_LEVEL" not in found

    def test_scan_dotenv_missing_file(self, tmp_path):
        found = _scan_dotenv(tmp_path / "nonexistent.env")
        assert found == {}

    def test_scan_dotenv_none_no_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        found = _scan_dotenv(None)
        assert found == {}

    def test_scan_dotenv_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('MY_API_KEY="quoted-value"\n')
        found = _scan_dotenv(env_file)
        assert found.get("MY_API_KEY") == "quoted-value"

    def test_scan_dotenv_skips_no_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("NO_EQUALS_LINE\nMY_TOKEN=abc\n")
        found = _scan_dotenv(env_file)
        assert "MY_TOKEN" in found
        assert "NO_EQUALS_LINE" not in found


# ---------------------------------------------------------------------------
# _scan_env_vars
# ---------------------------------------------------------------------------


class TestScanEnvVars:
    def test_finds_secret_env_vars(self):
        with patch.dict(os.environ, {"TEST_API_KEY": "secret123", "NORMAL_VAR": "value"}, clear=True):
            found = _scan_env_vars()
            assert "TEST_API_KEY" in found
            assert "NORMAL_VAR" not in found

    def test_skips_empty_values(self):
        with patch.dict(os.environ, {"TEST_API_KEY": "  "}, clear=True):
            found = _scan_env_vars()
            assert "TEST_API_KEY" not in found

    def test_skips_skip_keys(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin", "HOME": "/home/test"}, clear=True):
            found = _scan_env_vars()
            assert "PATH" not in found
            assert "HOME" not in found


# ---------------------------------------------------------------------------
# _scan_hermes_config
# ---------------------------------------------------------------------------


class TestScanHermesConfig:
    def test_scan_yaml_config(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("openai:\n  api_key: sk-test-123\nlogging:\n  level: debug\n")
        found = _scan_hermes_config(config_file)
        assert len(found) >= 1
        # Should find the api_key under openai
        values = list(found.values())
        assert "sk-test-123" in values

    def test_scan_missing_config(self, tmp_path):
        found = _scan_hermes_config(tmp_path / "nonexistent.yaml")
        assert found == {}

    def test_scan_none_no_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("hermes_katana.vault.migrate._safe_home", lambda: None)
        found = _scan_hermes_config(None)
        assert found == {}

    def test_scan_empty_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        found = _scan_hermes_config(config_file)
        assert found == {}


# ---------------------------------------------------------------------------
# _extract_secrets_from_dict
# ---------------------------------------------------------------------------


class TestExtractSecretsFromDict:
    def test_flat_dict(self):
        found = {}
        _extract_secrets_from_dict({"api_key": "val1", "name": "myapp"}, "", found)
        assert len(found) >= 1
        assert "val1" in found.values()

    def test_nested_dict(self):
        found = {}
        data = {"provider": {"api_key": "nested-key", "model": "gpt-4"}}
        _extract_secrets_from_dict(data, "", found)
        assert "nested-key" in found.values()

    def test_empty_values_skipped(self):
        found = {}
        _extract_secrets_from_dict({"api_key": "", "token": "  "}, "", found)
        assert len(found) == 0

    def test_non_string_values_skipped(self):
        found = {}
        _extract_secrets_from_dict({"api_key": 12345, "enabled": True}, "", found)
        assert len(found) == 0


# ---------------------------------------------------------------------------
# Secure delete
# ---------------------------------------------------------------------------


class TestSecureDeleteEnvVar:
    def test_delete_existing(self):
        with patch.dict(os.environ, {"TEST_SECRET": "value"}):
            assert _secure_delete_env_var("TEST_SECRET")
            assert "TEST_SECRET" not in os.environ

    def test_delete_nonexistent(self):
        assert not _secure_delete_env_var("DEFINITELY_NOT_SET_XYZ_123")


class TestSecureDeleteFromFile:
    def test_zero_overwrite_env_style(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("MY_API_KEY=secret123\nOTHER=val\n")
        result = _secure_delete_from_file(f, "MY_API_KEY")
        assert result is True
        content = f.read_text()
        assert "secret123" not in content
        assert "0" * 9 in content  # len("secret123") = 9

    def test_missing_file(self, tmp_path):
        result = _secure_delete_from_file(tmp_path / "nope", "KEY")
        assert result is False

    def test_key_not_found_in_file(self, tmp_path):
        f = tmp_path / ".env"
        f.write_text("OTHER_KEY=value\n")
        result = _secure_delete_from_file(f, "MISSING_KEY")
        assert result is False

    def test_yaml_style_zero_overwrite(self, tmp_path):
        f = tmp_path / "config.yaml"
        f.write_text("api_key: mysecret\nother: stuff\n")
        result = _secure_delete_from_file(f, "API_KEY")
        # The yaml pattern uses key.lower() matching
        assert result is True
        content = f.read_text()
        assert "mysecret" not in content


# ---------------------------------------------------------------------------
# discover_secrets
# ---------------------------------------------------------------------------


class TestDiscoverSecrets:
    def test_priority_env_over_dotenv(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_API_KEY=from-dotenv\n")
        with patch.dict(os.environ, {"MY_API_KEY": "from-env"}, clear=True):
            result = discover_secrets(dotenv_path=env_file)
            assert result["MY_API_KEY"] == ("from-env", "env")

    def test_discovers_from_dotenv(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("SPECIAL_TOKEN=abc123\n")
        with patch.dict(os.environ, {}, clear=True):
            result = discover_secrets(dotenv_path=env_file)
            assert "SPECIAL_TOKEN" in result

    def test_empty_sources(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with patch.dict(os.environ, {}, clear=True):
            result = discover_secrets(
                config_path=tmp_path / "nope.yaml",
                dotenv_path=tmp_path / "nope.env",
            )
            assert result == {}


# ---------------------------------------------------------------------------
# migrate_secrets
# ---------------------------------------------------------------------------


class TestMigrateSecrets:
    def test_dry_run(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_API_KEY=test-secret\n")
        vault = MagicMock()
        vault.list_keys.return_value = []
        with patch.dict(os.environ, {}, clear=True):
            result = migrate_secrets(vault, dotenv_path=env_file, dry_run=True)
        assert result.migrated >= 1
        vault.set.assert_not_called()

    def test_migrate_stores_in_vault(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_API_KEY=real-secret\n")
        vault = MagicMock()
        vault.list_keys.return_value = []
        with patch.dict(os.environ, {}, clear=True):
            result = migrate_secrets(vault, dotenv_path=env_file, secure_delete=False)
        assert result.migrated >= 1
        vault.set.assert_called()

    def test_skip_existing_keys(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_API_KEY=secret\n")
        vault = MagicMock()
        vault.list_keys.return_value = ["MY_API_KEY"]
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("hermes_katana.vault.migrate._safe_home", lambda: None)
        with patch.dict(os.environ, {}, clear=True):
            result = migrate_secrets(vault, dotenv_path=env_file)
        assert result.skipped >= 1
        assert result.migrated == 0

    def test_vault_error_recorded(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("MY_TOKEN=secret\n")
        vault = MagicMock()
        vault.list_keys.return_value = []
        vault.set.side_effect = RuntimeError("vault error")
        with patch.dict(os.environ, {}, clear=True):
            result = migrate_secrets(vault, dotenv_path=env_file, secure_delete=False)
        assert len(result.errors) >= 1

    def test_secure_delete_env_var(self, tmp_path):
        vault = MagicMock()
        vault.list_keys.return_value = []
        with patch.dict(os.environ, {"TEST_API_KEY": "from-env"}, clear=True):
            result = migrate_secrets(
                vault, secure_delete=True, dotenv_path=tmp_path / "nope.env", config_path=tmp_path / "nope.yaml"
            )
            if result.migrated > 0:
                assert "TEST_API_KEY" not in os.environ


class TestMigrationResult:
    def test_defaults(self):
        r = MigrationResult()
        assert r.migrated == 0
        assert r.skipped == 0
        assert r.deleted == 0
        assert r.errors == []
        assert r.sources == {}
