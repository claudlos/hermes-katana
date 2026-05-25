"""
Tests for src/hermes_katana/vault/honey_tokens.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_katana.vault.honey_tokens import (
    HoneyFileMonitor,
    HoneyTokenError,
    HoneyTokenVault,
    TokenKind,
    _generate_value,
    default_honey_token_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    """Return a temporary honey-token store path."""
    return tmp_path / "honey_tokens.json"


@pytest.fixture
def vault(tmp_store: Path) -> HoneyTokenVault:
    """Return a HoneyTokenVault backed by a temp file, audit disabled."""
    return HoneyTokenVault(path=tmp_store, audit_enabled=False)


# ---------------------------------------------------------------------------
# 1. default_honey_token_path returns a Path
# ---------------------------------------------------------------------------


def test_default_path_is_path():
    p = default_honey_token_path()
    assert isinstance(p, Path)
    assert p.name == "honey_tokens.json"


# ---------------------------------------------------------------------------
# 2. Token value generation — structural checks per kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, prefix",
    [
        (TokenKind.AWS_ACCESS_KEY, "AKIA"),
        (TokenKind.GITHUB, "ghp_"),
        (TokenKind.OPENAI, "sk-"),
        (TokenKind.SLACK, "xoxb-"),
        (TokenKind.TWILIO, "SK"),
        (TokenKind.SENDGRID, "SG."),
    ],
)
def test_generated_value_prefix(kind, prefix):
    value = _generate_value(kind)
    assert value.startswith(prefix), f"{kind}: expected prefix {prefix!r}, got {value!r}"


def test_generated_value_aws_secret_length():
    value = _generate_value(TokenKind.AWS_SECRET_KEY)
    assert len(value) == 40


def test_generated_value_jwt_structure():
    value = _generate_value(TokenKind.JWT)
    parts = value.split(".")
    assert len(parts) == 3, "JWT must have three dot-separated parts"


def test_generated_value_database_url_scheme():
    value = _generate_value(TokenKind.DATABASE_URL)
    assert value.startswith(("postgresql://", "mysql://", "mongodb://"))


def test_generated_value_password_length():
    value = _generate_value(TokenKind.PASSWORD)
    assert len(value) == 24


def test_generated_value_heroku_uuid_like():
    value = _generate_value(TokenKind.HEROKU)
    parts = value.split("-")
    assert len(parts) == 5


# ---------------------------------------------------------------------------
# 3. Create token
# ---------------------------------------------------------------------------


def test_create_stores_token(vault: HoneyTokenVault):
    token = vault.create("my_lure", TokenKind.OPENAI)
    assert token.name == "my_lure"
    assert token.kind == TokenKind.OPENAI
    assert token.value.startswith("sk-")
    assert "my_lure" in vault.list_tokens()


def test_create_with_custom_value(vault: HoneyTokenVault):
    token = vault.create("custom", TokenKind.GENERIC_API_KEY, value="DEADBEEF")
    assert token.value == "DEADBEEF"


def test_create_persists_to_disk(tmp_store: Path, vault: HoneyTokenVault):
    vault.create("disk_test", TokenKind.GITHUB)
    assert tmp_store.exists()
    data = json.loads(tmp_store.read_text())
    assert "disk_test" in data


# ---------------------------------------------------------------------------
# 4. Get token triggers alert
# ---------------------------------------------------------------------------


def test_get_increments_access_count(vault: HoneyTokenVault):
    vault.create("access_test", TokenKind.GENERIC_API_KEY, value="secret123")
    vault.get("access_test")
    vault.get("access_test")
    token = vault.get_token("access_test")
    assert token.access_count == 2


def test_get_returns_correct_value(vault: HoneyTokenVault):
    vault.create("val_test", TokenKind.GENERIC_API_KEY, value="my_fake_key")
    assert vault.get("val_test") == "my_fake_key"


def test_get_fires_alert_callback(vault: HoneyTokenVault):
    cb = MagicMock()
    vault._alert_callback = cb
    vault.create("cb_test", TokenKind.GITHUB)
    vault.get("cb_test")
    cb.assert_called_once()
    token_arg, detail_arg = cb.call_args[0]
    assert token_arg.name == "cb_test"
    assert "HONEY TOKEN ACCESSED" in detail_arg


def test_get_missing_raises(vault: HoneyTokenVault):
    with pytest.raises(HoneyTokenError, match="not found"):
        vault.get("nonexistent")


# ---------------------------------------------------------------------------
# 5. Remove token
# ---------------------------------------------------------------------------


def test_remove_deletes_token(vault: HoneyTokenVault):
    vault.create("rm_test", TokenKind.STRIPE)
    vault.remove("rm_test")
    assert "rm_test" not in vault.list_tokens()


def test_remove_missing_raises(vault: HoneyTokenVault):
    with pytest.raises(HoneyTokenError, match="not found"):
        vault.remove("no_such_token")


# ---------------------------------------------------------------------------
# 6. Plant in environment variable
# ---------------------------------------------------------------------------


def test_plant_env_sets_os_environ(vault: HoneyTokenVault):
    vault.create("env_lure", TokenKind.AWS_ACCESS_KEY)
    var = vault.plant_env("env_lure")
    assert os.environ.get(var) is not None
    assert os.environ[var].startswith("AKIA")
    # cleanup
    del os.environ[var]


def test_plant_env_custom_var(vault: HoneyTokenVault):
    vault.create("env_custom", TokenKind.GITHUB)
    vault.plant_env("env_custom", env_var="MY_FAKE_GH_TOKEN")
    assert os.environ.get("MY_FAKE_GH_TOKEN", "").startswith("ghp_")
    del os.environ["MY_FAKE_GH_TOKEN"]


def test_unplant_env_removes_var(vault: HoneyTokenVault):
    vault.create("unplant_test", TokenKind.GENERIC_API_KEY)
    var = vault.plant_env("unplant_test")
    vault.unplant_env("unplant_test")
    assert var not in os.environ


# ---------------------------------------------------------------------------
# 7. Plant into file
# ---------------------------------------------------------------------------


def test_plant_file_creates_file(tmp_path: Path, vault: HoneyTokenVault):
    vault.create("file_lure", TokenKind.OPENAI)
    dest = vault.plant_file("file_lure", file_path=tmp_path / "secrets.json")
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data["file_lure"].startswith("sk-")


def test_plant_file_custom_key(tmp_path: Path, vault: HoneyTokenVault):
    vault.create("ckey_lure", TokenKind.STRIPE)
    dest = vault.plant_file("ckey_lure", file_path=tmp_path / "cfg.json", config_key="STRIPE_SECRET")
    data = json.loads(dest.read_text())
    assert "STRIPE_SECRET" in data


# ---------------------------------------------------------------------------
# 8. Canary URL — fire-and-forget (no real network)
# ---------------------------------------------------------------------------


def test_canary_ping_launched_on_get(vault: HoneyTokenVault):
    vault.create("canary_lure", TokenKind.GENERIC_API_KEY, canary_url="http://canary.example.invalid")
    with patch("hermes_katana.vault.honey_tokens._canary_ping") as mock_ping:
        vault.get("canary_lure")
    mock_ping.assert_called_once_with("http://canary.example.invalid", "canary_lure")


def test_no_canary_when_url_is_none(vault: HoneyTokenVault):
    vault.create("no_canary", TokenKind.GENERIC_API_KEY)
    with patch("hermes_katana.vault.honey_tokens._canary_ping") as mock_ping:
        vault.get("no_canary")
    mock_ping.assert_not_called()


# ---------------------------------------------------------------------------
# 9. from_config factory
# ---------------------------------------------------------------------------


def test_from_config_respects_audit_flag():
    mock_cfg = MagicMock()
    mock_cfg.audit_enabled = False
    hv = HoneyTokenVault.from_config(mock_cfg)
    assert hv._audit_enabled is False


def test_from_config_fallback_when_attr_missing():
    class MinimalConfig:
        pass

    hv = HoneyTokenVault.from_config(MinimalConfig())
    assert hv._audit_enabled is True  # default


# ---------------------------------------------------------------------------
# 10. Persistence round-trip
# ---------------------------------------------------------------------------


def test_reload_from_disk(tmp_store: Path):
    v1 = HoneyTokenVault(path=tmp_store, audit_enabled=False)
    v1.create("persist_me", TokenKind.HEROKU, value="heroku-key-xyz")
    # New vault instance reading same file
    v2 = HoneyTokenVault(path=tmp_store, audit_enabled=False)
    assert "persist_me" in v2.list_tokens()
    tok = v2.get_token("persist_me")
    assert tok.value == "heroku-key-xyz"
    assert tok.kind == TokenKind.HEROKU


# ---------------------------------------------------------------------------
# 11. HoneyFileMonitor — plant + check
# ---------------------------------------------------------------------------


def test_honey_file_monitor_plant_creates_file(tmp_path: Path):
    monitor = HoneyFileMonitor(audit_enabled=False)
    dest = monitor.plant(tmp_path, filename="credentials.json")
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert "service_reference" in data


def test_honey_file_monitor_detects_access(tmp_path: Path):
    alerted: list[Path] = []
    monitor = HoneyFileMonitor(alert_callback=lambda p, d: alerted.append(p), audit_enabled=False)
    dest = monitor.plant(tmp_path, filename="service_account.json")

    # Simulate atime change by bumping stored baseline
    with monitor._lock:
        monitor._files[dest] = 0.0  # pretend atime was ancient

    accessed = monitor.check()
    assert dest in accessed
    assert dest in alerted


def test_honey_file_monitor_no_false_positive(tmp_path: Path):
    alerted: list[Path] = []
    monitor = HoneyFileMonitor(alert_callback=lambda p, d: alerted.append(p), audit_enabled=False)
    dest = monitor.plant(tmp_path, filename="tokens.json")
    # No access; check should return empty
    accessed = monitor.check()
    assert dest not in accessed
    assert not alerted


def test_honey_file_monitor_unmonitor(tmp_path: Path):
    monitor = HoneyFileMonitor(audit_enabled=False)
    dest = monitor.plant(tmp_path, filename="api_keys.txt")
    monitor.unmonitor(dest)
    assert dest not in monitor.monitored_paths()


# ---------------------------------------------------------------------------
# 12. get_token does NOT fire alert
# ---------------------------------------------------------------------------


def test_get_token_no_alert(vault: HoneyTokenVault):
    cb = MagicMock()
    vault._alert_callback = cb
    vault.create("silent_read", TokenKind.SENDGRID)
    _ = vault.get_token("silent_read")
    cb.assert_not_called()


# ---------------------------------------------------------------------------
# 13. Audit trail integration (smoke test — uses real trail if importable)
# ---------------------------------------------------------------------------


def test_audit_trail_written_on_access(tmp_store: Path, tmp_path: Path):
    """Verify that an audit entry is written when audit_enabled=True."""
    tmp_path / "audit.jsonl"
    with patch("hermes_katana.vault.honey_tokens._audit_honey_access") as mock_audit:
        hv = HoneyTokenVault(path=tmp_store, audit_enabled=True)
        hv.create("audit_lure", TokenKind.GENERIC_API_KEY)
        hv.get("audit_lure")
    mock_audit.assert_called_once()
    args = mock_audit.call_args[0]
    assert args[0] == "audit_lure"
    assert args[3] is True  # audit_enabled


def test_no_audit_when_disabled(tmp_store: Path):
    with patch("hermes_katana.vault.honey_tokens._audit_honey_access") as mock_audit:
        hv = HoneyTokenVault(path=tmp_store, audit_enabled=False)
        hv.create("no_audit_lure", TokenKind.GENERIC_API_KEY)
        hv.get("no_audit_lure")
    mock_audit.assert_called_once()
    args = mock_audit.call_args[0]
    assert args[3] is False  # audit_enabled=False passed through
