"""Tests for HermesKatana proxy addon_script (mitmproxy bootstrap)."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import hermes_katana.audit as audit_mod
import hermes_katana.vault as vault_mod

from hermes_katana.proxy.config import ProxyConfig


# ======================================================================
# _load_config
# ======================================================================


class TestLoadConfig:
    def test_default_config_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure KATANA_PROXY_CONFIG_JSON is not set
            os.environ.pop("KATANA_PROXY_CONFIG_JSON", None)
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert isinstance(cfg, ProxyConfig)
            assert cfg.port == 8443

    def test_config_from_env_json(self):
        payload = json.dumps({"port": 9999, "host": "0.0.0.0"})
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": payload}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert cfg.port == 9999
            assert cfg.host == "0.0.0.0"

    def test_invalid_json_falls_back(self):
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": "not-json!!!"}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            # Should fall back to defaults
            assert isinstance(cfg, ProxyConfig)


# ======================================================================
# _load_vault
# ======================================================================


class TestLoadVault:
    def test_vault_disabled(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_VAULT": "0"}):
            from hermes_katana.proxy.addon_script import _load_vault

            assert _load_vault() is None

    def test_vault_enabled_but_import_fails(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_VAULT": "1"}):
            from hermes_katana.proxy.addon_script import _load_vault

            # May return None if vault can't be initialized (no master key etc.)
            # Should not raise
            _load_vault()
            # Result is either a Vault or None — both acceptable


# ======================================================================
# _load_audit_trail
# ======================================================================


class TestLoadAuditTrail:
    def test_audit_disabled(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_AUDIT": "0"}):
            from hermes_katana.proxy.addon_script import _load_audit_trail

            assert _load_audit_trail() is None

    def test_audit_enabled_default(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_AUDIT": "1"}):
            from hermes_katana.proxy.addon_script import _load_audit_trail

            # May succeed or fail depending on filesystem — should not raise
            _load_audit_trail()

    def test_custom_paths_are_passed_to_backends(self, monkeypatch, tmp_path):
        from hermes_katana.proxy.addon_script import _load_audit_trail, _load_vault

        calls: dict[str, object] = {}

        class FakeVault:
            def __init__(self, *, path=None, auto_create=False):
                calls["vault_path"] = path
                calls["vault_auto_create"] = auto_create

        class FakeAuditTrail:
            def __init__(self, *, path=None):
                calls["audit_path"] = path

        monkeypatch.setattr(vault_mod, "Vault", FakeVault)
        monkeypatch.setattr(audit_mod, "AuditTrail", FakeAuditTrail)
        monkeypatch.setenv("KATANA_PROXY_ENABLE_VAULT", "1")
        monkeypatch.setenv("KATANA_PROXY_ENABLE_AUDIT", "1")
        monkeypatch.setenv("KATANA_PROXY_VAULT_PATH", str(tmp_path / "vault.db"))
        monkeypatch.setenv("KATANA_PROXY_AUDIT_PATH", str(tmp_path / "audit.jsonl"))

        vault = _load_vault()
        audit = _load_audit_trail()

        assert isinstance(vault, FakeVault)
        assert isinstance(audit, FakeAuditTrail)
        assert calls["vault_path"] == tmp_path / "vault.db"
        assert calls["vault_auto_create"] is False
        assert calls["audit_path"] == tmp_path / "audit.jsonl"
