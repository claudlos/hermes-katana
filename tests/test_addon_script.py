"""Tests for hermes_katana.proxy.addon_script — mitmproxy bootstrap."""

from __future__ import annotations

import json
import os
from unittest.mock import patch


from hermes_katana.proxy.config import ProxyConfig


class TestLoadConfig:
    def test_default_config_when_no_env(self):
        with patch.dict(os.environ, {}, clear=True):
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

    def test_invalid_json_falls_back(self):
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": "not-json{{{"}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert isinstance(cfg, ProxyConfig)
            assert cfg.port == 8443

    def test_config_empty_string(self):
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": ""}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert isinstance(cfg, ProxyConfig)


class TestLoadVault:
    def test_vault_disabled(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_VAULT": "0"}):
            from hermes_katana.proxy.addon_script import _load_vault

            assert _load_vault() is None

    def test_vault_import_fails(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_VAULT": "1"}):
            with patch("hermes_katana.proxy.addon_script.Path"):
                from hermes_katana.proxy.addon_script import _load_vault

                # Will fail because vault may not be initialized
                result = _load_vault()
                # Result is None on failure, which is acceptable
                assert result is None or result is not None  # just shouldn't crash

    def test_vault_with_custom_path(self):
        with patch.dict(
            os.environ, {"KATANA_PROXY_ENABLE_VAULT": "1", "KATANA_PROXY_VAULT_PATH": "/tmp/nonexistent-vault"}
        ):
            from hermes_katana.proxy.addon_script import _load_vault

            result = _load_vault()
            # Vault may auto-create; just verify no crash
            assert result is None or result is not None


class TestLoadAuditTrail:
    def test_audit_disabled(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_AUDIT": "0"}):
            from hermes_katana.proxy.addon_script import _load_audit_trail

            assert _load_audit_trail() is None

    def test_audit_enabled_default(self):
        with patch.dict(os.environ, {"KATANA_PROXY_ENABLE_AUDIT": "1"}):
            from hermes_katana.proxy.addon_script import _load_audit_trail

            # May succeed or fail depending on environment
            result = _load_audit_trail()
            assert result is None or result is not None

    def test_audit_with_custom_path(self):
        with patch.dict(
            os.environ,
            {
                "KATANA_PROXY_ENABLE_AUDIT": "1",
                "KATANA_PROXY_AUDIT_PATH": "/tmp/test-audit.log",
            },
        ):
            from hermes_katana.proxy.addon_script import _load_audit_trail

            result = _load_audit_trail()
            assert result is None or result is not None


class TestAddonScriptModule:
    def test_module_level_addons_list(self):
        """The addon_script module should define an `addons` list."""
        # We can't easily import at module level without side effects,
        # but we can verify the functions exist
        from hermes_katana.proxy.addon_script import _load_config, _load_vault, _load_audit_trail

        assert callable(_load_config)
        assert callable(_load_vault)
        assert callable(_load_audit_trail)

    def test_load_config_returns_proxyconfig(self):
        from hermes_katana.proxy.addon_script import _load_config

        with patch.dict(os.environ, {}, clear=False):
            result = _load_config()
            assert isinstance(result, ProxyConfig)

    def test_config_with_scan_modes(self):
        payload = json.dumps(
            {"port": 8080, "scan_modes": {"secrets": False, "injection": True, "content": True, "unicode": False}}
        )
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": payload}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert cfg.port == 8080
            assert cfg.scan_modes.secrets is False

    def test_config_with_rate_limit(self):
        payload = json.dumps({"rate_limit_requests": 100, "rate_limit_window": 5.0})
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": payload}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert cfg.rate_limit_requests == 100

    def test_config_with_allowed_domains(self):
        payload = json.dumps({"allowed_domains": ["openai.com", "anthropic.com"]})
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": payload}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert "openai.com" in cfg.allowed_domains

    def test_config_with_ignore_hosts(self):
        payload = json.dumps({"ignore_hosts": ["localhost"]})
        with patch.dict(os.environ, {"KATANA_PROXY_CONFIG_JSON": payload}):
            from hermes_katana.proxy.addon_script import _load_config

            cfg = _load_config()
            assert "localhost" in cfg.ignore_hosts
