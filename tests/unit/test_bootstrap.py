"""Tests for checkout-driven runtime bootstrap."""

from __future__ import annotations

import asyncio
from pathlib import Path

import hermes_katana.bootstrap as bootstrap_mod
from hermes_katana.installer.installer import KatanaInstaller
from tests.hermes_compat import (
    HERMES_V010_CORE_SNAPSHOT,
    HERMES_V010_EXTENDED_SNAPSHOT,
    fixture_checkout,
)


def _stub_ca_cert(self, target: Path) -> None:
    cert_dir = target / ".katana" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    (cert_dir / "katana-ca.pem").write_text("stub-cert", encoding="utf-8")
    (cert_dir / "katana-ca-key.pem").write_text("stub-key", encoding="utf-8")


class TestRuntimeBootstrap:
    def test_compose_runtime_env_uses_checkout_state(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        KatanaInstaller().install(checkout)
        bootstrap_mod.reset_runtime_cache()

        proxy_state = {"started": False}

        class FakeProxy:
            def __init__(self, config=None, vault=None, audit=None):
                self.config = config

            def start(self):
                proxy_state["started"] = True
                return 4242

            def status(self):
                return {
                    "running": proxy_state["started"],
                    "host": self.config.host,
                    "port": self.config.port,
                    "config": {"host": self.config.host, "port": self.config.port},
                }

        monkeypatch.setattr(bootstrap_mod, "KatanaProxy", FakeProxy)
        monkeypatch.setattr(bootstrap_mod, "_open_vault", lambda: None)
        monkeypatch.setattr(bootstrap_mod, "_collect_vault_values", lambda vault: set())

        env = bootstrap_mod.compose_runtime_env({}, checkout_root=checkout, start_proxy=True)

        assert env["KATANA_ACTIVE"] == "1"
        assert env["KATANA_CHECKOUT_ROOT"] == str(checkout.resolve())
        assert env["KATANA_CHECKOUT_CONFIG"].endswith(str(Path(".katana") / "katana.yaml"))
        assert env["KATANA_POLICY_SOURCE"] == "preset balanced"
        assert env["KATANA_PROXY_URL"] == "http://127.0.0.1:8080"
        assert proxy_state["started"] is True

    def test_ensure_dispatcher_bootstrap_attaches_chain_and_escalator(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_CORE_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        KatanaInstaller().install(checkout)
        bootstrap_mod.reset_runtime_cache()
        monkeypatch.setattr(bootstrap_mod, "_open_vault", lambda: None)
        monkeypatch.setattr(bootstrap_mod, "_collect_vault_values", lambda vault: set())

        class FakeProxy:
            def __init__(self, config=None, vault=None, audit=None):
                self.config = config

            def status(self):
                return {
                    "running": False,
                    "config": {"host": self.config.host, "port": self.config.port},
                }

        class DummyDispatcher:
            def approve_tool_call(self, ctx):
                return True

        monkeypatch.setattr(bootstrap_mod, "KatanaProxy", FakeProxy)

        dispatcher = DummyDispatcher()
        runtime = bootstrap_mod.ensure_dispatcher_bootstrap(dispatcher, checkout_root=checkout)

        assert runtime is not None
        assert getattr(dispatcher, "_katana_chain", None) is not None
        assert asyncio.run(dispatcher._katana_escalate(object())) is True
