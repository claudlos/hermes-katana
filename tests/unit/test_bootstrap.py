"""Tests for checkout-driven runtime bootstrap."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import hermes_katana.bootstrap as bootstrap_mod
import pytest
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

    def test_load_checkout_state_reads_escalate_action(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_CORE_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        KatanaInstaller().install(checkout)
        cfg = checkout / ".katana" / "katana.yaml"

        # Default install is fail-closed.
        bootstrap_mod.reset_runtime_cache()
        assert bootstrap_mod.load_checkout_state(checkout).escalate_action == "block"

        # Opt into interactive approval.
        cfg.write_text(
            cfg.read_text(encoding="utf-8").replace("escalate_action: block", "escalate_action: acp_prompt"),
            encoding="utf-8",
        )
        bootstrap_mod.reset_runtime_cache()
        assert bootstrap_mod.load_checkout_state(checkout).escalate_action == "acp_prompt"

        # Unknown values normalize back to block (fail-closed).
        cfg.write_text(
            cfg.read_text(encoding="utf-8").replace("escalate_action: acp_prompt", "escalate_action: nonsense"),
            encoding="utf-8",
        )
        bootstrap_mod.reset_runtime_cache()
        assert bootstrap_mod.load_checkout_state(checkout).escalate_action == "block"

    def test_runtime_cache_invalidates_when_checkout_config_changes(self, monkeypatch, tmp_dir):
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

        monkeypatch.setattr(bootstrap_mod, "KatanaProxy", FakeProxy)

        runtime1 = bootstrap_mod.get_runtime_bundle(checkout)
        cfg = checkout / ".katana" / "katana.yaml"
        cfg.write_text(cfg.read_text(encoding="utf-8").replace("preset: balanced", "preset: max"), encoding="utf-8")
        runtime2 = bootstrap_mod.get_runtime_bundle(checkout)

        assert runtime1 is not None
        assert runtime2 is not None
        assert runtime1.state.policy_preset == "balanced"
        assert runtime2.state.policy_preset == "max"
        assert runtime2 is not runtime1

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

    def test_prepare_runtime_environment_replaces_managed_env(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
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
                    "running": True,
                    "host": self.config.host,
                    "port": self.config.port,
                    "config": {"host": self.config.host, "port": self.config.port},
                }

        monkeypatch.setattr(bootstrap_mod, "KatanaProxy", FakeProxy)
        monkeypatch.setenv("KATANA_ACTIVE", "stale")
        monkeypatch.setenv("KATANA_PROXY_URL", "http://stale.invalid:1")
        monkeypatch.setenv("KATANA_CA_CERT", "/tmp/stale-cert.pem")
        monkeypatch.setenv("UNMANAGED_ENV", "keep-me")

        runtime = bootstrap_mod.prepare_runtime_environment(checkout_root=checkout)

        assert runtime is not None
        assert runtime.state.checkout_root == checkout.resolve()
        assert os.environ["KATANA_ACTIVE"] == "1"
        assert os.environ["KATANA_CHECKOUT_ROOT"] == str(checkout.resolve())
        assert os.environ["KATANA_PROXY_URL"] == "http://127.0.0.1:8080"
        assert os.environ["KATANA_CA_CERT"].endswith(str(Path(".katana") / "certs" / "katana-ca.pem"))
        assert os.environ["UNMANAGED_ENV"] == "keep-me"

    def test_compose_runtime_env_fails_closed_when_hermetic_ml_readiness_required(self, monkeypatch, tmp_dir, caplog):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        KatanaInstaller().install(checkout)
        bootstrap_mod.reset_runtime_cache()
        monkeypatch.setattr(
            "hermes_katana.cli._support.enforce_hermetic_ml_readiness",
            lambda config=None: (_ for _ in ()).throw(RuntimeError("degraded startup")),
        )

        with caplog.at_level(logging.WARNING, logger="hermes_katana.bootstrap"):
            with pytest.raises(RuntimeError, match="degraded startup"):
                bootstrap_mod.compose_runtime_env({}, checkout_root=checkout)

        assert any(
            getattr(record, "katana_event", "") == "bootstrap_runtime_blocked"
            and record.katana_payload["reason"] == "degraded startup"
            for record in caplog.records
        )

    def test_ensure_dispatcher_bootstrap_preserves_existing_handlers(self, monkeypatch, tmp_dir):
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
            def __init__(self):
                self._katana_chain = "existing-chain"
                self._katana_escalate = "existing-escalator"

        monkeypatch.setattr(bootstrap_mod, "KatanaProxy", FakeProxy)

        dispatcher = DummyDispatcher()
        runtime = bootstrap_mod.ensure_dispatcher_bootstrap(dispatcher, checkout_root=checkout)

        assert runtime is not None
        assert dispatcher._katana_chain == "existing-chain"
        assert dispatcher._katana_escalate == "existing-escalator"
        assert dispatcher._katana_runtime is runtime

    def test_resolve_checkout_path_rejects_escape(self, tmp_dir):
        checkout = tmp_dir / "checkout"
        checkout.mkdir()

        inside = bootstrap_mod._resolve_checkout_path(checkout, "nested/file.txt")
        escaped = bootstrap_mod._resolve_checkout_path(checkout, "../outside.txt")

        assert inside == (checkout / "nested" / "file.txt").resolve()
        assert escaped is None
