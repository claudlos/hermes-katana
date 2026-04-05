"""Regression tests for HermesKatana CLI wiring."""

from __future__ import annotations

import os
from io import StringIO
from types import SimpleNamespace
from pathlib import Path

from click.testing import CliRunner
from rich.console import Console

import hermes_katana.bootstrap as bootstrap_mod
import hermes_katana.cli.main as cli_main
import hermes_katana.config as config_mod
import hermes_katana.installer as installer_mod
import hermes_katana.proxy as proxy_mod


def _test_console(stream: StringIO) -> Console:
    return Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=120,
    )


class TestCLIContracts:
    def setup_method(self) -> None:
        self.runner = CliRunner()

    def test_policy_use_persists_selected_preset(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
        monkeypatch.setattr(config_mod, "_CONFIG_DIR", tmp_dir)
        monkeypatch.setattr(config_mod, "_CONFIG_FILE", tmp_dir / "config.yaml")
        # `policy use` writes KATANA_POLICY_PRESET into os.environ. Round-trip via
        # setenv+delenv so monkeypatch registers "originally absent" for restoration
        # (delenv alone with raising=False doesn't track nonexistent vars).
        if "KATANA_POLICY_PRESET" in os.environ:
            monkeypatch.setenv("KATANA_POLICY_PRESET", os.environ["KATANA_POLICY_PRESET"])
        else:
            monkeypatch.setenv("KATANA_POLICY_PRESET", "")
            monkeypatch.delenv("KATANA_POLICY_PRESET")

        result = self.runner.invoke(cli_main.main, ["policy", "use", "paranoid"])

        assert result.exit_code == 0
        config = config_mod.load_config()
        assert config.policy_preset == "paranoid"
        assert config.policy_path is None

        engine, source = cli_main._load_policy_engine()
        assert engine.policy_set_name == "paranoid"
        assert source == "preset paranoid"

    def test_vault_list_uses_real_vault_helper(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        calls: list[bool] = []

        class FakeVault:
            def list_keys(self):
                return ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

        def fake_open_vault(*, auto_create: bool):
            calls.append(auto_create)
            return FakeVault()

        monkeypatch.setattr(cli_main, "_open_vault", fake_open_vault)

        result = self.runner.invoke(cli_main.main, ["vault", "list"])

        assert result.exit_code == 0
        assert calls == [False]
        assert "OPENAI_API_KEY" in stdout.getvalue()

    def test_audit_stats_uses_default_audit_trail(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        class FakeTrail:
            def stats(self):
                return {"total_entries": 3, "file_exists": True}

        monkeypatch.setattr(cli_main, "_open_audit_trail", lambda: FakeTrail())

        result = self.runner.invoke(cli_main.main, ["audit", "stats"])

        assert result.exit_code == 0
        assert "total_entries" in stdout.getvalue()

    def test_proxy_start_builds_proxy_config(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        class FakeProxy:
            last_config = None

            def __init__(self, config=None):
                FakeProxy.last_config = config

            def start(self):
                return 1234

        monkeypatch.setattr(proxy_mod, "KatanaProxy", FakeProxy)

        result = self.runner.invoke(
            cli_main.main,
            ["proxy", "start", "--host", "0.0.0.0", "--port", "9000"],
        )

        assert result.exit_code == 0
        assert FakeProxy.last_config is not None
        assert FakeProxy.last_config.host == "0.0.0.0"
        assert FakeProxy.last_config.port == 9000

    def test_proxy_status_reads_structured_status(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        class FakeProxy:
            def __init__(self, config=None):
                self.config = config

            def status(self):
                return {
                    "running": True,
                    "host": "127.0.0.1",
                    "port": 8080,
                    "config": {
                        "host": "127.0.0.1",
                        "port": 8080,
                        "tls_verify": True,
                        "inject_credentials": True,
                    },
                }

        monkeypatch.setattr(proxy_mod, "KatanaProxy", FakeProxy)

        result = self.runner.invoke(cli_main.main, ["proxy", "status"])

        assert result.exit_code == 0
        assert "http://127.0.0.1:8080" in stdout.getvalue()

    def test_audit_show_renders_query_entries(self, monkeypatch):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        entry = SimpleNamespace(
            timestamp=SimpleNamespace(isoformat=lambda: "2026-03-24T00:00:00+00:00"),
            event_type=SimpleNamespace(value="tool_call"),
            tool_name="terminal",
            decision="allow",
            details="ok",
        )

        class FakeTrail:
            def query(self, limit=20):
                return [entry]

        monkeypatch.setattr(cli_main, "_open_audit_trail", lambda: FakeTrail())

        result = self.runner.invoke(cli_main.main, ["audit", "show", "--limit", "1"])

        assert result.exit_code == 0
        assert "terminal" in stdout.getvalue()

    def test_run_uses_checkout_bootstrap_environment(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        checkout = tmp_dir / "fixture-hermes"
        checkout.mkdir()

        captured: dict[str, object] = {}

        monkeypatch.setattr(
            bootstrap_mod,
            "load_checkout_state",
            lambda checkout_root=None: SimpleNamespace(
                checkout_root=Path(checkout_root) if checkout_root else checkout,
                policy_source="preset paranoid",
            ),
        )
        monkeypatch.setattr(
            bootstrap_mod,
            "compose_runtime_env",
            lambda env, checkout_root=None, start_proxy=False: {
                **env,
                "KATANA_ACTIVE": "1",
                "KATANA_CHECKOUT_ROOT": str(checkout),
                "KATANA_POLICY_SOURCE": "preset paranoid",
                "KATANA_PROXY_URL": "http://127.0.0.1:9000",
            },
        )
        monkeypatch.setattr(cli_main.shutil, "which", lambda cmd: "hermes.exe")

        def fake_run(args, env=None):
            captured["args"] = args
            captured["env"] = env
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(cli_main.subprocess, "run", fake_run)

        result = self.runner.invoke(
            cli_main.main,
            ["run", "--target", str(checkout), "--", "--task", "hello"],
        )

        assert result.exit_code == 0
        assert captured["args"] == ["hermes.exe", "--task", "hello"]
        env = captured["env"]
        assert env["KATANA_CHECKOUT_ROOT"] == str(checkout)
        assert env["KATANA_POLICY_SOURCE"] == "preset paranoid"
        assert env["KATANA_PROXY_URL"] == "http://127.0.0.1:9000"

    def test_restore_uses_installer_manifest(self, monkeypatch, tmp_dir):
        stdout = StringIO()
        stderr = StringIO()
        monkeypatch.setattr(cli_main, "console", _test_console(stdout))
        monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))

        manifest = tmp_dir / "manifest.json"
        manifest.write_text("{}", encoding="utf-8")

        class FakeInstaller:
            def restore(self, manifest_path, dry_run=False):
                return ["Restore hermes/tools/dispatch.py", "Remove .katana"]

        monkeypatch.setattr(installer_mod, "KatanaInstaller", FakeInstaller)

        result = self.runner.invoke(
            cli_main.main,
            ["restore", "--manifest", str(manifest), "--dry-run"],
        )

        assert result.exit_code == 0
        output = stdout.getvalue()
        assert "Restore hermes/tools/dispatch.py" in output
        assert "Dry run complete." in output
