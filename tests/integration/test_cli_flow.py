"""Golden-path CLI integration tests for HermesKatana."""

from __future__ import annotations

import secrets
from importlib import metadata
from io import StringIO
from pathlib import Path

from click.testing import CliRunner
from rich.console import Console

import hermes_katana.audit as audit_pkg
import hermes_katana.audit.trail as audit_mod
import hermes_katana.bootstrap as bootstrap_mod
import hermes_katana.cli.main as cli_main
import hermes_katana.config as config_mod
import hermes_katana.proxy as proxy_pkg
import hermes_katana.proxy.runner as proxy_runner_mod
import hermes_katana.vault as vault_pkg
import hermes_katana.vault.store as vault_mod
from hermes_katana.audit import AuditEntry, AuditEventType, AuditTrail
from tests.hermes_compat import HERMES_V010_EXTENDED_SNAPSHOT, fixture_checkout


def _test_console(stream: StringIO) -> Console:
    return Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        width=120,
    )


def _invoke(runner: CliRunner, monkeypatch, args: list[str]) -> tuple[object, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    monkeypatch.setattr(cli_main, "console", _test_console(stdout))
    monkeypatch.setattr(cli_main, "err_console", _test_console(stderr))
    result = runner.invoke(cli_main.main, args)
    return result, stdout.getvalue(), stderr.getvalue()


class TestCLIGoldenPath:
    def test_cli_operator_flow(self, monkeypatch, tmp_dir):
        runner = CliRunner()
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)

        config_dir = tmp_dir / "state" / "config"
        config_file = config_dir / "config.yaml"
        vault_path = tmp_dir / "state" / "vault" / "vault.json"
        audit_path = tmp_dir / "state" / "audit" / "audit.jsonl"
        pid_path = tmp_dir / "state" / "proxy" / "proxy.pid"

        monkeypatch.setattr(config_mod, "_CONFIG_DIR", config_dir)
        monkeypatch.setattr(config_mod, "_CONFIG_FILE", config_file)
        monkeypatch.setattr(vault_mod, "default_vault_path", lambda: vault_path)
        monkeypatch.setattr(vault_mod, "_default_vault_path", lambda: vault_path)
        monkeypatch.setattr(vault_pkg, "default_vault_path", lambda: vault_path)
        monkeypatch.setattr(audit_mod, "default_audit_path", lambda: audit_path)
        monkeypatch.setattr(audit_mod, "_default_audit_path", lambda: audit_path)
        monkeypatch.setattr(audit_pkg, "default_audit_path", lambda: audit_path)
        monkeypatch.setattr(proxy_runner_mod, "default_pid_path", lambda: pid_path)
        monkeypatch.setattr(proxy_runner_mod, "_default_pid_path", lambda: pid_path)
        monkeypatch.setattr(proxy_pkg, "default_pid_path", lambda: pid_path)

        keyring_state: dict[str, bytes] = {}
        monkeypatch.setattr(vault_mod, "_get_master_key", lambda: keyring_state.get("key"))
        monkeypatch.setattr(vault_mod, "_set_master_key", lambda key: keyring_state.__setitem__("key", key))
        monkeypatch.setattr(vault_mod, "_delete_master_key", lambda: keyring_state.pop("key", None))

        def fake_check_command(name: str) -> tuple[bool, str]:
            tool = Path(name).name.lower()
            if tool == "git":
                return True, "git version 2.50.0"
            if tool == "mitmdump":
                return True, "Mitmproxy: 10.0.0"
            if tool == "docker":
                return False, "not found"
            return True, "Python 3.13"

        monkeypatch.setattr(cli_main, "_check_command", fake_check_command)
        real_version = metadata.version

        def fake_version(distribution: str) -> str:
            if distribution == "mitmproxy":
                return "10.0.0"
            return real_version(distribution)

        monkeypatch.setattr(metadata, "version", fake_version)

        process_state = {"running": False}

        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

            def terminate(self):
                process_state["running"] = False

            def wait(self, timeout=None):
                return 0

            def kill(self):
                process_state["running"] = False

        class FakeThread:
            def __init__(self, target=None, daemon=None, name=None):
                self.target = target
                self.daemon = daemon
                self.name = name

            def start(self):
                return None

        def fake_popen(cmd, stdout=None, stderr=None, env=None):
            process_state["running"] = True
            return FakeProcess()

        monkeypatch.setattr(proxy_runner_mod.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(proxy_runner_mod.threading, "Thread", FakeThread)
        monkeypatch.setattr(
            proxy_runner_mod,
            "_is_process_running",
            lambda pid: process_state["running"] and pid == 4321,
        )
        monkeypatch.setattr(
            proxy_runner_mod.KatanaProxy,
            "_kill_process",
            staticmethod(lambda pid: process_state.__setitem__("running", False)),
        )

        result, stdout, _ = _invoke(
            runner,
            monkeypatch,
            ["doctor", "--target", str(checkout)],
        )
        assert result.exit_code == 0
        assert "Runtime State" in stdout
        assert "Target Checkout" in stdout

        result, stdout, _ = _invoke(runner, monkeypatch, ["policy", "use", "paranoid"])
        assert result.exit_code == 0
        assert config_file.exists()
        assert "Saved to:" in stdout

        result, _, _ = _invoke(
            runner,
            monkeypatch,
            ["vault", "set", "OPENAI_API_KEY", "sk-test-" + secrets.token_hex(4)],
        )
        assert result.exit_code == 0
        assert vault_path.exists()

        result, stdout, _ = _invoke(runner, monkeypatch, ["vault", "verify"])
        assert result.exit_code == 0
        assert "Vault integrity verified." in stdout

        trail = AuditTrail()
        trail.log(
            AuditEntry(
                event_type=AuditEventType.TOOL_CALL,
                tool_name="terminal",
                args_hash="abc123",
                decision="allow",
                details="fixture event",
            )
        )

        result, stdout, _ = _invoke(runner, monkeypatch, ["audit", "show", "--limit", "1"])
        assert result.exit_code == 0
        assert "terminal" in stdout

        result, stdout, _ = _invoke(runner, monkeypatch, ["install", "--target", str(checkout)])
        assert result.exit_code == 0
        assert "Installation complete." in stdout
        assert (checkout / ".katana" / "katana.yaml").exists()
        assert "[KATANA-PATCH] tool_dispatch_hook" in (checkout / "hermes" / "tools" / "dispatch.py").read_text(
            encoding="utf-8"
        )

        result, stdout, _ = _invoke(runner, monkeypatch, ["status", "--target", str(checkout)])
        assert result.exit_code == 0
        assert "Katana installed" in stdout
        assert "OK" in stdout

        result, stdout, _ = _invoke(
            runner,
            monkeypatch,
            ["proxy", "start", "--host", "127.0.0.1", "--port", "9000"],
        )
        assert result.exit_code == 0
        assert "Proxy started: http://127.0.0.1:9000" in stdout

        result, stdout, _ = _invoke(runner, monkeypatch, ["proxy", "status"])
        assert result.exit_code == 0
        assert "http://127.0.0.1:9000" in stdout

        bootstrap_mod.reset_runtime_cache()
        run_state: dict[str, object] = {}
        monkeypatch.setattr(cli_main.shutil, "which", lambda cmd: "hermes.exe" if cmd == "hermes" else None)

        def fake_cli_run(args, env=None):
            run_state["args"] = args
            run_state["env"] = env
            return type("Completed", (), {"returncode": 0})()

        monkeypatch.setattr(cli_main.subprocess, "run", fake_cli_run)

        result, stdout, _ = _invoke(
            runner,
            monkeypatch,
            ["run", "--target", str(checkout), "--", "--task", "hello"],
        )
        assert result.exit_code == 0
        assert run_state["args"] == ["hermes.exe", "--task", "hello"]
        assert run_state["env"]["KATANA_CHECKOUT_ROOT"] == str(checkout.resolve())
        assert run_state["env"]["KATANA_POLICY_SOURCE"] == "preset balanced"

        result, stdout, _ = _invoke(runner, monkeypatch, ["proxy", "stop"])
        assert result.exit_code == 0
        assert "Proxy stopped." in stdout

        result, stdout, _ = _invoke(
            runner,
            monkeypatch,
            ["uninstall", "--target", str(checkout), "--backup"],
        )
        assert result.exit_code == 0
        assert "Uninstallation complete." in stdout
        assert not (checkout / ".katana").exists()

        manifest_path = next((checkout / ".katana-backups").glob("uninstall-*/manifest.json"))
        result, stdout, _ = _invoke(
            runner,
            monkeypatch,
            ["restore", "--manifest", str(manifest_path)],
        )
        assert result.exit_code == 0
        assert "Restore complete." in stdout
        assert (checkout / ".katana").exists()
