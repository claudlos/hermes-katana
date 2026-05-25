"""Tests for HermesKatana proxy process startup."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.proxy.runner import KatanaProxy, _PidInfo, _write_pid_file, default_pid_path


class TestProxyRunner:
    def test_default_pid_path_is_user_scoped(self, monkeypatch, tmp_dir):
        monkeypatch.setattr("hermes_katana.proxy.runner.home_or_fallback", lambda: tmp_dir)

        path = default_pid_path()

        assert path == tmp_dir / ".config" / "hermes-katana" / "proxy" / "proxy.pid"

    def test_start_loads_katana_addon_script(self, monkeypatch, tmp_dir):
        seen: dict[str, object] = {}

        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

        class FakeThread:
            def __init__(self, target=None, daemon=None, name=None):
                self.target = target
                self.daemon = daemon
                self.name = name

            def start(self):
                return None

        def fake_popen(cmd, stdout=None, stderr=None, env=None):
            seen["cmd"] = cmd
            seen["env"] = env
            return FakeProcess()

        monkeypatch.setattr("hermes_katana.proxy.runner.subprocess.Popen", fake_popen)
        monkeypatch.setattr("hermes_katana.proxy.runner.threading.Thread", FakeThread)

        proxy = KatanaProxy(
            config=ProxyConfig(host="127.0.0.1", port=8080),
            pid_path=tmp_dir / "proxy.pid",
        )

        pid = proxy.start()

        assert pid == 4321
        assert "cmd" in seen
        cmd = seen["cmd"]
        assert "-s" in cmd
        script_path = Path(cmd[cmd.index("-s") + 1])
        assert script_path.name == "addon_script.py"

        env = seen["env"]
        assert env["KATANA_PROXY_ENABLE_VAULT"] == "1"
        assert "KATANA_PROXY_CONFIG_JSON" in env

    def test_start_scrubs_vault_key_from_child_env(self, monkeypatch, tmp_dir):
        seen: dict[str, object] = {}

        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

        class FakeThread:
            def __init__(self, target=None, daemon=None, name=None):
                self.target = target
                self.daemon = daemon
                self.name = name

            def start(self):
                return None

        def fake_popen(cmd, stdout=None, stderr=None, env=None):
            seen["env"] = env
            return FakeProcess()

        monkeypatch.setattr("hermes_katana.proxy.runner.subprocess.Popen", fake_popen)
        monkeypatch.setattr("hermes_katana.proxy.runner.threading.Thread", FakeThread)
        monkeypatch.setenv("HERMES_KATANA_VAULT_KEY", "secret")

        proxy = KatanaProxy(
            config=ProxyConfig(host="127.0.0.1", port=8080),
            pid_path=tmp_dir / "proxy.pid",
        )

        proxy.start()

        assert "HERMES_KATANA_VAULT_KEY" not in seen["env"]

    def test_start_scrubs_unrelated_secret_env_from_child(self, monkeypatch, tmp_dir):
        seen: dict[str, object] = {}

        class FakeProcess:
            pid = 4321

            def poll(self):
                return None

        class FakeThread:
            def __init__(self, target=None, daemon=None, name=None):
                self.target = target
                self.daemon = daemon
                self.name = name

            def start(self):
                return None

        def fake_popen(cmd, stdout=None, stderr=None, env=None):
            seen["env"] = env
            return FakeProcess()

        monkeypatch.setattr("hermes_katana.proxy.runner.subprocess.Popen", fake_popen)
        monkeypatch.setattr("hermes_katana.proxy.runner.threading.Thread", FakeThread)
        monkeypatch.setenv("OPENAI_API_KEY", "should-not-leak")
        monkeypatch.setenv("PATH", "/usr/bin")

        proxy = KatanaProxy(
            config=ProxyConfig(host="127.0.0.1", port=8080),
            pid_path=tmp_dir / "proxy.pid",
        )

        proxy.start()

        assert "OPENAI_API_KEY" not in seen["env"]
        assert seen["env"]["PATH"] == "/usr/bin"

    def test_start_refuses_insecure_credential_injection(self, tmp_dir):
        proxy = KatanaProxy(
            config=ProxyConfig(host="127.0.0.1", port=8080, tls_verify=False, inject_credentials=True),
            pid_path=tmp_dir / "proxy.pid",
        )

        with pytest.raises(RuntimeError, match="tls_verify is false"):
            proxy.start()

    def test_status_clears_stale_pid_file(self, monkeypatch, tmp_dir):
        pid_path = tmp_dir / "proxy.pid"
        _write_pid_file(
            pid_path,
            _PidInfo(
                pid=9876,
                host="127.0.0.1",
                port=8080,
                vault_hash="deadbeef",
                started_at=1.0,
            ),
        )
        monkeypatch.setattr(
            "hermes_katana.proxy.runner._is_process_running",
            lambda pid: False,
        )

        proxy = KatanaProxy(
            config=ProxyConfig(host="127.0.0.1", port=8080),
            pid_path=pid_path,
        )

        status = proxy.status()

        assert status == {
            "running": False,
            "config": {
                "host": "127.0.0.1",
                "port": 8080,
                "tls_verify": True,
                "inject_credentials": True,
            },
        }
        assert not pid_path.exists()

    def test_stop_refuses_to_kill_untrusted_pid(self, monkeypatch, tmp_dir):
        pid_path = tmp_dir / "proxy.pid"
        _write_pid_file(
            pid_path,
            _PidInfo(
                pid=9876,
                host="127.0.0.1",
                port=8080,
                vault_hash="deadbeef",
                started_at=1.0,
            ),
        )

        killed: list[int] = []
        monkeypatch.setattr("hermes_katana.proxy.runner._pid_matches_proxy_process", lambda pid, info=None: False)
        monkeypatch.setattr(KatanaProxy, "_kill_process", staticmethod(lambda pid, info=None: killed.append(pid)))

        proxy = KatanaProxy(
            config=ProxyConfig(host="127.0.0.1", port=8080),
            pid_path=pid_path,
        )
        proxy.stop()

        assert killed == []
        assert not pid_path.exists()
