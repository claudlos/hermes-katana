"""Tests for HermesKatana proxy process startup."""

from __future__ import annotations

from pathlib import Path

from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.proxy.runner import KatanaProxy, _PidInfo, _write_pid_file


class TestProxyRunner:
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
