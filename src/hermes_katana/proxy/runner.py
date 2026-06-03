"""
Proxy lifecycle management for HermesKatana.

Provides start/stop/restart/status operations for the MITM proxy with:
- PID file management with cross-platform file locking
- Watchdog thread for auto-restart on failure
- Health check endpoint
- Atomic PID file writes
"""

from __future__ import annotations

__all__ = [
    "KatanaProxy",
    "default_pid_path",
]


import hashlib
import importlib.util
import json
import logging
import os
import platform
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from hermes_katana._files import AdvisoryFileLock, atomic_write_text
from hermes_katana._paths import home_or_fallback
from hermes_katana.proxy.config import ProxyConfig

if TYPE_CHECKING:
    from hermes_katana.audit.trail import AuditTrail
    from hermes_katana.vault.store import Vault

logger = logging.getLogger(__name__)

_PROXY_ENV_EXACT = {
    "ALL_PROXY",
    "CURL_CA_BUNDLE",
    "HOME",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "LANG",
    "NO_PROXY",
    "PATH",
    "PYTHONHOME",
    "PYTHONIOENCODING",
    "PYTHONPATH",
    "PYTHONUTF8",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TERM",
    "TMP",
    "TMPDIR",
    "TEMP",
    "TZ",
    "VIRTUAL_ENV",
    "WINDIR",
}
_PROXY_ENV_PREFIXES = (
    "CONDA_",
    "KATANA_",
    "HERMES_KATANA_",
    "LC_",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
)


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


def default_pid_path() -> Path:
    """Return the default PID file path."""
    return home_or_fallback() / ".config" / "hermes-katana" / "proxy" / "proxy.pid"


def _default_pid_path() -> Path:
    """Return the default PID file path."""
    path = default_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _compute_vault_hash(vault: Optional["Vault"]) -> str:
    """Compute a hash of vault state for change detection."""
    if vault is None:
        return "no-vault"
    try:
        keys = sorted(vault.list_keys())
        return hashlib.sha256("|".join(keys).encode()).hexdigest()[:12]
    except Exception:
        return "vault-error"


class _PidInfo:
    """Data stored in the PID file."""

    def __init__(
        self,
        pid: int,
        host: str,
        port: int,
        vault_hash: str,
        started_at: float,
    ) -> None:
        self.pid = pid
        self.host = host
        self.port = port
        self.vault_hash = vault_hash
        self.started_at = started_at

    def to_dict(self) -> dict[str, Any]:
        """Serialize PID info to a dictionary."""
        return {
            "pid": self.pid,
            "host": self.host,
            "port": self.port,
            "vault_hash": self.vault_hash,
            "started_at": self.started_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_PidInfo":
        """Deserialize PID info from a dictionary."""
        return cls(
            pid=data["pid"],
            host=data.get("host", "127.0.0.1"),
            port=data["port"],
            vault_hash=data.get("vault_hash", "unknown"),
            started_at=data.get("started_at", 0.0),
        )

    def to_json(self) -> str:
        """Serialize PID info to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> "_PidInfo":
        """Deserialize PID info from JSON."""
        return cls.from_dict(json.loads(raw))


def _write_pid_file(path: Path, info: _PidInfo) -> None:
    """Atomically write a PID file with file locking."""
    try:
        with AdvisoryFileLock(path):
            atomic_write_text(path, info.to_json())
    except Exception as exc:
        logger.error("Failed to write PID file: %s", exc)
        raise


def _read_pid_file(path: Path) -> Optional[_PidInfo]:
    """Read a PID file with file locking."""
    if not path.exists():
        return None
    try:
        with AdvisoryFileLock(path):
            raw = path.read_text(encoding="utf-8")
        return _PidInfo.from_json(raw)
    except Exception as exc:
        logger.debug("Could not read PID file: %s", exc)
        return None


def _remove_pid_file(path: Path) -> None:
    """Remove the PID file."""
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.debug("Could not remove PID file: %s", exc)


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if pid <= 0:
        return False
    try:
        if platform.system() == "Windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def _read_process_command(pid: int) -> str:
    """Read a process command line for validation when possible."""
    if pid <= 0:
        return ""

    if os.name == "posix":
        proc_cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            raw = proc_cmdline.read_bytes()
            return raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except OSError:
            ps_run = getattr(subprocess, "run", None)
            if ps_run is None:
                return ""
            try:
                result = ps_run(
                    ["ps", "-p", str(pid), "-o", "command="],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
            except (AttributeError, OSError, subprocess.TimeoutExpired):
                return ""
            return str(getattr(result, "stdout", "")).strip()

    return ""


def _pid_matches_proxy_process(pid: int, info: Optional[_PidInfo] = None) -> bool:
    """Validate that a PID belongs to a Katana-managed mitmproxy process."""
    if not _is_process_running(pid):
        return False

    command = _read_process_command(pid)
    if not command:
        return False

    markers = ["mitmproxy.tools.main", "mitmdump", "addon_script.py"]
    if info is not None:
        markers.append(f"--listen-port {info.port}")

    return all(marker in command for marker in markers)


def _mitmproxy_runtime_available() -> bool:
    """Return True when mitmproxy is importable by the active Python runtime."""
    try:
        return importlib.util.find_spec("mitmproxy.tools.main") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _invoke_kill_process(kill_process: Any, pid: int, info: Optional[_PidInfo]) -> None:
    """Call a kill handler with backward-compatible support for legacy single-arg stubs."""
    try:
        kill_process(pid, info)
    except TypeError:
        kill_process(pid)


def _build_proxy_env(
    *,
    config_json: str,
    inject_credentials_enabled: bool,
    vault_path: Optional[Path],
    audit_path: Optional[Path],
) -> dict[str, str]:
    """Build a minimally-scoped child environment for the proxy process."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key in _PROXY_ENV_EXACT or key.startswith(_PROXY_ENV_PREFIXES)
    }
    env.pop("HERMES_KATANA_VAULT_KEY", None)
    env["KATANA_PROXY_CONFIG_JSON"] = config_json
    env["KATANA_PROXY_ENABLE_VAULT"] = "1" if inject_credentials_enabled else "0"
    env["KATANA_PROXY_ENABLE_AUDIT"] = "1"
    if vault_path is not None:
        env["KATANA_PROXY_VAULT_PATH"] = str(vault_path)
    if audit_path is not None:
        env["KATANA_PROXY_AUDIT_PATH"] = str(audit_path)
    return env


# ---------------------------------------------------------------------------
# Health check server
# ---------------------------------------------------------------------------


class _HealthCheckServer(threading.Thread):
    """Simple HTTP health check endpoint."""

    def __init__(self, port: int, proxy_ref: "KatanaProxy") -> None:
        super().__init__(daemon=True, name="katana-health-check")
        self.port = port
        self.proxy_ref = proxy_ref
        self._server: Any = None

    def run(self) -> None:
        """Run the health check HTTP server."""
        import http.server
        import json as _json

        proxy_ref = self.proxy_ref

        class Handler(http.server.BaseHTTPRequestHandler):
            """HTTP request handler for health check endpoint."""

            def do_GET(self) -> None:  # noqa: N802
                """Handle GET requests for health checks."""
                if self.path == "/health":
                    status = proxy_ref.status()
                    code = 200 if status.get("running") else 503
                    body = _json.dumps(status).encode()
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                """Suppress default HTTP access logging."""
                pass  # Suppress default HTTP logging

        try:
            self._server = http.server.HTTPServer(("127.0.0.1", self.port), Handler)
            self._server.serve_forever()
        except Exception as exc:
            logger.debug("Health check server error: %s", exc)

    def stop(self) -> None:
        """Stop the health check server."""
        if self._server:
            self._server.shutdown()


# ---------------------------------------------------------------------------
# KatanaProxy
# ---------------------------------------------------------------------------


class KatanaProxy:
    """Manages the HermesKatana MITM proxy lifecycle.

    Provides start/stop/restart/status operations with PID file management,
    watchdog auto-restart, and optional health check endpoint.

    Args:
        config: Proxy configuration.
        vault: Optional vault for credential injection.
        audit: Optional audit trail for event logging.
        pid_path: Optional custom PID file path.

    Example:
        >>> config = ProxyConfig(port=8443)
        >>> proxy = KatanaProxy(config)
        >>> pid = proxy.start()
        >>> proxy.is_running()
        True
        >>> proxy.stop()
    """

    def __init__(
        self,
        config: Optional[ProxyConfig] = None,
        vault: Optional["Vault"] = None,
        audit: Optional["AuditTrail"] = None,
        pid_path: Optional[Path] = None,
    ) -> None:
        self.config = config or ProxyConfig()
        self.vault = vault
        self.audit = audit
        self._pid_path = pid_path or _default_pid_path()
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._watchdog: Optional[threading.Thread] = None
        self._watchdog_stop = threading.Event()
        self._health_server: Optional[_HealthCheckServer] = None
        self._lock = threading.Lock()
        # Runtime counters for enhanced health/status reporting
        self._request_count = 0
        self._started_at: Optional[float] = None
        self._shutting_down = threading.Event()

    def start(self) -> int:
        """Start the proxy process.

        Returns:
            The PID of the started proxy process.

        Raises:
            RuntimeError: If the proxy is already running on the same port.
        """
        with self._lock:
            # Check for existing instance
            existing = _read_pid_file(self._pid_path)
            if existing and _pid_matches_proxy_process(existing.pid, existing):
                if existing.port == self.config.port:
                    logger.info(
                        "Proxy already running on port %d (PID %d)",
                        existing.port,
                        existing.pid,
                    )
                    return existing.pid
                else:
                    # Different port - stop the old one first
                    logger.info(
                        "Stopping existing proxy on port %d before starting on %d",
                        existing.port,
                        self.config.port,
                    )
                    _invoke_kill_process(self._kill_process, existing.pid, existing)
            elif existing is not None:
                logger.warning("Ignoring stale or untrusted proxy PID file at %s", self._pid_path)
                _remove_pid_file(self._pid_path)

            return self._start_proxy()

    def _start_proxy(self) -> int:
        """Internal: start the mitmproxy process."""
        addon_script = Path(__file__).with_name("addon_script.py")

        if self.config.inject_credentials and not self.config.tls_verify:
            raise RuntimeError("Proxy refuses to start with credential injection enabled while tls_verify is false.")

        if not _mitmproxy_runtime_available():
            raise RuntimeError(
                "mitmproxy is not importable in the active Python environment. "
                "Install the proxy extra with: pip install 'hermes-katana[proxy]'"
            )

        # Build the mitmdump command
        cmd = [
            sys.executable,
            "-m",
            "mitmproxy.tools.main",
            "mitmdump",
            "--listen-host",
            self.config.host,
            "--listen-port",
            str(self.config.port),
            "--set",
            f"ssl_insecure={'true' if not self.config.tls_verify else 'false'}",
        ]

        cmd.extend(["-s", str(addon_script)])

        # Add ignore hosts
        for host in self.config.ignore_hosts:
            cmd.extend(["--ignore-hosts", host])

        # For now, use a marker approach - the actual addon is loaded
        # via mitmproxy script. In production, this would reference the
        # addon script path.
        logger.info(
            "Starting proxy on %s:%d",
            self.config.host,
            self.config.port,
        )

        vault_path = getattr(self.vault, "path", None)
        audit_path = getattr(self.audit, "path", None)
        env = _build_proxy_env(
            config_json=self.config.model_dump_json(),
            inject_credentials_enabled=self.config.inject_credentials,
            vault_path=vault_path,
            audit_path=audit_path,
        )

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            # Fail fast when mitmproxy is missing or startup aborts immediately.
            time.sleep(0.2)
            returncode = self._process.poll()
            if returncode is not None:
                self._process = None
                raise RuntimeError(
                    f"Proxy process exited during startup (exit code {returncode}). Check that mitmproxy is installed."
                )
            pid = self._process.pid
        except FileNotFoundError:
            raise RuntimeError(
                "mitmproxy is not installed or not found on PATH. Install it with: pip install mitmproxy"
            )

        # Write PID file
        vault_hash = _compute_vault_hash(self.vault)
        info = _PidInfo(
            pid=pid,
            host=self.config.host,
            port=self.config.port,
            vault_hash=vault_hash,
            started_at=time.time(),
        )
        _write_pid_file(self._pid_path, info)
        self._started_at = time.time()
        self._shutting_down.clear()
        logger.info("Proxy started with PID %d", pid)

        # Start watchdog
        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="katana-proxy-watchdog",
        )
        self._watchdog.start()

        # Start health check server if configured
        if self.config.health_check_port:
            self._health_server = _HealthCheckServer(self.config.health_check_port, self)
            self._health_server.start()
            logger.info(
                "Health check endpoint at http://127.0.0.1:%d/health",
                self.config.health_check_port,
            )

        return pid

    def stop(self, *, graceful: bool = True) -> None:
        """Stop the proxy process and clean up.

        Args:
            graceful: If True, send SIGTERM and wait for the configured
                graceful_shutdown_timeout before force-killing.
        """
        with self._lock:
            self._shutting_down.set()

            # Stop watchdog
            self._watchdog_stop.set()

            # Stop health check server
            if self._health_server:
                self._health_server.stop()
                self._health_server = None

            timeout = self.config.graceful_shutdown_timeout if graceful else 1.0

            # Stop proxy process
            info = _read_pid_file(self._pid_path)
            if info and _pid_matches_proxy_process(info.pid, info):
                _invoke_kill_process(self._kill_process, info.pid, info)
            elif info is not None:
                logger.warning("Refusing to kill PID %d because it does not look like a Katana proxy", info.pid)

            if self._process:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=timeout)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass
                self._process = None

            self._started_at = None
            _remove_pid_file(self._pid_path)
            logger.info("Proxy stopped")

    def is_running(self) -> bool:
        """Check if the proxy is currently running.

        Returns:
            True if the proxy process is active.
        """
        info = _read_pid_file(self._pid_path)
        if info is None:
            return False
        running = _pid_matches_proxy_process(info.pid, info)
        if not running:
            _remove_pid_file(self._pid_path)
        return running

    def status(self) -> dict[str, Any]:
        """Get comprehensive proxy status.

        Returns:
            Dict with running state, PID, port, uptime, vault hash, etc.
        """
        info = _read_pid_file(self._pid_path)
        running = False
        if info:
            running = _pid_matches_proxy_process(info.pid, info)
            if not running:
                _remove_pid_file(self._pid_path)
                info = None

        result: dict[str, Any] = {
            "running": running,
            "config": {
                "host": info.host if info else self.config.host,
                "port": info.port if info else self.config.port,
                "tls_verify": self.config.tls_verify,
                "inject_credentials": self.config.inject_credentials,
            },
        }

        if info:
            result["pid"] = info.pid
            result["host"] = info.host
            result["port"] = info.port
            result["vault_hash"] = info.vault_hash
            result["started_at"] = info.started_at
            if running and info.started_at > 0:
                result["uptime_seconds"] = time.time() - info.started_at

        return result

    def _watchdog_loop(self) -> None:
        """Watchdog thread that monitors the proxy and restarts on failure."""
        while not self._watchdog_stop.wait(timeout=5.0):
            info = _read_pid_file(self._pid_path)
            if info is None:
                continue
            if not _pid_matches_proxy_process(info.pid, info):
                logger.warning("Proxy process (PID %d) died, restarting...", info.pid)
                with self._lock:
                    try:
                        self._start_proxy()
                        logger.info("Proxy restarted successfully")
                    except Exception as exc:
                        logger.error("Failed to restart proxy: %s", exc)

    @staticmethod
    def _kill_process(pid: int, info: Optional[_PidInfo] = None) -> None:
        """Kill a process by PID after validating that it is the proxy."""
        if not _pid_matches_proxy_process(pid, info):
            raise RuntimeError(f"Refusing to kill unrelated process {pid}")
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True,
                    timeout=5,
                )
            else:
                os.kill(pid, signal.SIGTERM)
                # Give it a moment to clean up
                time.sleep(0.5)
                if _is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
