"""
HermesKatana Honey Token Vault - Decoy secrets that alert when accessed.

Generates realistic-looking but fake credentials (API keys, tokens, passwords,
certificates) and plants them as traps.  Any access to a honey token triggers
an alert: an audit-trail entry plus an optional user-supplied callback.

Features
--------
- Realistic decoy generation: AWS keys, GitHub tokens, Stripe keys, JWTs, etc.
- Plant tokens in environment variables or config files
- Vault integration: read-through hooks detect honey-token access
- Canary URL registration: tokens that trigger HTTP pings when observed
- Honey files: dummy config paths that log read attempts
- Configurable alert callback
- Thread-safe; atomic file operations

Usage::

    from hermes_katana.vault.honey_tokens import HoneyTokenVault, TokenKind

    hv = HoneyTokenVault()
    token = hv.create("lure_openai", TokenKind.OPENAI)
    # Plant in env
    hv.plant_env("lure_openai")
    # Read detection
    value = hv.get("lure_openai")   # → triggers alert automatically
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import string
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "HoneyTokenVault",
    "TokenKind",
    "HoneyToken",
    "HoneyFileMonitor",
    "HoneyTokenError",
    "default_honey_token_path",
]

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_CANARY_TIMEOUT = 5  # seconds; fire-and-forget HTTP canary ping
_DEFAULT_STORE_FILE = "honey_tokens.json"
_DEFAULT_HONEY_DIR = "honey"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class HoneyTokenError(Exception):
    """Base exception for the honey-token vault."""


# ---------------------------------------------------------------------------
# Token kinds
# ---------------------------------------------------------------------------


class TokenKind(str, Enum):
    """Categories of realistic-looking decoy credentials."""

    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB = "github"
    OPENAI = "openai"
    STRIPE = "stripe"
    GENERIC_API_KEY = "generic_api_key"
    JWT = "jwt"
    PASSWORD = "password"
    SLACK = "slack"
    TWILIO = "twilio"
    SENDGRID = "sendgrid"
    HEROKU = "heroku"
    DATABASE_URL = "database_url"


# ---------------------------------------------------------------------------
# Generator helpers
# ---------------------------------------------------------------------------

_ALPHA = string.ascii_letters
_ALNUM = string.ascii_letters + string.digits
_HEX = string.hexdigits[:16]  # lowercase hex


def _rand_hex(n: int) -> str:
    return secrets.token_hex(n)


def _rand_alnum(n: int) -> str:
    return "".join(secrets.choice(_ALNUM) for _ in range(n))


def _rand_upper_alnum(n: int) -> str:
    return "".join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(n))


def _rand_b64url(n: int) -> str:
    """n bytes → url-safe base64, no padding."""
    import base64

    return base64.urlsafe_b64encode(secrets.token_bytes(n)).rstrip(b"=").decode()


_FAKE_USERS = ["root", "admin", "sysadmin", "deploy", "app", "service"]
_FAKE_HOSTS = ["db.internal", "prod-db.company.com", "postgres.cluster", "mysql.internal"]
_FAKE_DBNAMES = ["production", "app_prod", "main", "core"]


def _generate_value(kind: TokenKind) -> str:
    """Return a realistic-looking but fake credential value."""
    if kind == TokenKind.AWS_ACCESS_KEY:
        return "AKIA" + _rand_upper_alnum(16)

    if kind == TokenKind.AWS_SECRET_KEY:
        chars = string.ascii_letters + string.digits + "/+"
        return "".join(secrets.choice(chars) for _ in range(40))

    if kind == TokenKind.GITHUB:
        return "ghp_" + _rand_alnum(36)

    if kind == TokenKind.OPENAI:
        return "sk-" + _rand_alnum(48)

    if kind == TokenKind.STRIPE:
        prefix = secrets.choice(["sk_live_", "rk_live_"])
        return prefix + _rand_alnum(24)

    if kind == TokenKind.GENERIC_API_KEY:
        return _rand_hex(16)

    if kind == TokenKind.JWT:
        # Fake JWT: header.payload.signature
        header = _rand_b64url(20)
        payload = _rand_b64url(30)
        sig = _rand_b64url(32)
        return f"{header}.{payload}.{sig}"

    if kind == TokenKind.PASSWORD:
        chars = string.ascii_letters + string.digits + "!@#$%^&*"
        return "".join(secrets.choice(chars) for _ in range(24))

    if kind == TokenKind.SLACK:
        return "xoxb-" + "-".join(_rand_alnum(10) for _ in range(3))

    if kind == TokenKind.TWILIO:
        return "SK" + _rand_hex(16)

    if kind == TokenKind.SENDGRID:
        return "SG." + _rand_alnum(22) + "." + _rand_alnum(43)

    if kind == TokenKind.HEROKU:
        # Heroku API key format: uuid-like
        parts = [_rand_hex(4), _rand_hex(2), _rand_hex(2), _rand_hex(2), _rand_hex(6)]
        return "-".join(parts)

    if kind == TokenKind.DATABASE_URL:
        user = secrets.choice(_FAKE_USERS)
        password = _rand_alnum(20)
        host = secrets.choice(_FAKE_HOSTS)
        port = secrets.choice([5432, 3306, 27017])
        dbname = secrets.choice(_FAKE_DBNAMES)
        scheme = "postgresql" if port == 5432 else ("mongodb" if port == 27017 else "mysql")
        return f"{scheme}://{user}:{password}@{host}:{port}/{dbname}"

    # Fallback
    return _rand_hex(24)


# ---------------------------------------------------------------------------
# Honey-token data model
# ---------------------------------------------------------------------------


@dataclass
class HoneyToken:
    """A single decoy credential entry."""

    name: str
    kind: TokenKind
    value: str
    canary_url: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: Optional[float] = None
    planted_env: Optional[str] = None  # env var name if planted
    planted_file: Optional[str] = None  # file path if planted

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "value": self.value,
            "canary_url": self.canary_url,
            "created_at": self.created_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "planted_env": self.planted_env,
            "planted_file": self.planted_file,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HoneyToken":
        return cls(
            name=d["name"],
            kind=TokenKind(d["kind"]),
            value=d["value"],
            canary_url=d.get("canary_url"),
            created_at=d.get("created_at", time.time()),
            access_count=d.get("access_count", 0),
            last_accessed=d.get("last_accessed"),
            planted_env=d.get("planted_env"),
            planted_file=d.get("planted_file"),
        )


# ---------------------------------------------------------------------------
# Default path helpers
# ---------------------------------------------------------------------------


def default_honey_token_path() -> Path:
    """Return the default honey-token store path."""
    from hermes_katana._paths import home_or_fallback

    return home_or_fallback() / ".config" / "hermes-katana" / _DEFAULT_STORE_FILE


def _default_honey_dir() -> Path:
    """Return the default directory for honey files."""
    from hermes_katana._paths import home_or_fallback

    return home_or_fallback() / ".config" / "hermes-katana" / _DEFAULT_HONEY_DIR


# ---------------------------------------------------------------------------
# Audit helper (graceful degradation if audit not available)
# ---------------------------------------------------------------------------


def _audit_honey_access(name: str, kind: str, detail: str, audit_enabled: bool) -> None:
    """Write a honey-token access event to the audit trail."""
    if not audit_enabled:
        return
    try:
        from hermes_katana.audit.trail import AuditEntry, AuditEventType, AuditTrail

        trail = AuditTrail()
        # Use SECRET_BLOCKED as closest event type; detail carries honey-token context.
        entry = AuditEntry(
            event_type=AuditEventType.SECRET_BLOCKED,
            tool_name="honey_token_vault",
            args_hash=hashlib.sha256(f"{name}:{kind}".encode()).hexdigest()[:16],
            decision="honey_token_accessed",
            details=detail,
        )
        trail.log(entry)
    except Exception:  # noqa: BLE001
        logger.warning("Failed to write decoy audit entry", exc_info=True)


# ---------------------------------------------------------------------------
# Canary ping (fire-and-forget)
# ---------------------------------------------------------------------------


def _canary_ping(url: str, token_name: str) -> None:
    """Send a canary HTTP GET in a daemon thread; never blocks caller."""

    def _ping() -> None:
        try:
            full_url = f"{url.rstrip('/')}?token={urllib.parse.quote(token_name)}"
            req = urllib.request.Request(full_url, headers={"User-Agent": "HermesKatana-Canary/1.0"})
            with urllib.request.urlopen(req, timeout=_CANARY_TIMEOUT):
                pass
            logger.info("Canary ping sent")
        except Exception:  # noqa: BLE001
            logger.debug("Canary ping failed", exc_info=True)

    t = threading.Thread(target=_ping, daemon=True, name=f"canary-{token_name}")
    t.start()


# ---------------------------------------------------------------------------
# Main vault class
# ---------------------------------------------------------------------------


class HoneyTokenVault:
    """Manages a collection of honey tokens (decoy credentials).

    Parameters
    ----------
    path :
        JSON store file.  Defaults to ``~/.config/hermes-katana/honey_tokens.json``.
    alert_callback :
        Optional callable invoked on every honey-token access.  Receives
        ``(token: HoneyToken, detail: str)``.
    audit_enabled :
        Whether to write to the audit trail on access.  Defaults to True.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        alert_callback: Optional[Callable[[HoneyToken, str], None]] = None,
        audit_enabled: bool = True,
    ) -> None:
        self._path = Path(path) if path else default_honey_token_path()
        self._alert_callback = alert_callback
        self._audit_enabled = audit_enabled
        self._lock = threading.RLock()
        self._tokens: dict[str, HoneyToken] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load tokens from JSON store (create empty if missing)."""
        if not self._path.exists():
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            self._tokens = {name: HoneyToken.from_dict(d) for name, d in data.items()}
        except Exception:  # noqa: BLE001
            logger.warning("Could not load decoy store", exc_info=True)
            self._tokens = {}

    def _save(self) -> None:
        """Atomically persist tokens to JSON store."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: tok.to_dict() for name, tok in self._tokens.items()}
        payload = json.dumps(data, indent=2)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".honey_tokens_", suffix=".tmp")
        try:
            os.write(fd, payload.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        Path(tmp).replace(self._path)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def create(
        self,
        name: str,
        kind: TokenKind = TokenKind.GENERIC_API_KEY,
        canary_url: Optional[str] = None,
        value: Optional[str] = None,
    ) -> HoneyToken:
        """Create and store a new honey token.

        Parameters
        ----------
        name :
            Logical key name (e.g. ``"lure_aws_key"``).
        kind :
            Token flavour; controls the generated value format.
        canary_url :
            If provided, an HTTP GET is fired to this URL when the token is accessed.
        value :
            Supply a custom value instead of auto-generating one.
        """
        with self._lock:
            token = HoneyToken(
                name=name,
                kind=kind,
                value=value if value is not None else _generate_value(kind),
                canary_url=canary_url,
            )
            self._tokens[name] = token
            self._save()
            logger.debug("Created decoy entry")
            return token

    def get(self, name: str) -> str:
        """Return the decoy value and fire an alert (this is a trap!).

        Raises
        ------
        HoneyTokenError
            If no token with *name* exists.
        """
        with self._lock:
            token = self._tokens.get(name)
            if token is None:
                raise HoneyTokenError(f"Honey token {name!r} not found")

            # Record access
            token.access_count += 1
            token.last_accessed = time.time()
            self._save()

        detail = f"HONEY TOKEN ACCESSED: name={name!r} kind={token.kind.value} access_count={token.access_count}"
        logger.warning(detail)
        _audit_honey_access(name, token.kind.value, detail, self._audit_enabled)

        if token.canary_url:
            _canary_ping(token.canary_url, name)

        if self._alert_callback is not None:
            try:
                self._alert_callback(token, detail)
            except Exception:  # noqa: BLE001
                logger.warning("Decoy alert callback raised", exc_info=True)

        return token.value

    def remove(self, name: str) -> None:
        """Delete a honey token by name."""
        with self._lock:
            if name not in self._tokens:
                raise HoneyTokenError(f"Honey token {name!r} not found")
            del self._tokens[name]
            self._save()

    def list_tokens(self) -> list[str]:
        """Return names of all registered honey tokens."""
        with self._lock:
            return list(self._tokens.keys())

    def get_token(self, name: str) -> HoneyToken:
        """Return the HoneyToken metadata without firing an alert.

        For internal inspection only — does NOT increment access_count.
        """
        with self._lock:
            token = self._tokens.get(name)
            if token is None:
                raise HoneyTokenError(f"Honey token {name!r} not found")
            return token

    # ------------------------------------------------------------------
    # Planting helpers
    # ------------------------------------------------------------------

    def plant_env(self, name: str, env_var: Optional[str] = None) -> str:
        """Plant a honey token into the current process environment.

        Parameters
        ----------
        name :
            Honey token to plant.
        env_var :
            Environment variable name.  Defaults to ``name.upper()``.

        Returns
        -------
        str
            The environment variable name used.
        """
        with self._lock:
            token = self._tokens.get(name)
            if token is None:
                raise HoneyTokenError(f"Honey token {name!r} not found")
            var = env_var or name.upper()
            os.environ[var] = token.value
            token.planted_env = var
            self._save()
            logger.debug("Planted decoy env var")
            return var

    def plant_file(
        self,
        name: str,
        file_path: Optional[Path] = None,
        config_key: Optional[str] = None,
    ) -> Path:
        """Write a honey token into a JSON config file.

        The file will look like a real secrets file (``{"<config_key>": "<value>"}``).

        Parameters
        ----------
        name :
            Honey token to plant.
        file_path :
            Destination path.  Defaults to ``<honey_dir>/<name>.json``.
        config_key :
            Key name inside the file.  Defaults to the token name.

        Returns
        -------
        Path
            Path of the written file.
        """
        with self._lock:
            token = self._tokens.get(name)
            if token is None:
                raise HoneyTokenError(f"Honey token {name!r} not found")

            dest = Path(file_path) if file_path else _default_honey_dir() / f"{name}.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            key = config_key or name
            payload = json.dumps({key: token.value, "_type": token.kind.value}, indent=2)
            dest.write_text(payload, encoding="utf-8")

            token.planted_file = str(dest)
            self._save()
            logger.debug("Planted decoy config file")
            return dest

    def unplant_env(self, name: str) -> None:
        """Remove a planted honey token from the process environment."""
        with self._lock:
            token = self._tokens.get(name)
            if token is None:
                raise HoneyTokenError(f"Honey token {name!r} not found")
            if token.planted_env and token.planted_env in os.environ:
                del os.environ[token.planted_env]
            token.planted_env = None
            self._save()

    # ------------------------------------------------------------------
    # Configurable via KatanaConfig
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Any) -> "HoneyTokenVault":
        """Create a HoneyTokenVault from a KatanaConfig instance.

        Uses ``config.audit_enabled`` to gate audit writes.
        Falls back to defaults when attributes are absent.
        """
        audit_enabled = getattr(config, "audit_enabled", True)
        return cls(audit_enabled=audit_enabled)


# ---------------------------------------------------------------------------
# Honey file monitor
# ---------------------------------------------------------------------------


class HoneyFileMonitor:
    """Plants honey files and detects read attempts via stat-time polling.

    Honey files look like real sensitive config files (``credentials.json``,
    ``secrets.yaml``) but carry decoy data.  Polling via :meth:`check` detects
    atime changes indicating someone (or something) read the file.

    Parameters
    ----------
    alert_callback :
        Called with ``(path: Path, detail: str)`` when a honey file is accessed.
    audit_enabled :
        Whether to write to the audit trail on access.
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[Path, str], None]] = None,
        audit_enabled: bool = True,
    ) -> None:
        self._alert_callback = alert_callback
        self._audit_enabled = audit_enabled
        self._lock = threading.RLock()
        # path → last observed atime
        self._files: dict[Path, float] = {}

    # realistic-looking filenames for decoy files
    _HONEY_NAMES: list[str] = [
        "credentials.json",
        ".aws_credentials",
        "secrets.yaml",
        "service_account.json",
        ".env.production",
        "api_keys.txt",
        "tokens.json",
    ]

    def plant(
        self,
        directory: Path,
        filename: Optional[str] = None,
        content: Optional[str] = None,
        kind: TokenKind = TokenKind.GENERIC_API_KEY,
    ) -> Path:
        """Write a honey file and begin monitoring it.

        Parameters
        ----------
        directory :
            Directory in which to create the file.
        filename :
            Filename inside *directory*.  Picks a realistic name if omitted.
        content :
            File content.  Auto-generates a JSON decoy blob if omitted.
        kind :
            TokenKind used for generating decoy content.

        Returns
        -------
        Path
            Absolute path of the created honey file.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        fname = filename or secrets.choice(self._HONEY_NAMES)
        dest = directory / fname

        if content is None:
            content = json.dumps({"service_reference": f"decoy-{uuid.uuid4().hex}"}, indent=2)

        dest.write_bytes(content.encode("utf-8"))

        with self._lock:
            try:
                self._files[dest] = dest.stat().st_atime
            except FileNotFoundError:
                pass

        logger.debug("Planted honey file at %s", dest)
        return dest

    def monitor(self, path: Path) -> None:
        """Start monitoring an existing file for access."""
        path = Path(path)
        with self._lock:
            try:
                self._files[path] = path.stat().st_atime
            except FileNotFoundError:
                logger.warning("Honey file not found for monitoring: %s", path)

    def unmonitor(self, path: Path) -> None:
        """Stop monitoring a file."""
        with self._lock:
            self._files.pop(Path(path), None)

    def monitored_paths(self) -> list[Path]:
        """Return all currently monitored paths."""
        with self._lock:
            return list(self._files.keys())

    def check(self) -> list[Path]:
        """Poll all monitored files; return paths that were accessed since last check.

        Fires alerts for newly accessed files.
        """
        accessed: list[Path] = []
        with self._lock:
            for path, last_atime in list(self._files.items()):
                try:
                    current = path.stat().st_atime
                except FileNotFoundError:
                    continue
                if current > last_atime:
                    self._files[path] = current
                    accessed.append(path)
                    self._alert(path)
        return accessed

    def _alert(self, path: Path) -> None:
        detail = f"HONEY FILE ACCESSED: path={path}"
        logger.warning(detail)
        _audit_honey_access(str(path), "honey_file", detail, self._audit_enabled)
        if self._alert_callback is not None:
            try:
                self._alert_callback(path, detail)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Honey file alert callback raised for %s: %s", path, exc)
