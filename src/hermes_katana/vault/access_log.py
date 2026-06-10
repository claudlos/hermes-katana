"""Vault access audit log.

Records every vault operation (get, set, delete, rotate, list) with
caller information for post-incident forensics.  Stored as append-only
JSONL alongside the vault file.

Thread-safe with file locking.  Rotation happens when the log exceeds
a configurable size threshold.

Usage::

    from hermes_katana.vault.access_log import VaultAccessLog

    log = VaultAccessLog()
    log.log_access("OPENAI_API_KEY", "GET", caller="hermes_plugin:pre_tool_call")
    history = log.get_access_history("OPENAI_API_KEY", limit=10)
"""

from __future__ import annotations

import inspect
import hashlib
import hmac as _hmac_mod
import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "AccessEntry",
    "VaultAccessLog",
]

# Default max log size before rotation (5 MB)
_DEFAULT_MAX_SIZE = 5 * 1024 * 1024

# Sentinel written instead of an HMAC when no integrity key is available.
# Lines carrying it can never pass verify_integrity() — absence of a key
# must not silently produce forgeable "tamper evidence".
_UNKEYED_MARKER = "UNKEYED"


@dataclass(frozen=True, slots=True)
class AccessEntry:
    """A single vault access record.

    Attributes:
        key_name: The secret key accessed (or '*' for list/rotate/lock operations).
        operation: One of GET, SET, DELETE, ROTATE, LIST, LOCK, UNLOCK, VERIFY.
        timestamp: Unix epoch of the access.
        caller: Module:function string identifying who accessed the vault.
        success: Whether the operation succeeded.
        detail: Optional extra context (e.g., error message on failure).
    """

    key_name: str
    operation: str
    timestamp: float = field(default_factory=time.time)
    caller: str = ""
    success: bool = True
    detail: str = ""


def _infer_caller(skip: int = 3) -> str:
    """Walk the stack to find the nearest non-vault caller.

    Returns 'module:function' string, or 'unknown' if detection fails.
    """
    try:
        frame = inspect.currentframe()
        for _ in range(skip):
            if frame is not None:
                frame = frame.f_back
        if frame is None:
            return "unknown"

        module = frame.f_globals.get("__name__", "unknown")
        func = frame.f_code.co_name

        # Skip internal vault frames
        if "vault" in module and func.startswith("_"):
            if frame.f_back is not None:
                frame = frame.f_back
                module = frame.f_globals.get("__name__", "unknown")
                func = frame.f_code.co_name

        return f"{module}:{func}"
    except Exception:
        return "unknown"


def _default_access_log_path() -> Path:
    """Default path for the vault access log."""
    from hermes_katana._paths import home_or_fallback

    return home_or_fallback() / ".config" / "hermes-katana" / "vault_access.jsonl"


def _owner_only_opener(path: str, flags: int) -> int:
    """Open new access-log files with owner-only permissions."""
    return os.open(path, flags, 0o600)


class VaultAccessLog:
    """Append-only access log for vault operations.

    Args:
        path: Path to the JSONL log file.
        max_size: Max file size in bytes before rotation (default 5 MB).
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self._path = path or _default_access_log_path()
        self._max_size = max_size
        self._lock = threading.Lock()
        self._hmac_key: Optional[bytes] = None
        self._warned_unkeyed = False
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Path to the JSONL access log file."""
        return self._path

    def log_access(
        self,
        key_name: str,
        operation: str,
        *,
        caller: str = "",
        success: bool = True,
        detail: str = "",
    ) -> None:
        """Record a vault access event.

        Args:
            key_name: The secret key name (use '*' for bulk ops).
            operation: Operation type (GET, SET, DELETE, ROTATE, etc.).
            caller: Who performed the access (auto-detected if empty).
            success: Whether the operation succeeded.
            detail: Optional extra context.
        """
        if not caller:
            caller = _infer_caller()

        entry = AccessEntry(
            key_name=key_name,
            operation=operation.upper(),
            caller=caller,
            success=success,
            detail=detail,
        )

        with self._lock:
            try:
                self._maybe_rotate()
                created = not self._path.exists()
                with open(self._path, "a", encoding="utf-8", opener=_owner_only_opener) as f:
                    line_data = json.dumps(asdict(entry), default=str)
                    line_hmac = self._compute_line_hmac(line_data)
                    f.write(line_data + "|" + line_hmac + "\n")
                    f.flush()
                self._path.chmod(0o600)
                if created:
                    from hermes_katana._files import harden_owner_only

                    harden_owner_only(self._path)
            except Exception:
                logger.debug("Failed to write vault access log", exc_info=True)

    def get_access_history(
        self,
        key_name: str,
        limit: int = 50,
    ) -> list[AccessEntry]:
        """Get recent access history for a specific key.

        Args:
            key_name: The secret key to look up.
            limit: Maximum number of entries to return.

        Returns:
            List of AccessEntry, most recent first.
        """
        return self._query(key_name=key_name, limit=limit)

    def get_all_access(
        self,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[AccessEntry]:
        """Get all recent access entries.

        Args:
            since: Only return entries after this Unix timestamp.
            limit: Maximum number of entries to return.

        Returns:
            List of AccessEntry, most recent first.
        """
        return self._query(since=since, limit=limit)

    def _query(
        self,
        key_name: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> list[AccessEntry]:
        """Query the access log with optional filters."""
        if not self._path.exists():
            return []

        entries: list[AccessEntry] = []
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        if "|" in line:
                            line_data, _line_hmac = line.rsplit("|", 1)
                        else:
                            line_data = line
                        d = json.loads(line_data)
                        entry = AccessEntry(**d)
                        if key_name and entry.key_name != key_name:
                            continue
                        if since and entry.timestamp < since:
                            continue
                        entries.append(entry)
                    except (json.JSONDecodeError, TypeError):
                        continue
        except OSError:
            return []

        # Most recent first, capped at limit
        entries.reverse()
        return entries[:limit]

    def _maybe_rotate(self) -> None:
        """Rotate the log file if it exceeds max size."""
        try:
            if self._path.exists() and self._path.stat().st_size > self._max_size:
                rotated = self._path.with_suffix(".jsonl.1")
                if rotated.exists():
                    rotated.unlink()
                self._path.rename(rotated)
                logger.debug("Vault access log rotated")
        except OSError:
            pass

    def clear(self) -> None:
        """Clear the access log (for testing)."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def _compute_line_hmac(self, line_data: str) -> str:
        """Compute HMAC-SHA256 for a single log line for tamper evidence.

        Returns the UNKEYED sentinel when no secret key is available — an
        attacker-recomputable digest would be worse than an honest gap.
        """
        hmac_key = self._get_hmac_key()
        if hmac_key is None:
            return _UNKEYED_MARKER
        return _hmac_mod.new(hmac_key, line_data.encode("utf-8"), hashlib.sha256).hexdigest()

    def _get_hmac_key(self) -> Optional[bytes]:
        """Resolve the log-integrity HMAC key — never from public constants.

        Resolution order:

        1. ``HERMES_KATANA_LOG_KEY`` environment variable (explicit override).
        2. A subkey derived from the vault master key in the OS keyring
           (same pattern store._compute_hmac uses for the vault itself).

        Returns None when neither source is available. Entries are then
        written UNKEYED and ``verify_integrity()`` fails closed; the old
        behaviour (key derived from the public log path) let anyone
        recompute valid HMACs after editing the log (audit finding B1).
        """
        if self._hmac_key is not None:
            return self._hmac_key

        env_key = os.environ.get("HERMES_KATANA_LOG_KEY")
        if env_key:
            self._hmac_key = hashlib.sha256(env_key.encode()).digest()
            return self._hmac_key

        master: Optional[bytes] = None
        try:
            from hermes_katana.vault.store import _get_master_key

            master = _get_master_key()
        except Exception:  # noqa: BLE001
            master = None
        if master:
            self._hmac_key = hashlib.sha256(b"hmac:access-log:" + master).digest()
            return self._hmac_key

        if not self._warned_unkeyed:
            self._warned_unkeyed = True
            logger.warning(
                "No integrity key available for the vault access log (no "
                "HERMES_KATANA_LOG_KEY and no vault master key in the keyring). "
                "Entries will be written UNKEYED and verify_integrity() will "
                "report failure until a key is available."
            )
        return None

    def verify_integrity(self) -> bool:
        """Verify HMAC integrity of all log entries.

        Returns True only when every line carries a valid HMAC under the
        current key. UNKEYED lines, lines written under a key that is no
        longer available, and tampered lines all fail (fail closed).
        """
        if not self._path.exists():
            return True
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if "|" not in line:
                        return False
                    line_data, line_hmac = line.rsplit("|", 1)
                    if line_hmac == _UNKEYED_MARKER:
                        return False
                    expected = self._compute_line_hmac(line_data)
                    if expected == _UNKEYED_MARKER:
                        return False
                    if not _hmac_mod.compare_digest(line_hmac, expected):
                        return False
            return True
        except Exception:
            return False
