"""Secret expiry management for the vault.

Tracks TTL metadata for vault secrets.  Expired secrets return None
on access (with a logged warning), and ``check_expired()`` returns a
list of key names that have passed their expiry time.

Metadata is stored in a separate JSON file alongside the vault so
that expiry concerns don't complicate the encryption layer.

Usage::

    from hermes_katana.vault.expiry import SecretExpiry

    expiry = SecretExpiry()
    expiry.set_expiry("TEMP_TOKEN", ttl_seconds=3600)
    expiry.is_expired("TEMP_TOKEN")   # False (within 1 hour)
    # ... 1 hour later ...
    expiry.check_expired()             # ["TEMP_TOKEN"]
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _default_expiry_path() -> Path:
    """Default path for the expiry metadata file."""
    return Path.home() / ".config" / "hermes-katana" / "vault_expiry.json"


class SecretExpiry:
    """Manages TTL-based expiry for vault secrets.

    Stores expiry timestamps in a JSON file. Thread-safe.

    Args:
        path: Path to the expiry metadata file.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _default_expiry_path()
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def set_expiry(self, key_name: str, ttl_seconds: float) -> None:
        """Set a TTL on a secret.

        Args:
            key_name: The vault key name.
            ttl_seconds: Seconds from now until the secret expires.
        """
        expires_at = time.time() + ttl_seconds
        with self._lock:
            data = self._read()
            data[key_name] = {
                "expires_at": expires_at,
                "set_at": time.time(),
                "ttl_seconds": ttl_seconds,
            }
            self._write(data)
        logger.debug(
            "Set expiry for %s: %.0f seconds (expires %s)",
            key_name,
            ttl_seconds,
            datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat(),
        )

    def get_expiry(self, key_name: str) -> Optional[datetime]:
        """Get the expiry time for a secret.

        Args:
            key_name: The vault key name.

        Returns:
            Expiry datetime (UTC), or None if no expiry is set.
        """
        with self._lock:
            data = self._read()
        entry = data.get(key_name)
        if entry is None:
            return None
        ts = entry.get("expires_at")
        if ts is None:
            return None
        return datetime.fromtimestamp(ts, tz=timezone.utc)

    def is_expired(self, key_name: str) -> bool:
        """Check if a secret has expired.

        Args:
            key_name: The vault key name.

        Returns:
            True if the secret has an expiry AND it has passed.
            False if no expiry is set or it hasn't expired yet.
        """
        with self._lock:
            data = self._read()
        entry = data.get(key_name)
        if entry is None:
            return False
        expires_at = entry.get("expires_at")
        if expires_at is None:
            return False
        return time.time() > expires_at

    def check_expired(self) -> list[str]:
        """Return a list of all expired key names.

        Returns:
            List of key names whose TTL has passed.
        """
        now = time.time()
        with self._lock:
            data = self._read()
        expired = []
        for key_name, entry in data.items():
            expires_at = entry.get("expires_at")
            if expires_at is not None and now > expires_at:
                expired.append(key_name)
        return sorted(expired)

    def extend_expiry(self, key_name: str, additional_seconds: float) -> None:
        """Extend the TTL of an existing secret.

        Args:
            key_name: The vault key name.
            additional_seconds: Seconds to add to the current expiry.

        Raises:
            KeyError: If no expiry is set for this key.
        """
        with self._lock:
            data = self._read()
            entry = data.get(key_name)
            if entry is None:
                raise KeyError(f"No expiry set for key: {key_name}")
            entry["expires_at"] = entry["expires_at"] + additional_seconds
            self._write(data)
        logger.debug("Extended expiry for %s by %.0f seconds", key_name, additional_seconds)

    def remove_expiry(self, key_name: str) -> None:
        """Remove the TTL from a secret (make it permanent).

        Args:
            key_name: The vault key name.
        """
        with self._lock:
            data = self._read()
            if key_name in data:
                del data[key_name]
                self._write(data)
        logger.debug("Removed expiry for %s", key_name)

    def list_expiries(self) -> dict[str, datetime]:
        """List all keys with expiry times.

        Returns:
            Dict mapping key names to their expiry datetimes (UTC).
        """
        with self._lock:
            data = self._read()
        result = {}
        for key_name, entry in data.items():
            ts = entry.get("expires_at")
            if ts is not None:
                result[key_name] = datetime.fromtimestamp(ts, tz=timezone.utc)
        return result

    def _read(self) -> dict:
        """Read the expiry metadata file."""
        if not self._path.exists():
            return {}
        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict) -> None:
        """Atomically write the expiry metadata file.

        Uses temp file + os.replace to prevent partial writes.
        """
        content = json.dumps(data, indent=2, default=str)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=".expiry_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fp:
                    fp.write(content)
                    fp.flush()
                    os.fsync(fp.fileno())
                os.replace(tmp_path, str(self._path))
            except Exception:
                os.unlink(tmp_path)
                raise
        except OSError:
            logger.warning("Failed to write expiry metadata", exc_info=True)

    def clear(self) -> None:
        """Clear all expiry data (for testing)."""
        with self._lock:
            if self._path.exists():
                self._path.unlink()

    def sync_with_vault(self, vault_keys: list[str]) -> list[str]:
        """Remove orphaned expiry entries not present in the vault (GAP 2.8).

        Args:
            vault_keys: List of key names currently in the vault.

        Returns:
            List of orphaned key names that were removed.
        """
        with self._lock:
            data = self._read()
            vault_set = set(vault_keys)
            orphaned = [k for k in data if k not in vault_set]
            if orphaned:
                for k in orphaned:
                    del data[k]
                self._write(data)
                logger.debug("Removed %d orphaned expiry entries: %s", len(orphaned), orphaned)
        return sorted(orphaned)
