"""Shared file locking and atomic write helpers for HermesKatana."""

from __future__ import annotations

import logging
import os
import secrets
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AdvisoryFileLock",
    "atomic_write_text",
    "harden_owner_only",
]


class AdvisoryFileLock:
    """Cross-platform advisory lock backed by a sidecar file."""

    def __init__(self, path: Path, *, suffix: str = ".lock") -> None:
        self._lock_path = path.with_suffix(path.suffix + suffix)
        self._fp: Any = None

    @property
    def path(self) -> Path:
        """Path to the underlying sidecar lock file."""
        return self._lock_path

    def acquire(self) -> None:
        """Acquire an exclusive blocking lock."""
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._lock_path, "a+b")
        if os.name == "nt":
            import msvcrt

            self._fp.seek(0)
            msvcrt.locking(self._fp.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        """Release the lock and close the sidecar file."""
        if self._fp is None:
            return

        try:
            if os.name == "nt":
                import msvcrt

                self._fp.seek(0)
                msvcrt.locking(self._fp.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
        finally:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> "AdvisoryFileLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()


def harden_owner_only(path: Path) -> bool:
    """Best-effort restriction of *path* to the current user only.

    POSIX: ``chmod 0600``. Windows: POSIX modes are a no-op, so strip
    inherited ACEs and grant only the current account via ``icacls``
    (vault/honey/lock files otherwise inherit whatever the parent
    directory allows — audit finding B2, 2026-06-09).

    Returns True when the restriction was applied; failures are logged
    and return False so callers can decide whether to warn loudly.
    """
    try:
        if os.name != "nt":
            os.chmod(path, 0o600)
            return True

        import getpass
        import subprocess

        user = getpass.getuser()
        proc = subprocess.run(  # noqa: S603, S607 — fixed binary, no shell
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:F"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            logger.warning(
                "Could not restrict ACL on %s (icacls exit %d): %s",
                path,
                proc.returncode,
                (proc.stderr or proc.stdout or "").strip(),
            )
            return False
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Could not restrict permissions on %s", path, exc_info=True)
        return False


def atomic_write_text(path: Path, content: str, *, mode: int = 0o600, encoding: str = "utf-8") -> None:
    """Atomically replace *path* with *content* using a securely created temp file.

    When *mode* grants no group/other access, the final file is also
    ACL-restricted to the current user on Windows (where the POSIX *mode*
    bits are otherwise meaningless).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")

    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fp:
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        if os.name != "nt":
            os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
        if os.name == "nt" and (mode & 0o077) == 0:
            harden_owner_only(path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
