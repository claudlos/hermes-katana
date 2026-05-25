"""
Centralized path resolution helpers for HermesKatana.

Modules that need user-scoped paths (config, vault, audit trail,
access log, expiry) go through this module to resolve the user's
home directory. This gives a single place to handle sandboxed
environments and platform differences.

Background: ``Path.home()`` raises ``RuntimeError`` on Windows when
``USERPROFILE``/``HOMEDRIVE``/``HOMEPATH`` are all unset - which
happens in isolated test environments, restricted service accounts,
and some containers. Without a safe resolver, any module that calls
``Path.home()`` at import time crashes the entire process.

Security note
-------------
When home is unresolvable the helpers route paths through a fallback
under the system tempdir. That is a safety net for sandboxed tests /
misconfigured containers - NOT a production deployment target. If
the fallback fires, HermesKatana emits a one-time ``logger.warning``
so misconfigurations surface instead of silently writing the vault
or audit log to a shared location.

The fallback root is user-scoped (``hermes-katana-fallback-<user>``)
and created with ``0o700`` permissions on POSIX so other accounts on
the machine cannot read secrets or tamper with audit entries.
"""

from __future__ import annotations

import getpass
import logging
import os
import stat
import tempfile
from pathlib import Path
from typing import Optional

__all__ = [
    "safe_home",
    "fallback_root",
    "home_or_fallback",
    "resolve_home_relative",
]

logger = logging.getLogger(__name__)
_PRIVATE_DIR_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR

# Module-level flags so warnings fire at most once per process even
# though safe_home() / fallback_root() are called many times.
_home_warning_emitted = False
_fallback_dir_created = False


def _fallback_user_token() -> str:
    """Return a per-user token used to namespace the fallback dir.

    Tries ``getpass.getuser()`` first (honors USER/LOGNAME env vars and
    falls back to pwd lookup). If that fails (e.g. cleared environ +
    missing pwd entry), falls back to ``uid-<n>`` on POSIX or
    ``nouser`` as a last resort. Never raises.
    """
    try:
        return getpass.getuser()
    except Exception:
        try:
            if hasattr(os, "getuid"):
                return f"uid-{os.getuid()}"
        except Exception:
            pass
        return "nouser"


def safe_home() -> Optional[Path]:
    """Return ``Path.home()`` or ``None`` if home cannot be resolved.

    On Windows, ``Path.home()`` raises ``RuntimeError`` when none of
    ``USERPROFILE``, ``HOMEDRIVE``, or ``HOMEPATH`` are set. On POSIX,
    it raises ``KeyError`` or ``RuntimeError`` when ``HOME`` is unset
    and the ``pwd`` database lookup fails.

    When home is unresolvable, this emits a one-time warning pointing
    operators at the fallback location, then returns ``None``.

    Callers that can gracefully skip home-scoped paths should use this
    directly (see ``vault.migrate``). Callers that need a Path no
    matter what should use :func:`home_or_fallback` or
    :func:`resolve_home_relative`.
    """
    global _home_warning_emitted
    try:
        return Path.home()
    except (RuntimeError, KeyError) as exc:
        if not _home_warning_emitted:
            logger.warning(
                "Path.home() failed (%s); HermesKatana will write user-scoped "
                "files under %s. Set HOME/USERPROFILE explicitly to avoid "
                "this — the tempdir fallback is intended for sandboxed tests, "
                "NOT production deployments.",
                exc,
                fallback_root(),
            )
            _home_warning_emitted = True
        return None


def fallback_root() -> Path:
    """Return the user-scoped tempdir fallback root.

    The returned path is ``<tempdir>/hermes-katana-fallback-<user>``,
    created with ``0o700`` permissions on POSIX so other accounts on
    the host cannot read or tamper with files written underneath.
    Only used by :func:`home_or_fallback` and
    :func:`resolve_home_relative` when :func:`safe_home` returns
    ``None``.

    The directory IS created the first time this function is called
    with missing parents — this is intentional because the fallback
    holds secrets (vault) and audit entries that must land in a
    restricted-permission directory. On Windows permissions rely on
    the tempdir ACL (each user's tempdir is already private).
    """
    global _fallback_dir_created
    root = Path(tempfile.gettempdir()) / f"hermes-katana-fallback-{_fallback_user_token()}"
    if not _fallback_dir_created:
        try:
            root.mkdir(parents=True, exist_ok=True)
            # If the dir already existed with looser perms, tighten them.
            if hasattr(os, "chmod"):
                try:
                    os.chmod(root, _PRIVATE_DIR_MODE)
                except OSError:
                    pass
        except OSError:
            # Don't crash — callers can still build Path objects even if
            # mkdir failed. Writes will fail later with a clearer error.
            pass
        _fallback_dir_created = True
    return root


def home_or_fallback() -> Path:
    """Return the user's home, or the tempdir fallback if unresolvable.

    Use this when a function must return a single root ``Path`` but
    cannot tolerate a crash on sandboxed environments. The returned
    path is always usable (tests can mkdir under it, etc.).
    """
    home = safe_home()
    return home if home is not None else fallback_root()


def resolve_home_relative(*parts: str) -> Path:
    """Join path ``parts`` under the user's home directory.

    Normal case::

        resolve_home_relative(".config", "hermes-katana", "vault.json")
        # -> ~/.config/hermes-katana/vault.json

    When home is unresolvable, the parts are joined under
    :func:`fallback_root` instead, so callers always get a usable Path
    object. This function NEVER raises. The leaf directory is NOT
    created; callers are responsible for ``mkdir(parents=True,
    exist_ok=True)`` on the parent before writing. (The fallback root
    itself IS created with ``0o700``, see :func:`fallback_root`.)
    """
    return home_or_fallback().joinpath(*parts)
