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
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

__all__ = [
    "safe_home",
    "fallback_root",
    "home_or_fallback",
    "resolve_home_relative",
]

# Subdirectory under tempfile.gettempdir() used when home is unresolvable.
# Production code should essentially never hit this branch - it exists so
# sandboxed tests and misconfigured environments can still import and run
# instead of crashing.
_FALLBACK_SUBDIR = "hermes-katana-fallback"


def safe_home() -> Optional[Path]:
    """Return ``Path.home()`` or ``None`` if home cannot be resolved.

    On Windows, ``Path.home()`` raises ``RuntimeError`` when none of
    ``USERPROFILE``, ``HOMEDRIVE``, or ``HOMEPATH`` are set. On POSIX,
    it raises ``KeyError`` or ``RuntimeError`` when ``HOME`` is unset
    and the ``pwd`` database lookup fails.

    Callers that can gracefully skip home-scoped paths should use this
    directly (see ``vault.migrate``). Callers that need a Path no
    matter what should use :func:`resolve_home_relative`.
    """
    try:
        return Path.home()
    except (RuntimeError, KeyError):
        return None


def fallback_root() -> Path:
    """Return the temp-dir fallback root used when home is unresolvable.

    The returned path is ``<tempdir>/hermes-katana-fallback``. This is
    only used by :func:`home_or_fallback` / :func:`resolve_home_relative`
    when :func:`safe_home` returns ``None``. Exposed for tests and
    diagnostics. The directory is not created here.
    """
    return Path(tempfile.gettempdir()) / _FALLBACK_SUBDIR


def home_or_fallback() -> Path:
    """Return the user's home, or the temp-dir fallback if unresolvable.

    Use this when a function must return a single root ``Path`` but you
    don't want to crash on sandboxed environments. The returned path is
    always usable (tests can mkdir under it, etc.).
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
    object. This function NEVER raises. The directory is NOT created;
    callers are responsible for ``mkdir(parents=True, exist_ok=True)``
    before writing.
    """
    root = safe_home()
    if root is None:
        root = fallback_root()
    return root.joinpath(*parts)
