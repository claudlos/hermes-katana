"""Portable Proving Ground path resolution.

The public GitHub package bundles Proving Ground code, but not private corpora,
session workspaces, or result databases. Runtime paths therefore default to the
current working directory and can be overridden with environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

_THIS = Path(__file__).resolve()


def pg_root() -> Path:
    """Return the Proving Ground runtime root.

    Override with ``KATANA_PROVING_GROUND_ROOT``.
    """
    env = os.environ.get("KATANA_PROVING_GROUND_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def katana_root() -> Path:
    """Return the HermesKatana checkout/package root.

    Override with ``HERMES_KATANA_ROOT``. Default: current working directory.
    """
    env = os.environ.get("HERMES_KATANA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() or (cwd / "src/hermes_katana").exists():
        return cwd
    return _THIS.parents[3]


def default_corpus_path() -> Path:
    """Return the bundled tiny sample corpus path."""
    return _THIS.parents[3] / "examples" / "proving_ground" / "sample_attacks.jsonl"


def require_dir(path: Path, *, env_var: str, hint: str = "") -> Path:
    """Validate that ``path`` exists; raise an actionable error if not."""
    if not path.exists():
        msg = f"Path not found: {path}\n  Set {env_var} to override, or place the repo at the default location."
        if hint:
            msg += f"\n  Hint: {hint}"
        raise FileNotFoundError(msg)
    return path
