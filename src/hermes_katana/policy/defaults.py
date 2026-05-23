"""Built-in policy preset loader.

The canonical built-in policy definitions live in the repository-level
``policies/*.yaml`` files.  This module keeps the historical Python exports
available for compatibility while ensuring runtime defaults and documented YAML
presets are loaded from the same source.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]

__all__ = [
    "BUILTIN_POLICY_PRESETS",
    "MAX_POLICIES",
    "BALANCED_POLICIES",
    "PERMISSIVE_POLICIES",
    "BUILTIN_POLICY_SETS",
    "builtin_policy_path",
    "builtin_policy_sets",
    "load_builtin_policy_set",
]

BUILTIN_POLICY_PRESETS = ("max", "balanced", "permissive")


def _repo_policy_dir() -> Path | None:
    """Return the source-tree policy directory when running from checkout."""
    candidate = Path(__file__).resolve().parents[3] / "policies"
    return candidate if candidate.is_dir() else None


def _package_policy_dir() -> Path | None:
    """Return the packaged policy directory when running from an install."""
    candidate = Path(__file__).resolve().parents[1] / "policies"
    return candidate if candidate.is_dir() else None


def _policy_dir() -> Path:
    for candidate in (_repo_policy_dir(), _package_policy_dir()):
        if candidate is not None:
            return candidate
    raise RuntimeError(
        "Built-in policy presets are unavailable; expected policies/*.yaml in "
        "the source tree or hermes_katana/policies in the installed package."
    )


def builtin_policy_path(name: str) -> Path | None:
    """Return the canonical YAML path for a built-in policy preset."""
    if name not in BUILTIN_POLICY_PRESETS:
        return None

    path = _policy_dir() / f"{name}.yaml"
    return path if path.is_file() else None


def load_builtin_policy_set(name: str) -> dict[str, Any]:
    """Load one built-in policy preset from its canonical YAML file."""
    if name not in BUILTIN_POLICY_PRESETS:
        raise KeyError(f"Unknown built-in policy preset: {name}")

    path = builtin_policy_path(name)
    if path is None:
        raise RuntimeError(f"Built-in policy preset '{name}' is missing from {_policy_dir()}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise RuntimeError(f"Built-in policy preset '{name}' must load as a YAML mapping")
    return data


def _load_all_builtin_policy_sets() -> dict[str, dict[str, Any]]:
    return {name: load_builtin_policy_set(name) for name in BUILTIN_POLICY_PRESETS}


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


_BUILTIN_POLICY_SETS: dict[str, dict[str, Any]] = _load_all_builtin_policy_sets()
BUILTIN_POLICY_SETS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {name: _freeze(data) for name, data in _BUILTIN_POLICY_SETS.items()}
)

# Named exports used by tests and downstream integrations.
MAX_POLICIES = BUILTIN_POLICY_SETS["max"]
BALANCED_POLICIES = BUILTIN_POLICY_SETS["balanced"]
PERMISSIVE_POLICIES = BUILTIN_POLICY_SETS["permissive"]


def builtin_policy_sets() -> dict[str, dict[str, Any]]:
    """Return a defensive copy of every built-in policy preset."""
    return deepcopy(_BUILTIN_POLICY_SETS)
