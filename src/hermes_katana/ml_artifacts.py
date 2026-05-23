"""Helpers for safely loading local ML artifacts.

Pickle, joblib, and ``torch.load(weights_only=False)`` can execute code while
loading. HermesKatana treats those formats as untrusted unless the caller pins a
SHA-256 hash or the operator explicitly opts in with
``HERMES_KATANA_TRUST_ML_ARTIFACTS=1``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Any

__all__ = [
    "UnsafeArtifactError",
    "artifact_sha256",
    "ml_artifacts_trusted",
    "require_trusted_artifact",
    "safe_joblib_load",
    "safe_pickle_load",
    "safe_torch_load",
]

_TRUST_ENV = "HERMES_KATANA_TRUST_ML_ARTIFACTS"
_TRUE_VALUES = {"1", "true", "yes", "on"}


class UnsafeArtifactError(RuntimeError):
    """Raised when an unsafe ML artifact load is not explicitly trusted."""


def ml_artifacts_trusted() -> bool:
    """Return True when unsafe ML artifact loading is operator-enabled."""
    return os.environ.get(_TRUST_ENV, "").strip().lower() in _TRUE_VALUES


def artifact_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of an artifact file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_trusted_artifact(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    allow_env: bool = True,
) -> Path:
    """Validate that an unsafe artifact may be loaded.

    A matching SHA-256 pin is preferred. The environment opt-in exists for
    local/offline deployments that manage artifact provenance externally.
    """
    artifact = Path(path)
    if expected_sha256 is not None:
        actual = artifact_sha256(artifact)
        if not hmac.compare_digest(actual.lower(), expected_sha256.strip().lower()):
            raise UnsafeArtifactError(
                f"ML artifact hash mismatch for {artifact}: expected {expected_sha256}, got {actual}"
            )
        return artifact

    if allow_env and ml_artifacts_trusted():
        return artifact

    raise UnsafeArtifactError(
        f"Refusing to load unsafe ML artifact {artifact}. Set {_TRUST_ENV}=1 or provide an expected SHA-256 hash."
    )


def safe_joblib_load(path: str | Path, *, expected_sha256: str | None = None) -> Any:
    """Load a joblib artifact only after trust/hash validation."""
    artifact = require_trusted_artifact(path, expected_sha256=expected_sha256)
    import joblib

    return joblib.load(str(artifact))


def safe_pickle_load(path: str | Path, *, expected_sha256: str | None = None) -> Any:
    """Load a pickle artifact only after trust/hash validation."""
    artifact = require_trusted_artifact(path, expected_sha256=expected_sha256)
    import pickle  # noqa: S403

    with artifact.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301


def safe_torch_load(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    weights_only: bool = True,
    **kwargs: Any,
) -> Any:
    """Load a torch artifact using safe weights-only mode unless trusted."""
    if not weights_only:
        require_trusted_artifact(path, expected_sha256=expected_sha256)
    elif expected_sha256 is not None:
        require_trusted_artifact(path, expected_sha256=expected_sha256, allow_env=False)

    import torch

    return torch.load(str(path), weights_only=weights_only, **kwargs)
