"""Runtime artifact manifest verification for hermetic deployments."""

from __future__ import annotations

import json
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from hermes_katana.installer.compat_snapshots import (
    compute_file_sha256,
    compute_tree_sha256,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MANIFEST = _REPO_ROOT / "training" / "runtime_artifact_manifest.json"


def runtime_artifact_manifest_path() -> Path:
    """Return the canonical runtime artifact manifest path."""
    return _DEFAULT_MANIFEST


def _safe_manifest_path(rel_path: str) -> bool:
    if not rel_path or rel_path.startswith("/") or "\\" in rel_path or "\x00" in rel_path:
        return False
    if len(rel_path) >= 2 and rel_path[1] == ":":
        return False
    return ".." not in PurePosixPath(rel_path).parts


def _resolve_manifest_artifact_path(repo_root: Path, rel_path: str) -> Path | None:
    if not _safe_manifest_path(rel_path):
        return None
    try:
        target = (repo_root / rel_path).resolve()
        target.relative_to(repo_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return target


def verify_runtime_artifact_manifest(
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the runtime artifact manifest against the local checkout."""
    repo_root = _REPO_ROOT.resolve()
    manifest_file = Path(manifest_path or _DEFAULT_MANIFEST).expanduser().resolve()
    if not manifest_file.exists():
        return {
            "ready": False,
            "manifest_path": str(manifest_file),
            "verified": 0,
            "total": 0,
            "missing": [f"runtime artifact manifest missing at {manifest_file}"],
            "mismatched": [],
            "empty": [],
            "errors": [],
        }

    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ready": False,
            "manifest_path": str(manifest_file),
            "verified": 0,
            "total": 0,
            "missing": [],
            "mismatched": [],
            "empty": [],
            "errors": [f"failed to parse runtime artifact manifest: {exc}"],
        }

    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {
            "ready": False,
            "manifest_path": str(manifest_file),
            "verified": 0,
            "total": 0,
            "missing": [],
            "mismatched": [],
            "empty": [],
            "errors": ["runtime artifact manifest is missing an artifacts mapping"],
        }

    missing: list[str] = []
    mismatched: list[str] = []
    empty: list[str] = []
    errors: list[str] = []
    verified = 0

    for name, raw_entry in artifacts.items():
        if not isinstance(raw_entry, dict):
            errors.append(f"{name}: manifest entry must be an object")
            continue

        rel_path = raw_entry.get("path")
        kind = raw_entry.get("kind")
        expected_sha = raw_entry.get("sha256")
        require_entries = bool(raw_entry.get("require_entries", False))
        if not isinstance(rel_path, str) or not isinstance(kind, str) or not isinstance(expected_sha, str):
            errors.append(f"{name}: manifest entry is missing path/kind/sha256")
            continue

        target = _resolve_manifest_artifact_path(repo_root, rel_path)
        if target is None:
            errors.append(f"{name}: unsafe artifact path {rel_path!r}")
            continue

        if not target.exists():
            missing.append(f"{name}: missing {rel_path}")
            continue

        try:
            if kind == "file":
                if not target.is_file():
                    errors.append(f"{name}: expected file at {rel_path}")
                    continue
                actual_sha = compute_file_sha256(target)
            elif kind == "directory":
                if not target.is_dir():
                    errors.append(f"{name}: expected directory at {rel_path}")
                    continue
                if require_entries and not any(target.iterdir()):
                    empty.append(f"{name}: directory is empty at {rel_path}")
                    continue
                actual_sha = compute_tree_sha256(target)
            else:
                errors.append(f"{name}: unsupported kind {kind!r}")
                continue
        except OSError as exc:
            errors.append(f"{name}: failed to hash {rel_path}: {exc}")
            continue

        if actual_sha != expected_sha:
            mismatched.append(f"{name}: checksum mismatch for {rel_path} (expected {expected_sha}, got {actual_sha})")
            continue

        verified += 1

    return {
        "ready": not missing and not mismatched and not empty and not errors,
        "manifest_path": str(manifest_file),
        "verified": verified,
        "total": len(artifacts),
        "missing": missing,
        "mismatched": mismatched,
        "empty": empty,
        "errors": errors,
    }
