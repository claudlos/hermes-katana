#!/usr/bin/env python3
"""Refresh the pinned "hermes-current" compatibility snapshot from a live checkout.

This is the deterministic half of the Hermes update repair loop. When the
upstream Hermes Agent tree changes, the exact-anchor patch tests need a fresh
copy of the current files they patch. This script copies the small allowlisted
file set into tests/fixtures/hermes_compat/hermes-current-snapshot, rewrites its
MANIFEST.json, and updates the test constant that pins the expected Hermes
commit.

It intentionally does not edit patch anchors in src/hermes_katana/installer/
patches.py. If the drift canary still fails after this refresh, the new
snapshot gives a local, reviewable fixture for repairing those anchors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "tests" / "fixtures" / "hermes_compat" / "hermes-current-snapshot"
DEFAULT_COMPAT_TEST = REPO_ROOT / "tests" / "unit" / "test_compat_snapshots.py"

# Keep this list in sync with tests/unit/test_compat_snapshots.py::_EXPECTED_FILES.
SNAPSHOT_FILES = (
    "model_tools.py",
    "tools/registry.py",
    "tools/terminal_tool.py",
    "hermes_cli/__init__.py",
    "hermes_cli/banner.py",
    "tools/environments/docker.py",
    "gateway/platforms/base.py",
    "gateway/run.py",
    "pyproject.toml",
)

_EXPECTED_COMMIT_RE = re.compile(r'(_EXPECTED_HERMES_COMMIT\s*=\s*)"[0-9a-f]{40}"')


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(source: Path) -> str:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not infer Hermes commit from {source}; pass --commit explicitly") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"Unexpected git commit from {source}: {commit!r}")
    return commit


def _prepare_snapshot_dir(snapshot_dir: Path) -> None:
    """Reset the snapshot directory so removed allowlist entries cannot linger."""
    if snapshot_dir.exists():
        if not snapshot_dir.is_dir():
            raise NotADirectoryError(f"Snapshot path exists but is not a directory: {snapshot_dir}")
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)


def _copy_snapshot_files(source: Path, snapshot_dir: Path) -> dict[str, dict[str, int | str]]:
    _prepare_snapshot_dir(snapshot_dir)
    manifest_files: dict[str, dict[str, int | str]] = {}
    for relative in SNAPSHOT_FILES:
        src = source / relative
        if not src.is_file():
            raise FileNotFoundError(f"Hermes checkout is missing required snapshot file: {relative}")
        dst = snapshot_dir / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_files[relative] = {
            "sha256": _sha256(dst),
            "size": dst.stat().st_size,
        }
    return manifest_files


def _read_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _current_manifest_timestamp(
    existing: dict | None,
    commit: str,
    files: dict[str, dict[str, int | str]],
) -> str | None:
    """Return the existing timestamp when the manifest content is unchanged."""
    if existing is None:
        return None
    if existing.get("hermes_commit") != commit or existing.get("files") != dict(sorted(files.items())):
        return None
    captured_at = existing.get("captured_at")
    return captured_at if isinstance(captured_at, str) and captured_at else None


def _write_manifest(
    snapshot_dir: Path,
    commit: str,
    files: dict[str, dict[str, int | str]],
    *,
    previous_manifest: dict | None = None,
) -> Path:
    manifest_path = snapshot_dir / "MANIFEST.json"
    captured_at = _current_manifest_timestamp(previous_manifest, commit, files)
    if captured_at is None:
        captured_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "hermes_commit": commit,
        "captured_at": captured_at,
        "files": dict(sorted(files.items())),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return manifest_path


def _update_expected_commit(test_path: Path, commit: str) -> bool:
    text = test_path.read_text(encoding="utf-8")
    updated, count = _EXPECTED_COMMIT_RE.subn(rf'\1"{commit}"', text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not update _EXPECTED_HERMES_COMMIT in {test_path}")
    if updated == text:
        return False
    test_path.write_text(updated, encoding="utf-8")
    return True


def refresh_current_snapshot(
    source: Path,
    *,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    compat_test: Path = DEFAULT_COMPAT_TEST,
    commit: str | None = None,
) -> tuple[str, Path]:
    """Refresh hermes-current-snapshot and return (commit, manifest_path)."""
    source = source.resolve()
    snapshot_dir = snapshot_dir.resolve()
    compat_test = compat_test.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Hermes source checkout does not exist: {source}")
    if not compat_test.is_file():
        raise FileNotFoundError(f"Compatibility test file does not exist: {compat_test}")
    resolved_commit = commit or _git_commit(source)
    if not re.fullmatch(r"[0-9a-f]{40}", resolved_commit):
        raise ValueError("Hermes commit must be a 40-character lowercase hex SHA")

    previous_manifest = _read_manifest(snapshot_dir / "MANIFEST.json")
    files = _copy_snapshot_files(source, snapshot_dir)
    manifest_path = _write_manifest(snapshot_dir, resolved_commit, files, previous_manifest=previous_manifest)
    _update_expected_commit(compat_test, resolved_commit)
    return resolved_commit, manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Path to a Hermes Agent checkout")
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--compat-test", type=Path, default=DEFAULT_COMPAT_TEST)
    parser.add_argument("--commit", help="Hermes commit SHA override; defaults to git rev-parse HEAD in --source")
    args = parser.parse_args()

    commit, manifest_path = refresh_current_snapshot(
        args.source,
        snapshot_dir=args.snapshot_dir,
        compat_test=args.compat_test,
        commit=args.commit,
    )
    print(f"Refreshed hermes-current snapshot from Hermes {commit}")
    print(f"Updated manifest: {manifest_path}")
    print(f"Updated expected-commit test: {args.compat_test.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
