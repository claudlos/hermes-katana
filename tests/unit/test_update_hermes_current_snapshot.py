"""Tests for the hermes-current snapshot refresh helper script."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "update_hermes_current_snapshot.py"


def _load_refresh_script():
    spec = importlib.util.spec_from_file_location("update_hermes_current_snapshot", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_hermes_checkout(root: Path, snapshot_files: tuple[str, ...]) -> Path:
    source = root / "hermes-agent"
    source.mkdir(parents=True)
    for relative in snapshot_files:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# fake Hermes file: {relative}\n", encoding="utf-8")
    return source


def _write_compat_test(path: Path, commit: str = "0" * 40) -> None:
    path.write_text(f'_EXPECTED_HERMES_COMMIT = "{commit}"\n', encoding="utf-8")


def test_refresh_current_snapshot_copies_allowlisted_files_and_updates_commit(tmp_dir):
    script = _load_refresh_script()
    source = _write_fake_hermes_checkout(tmp_dir, script.SNAPSHOT_FILES)
    snapshot_dir = tmp_dir / "snapshot"
    compat_test = tmp_dir / "test_compat_snapshots.py"
    _write_compat_test(compat_test)
    commit = "a" * 40

    refreshed_commit, manifest_path = script.refresh_current_snapshot(
        source,
        snapshot_dir=snapshot_dir,
        compat_test=compat_test,
        commit=commit,
    )

    assert refreshed_commit == commit
    assert manifest_path == snapshot_dir / "MANIFEST.json"
    assert f'_EXPECTED_HERMES_COMMIT = "{commit}"' in compat_test.read_text(encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["hermes_commit"] == commit
    assert set(manifest["files"]) == set(script.SNAPSHOT_FILES)
    for relative in script.SNAPSHOT_FILES:
        copied = snapshot_dir / relative
        assert copied.read_text(encoding="utf-8") == f"# fake Hermes file: {relative}\n"
        assert manifest["files"][relative]["sha256"] == _sha256(copied)
        assert manifest["files"][relative]["size"] == copied.stat().st_size


def test_refresh_current_snapshot_removes_stale_snapshot_files(tmp_dir):
    script = _load_refresh_script()
    source = _write_fake_hermes_checkout(tmp_dir, script.SNAPSHOT_FILES)
    snapshot_dir = tmp_dir / "snapshot"
    stale_file = snapshot_dir / "removed.py"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text("stale\n", encoding="utf-8")
    compat_test = tmp_dir / "test_compat_snapshots.py"
    _write_compat_test(compat_test)

    script.refresh_current_snapshot(
        source,
        snapshot_dir=snapshot_dir,
        compat_test=compat_test,
        commit="b" * 40,
    )

    assert not stale_file.exists()


def test_refresh_current_snapshot_preserves_manifest_timestamp_when_unchanged(tmp_dir):
    script = _load_refresh_script()
    source = _write_fake_hermes_checkout(tmp_dir, script.SNAPSHOT_FILES)
    snapshot_dir = tmp_dir / "snapshot"
    compat_test = tmp_dir / "test_compat_snapshots.py"
    _write_compat_test(compat_test)
    commit = "c" * 40

    script.refresh_current_snapshot(
        source,
        snapshot_dir=snapshot_dir,
        compat_test=compat_test,
        commit=commit,
    )
    manifest_path = snapshot_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["captured_at"] = "2026-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    script.refresh_current_snapshot(
        source,
        snapshot_dir=snapshot_dir,
        compat_test=compat_test,
        commit=commit,
    )

    refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert refreshed["captured_at"] == "2026-01-01T00:00:00+00:00"


def test_refresh_current_snapshot_rejects_invalid_commit_override(tmp_dir):
    script = _load_refresh_script()
    source = _write_fake_hermes_checkout(tmp_dir, script.SNAPSHOT_FILES)
    compat_test = tmp_dir / "test_compat_snapshots.py"
    _write_compat_test(compat_test)

    try:
        script.refresh_current_snapshot(
            source,
            snapshot_dir=tmp_dir / "snapshot",
            compat_test=compat_test,
            commit="not-a-sha",
        )
    except ValueError as exc:
        assert "40-character lowercase hex SHA" in str(exc)
    else:  # pragma: no cover - failure branch for assertion readability
        raise AssertionError("invalid commit override was accepted")
