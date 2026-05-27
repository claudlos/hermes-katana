from __future__ import annotations

import json

import pytest

from hermes_katana import runtime_artifacts as runtime_artifacts_mod


def test_verify_runtime_artifact_manifest_accepts_matching_file_and_directory(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    payload_file = repo_root / "artifact.bin"
    payload_file.write_bytes(b"katana")
    payload_dir = repo_root / "tree"
    payload_dir.mkdir()
    (payload_dir / "nested.txt").write_text("secure", encoding="utf-8")

    monkeypatch.setattr(runtime_artifacts_mod, "_REPO_ROOT", repo_root)

    manifest = {
        "version": 1,
        "artifacts": {
            "payload_file": {
                "kind": "file",
                "path": "artifact.bin",
                "sha256": runtime_artifacts_mod.compute_file_sha256(payload_file),
                "require_entries": False,
            },
            "payload_dir": {
                "kind": "directory",
                "path": "tree",
                "sha256": runtime_artifacts_mod.compute_tree_sha256(payload_dir),
                "require_entries": True,
            },
        },
    }
    manifest_path = repo_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = runtime_artifacts_mod.verify_runtime_artifact_manifest(manifest_path)

    assert result["ready"] is True
    assert result["verified"] == 2
    assert result["missing"] == []
    assert result["mismatched"] == []
    assert result["empty"] == []


def test_verify_runtime_artifact_manifest_flags_empty_required_directory(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    empty_dir = repo_root / "empty"
    empty_dir.mkdir()

    monkeypatch.setattr(runtime_artifacts_mod, "_REPO_ROOT", repo_root)

    manifest = {
        "version": 1,
        "artifacts": {
            "empty_dir": {
                "kind": "directory",
                "path": "empty",
                "sha256": runtime_artifacts_mod.compute_tree_sha256(empty_dir),
                "require_entries": True,
            }
        },
    }
    manifest_path = repo_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = runtime_artifacts_mod.verify_runtime_artifact_manifest(manifest_path)

    assert result["ready"] is False
    assert result["verified"] == 0
    assert result["empty"] == ["empty_dir: directory is empty at empty"]


def _write_single_artifact_manifest(repo_root, rel_path: str) -> None:
    manifest = {
        "version": 1,
        "artifacts": {
            "escape": {
                "kind": "file",
                "path": rel_path,
                "sha256": "0" * 64,
            }
        },
    }
    (repo_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_runtime_artifact_manifest_rejects_parent_traversal(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"outside")

    monkeypatch.setattr(runtime_artifacts_mod, "_REPO_ROOT", repo_root)

    manifest = {
        "version": 1,
        "artifacts": {
            "escape": {
                "kind": "file",
                "path": "../outside.bin",
                "sha256": runtime_artifacts_mod.compute_file_sha256(outside_file),
            }
        },
    }
    manifest_path = repo_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = runtime_artifacts_mod.verify_runtime_artifact_manifest(manifest_path)

    assert result["ready"] is False
    assert result["verified"] == 0
    assert result["errors"] == ["escape: unsafe artifact path '../outside.bin'"]


def test_verify_runtime_artifact_manifest_rejects_absolute_backslash_drive_and_nul_paths(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(runtime_artifacts_mod, "_REPO_ROOT", repo_root)

    for rel_path in ("/etc/passwd", r"models\artifact.bin", "C:/Users/example/artifact.bin", "safe\x00name.bin"):
        _write_single_artifact_manifest(repo_root, rel_path)

        result = runtime_artifacts_mod.verify_runtime_artifact_manifest(repo_root / "manifest.json")

        assert result["ready"] is False
        assert result["verified"] == 0
        assert result["errors"] == [f"escape: unsafe artifact path {rel_path!r}"]


def test_verify_runtime_artifact_manifest_rejects_symlink_escape(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"outside")
    symlink_path = repo_root / "artifact-link.bin"
    try:
        symlink_path.symlink_to(outside_file)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    monkeypatch.setattr(runtime_artifacts_mod, "_REPO_ROOT", repo_root)

    manifest = {
        "version": 1,
        "artifacts": {
            "escape": {
                "kind": "file",
                "path": "artifact-link.bin",
                "sha256": runtime_artifacts_mod.compute_file_sha256(outside_file),
            }
        },
    }
    manifest_path = repo_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = runtime_artifacts_mod.verify_runtime_artifact_manifest(manifest_path)

    assert result["ready"] is False
    assert result["verified"] == 0
    assert result["errors"] == ["escape: unsafe artifact path 'artifact-link.bin'"]
