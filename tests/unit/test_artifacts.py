from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hermes_katana.artifacts import (
    ARTIFACT_MANIFEST,
    MINILM_ONNX_REQUIRED_FILES,
    V15_LARGE_REQUIRED_FILES,
    ArtifactNotFoundError,
    ArtifactStatus,
    UnknownArtifactError,
    artifact_spec,
    artifact_specs,
    artifact_status,
    download_artifact,
    minilm_onnx_spec,
    resolve_minilm_onnx,
    v15_large_spec,
)


def _write_artifact(path: Path, files: tuple[str, ...] = MINILM_ONNX_REQUIRED_FILES) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload_files = tuple(name for name in files if name != ARTIFACT_MANIFEST)
    for name in files:
        if name != ARTIFACT_MANIFEST:
            (path / name).write_text("x")
    manifest = {
        "schema_version": 1,
        "files": {name: {"sha256": hashlib.sha256(b"x").hexdigest(), "size": 1} for name in payload_files},
    }
    (path / ARTIFACT_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


def test_artifact_status_reports_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_path))
    status = artifact_status(minilm_onnx_spec())

    assert not status.present
    assert set(status.missing_files) == set(MINILM_ONNX_REQUIRED_FILES)
    assert str(status.path).startswith(str(tmp_path))


def test_resolve_minilm_uses_explicit_local_dir(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "onnx"
    _write_artifact(artifact_dir)
    monkeypatch.setenv("KATANA_MINILM_ONNX_DIR", str(artifact_dir))

    assert resolve_minilm_onnx(download=False) == artifact_dir.resolve()


def test_artifact_status_rejects_manifest_hash_mismatch(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "onnx"
    _write_artifact(artifact_dir)
    manifest = json.loads((artifact_dir / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
    manifest["files"]["model.onnx"]["sha256"] = "0" * 64
    (artifact_dir / ARTIFACT_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("KATANA_MINILM_ONNX_DIR", str(artifact_dir))

    status = artifact_status(minilm_onnx_spec())

    assert not status.present
    assert "model.onnx: sha256 mismatch" in status.errors


def test_registry_lists_small_and_large_models():
    specs = artifact_specs()
    names = {spec.name for spec in specs}

    assert "katana_v15_distill_minilm_onnx" in names
    assert "katana_v15_large" in names
    assert artifact_spec("small").name == "katana_v15_distill_minilm_onnx"
    assert artifact_spec("large").name == "katana_v15_large"


def test_registry_includes_v17_research_models():
    specs = {spec.name: spec for spec in artifact_specs()}
    assert "katana_v17_large" in specs
    assert "katana_v17_minilm" in specs
    assert artifact_spec("v17_large").repo_id == "Carlosian/hermes-katana-17"
    assert artifact_spec("v17_minilm").repo_id == "Carlosian/hermes-katana-90"
    # research models are explicit-download-only (excluded from managed setup --all/full)
    assert specs["katana_v17_large"].managed_setup is False
    assert specs["katana_v17_minilm"].managed_setup is False
    # pinned to a commit revision and integrity-verified via the manifest
    assert specs["katana_v17_minilm"].revision
    assert ARTIFACT_MANIFEST in specs["katana_v17_minilm"].required_files
    # v15 aliases are unchanged
    assert artifact_spec("minilm").name == "katana_v15_distill_minilm_onnx"


def test_unknown_registry_model_fails_closed():
    with pytest.raises(UnknownArtifactError):
        artifact_spec("mystery-model")


def test_large_artifact_uses_its_own_env_overrides(tmp_path, monkeypatch):
    large_dir = tmp_path / "large"
    _write_artifact(large_dir, V15_LARGE_REQUIRED_FILES)
    monkeypatch.setenv("KATANA_V15_LARGE_DIR", str(large_dir))
    monkeypatch.setenv("KATANA_V15_LARGE_HF_REPO_ID", "local/large")
    monkeypatch.setenv("KATANA_V15_LARGE_HF_REVISION", "test-rev")

    spec = v15_large_spec()
    status = artifact_status(spec)

    assert spec.repo_id == "local/large"
    assert spec.revision == "test-rev"
    assert status.present
    assert status.path == large_dir.resolve()
    assert status.source == "KATANA_V15_LARGE_DIR"


def test_download_artifact_accepts_model_alias(monkeypatch, tmp_path):
    calls = []

    def fake_download(spec, target, force):
        calls.append((spec.name, target, force))
        _write_artifact(target, spec.required_files)
        return target

    monkeypatch.setattr("hermes_katana.artifacts._download_with_huggingface_hub", fake_download)

    status = download_artifact("large", tmp_path / "large-cache", force=True)

    assert isinstance(status, ArtifactStatus)
    assert status.present
    assert calls == [("katana_v15_large", tmp_path / "large-cache", True)]


def test_resolve_minilm_does_not_download_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("KATANA_ARTIFACT_AUTO_DOWNLOAD", raising=False)

    with pytest.raises(ArtifactNotFoundError):
        resolve_minilm_onnx(download=False)
