from __future__ import annotations

from pathlib import Path

import pytest

from hermes_katana.artifacts import (
    MINILM_ONNX_REQUIRED_FILES,
    ArtifactNotFoundError,
    artifact_status,
    minilm_onnx_spec,
    resolve_minilm_onnx,
)


def _write_artifact(path: Path) -> None:
    path.mkdir(parents=True)
    for name in MINILM_ONNX_REQUIRED_FILES:
        (path / name).write_text("x")


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


def test_resolve_minilm_does_not_download_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("KATANA_ARTIFACT_DIR", str(tmp_path))
    monkeypatch.delenv("KATANA_ARTIFACT_AUTO_DOWNLOAD", raising=False)

    with pytest.raises(ArtifactNotFoundError):
        resolve_minilm_onnx(download=False)
