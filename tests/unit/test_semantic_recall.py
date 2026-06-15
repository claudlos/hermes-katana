from __future__ import annotations

import sys
import types

import pytest

from hermes_katana.scanner import semantic_recall as sr


def _reset_lazy() -> None:
    sr._Lazy.collection = None
    sr._Lazy.model = None
    sr._Lazy.projector = None
    sr._Lazy.embedder = None
    sr._Lazy._zvec = None


def test_semantic_backend_status_falls_back_when_artifacts_missing(tmp_path, monkeypatch):
    contrastive_model = tmp_path / "models" / "contrastive_zvec_v1"
    quantized_model = tmp_path / "models" / "zvec_quantized" / "zvec_quantized"
    semantic_index = tmp_path / "data" / "contrastive_zvec_index"
    original_index = tmp_path / "data" / "translations_clean" / "zvec_semantic_index"
    original_index.mkdir(parents=True)
    (original_index / "stub.idx").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(sr, "CONTRASTIVE_MODEL_DIR", contrastive_model)
    monkeypatch.setattr(sr, "QUANTIZED_MODEL_DIR", quantized_model)
    monkeypatch.setattr(sr, "SEMANTIC_INDEX_DIR", semantic_index)
    monkeypatch.setattr(sr, "ORIGINAL_INDEX_DIR", original_index)

    status = sr.semantic_backend_status()

    assert status["backend"] == "minilm_fallback"
    assert status["contrastive_ready"] is False
    assert status["quantized_ready"] is False
    assert "missing semantic index" in str(status["reason"])
    assert status["active_index_dir"] == str(original_index)


def test_semantic_backend_status_prefers_contrastive_when_all_artifacts_exist(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    contrastive_model = tmp_path / "models" / "contrastive_zvec_v1"
    quantized_model = tmp_path / "models" / "zvec_quantized" / "zvec_quantized"
    semantic_index = tmp_path / "data" / "contrastive_zvec_index"
    original_index = tmp_path / "data" / "translations_clean" / "zvec_semantic_index"

    backbone = contrastive_model / "best" / "backbone"
    backbone.mkdir(parents=True)
    (backbone / "config.json").write_text("{}", encoding="utf-8")
    projector = contrastive_model / "best" / "projector.pt"
    torch.save({"projector": {}, "embed": 128, "temp": 0.07}, projector)

    semantic_index.mkdir(parents=True)
    (semantic_index / "index.bin").write_text("ok", encoding="utf-8")

    original_index.mkdir(parents=True)
    (original_index / "stub.idx").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(sr, "CONTRASTIVE_MODEL_DIR", contrastive_model)
    monkeypatch.setattr(sr, "QUANTIZED_MODEL_DIR", quantized_model)
    monkeypatch.setattr(sr, "SEMANTIC_INDEX_DIR", semantic_index)
    monkeypatch.setattr(sr, "ORIGINAL_INDEX_DIR", original_index)

    status = sr.semantic_backend_status()

    assert status["backend"] == "contrastive"
    assert status["contrastive_ready"] is True
    assert status["model_layout"] == "contrastive_zvec_v1"
    assert status["active_index_dir"] == str(semantic_index)


def test_semantic_backend_status_rejects_malformed_contrastive_projector(tmp_path, monkeypatch):
    contrastive_model = tmp_path / "models" / "contrastive_zvec_v1"
    quantized_model = tmp_path / "models" / "zvec_quantized" / "zvec_quantized"
    semantic_index = tmp_path / "data" / "contrastive_zvec_index"
    original_index = tmp_path / "data" / "translations_clean" / "zvec_semantic_index"

    backbone = contrastive_model / "best" / "backbone"
    backbone.mkdir(parents=True)
    (backbone / "config.json").write_text("{}", encoding="utf-8")
    (contrastive_model / "best" / "projector.pt").write_bytes(b"not a torch checkpoint")

    semantic_index.mkdir(parents=True)
    (semantic_index / "index.bin").write_text("ok", encoding="utf-8")

    original_index.mkdir(parents=True)
    (original_index / "stub.idx").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(sr, "CONTRASTIVE_MODEL_DIR", contrastive_model)
    monkeypatch.setattr(sr, "QUANTIZED_MODEL_DIR", quantized_model)
    monkeypatch.setattr(sr, "SEMANTIC_INDEX_DIR", semantic_index)
    monkeypatch.setattr(sr, "ORIGINAL_INDEX_DIR", original_index)

    status = sr.semantic_backend_status()

    assert status["backend"] == "minilm_fallback"
    assert status["contrastive_ready"] is False
    assert "compatible semantic model" in str(status["reason"])


def test_lazy_contrastive_loader_accepts_deployed_projector_schema(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    contrastive_model = tmp_path / "models" / "contrastive_zvec_v1"
    semantic_index = tmp_path / "data" / "contrastive_zvec_index"

    backbone = contrastive_model / "best" / "backbone"
    backbone.mkdir(parents=True)
    (backbone / "config.json").write_text("{}", encoding="utf-8")
    projector = torch.nn.Sequential(
        torch.nn.Linear(384, 384),
        torch.nn.GELU(),
        torch.nn.Linear(384, 128),
    )
    torch.save(
        {"projector": projector.state_dict(), "embed": 128, "temp": 0.07},
        contrastive_model / "best" / "projector.pt",
    )

    semantic_index.mkdir(parents=True)
    (semantic_index / "index.bin").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(sr, "CONTRASTIVE_MODEL_DIR", contrastive_model)
    monkeypatch.setattr(sr, "SEMANTIC_INDEX_DIR", semantic_index)
    monkeypatch.setattr(sr, "_ZVEC_INDEX_DIR", str(semantic_index))
    monkeypatch.setattr(sr, "_USE_CONTRASTIVE", True)
    monkeypatch.setattr(sr, "_USE_QUANTIZED_ZVEC", False)

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, values):
            return [[0.0] * 384 for _ in values]

    fake_zvec = types.SimpleNamespace(open=lambda *args, **kwargs: object())
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    )
    monkeypatch.setitem(sys.modules, "zvec", fake_zvec)
    _reset_lazy()

    try:
        sr._Lazy.ensure()
        assert sr._Lazy.projector is not None
    finally:
        _reset_lazy()


def test_encode_text_runs_contrastive_projector_in_inference_mode(monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeModel:
        def encode(self, values):
            assert values == ["benign note"]
            return [[0.0] * 384]

    monkeypatch.setattr(sr._Lazy, "ensure", lambda: None)
    setattr(sr._Lazy, "model", FakeModel())
    sr._Lazy.embedder = None
    sr._Lazy.projector = torch.nn.Sequential(
        torch.nn.Linear(384, 384),
        torch.nn.GELU(),
        torch.nn.Linear(384, 128),
    )

    try:
        encoded = sr._encode_text("benign note")
    finally:
        _reset_lazy()

    assert len(encoded) == 128


def test_semantic_backend_status_uses_quantized_layout_when_index_exists(tmp_path, monkeypatch):
    contrastive_model = tmp_path / "models" / "contrastive_zvec_v1"
    quantized_model = tmp_path / "models" / "zvec_quantized" / "zvec_quantized"
    semantic_index = tmp_path / "data" / "contrastive_zvec_index"
    original_index = tmp_path / "data" / "translations_clean" / "zvec_semantic_index"

    backbone = quantized_model / "backbone_fp32"
    tokenizer = quantized_model / "tokenizer"
    backbone.mkdir(parents=True)
    tokenizer.mkdir(parents=True)
    (backbone / "config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
    (quantized_model / "projector_fp32.pt").write_bytes(b"pt")

    semantic_index.mkdir(parents=True)
    (semantic_index / "index.bin").write_text("ok", encoding="utf-8")

    original_index.mkdir(parents=True)
    (original_index / "stub.idx").write_text("ok", encoding="utf-8")

    monkeypatch.setattr(sr, "CONTRASTIVE_MODEL_DIR", contrastive_model)
    monkeypatch.setattr(sr, "QUANTIZED_MODEL_DIR", quantized_model)
    monkeypatch.setattr(sr, "SEMANTIC_INDEX_DIR", semantic_index)
    monkeypatch.setattr(sr, "ORIGINAL_INDEX_DIR", original_index)

    status = sr.semantic_backend_status()

    assert status["backend"] == "zvec_quantized"
    assert status["quantized_ready"] is True
    assert status["model_layout"] == "zvec_quantized"
    assert status["active_index_dir"] == str(semantic_index)
