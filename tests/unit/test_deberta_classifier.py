from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import hermes_katana.scanner.deberta_classifier as module
import pytest
import torch


@pytest.fixture(autouse=True)
def _clear_resolver_cache():
    module._resolve_model_dir.cache_clear()
    module._LazyDeBERTa.threshold = module.DEFAULT_THRESHOLD
    yield
    module._resolve_model_dir.cache_clear()
    module._LazyDeBERTa.threshold = module.DEFAULT_THRESHOLD


def _make_model_dir(path, *, onnx_name: str | None = module.ONNX_FILENAME) -> None:
    (path / "best").mkdir(parents=True)
    if onnx_name is not None:
        (path / onnx_name).touch()


def test_resolve_model_dir_discovers_timestamped_small_artifact(tmp_path, monkeypatch):
    training_models = tmp_path / "training" / "models"
    model_dir = (
        training_models / "deberta_v3_small_katana_phase1-20260410T041018Z-3-001" / "deberta_v3_small_katana_phase1"
    )
    _make_model_dir(model_dir)

    monkeypatch.setattr(module, "TRAINING_MODELS_DIR", training_models)
    module._resolve_model_dir.cache_clear()

    assert module._resolve_model_dir() == model_dir


def test_resolve_model_dir_honors_env_override(tmp_path, monkeypatch):
    override_dir = tmp_path / "custom-small-model"
    _make_model_dir(override_dir)

    monkeypatch.setenv(module.DEBERTA_MODEL_DIR_ENV, str(override_dir))
    module._resolve_model_dir.cache_clear()

    assert module._resolve_model_dir() == override_dir


def test_resolve_checkpoint_dir_prefers_best(tmp_path):
    model_dir = tmp_path / "deberta_v3_small_katana_phase1"
    (model_dir / "best").mkdir(parents=True)
    (model_dir / "final").mkdir()

    assert module._resolve_checkpoint_dir(model_dir) == model_dir / "best"


def test_resolve_onnx_path_falls_back_to_any_onnx_file(tmp_path):
    model_dir = tmp_path / "deberta_v3_small_katana_phase1"
    model_dir.mkdir()
    fallback_onnx = model_dir / "model.onnx"
    fallback_onnx.touch()

    assert module._resolve_onnx_path(model_dir) == fallback_onnx


def test_probability_from_logits_accepts_scalar_binary_output():
    assert module._probability_from_logits(0.0) == 0.5


def test_probability_from_logits_accepts_two_class_output():
    probability = module._probability_from_logits([0.0, 1.0])
    assert 0.73 < probability < 0.74


def test_detect_deberta_does_not_emit_safe_findings(monkeypatch):
    monkeypatch.setattr(module, "classify_deberta", lambda text: (0.99, "safe"))

    assert module.detect_deberta("clean text") == []


def test_detect_deberta_emits_attack_findings(monkeypatch):
    monkeypatch.setattr(module, "classify_deberta", lambda text: (0.91, "attack"))

    findings = module.detect_deberta("suspicious text")

    assert len(findings) == 1
    assert findings[0].label == "attack"
    assert findings[0].confidence == 0.91


def test_resolve_threshold_reads_artifact_file(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "thresholds.json").write_text('{"operating_threshold": 0.95}', encoding="utf-8")

    assert module._resolve_threshold(model_dir) == 0.95


def test_resolve_threshold_env_override_wins(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "thresholds.json").write_text('{"operating_threshold": 0.95}', encoding="utf-8")
    monkeypatch.setenv(module.DEBERTA_THRESHOLD_ENV, "0.7")

    assert module._resolve_threshold(model_dir) == 0.7


def test_classify_torch_accepts_two_class_logits(monkeypatch):
    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["scan me"]
            return {
                "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }

    class FakeModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]], dtype=torch.float32))

    original_model = module._LazyDeBERTa.model
    original_tokenizer = module._LazyDeBERTa.tokenizer
    original_device = module._LazyDeBERTa.device
    monkeypatch.setattr(module._LazyDeBERTa, "model", FakeModel())
    monkeypatch.setattr(module._LazyDeBERTa, "tokenizer", FakeTokenizer())
    monkeypatch.setattr(module._LazyDeBERTa, "device", torch.device("cpu"))

    try:
        score, label = module._classify_torch(["scan me"])[0]
    finally:
        module._LazyDeBERTa.model = original_model
        module._LazyDeBERTa.tokenizer = original_tokenizer
        module._LazyDeBERTa.device = original_device

    assert 0.88 < score < 0.89
    assert label == "attack"


def test_lazy_deberta_ensure_initializes_once_under_concurrency(monkeypatch, tmp_path):
    model_dir = tmp_path / "deberta_v3_small_katana_phase1"
    _make_model_dir(model_dir, onnx_name=None)

    class FakeLoadedModel:
        def to(self, device):
            return self

        def eval(self):
            return None

    load_counts = {"model": 0, "tokenizer": 0}

    def fake_model_loader(path):
        time.sleep(0.05)
        load_counts["model"] += 1
        return FakeLoadedModel()

    def fake_tokenizer_loader(path):
        time.sleep(0.05)
        load_counts["tokenizer"] += 1
        return object()

    monkeypatch.setattr(module, "_resolve_model_dir", lambda: model_dir)
    monkeypatch.setattr(module, "_resolve_checkpoint_dir", lambda path: model_dir / "best")
    monkeypatch.setattr(module, "_resolve_onnx_path", lambda path: None)
    monkeypatch.setattr(
        module,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=fake_model_loader),
    )
    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=fake_tokenizer_loader),
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)

    original_model = module._LazyDeBERTa.model
    original_tokenizer = module._LazyDeBERTa.tokenizer
    original_device = module._LazyDeBERTa.device
    original_session = module._LazyDeBERTa.ort_session
    original_model_dir = module._LazyDeBERTa.model_dir
    original_threshold = module._LazyDeBERTa.threshold
    module._LazyDeBERTa.model = None
    module._LazyDeBERTa.tokenizer = None
    module._LazyDeBERTa.device = None
    module._LazyDeBERTa.ort_session = None
    module._LazyDeBERTa.model_dir = None
    module._LazyDeBERTa.threshold = module.DEFAULT_THRESHOLD

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: module._LazyDeBERTa.ensure(), range(4)))
    finally:
        module._LazyDeBERTa.model = original_model
        module._LazyDeBERTa.tokenizer = original_tokenizer
        module._LazyDeBERTa.device = original_device
        module._LazyDeBERTa.ort_session = original_session
        module._LazyDeBERTa.model_dir = original_model_dir
        module._LazyDeBERTa.threshold = original_threshold

    assert load_counts["model"] == 1
    assert load_counts["tokenizer"] == 1


def test_lazy_deberta_cold_start_falls_back_to_torch_when_onnx_load_fails(monkeypatch, tmp_path):
    model_dir = tmp_path / "deberta_v3_small_katana_phase1"
    _make_model_dir(model_dir)

    class FakeLoadedModel:
        def to(self, device):
            return self

        def eval(self):
            return None

    load_counts = {"model": 0, "tokenizer": 0}

    def fake_model_loader(path):
        load_counts["model"] += 1
        return FakeLoadedModel()

    def fake_tokenizer_loader(path):
        load_counts["tokenizer"] += 1
        return object()

    class FakeSessionOptions:
        graph_optimization_level = None

    class FakeGraphOptimizationLevel:
        ORT_ENABLE_ALL = "all"

    class FakeOnnxRuntime:
        SessionOptions = FakeSessionOptions
        GraphOptimizationLevel = FakeGraphOptimizationLevel

        @staticmethod
        def InferenceSession(*args, **kwargs):
            raise RuntimeError("onnx cold-start failed")

    monkeypatch.setattr(module, "_resolve_model_dir", lambda: model_dir)
    monkeypatch.setattr(module, "_resolve_checkpoint_dir", lambda path: model_dir / "best")
    monkeypatch.setattr(module, "_resolve_onnx_path", lambda path: model_dir / module.ONNX_FILENAME)
    monkeypatch.setattr(
        module,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=fake_model_loader),
    )
    monkeypatch.setattr(
        module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=fake_tokenizer_loader),
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "onnxruntime", FakeOnnxRuntime)

    original_model = module._LazyDeBERTa.model
    original_tokenizer = module._LazyDeBERTa.tokenizer
    original_device = module._LazyDeBERTa.device
    original_session = module._LazyDeBERTa.ort_session
    original_model_dir = module._LazyDeBERTa.model_dir
    original_threshold = module._LazyDeBERTa.threshold
    module._LazyDeBERTa.model = None
    module._LazyDeBERTa.tokenizer = None
    module._LazyDeBERTa.device = None
    module._LazyDeBERTa.ort_session = None
    module._LazyDeBERTa.model_dir = None
    module._LazyDeBERTa.threshold = module.DEFAULT_THRESHOLD

    observed: dict[str, object] = {}

    try:
        module._LazyDeBERTa.ensure()
        observed["ort_session"] = module._LazyDeBERTa.ort_session
        observed["model"] = module._LazyDeBERTa.model
    finally:
        module._LazyDeBERTa.model = original_model
        module._LazyDeBERTa.tokenizer = original_tokenizer
        module._LazyDeBERTa.device = original_device
        module._LazyDeBERTa.ort_session = original_session
        module._LazyDeBERTa.model_dir = original_model_dir
        module._LazyDeBERTa.threshold = original_threshold
        sys.modules.pop("onnxruntime", None)

    assert load_counts["model"] == 1
    assert load_counts["tokenizer"] == 1
    assert observed["ort_session"] is None
    assert observed["model"] is not None and observed["model"] != "onnx"


def test_lazy_deberta_loads_local_binary_classifier_head(monkeypatch, tmp_path):
    model_dir = tmp_path / "deberta_v3_small_katana_v8"
    _make_model_dir(model_dir, onnx_name=None)
    checkpoint_dir = model_dir / "best"

    head = torch.nn.Linear(3, 1)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
        head.bias.zero_()
    torch.save({"classifier": head.state_dict()}, checkpoint_dir / "classifier_head.pt")

    class FakeBackbone(torch.nn.Module):
        config = SimpleNamespace(hidden_size=3, hidden_dropout_prob=0.0)

        def forward(self, input_ids, attention_mask):
            batch, seq_len = input_ids.shape
            hidden = torch.zeros(batch, seq_len, 3, dtype=torch.float32)
            hidden[..., 0] = 2.0
            return SimpleNamespace(last_hidden_state=hidden)

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            assert texts == ["scan me"]
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }

    load_counts = {"backbone": 0, "tokenizer": 0}

    def fake_backbone_loader(path):
        assert path == str(checkpoint_dir)
        load_counts["backbone"] += 1
        return FakeBackbone()

    def fake_tokenizer_loader(path):
        assert path == str(checkpoint_dir)
        load_counts["tokenizer"] += 1
        return FakeTokenizer()

    monkeypatch.setattr(module, "_resolve_model_dir", lambda: model_dir)
    monkeypatch.setattr(module, "_resolve_checkpoint_dir", lambda path: checkpoint_dir)
    monkeypatch.setattr(module, "_resolve_onnx_path", lambda path: None)
    monkeypatch.setattr(module, "AutoModel", SimpleNamespace(from_pretrained=fake_backbone_loader))
    monkeypatch.setattr(module, "_load_tokenizer", fake_tokenizer_loader)
    monkeypatch.setenv("HERMES_KATANA_DEVICE", "cpu")

    original_model = module._LazyDeBERTa.model
    original_tokenizer = module._LazyDeBERTa.tokenizer
    original_device = module._LazyDeBERTa.device
    original_session = module._LazyDeBERTa.ort_session
    original_model_dir = module._LazyDeBERTa.model_dir
    original_threshold = module._LazyDeBERTa.threshold
    module._LazyDeBERTa.model = None
    module._LazyDeBERTa.tokenizer = None
    module._LazyDeBERTa.device = None
    module._LazyDeBERTa.ort_session = None
    module._LazyDeBERTa.model_dir = None
    module._LazyDeBERTa.threshold = module.DEFAULT_THRESHOLD

    try:
        module._LazyDeBERTa.ensure()
        score, label = module._classify_torch(["scan me"])[0]
    finally:
        module._LazyDeBERTa.model = original_model
        module._LazyDeBERTa.tokenizer = original_tokenizer
        module._LazyDeBERTa.device = original_device
        module._LazyDeBERTa.ort_session = original_session
        module._LazyDeBERTa.model_dir = original_model_dir
        module._LazyDeBERTa.threshold = original_threshold

    assert load_counts == {"backbone": 1, "tokenizer": 1}
    assert 0.88 < score < 0.89
    assert label == "attack"
