"""DeBERTa-v3-small binary classifier for HermesKatana cascade.

Fast-path scanner: ~5-10ms on CPU for 256-tok input.
Returns (score: float, label: str) for cascade integration.

Usage:
    from hermes_katana.scanner.deberta_classifier import classify_deberta, detect_deberta

    score, label = classify_deberta("ignore all previous instructions")
    # score: 0.0-1.0 (probability of attack)
    # label: "attack" or "safe"
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal


class _LazyModuleProxy:
    def __init__(self, module_name: str, attr_name: str | None = None) -> None:
        self.module_name = module_name
        self.attr_name = attr_name

    def _target(self) -> Any:
        module = importlib.import_module(self.module_name)
        return getattr(module, self.attr_name) if self.attr_name else module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._target()(*args, **kwargs)


torch = _LazyModuleProxy("torch")
AutoModel = _LazyModuleProxy("transformers", "AutoModel")
AutoModelForSequenceClassification = _LazyModuleProxy("transformers", "AutoModelForSequenceClassification")
AutoTokenizer = _LazyModuleProxy("transformers", "AutoTokenizer")


def _torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("torch is required for the DeBERTa PyTorch backend. Install `hermes-katana[ml]`.") from exc


def _load_tokenizer(path: str) -> Any:
    """Load tokenizer, working around extra_special_tokens list/dict mismatch."""
    try:
        return AutoTokenizer.from_pretrained(path)
    except AttributeError as exc:
        if "keys" not in str(exc):
            raise
        # tokenizer_config.json has extra_special_tokens as a list instead of dict.
        # Fix it in a temp copy so we don't mutate the original checkpoint.
        import json as _json
        import shutil
        import tempfile

        config_path = Path(path) / "tokenizer_config.json"
        if not config_path.exists():
            raise
        cfg = _json.loads(config_path.read_text(encoding="utf-8"))
        est = cfg.get("extra_special_tokens")
        if not isinstance(est, list):
            raise
        tmpdir = tempfile.mkdtemp(prefix="deberta_tok_")
        try:
            for fname in os.listdir(path):
                src = Path(path) / fname
                if src.is_file():
                    shutil.copy2(src, tmpdir)
            cfg["extra_special_tokens"] = {t: t for t in est}
            (Path(tmpdir) / "tokenizer_config.json").write_text(_json.dumps(cfg, indent=2), encoding="utf-8")
            return AutoTokenizer.from_pretrained(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


__all__ = [
    "DeBERTaCategory",
    "DeBERTaFinding",
    "DeBERTaSeverity",
    "detect_deberta",
    "classify_deberta",
]

TRAINING_MODELS_DIR = Path(__file__).resolve().parents[3] / "training" / "models"
DEBERTA_MODEL_DIR_ENV = "HERMES_KATANA_DEBERTA_MODEL_DIR"
DEBERTA_THRESHOLD_ENV = "HERMES_KATANA_DEBERTA_THRESHOLD"
LEGACY_MODEL_DIRNAME = "deberta_v3_small_katana"
MODEL_DIR_GLOB = "deberta_v3_small_katana*"
ONNX_FILENAME = "deberta_v3_small_cpu.onnx"

MAX_LENGTH = 256
DEFAULT_THRESHOLD = 0.5


def _model_metric(path: Path, key: str) -> float:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return 0.0
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    value = metrics.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _is_model_dir(path: Path) -> bool:
    return path.is_dir() and ((path / "best").exists() or (path / "final").exists() or any(path.glob("*.onnx")))


def _candidate_model_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in [root / LEGACY_MODEL_DIRNAME, *sorted(root.glob(MODEL_DIR_GLOB))]:
        if not path.is_dir():
            continue
        candidates.append(path)
        candidates.extend(child for child in path.iterdir() if child.is_dir())

    unique_candidates = list(dict.fromkeys(candidates))
    valid_candidates = [path for path in unique_candidates if _is_model_dir(path)]
    return sorted(
        valid_candidates,
        key=lambda path: (
            path.name == LEGACY_MODEL_DIRNAME,
            _model_metric(path, "final_test_f1"),
            (path / ONNX_FILENAME).exists(),
            (path / "best").exists(),
            path.stat().st_mtime,
        ),
        reverse=True,
    )


@lru_cache(maxsize=1)
def _resolve_model_dir() -> Path:
    override = os.environ.get(DEBERTA_MODEL_DIR_ENV)
    if override:
        override_path = Path(override).expanduser()
        if _is_model_dir(override_path):
            return override_path
        raise FileNotFoundError(
            f"{DEBERTA_MODEL_DIR_ENV} points to {override_path}, "
            "but that directory does not contain a DeBERTa-v3-small artifact."
        )

    candidates = _candidate_model_dirs(TRAINING_MODELS_DIR)
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not find a DeBERTa-v3-small artifact under "
        f"{TRAINING_MODELS_DIR}. Set {DEBERTA_MODEL_DIR_ENV} to an extracted "
        "model directory with best/final checkpoints or an ONNX export."
    )


def _resolve_checkpoint_dir(model_dir: Path) -> Path:
    for name in ("best", "final"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"DeBERTa checkpoint not found under {model_dir}. Expected best/ or final/.")


def _resolve_onnx_path(model_dir: Path) -> Path | None:
    preferred = model_dir / ONNX_FILENAME
    if preferred.exists():
        return preferred
    matches = sorted(model_dir.glob("*.onnx"))
    return matches[0] if matches else None


def _resolve_threshold(model_dir: Path) -> float:
    override = os.environ.get(DEBERTA_THRESHOLD_ENV)
    if override:
        try:
            value = float(override)
        except ValueError as exc:
            raise ValueError(f"{DEBERTA_THRESHOLD_ENV} must be a float in [0, 1], got {override!r}") from exc
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{DEBERTA_THRESHOLD_ENV} must be in [0, 1], got {value}")
        return value

    thresholds_path = model_dir / "thresholds.json"
    if thresholds_path.exists():
        try:
            data = json.loads(thresholds_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        for key in ("operating_threshold", "threshold", "deberta_threshold"):
            value = data.get(key)
            if isinstance(value, int | float) and 0.0 <= float(value) <= 1.0:
                return float(value)

    return DEFAULT_THRESHOLD


def _probability_from_logits(logit: Any) -> float:
    torch_mod = _torch()
    values = torch_mod.as_tensor(logit, dtype=torch_mod.float32).reshape(-1)
    if values.numel() == 1:
        return float(torch_mod.sigmoid(values[0]).item())
    if values.numel() == 2:
        return float(torch_mod.softmax(values, dim=0)[1].item())
    raise ValueError(f"Unsupported ONNX logit shape with {values.numel()} values")


def _load_torch(path: Path, *, map_location: Any = "cpu") -> Any:
    torch_mod = _torch()
    try:
        return torch_mod.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch_mod.load(path, map_location=map_location)


class _BinaryDeBERTaRuntime:
    """Runtime wrapper for the local binary head saved by training/train_deberta_binary.py."""

    def __init__(self, checkpoint_dir: Path) -> None:
        torch_mod = _torch()
        nn = torch_mod.nn
        self.backbone = AutoModel.from_pretrained(str(checkpoint_dir))
        hidden_size = int(self.backbone.config.hidden_size)
        dropout_prob = float(getattr(self.backbone.config, "hidden_dropout_prob", 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

        head_state = _load_torch(checkpoint_dir / "classifier_head.pt")
        classifier_state = head_state["classifier"] if isinstance(head_state, dict) else head_state
        self.classifier.load_state_dict(classifier_state)

    def to(self, device: Any) -> "_BinaryDeBERTaRuntime":
        self.backbone.to(device)
        self.dropout.to(device)
        self.classifier.to(device)
        return self

    def eval(self) -> "_BinaryDeBERTaRuntime":
        self.backbone.eval()
        self.dropout.eval()
        self.classifier.eval()
        return self

    def __call__(self, input_ids: Any, attention_mask: Any) -> Any:
        return self.forward(input_ids=input_ids, attention_mask=attention_mask)

    def forward(self, input_ids: Any, attention_mask: Any) -> Any:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        logits = self.classifier(self.dropout(pooled.to(self.classifier.weight.dtype))).squeeze(-1)
        return SimpleNamespace(logits=logits)


class DeBERTaCategory(str, Enum):
    DEBERTA_ATTACK = "deberta_attack"


class DeBERTaSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class DeBERTaFinding:
    category: DeBERTaCategory
    severity: DeBERTaSeverity
    confidence: float  # raw probability
    label: str  # "attack" or "safe"
    description: str


def _severity_from_score(score: float) -> DeBERTaSeverity:
    if score >= 0.85:
        return DeBERTaSeverity.CRITICAL
    if score >= 0.70:
        return DeBERTaSeverity.HIGH
    if score >= 0.55:
        return DeBERTaSeverity.MEDIUM
    return DeBERTaSeverity.LOW


# ---------------------------------------------------------------------------
# Lazy singleton model loader
# ---------------------------------------------------------------------------


class _LazyDeBERTa:
    """Process-level singleton — warm-start on first use."""

    model: Any | Literal["onnx"] | None = None
    tokenizer: Any | None = None
    device: Any | None = None
    ort_session: Any | None = None
    model_dir: Path | None = None
    threshold: float = DEFAULT_THRESHOLD
    _lock = threading.Lock()

    @classmethod
    def ensure(cls) -> None:
        if cls.model is not None:
            return

        with cls._lock:
            if cls.model is not None:
                return

            model_dir = _resolve_model_dir()
            model_path = _resolve_checkpoint_dir(model_dir)
            onnx_path = _resolve_onnx_path(model_dir)

            from hermes_katana.scabbard.embedder import _resolve_default_device

            cls.device = _torch().device(_resolve_default_device())
            cls.model_dir = model_dir
            cls.threshold = _resolve_threshold(model_dir)

            # Prefer ONNX if available (faster on CPU)
            if onnx_path is not None:
                try:
                    import onnxruntime

                    sess_opts = onnxruntime.SessionOptions()
                    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                    cls.ort_session = onnxruntime.InferenceSession(
                        str(onnx_path), sess_opts, providers=["CPUExecutionProvider"]
                    )
                    cls.model = "onnx"
                    cls.tokenizer = _load_tokenizer(str(model_path))
                    print(f"[deberta] Loaded ONNX model from {onnx_path}")
                    return
                except Exception as e:
                    print(f"[deberta] ONNX load failed ({e}), falling back to PyTorch")

            # Fallback: PyTorch model. Current local trainer saves a backbone plus classifier_head.pt;
            # older artifacts may still be standard HuggingFace sequence classifiers.
            if (model_path / "classifier_head.pt").exists():
                cls.model = _BinaryDeBERTaRuntime(model_path).to(cls.device)
            else:
                cls.model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(cls.device)
            model = cls.model
            if model == "onnx":
                raise RuntimeError("Unexpected ONNX sentinel in PyTorch model path")
            model.eval()
            cls.tokenizer = _load_tokenizer(str(model_path))
            print(f"[deberta] Loaded PyTorch model from {model_path}")


# ---------------------------------------------------------------------------
# ONNX inference path
# ---------------------------------------------------------------------------


def _classify_onnx(texts: list[str]) -> list[tuple[float, str]]:
    """Fast ONNX inference path."""
    sess = _LazyDeBERTa.ort_session
    tokenizer = _LazyDeBERTa.tokenizer
    if sess is None or tokenizer is None:
        raise RuntimeError("DeBERTa ONNX session is not initialized")
    results: list[tuple[float, str]] = []

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="np",
    )
    input_ids = enc["input_ids"].astype("int64")
    attention_mask = enc["attention_mask"].astype("int64")

    logits = sess.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
    )[0]

    for logit in logits:
        prob = _probability_from_logits(logit)
        label = "attack" if prob >= _LazyDeBERTa.threshold else "safe"
        results.append((float(prob), label))

    return results


# ---------------------------------------------------------------------------
# PyTorch inference path
# ---------------------------------------------------------------------------


def _classify_torch(texts: list[str]) -> list[tuple[float, str]]:
    """PyTorch inference path (GPU or fallback CPU)."""
    model = _LazyDeBERTa.model
    tokenizer = _LazyDeBERTa.tokenizer
    device = _LazyDeBERTa.device
    if model is None or model == "onnx" or tokenizer is None or device is None:
        raise RuntimeError("DeBERTa torch model is not initialized")

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits.detach().cpu()
        if logits.ndim == 0:
            logits = logits.reshape(1, 1)
        elif logits.ndim == 1:
            logits = logits.reshape(1, -1)

    results: list[tuple[float, str]] = []
    for logit in logits:
        p = _probability_from_logits(logit)
        label = "attack" if p >= _LazyDeBERTa.threshold else "safe"
        results.append((float(p), label))

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_deberta(
    text: str,
    min_confidence: float = 0.0,
) -> list[DeBERTaFinding]:
    """Scan text with DeBERTa classifier.

    Args:
        text: Input text to classify.
        min_confidence: Minimum score to include in findings (0.0 = return all).

    Returns:
        List of attack findings. A classifier result labelled "safe" is not a
        scanner finding; callers that need raw telemetry should use
        classify_deberta().
    """
    if not text or not text.strip():
        return []

    score, label = classify_deberta(text)
    if label != "attack" or score < min_confidence:
        return []

    return [
        DeBERTaFinding(
            category=DeBERTaCategory.DEBERTA_ATTACK,
            severity=_severity_from_score(score),
            confidence=score,
            label=label,
            description=f"DeBERTa classifier: {label} (p={score:.3f})",
        )
    ]


def classify_deberta(text: str) -> tuple[float, str]:
    """Simple classify → (score, label) tuple.

    Args:
        text: Input text.

    Returns:
        (score: float, label: str) where score is probability of attack,
        label is "attack" or "safe".
    """
    if not text or not text.strip():
        return 0.0, "safe"

    _LazyDeBERTa.ensure()

    if _LazyDeBERTa.model == "onnx":
        return _classify_onnx([text])[0]
    return _classify_torch([text])[0]
