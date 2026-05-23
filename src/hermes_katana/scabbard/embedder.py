"""zvec Embedder: ZvecEmbedder backed by trained MiniLM + projector.

Provides 128-dim L2-normalized embeddings from the contrastive zvec model
(backbone: sentence-transformers/all-MiniLM-L6-v2 + learned projector).

Architecture
------------
  MiniLM-L6-v2 (384-dim [CLS]) → Projection MLP (384→128) → L2-norm → 128-dim

The projector was trained with SupConLoss to cluster attacks by semantic type
regardless of surface obfuscation (base64, homoglyphs, zero-width chars, etc.).

Usage
-----
  from hermes_katana.scabbard.embedder import ZvecEmbedder

  embedder = ZvecEmbedder()
  vec = embedder.encode("Ignore previous instructions")
  # vec.shape == (128,), unit norm

  vec_batch = embedder.encode_batch([
      "Ignore previous instructions",
      "What's the weather?",
  ])
  # vec_batch.shape == (2, 128)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_default_device() -> str:
    """Pick a torch device, honoring an env-var override.

    Set ``HERMES_KATANA_DEVICE`` to ``"cpu"`` or ``"cuda"`` to bypass the
    runtime probe. Useful on hosts where the NVIDIA driver/nvml is hung,
    in which case ``torch.cuda.is_available()`` itself stalls.
    """
    override = os.environ.get("HERMES_KATANA_DEVICE")
    if override:
        device = override.strip().lower()
        if device == "cpu" or device == "cuda" or device.startswith("cuda:") or device == "mps":
            return device
        logger.warning(
            "Ignoring invalid HERMES_KATANA_DEVICE=%r; falling back to CPU",
            override,
        )
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


if TYPE_CHECKING:
    from hermes_katana.scabbard.fusion import ClassificationResult

# -----------------------------------------------------------------------
# Default model paths (relative to training/models/)
# -----------------------------------------------------------------------

DEFAULT_ZVEC_BASE = (
    Path(__file__).parent.parent.parent.parent
    / "training"
    / "models"
    / "zvec_quantized-20260408T061203Z-3-001"
    / "zvec_quantized"
)
DEFAULT_BACKBONE = DEFAULT_ZVEC_BASE / "backbone_fp32"
DEFAULT_PROJECTOR = DEFAULT_ZVEC_BASE / "projector_fp32.pt"
DEFAULT_TOKENIZER = DEFAULT_ZVEC_BASE / "tokenizer"


# -----------------------------------------------------------------------
# Lazy torch import (keeps minimal profile importable without torch)
# -----------------------------------------------------------------------


def _torch() -> Any:
    """Lazy import for torch — avoids hard dependency in minimal deployments."""
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError("torch is required for zvec embedding. Install with: pip install torch") from exc


def _transformers() -> Any:
    """Lazy import for transformers."""
    try:
        import transformers

        return transformers
    except ImportError as exc:
        raise ImportError(
            "transformers is required for zvec embedding. Install with: pip install transformers"
        ) from exc


def _np() -> Any:
    """numpy is already a hard dep but kept lazy for symmetry."""
    return np


def _move_module_to_device(module: Any, device: str, *, module_name: str) -> tuple[Any, str]:
    """Move a torch module to *device*, falling back to CPU on accelerator failure.

    CUDA can be available but unusable because of OOM, driver resets, or a
    partially occupied consumer GPU. Classification should degrade to CPU
    instead of failing open/closed unpredictably at model-load time.
    """
    try:
        return module.to(device), device
    except Exception as exc:
        if device == "cpu":
            raise
        try:
            torch = _torch()
            if hasattr(torch, "cuda") and hasattr(torch.cuda, "empty_cache"):
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.warning(
            "Could not move %s to %s; falling back to CPU: %s",
            module_name,
            device,
            exc,
        )
        return module.to("cpu"), "cpu"


# -----------------------------------------------------------------------
# Projection MLP (must match architecture used during training)
# -----------------------------------------------------------------------


class _ProjectorMLP:
    """2-layer GELU projection: 384 → 384 → 128, followed by L2-normalize.

    Matches the trained projector architecture from
    ``train_contrastive.py`` → SupConLoss projection head.
    """

    def __init__(self, h: int = 384, e: int = 128) -> None:
        self.h = h
        self.e = e
        self._w0: Optional[np.ndarray] = None
        self._b0: Optional[np.ndarray] = None
        self._w2: Optional[np.ndarray] = None
        self._b2: Optional[np.ndarray] = None
        self._loaded: bool = False

    def load(self, state_dict: dict[str, Any]) -> None:
        """Load from a PyTorch state_dict (float32)."""
        # Checkpoints may save the Sequential state directly or under a
        # top-level "projector" key alongside metadata such as temperature.
        projector_state = state_dict.get("projector")
        if isinstance(projector_state, dict):
            state_dict = projector_state

        self._w0 = np.asarray(state_dict["net.0.weight"].cpu().numpy(), dtype=np.float32)
        self._b0 = np.asarray(state_dict["net.0.bias"].cpu().numpy(), dtype=np.float32)
        self._w2 = np.asarray(state_dict["net.2.weight"].cpu().numpy(), dtype=np.float32)
        self._b2 = np.asarray(state_dict["net.2.bias"].cpu().numpy(), dtype=np.float32)
        self._loaded = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project 384-dim MiniLM output to 128-dim and L2-normalize.

        Args:
            x: array of shape (batch, 384) or (384,)

        Returns:
            L2-normalized array of shape (batch, 128) or (128,)
        """
        if not self._loaded:
            raise RuntimeError("Projector not loaded. Call .load() first.")
        if self._w0 is None or self._b0 is None or self._w2 is None or self._b2 is None:
            raise RuntimeError("Projector weights are missing. Call .load() first.")

        w0 = self._w0
        b0 = self._b0
        w2 = self._w2
        b2 = self._b2

        x = np.asarray(x, dtype=np.float32)

        # Layer 0: 384 → 384 + GELU
        h = np.dot(x, w0.T) + b0
        # GELU: 0.5 * h * (1 + tanh(sqrt(2/pi) * (h + 0.044715*h^3)))
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        h = 0.5 * h * (1.0 + np.tanh(sqrt_2_over_pi * (h + 0.044715 * h**3)))

        # Layer 2: 384 → 128
        out = np.dot(h, w2.T) + b2

        # L2-normalize rows
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        out = out / norms

        return cast(np.ndarray, np.asarray(out, dtype=np.float32))


# -----------------------------------------------------------------------
# Main embedder
# -----------------------------------------------------------------------


class ZvecEmbedder:
    """128-dim semantic embedder using trained zvec MiniLM model.

    Wraps ``sentence-transformers/all-MiniLM-L6-v2`` + learned projection head
    to produce L2-normalized attack-semantic embeddings.  Designed as a
    drop-in for the ZvecEmbedder interface expected by
    :class:`~hermes_katana.scabbard.feature_extractor.FeatureExtractor`.

    Parameters
    ----------
    model_path :
        Path to ``backbone_fp32/`` (transformers model dir).
        Defaults to the trained zvec backbone.
    projector_path :
        Path to ``projector_fp32.pt`` (trained projection head).
        Defaults to the trained zvec projector.
    tokenizer_path :
        Path to the tokenizer directory.
        Defaults to the zvec tokenizer.
    device :
        ``"cuda"`` to use GPU (if available), ``"cpu"`` otherwise.
        Note: zvec is lightweight (~87MB) and fast on CPU (~5K rec/s).

    Attributes
    ----------
    embedding_dim : int
        Always 128 (output dimension of the trained projector).
    backbone_name : str
        The base model identifier (``"sentence-transformers/all-MiniLM-L6-v2"``).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        projector_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_BACKBONE
        self.projector_path = Path(projector_path) if projector_path else DEFAULT_PROJECTOR
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER
        self.embedding_dim: int = 128

        # Lazy-load torch objects
        self._backbone: Optional[Any] = None
        self._projector: Optional[_ProjectorMLP] = None
        self._tokenizer: Optional[Any] = None

        # Determine device (honors HERMES_KATANA_DEVICE override).
        if device:
            self._device = device
        else:
            self._device = _resolve_default_device()

        logger.info(
            "ZvecEmbedder init: model=%s, projector=%s, device=%s",
            self.model_path,
            self.projector_path,
            self._device,
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_backbone(self) -> Any:
        if self._backbone is None:
            transformers = _transformers()
            model = transformers.AutoModel.from_pretrained(
                str(self.model_path),
            )
            self._backbone, self._device = _move_module_to_device(
                model,
                self._device,
                module_name="ZvecEmbedder backbone",
            )
            self._backbone.eval()
        return self._backbone

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            transformers = _transformers()
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.tokenizer_path),
            )
        return self._tokenizer

    def _ensure_projector(self) -> _ProjectorMLP:
        if self._projector is None:
            _torch()
            projector = _ProjectorMLP(h=384, e=128)
            from hermes_katana.ml_artifacts import safe_torch_load

            state = safe_torch_load(
                self.projector_path,
                map_location="cpu",
                weights_only=True,
            )
            projector.load(state)
            self._projector = projector
        return self._projector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string to a 128-dim embedding vector.

        Args:
            text: Input text string.

        Returns:
            L2-normalized 128-dim numpy array (unit vector).
        """
        backbone = self._ensure_backbone()
        tokenizer = self._ensure_tokenizer()
        projector = self._ensure_projector()

        torch = _torch()
        with torch.no_grad():
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = backbone(**inputs)
            # [CLS] token
            cls_hidden = outputs.last_hidden_state[:, 0, :]
            # Project to 128-dim and L2-normalize
            cls_np = np.asarray(cls_hidden.cpu().numpy(), dtype=np.float32)
            emb = np.asarray(projector.forward(cls_np)[0], dtype=np.float32)

        return cast(np.ndarray, emb)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts to a (N, 128) embedding matrix.

        Args:
            texts: List of N input text strings.

        Returns:
            L2-normalized array of shape (N, 128).
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        backbone = self._ensure_backbone()
        tokenizer = self._ensure_tokenizer()
        projector = self._ensure_projector()

        torch = _torch()
        with torch.no_grad():
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = backbone(**inputs)
            cls_hidden = outputs.last_hidden_state[:, 0, :]
            cls_np = np.asarray(cls_hidden.cpu().numpy(), dtype=np.float32)
            embs = np.asarray(projector.forward(cls_np), dtype=np.float32)

        return cast(np.ndarray, embs)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text embeddings.

        Since embeddings are L2-normalized, this is just dot product.
        """
        a = self.encode(text_a)
        b = self.encode(text_b)
        return float(np.dot(a, b))

    def __repr__(self) -> str:
        return (
            f"ZvecEmbedder(embedding_dim={self.embedding_dim}, backbone={self.model_path.name}, device={self._device})"
        )


# -----------------------------------------------------------------------
# DeBERTa Sequence Classifier (replaces zvec+centroids+XGBoost)
# -----------------------------------------------------------------------


class DeBERTaClassifier:
    """DeBERTa-v2-base sequence classifier wired to Scabbard runtime labels.

    Loads the trained ``deberta-scabbard`` model (8 classes) and maps its
    outputs to scabbard's standard label set via the label_map defined in
    the task spec (LABEL_6 → behavioral_control, LABEL_7 → jailbreak).

    Parameters
    ----------
    model_path :
        Path to the trained DeBERTa-v2-base model directory
        (contains ``config.json``, ``model.safetensors``, ``tokenizer``).
    device :
        ``"cuda"`` to use GPU, ``"cpu"`` otherwise.
    allow_threshold :
        Attack-score below this → ALLOW decision.
    block_threshold :
        Attack-score above this → BLOCK decision; between thresholds → FLAG.

    Attributes
    ----------
    label_map : dict[int, str]
        Maps the legacy model's 8 class indices (0-7) to scabbard label names.

    Example
    -------
        clf = DeBERTaClassifier("/path/to/deberta-scabbard")
        scores = clf.classify("Ignore previous instructions")
        # scores == {"clean": 0.01, "content_injection": 0.05, ...}
    """

    # 8-class to scabbard-label mapping
    _LABEL_MAP: dict[int, str] = {
        0: "clean",
        1: "content_injection",
        2: "semantic_manipulation",
        3: "behavioral_control",
        4: "exfiltration_attempt",
        5: "jailbreak",
        6: "behavioral_control",  # role_override
        7: "jailbreak",  # prompt_extraction
    }

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        allow_threshold: float = 0.3,
        block_threshold: float = 0.85,
    ) -> None:
        self.model_path = Path(model_path)
        self._device = device or _resolve_default_device()
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold

        # Lazy-load model and tokenizer
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

        logger.info(
            "DeBERTaClassifier init: model=%s, device=%s",
            self.model_path,
            self._device,
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                import transformers
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for DeBERTaClassifier. Install with: pip install transformers"
                ) from exc
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
            )
            self._model, self._device = _move_module_to_device(
                model,
                self._device,
                module_name="DeBERTaClassifier model",
            )
            self._model.eval()
        return self._model

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is None:
            import transformers

            self._tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self.model_path),
            )
        return self._tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict[str, float]:
        """Classify text, returning per-label probabilities for runtime labels.

        Returns
        -------
        dict[str, float]
            Mapping from label name to probability (sum not guaranteed to be 1.0
            because duplicate targets take the max probability across their
            source classes).
        """
        tokenizer = self._ensure_tokenizer()
        model = self._ensure_model()
        torch = _torch()

        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(self._device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits[0]  # shape (8,)

        probs = torch.softmax(logits, dim=0)

        # Initialize scores dict with all runtime labels. Labels not produced
        # by the legacy 8-class model remain at 0.0.
        scores: dict[str, float] = {label: 0.0 for label in _ATTACK_LABELS}

        # LABEL_0 → clean
        scores["clean"] = float(probs[0])

        # LABEL_1 … LABEL_7 → mapped targets; take max for duplicates
        for orig_idx in range(1, 8):
            target = self._LABEL_MAP[orig_idx]
            scores[target] = max(scores[target], float(probs[orig_idx]))

        return scores

    def classify_result(self, text: str) -> ClassificationResult:
        """Classify and return a full ClassificationResult (decision + scores).

        Mirrors the FusionClassifier.classify() interface so it can be used as
        a drop-in override signal.
        """
        from hermes_katana.scabbard.fusion import ClassificationResult, Decision

        scores = self.classify(text)

        attack_scores = {k: v for k, v in scores.items() if k != "clean"}
        top_category = max(attack_scores, key=lambda k: attack_scores[k])
        max_attack_score = attack_scores[top_category]

        if max_attack_score < self.allow_threshold:
            decision = Decision.ALLOW
        elif max_attack_score > self.block_threshold:
            decision = Decision.BLOCK
        else:
            decision = Decision.FLAG

        return ClassificationResult(
            scores=scores,
            decision=decision,
            top_category=top_category,
            confidence=max_attack_score,
        )

    def __repr__(self) -> str:
        return f"DeBERTaClassifier(model={self.model_path.name}, device={self._device})"


# -----------------------------------------------------------------------
# katana_v11 (DeBERTa-v3-large, 9-class, origin-aware) classifier
# -----------------------------------------------------------------------


class KatanaV11Classifier:
    """v1.0 classifier wired to the katana_v11 (DeBERTa-v3-large, 9-class) model.

    Differences from the legacy ``DeBERTaClassifier`` (which targets the older
    DeBERTa-v2-base 8-class model):

    * **9-class output** — adds ``encoding_evasion`` and ``persona_jailbreak``
      as distinct labels, matching the v5.1 corpus taxonomy.
    * **Origin-aware** — prepends ``[ORIGIN=<tier>] `` to each text before
      tokenization, matching what the trainer did. The 6 tier tokens were
      registered as added-special-tokens in the saved tokenizer at IDs
      128001-128006, so vocab continuity is preserved.
    * **Batched inference** — :meth:`classify_batch` runs N texts (each with
      its own origin) through one tokenizer call + one forward pass.

    Threshold semantics match :class:`DeBERTaClassifier`: attack-score below
    ``allow_threshold`` -> ALLOW, above ``block_threshold`` -> BLOCK, between
    -> FLAG. Attack score is the max non-clean class probability.

    Parameters
    ----------
    model_path :
        Path to the saved checkpoint dir (must contain ``config.json``,
        ``model.safetensors``, and tokenizer files). Typically
        ``training/checkpoints/katana_v11/best``.
    device :
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
    allow_threshold, block_threshold :
        See class docstring.
    default_origin :
        Used when the caller doesn't supply an origin explicitly. Defaults to
        ``"user_input"`` to match the trainer's fallback for missing-origin rows.
    """

    LABELS: tuple[str, ...] = (
        "clean",
        "content_injection",
        "semantic_manipulation",
        "behavioral_control",
        "exfiltration_attempt",
        "jailbreak",
        "cognitive_state_attack",
        "encoding_evasion",
        "persona_jailbreak",
    )

    ORIGIN_TIERS: tuple[str, ...] = (
        "user_input",
        "retrieved_web",
        "mcp_tool_description",
        "mcp_tool_result",
        "prior_session_memory",
        "delegated_agent_output",
    )

    BACKENDS: tuple[str, ...] = ("torch", "onnx", "onnx_int8")

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        allow_threshold: float = 0.3,
        block_threshold: float = 0.85,
        default_origin: str = "user_input",
        max_length: int = 256,
        backend: str = "torch",
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, got {backend!r}")
        self.backend = backend
        self.model_path = Path(model_path)
        # ONNX runtime always runs on CPU (no GPU EP wired here); torch
        # auto-selects.
        if backend == "torch":
            self._device = device or _resolve_default_device()
        else:
            self._device = "cpu"
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold
        self.default_origin = default_origin if default_origin in self.ORIGIN_TIERS else "user_input"
        self.max_length = max_length
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

        logger.info(
            "KatanaV11Classifier init: model=%s, backend=%s, device=%s, default_origin=%s",
            self.model_path,
            self.backend,
            self._device,
            self.default_origin,
        )

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        transformers = _transformers()
        # Silence the "incorrect regex pattern (fix_mistral_regex=True)"
        # warning emitted on every load. Setting fix_mistral_regex=True
        # *changes* tokenization (verified: 4/4 sample IDs differ), which
        # would break v1.0 model compatibility. Instead, drop the message
        # at the transformers logger level for the duration of the load.
        try:
            tform_logging = transformers.utils.logging
        except AttributeError:
            tform_logging = None
        if tform_logging is not None:
            prev_level = tform_logging.get_verbosity()
            tform_logging.set_verbosity_error()
        try:
            self._tokenizer = transformers.AutoTokenizer.from_pretrained(str(self.model_path))
        finally:
            if tform_logging is not None:
                tform_logging.set_verbosity(prev_level)
        if self.backend == "torch":
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path),
            )
            self._model, self._device = _move_module_to_device(
                model,
                self._device,
                module_name="KatanaV11Classifier model",
            )
            self._model.eval()
        else:
            try:
                import onnxruntime
            except ImportError as exc:
                raise ImportError(
                    "onnxruntime is required for ONNX backends. Install with: pip install 'hermes-katana[fast-cpu]'"
                ) from exc
            onnx_path = self.model_path if self.model_path.suffix == ".onnx" else self.model_path / "model.onnx"
            if not onnx_path.is_file():
                candidates = sorted(self.model_path.glob("*.onnx")) if self.model_path.is_dir() else []
                if candidates:
                    onnx_path = candidates[0]
                else:
                    raise FileNotFoundError(f"ONNX model file not found under {self.model_path}")
            sess_opts = onnxruntime.SessionOptions()
            sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._model = onnxruntime.InferenceSession(
                str(onnx_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
        return self._model, self._tokenizer

    # ------------------------------------------------------------------
    # Origin handling
    # ------------------------------------------------------------------

    def _origin_prefix(self, origin: Optional[str]) -> str:
        tier = origin if origin in self.ORIGIN_TIERS else self.default_origin
        return f"[ORIGIN={tier}] "

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str, origin: Optional[str] = None) -> dict[str, float]:
        """Single-text classification. Returns label -> probability."""
        return self.classify_batch([text], [origin])[0]

    def classify_result(self, text: str, origin: Optional[str] = None) -> "ClassificationResult":
        """Single-text classification returning a full ClassificationResult."""
        return self.classify_batch_results([text], [origin])[0]

    def classify_batch(
        self,
        texts: list[str],
        origins: Optional[list[Optional[str]]] = None,
    ) -> list[dict[str, float]]:
        """Batched classification.

        Parameters
        ----------
        texts :
            List of N raw texts.
        origins :
            Optional per-text origin tier. ``None`` entries fall back to
            ``self.default_origin``. If the whole list is None, every text
            uses the default.

        Returns
        -------
        list[dict[str, float]]
            One label -> probability dict per input text.
        """
        if not texts:
            return []
        if origins is None:
            origins = [None] * len(texts)
        if len(origins) != len(texts):
            raise ValueError(f"len(origins)={len(origins)} != len(texts)={len(texts)}")

        prefixed = [self._origin_prefix(origins[i]) + (texts[i] or "") for i in range(len(texts))]

        model, tokenizer = self._ensure_loaded()
        if self.backend == "torch":
            torch = _torch()
            enc = tokenizer(
                prefixed,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc).logits  # (N, num_labels)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            enc = tokenizer(
                prefixed,
                return_tensors="np",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            input_names = {inp.name for inp in model.get_inputs()}
            np_enc = {k: v for k, v in enc.items() if k in input_names}
            logits = model.run(None, np_enc)[0]
            # numerically stable softmax
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = _np().exp(shifted)
            probs = exp / exp.sum(axis=-1, keepdims=True)

        out: list[dict[str, float]] = []
        for row in probs:
            scores = {self.LABELS[i]: float(row[i]) for i in range(len(self.LABELS))}
            out.append(scores)
        return out

    def classify_batch_results(
        self,
        texts: list[str],
        origins: Optional[list[Optional[str]]] = None,
    ) -> list["ClassificationResult"]:
        """Batched classification returning ClassificationResult objects."""
        from hermes_katana.scabbard.fusion import ClassificationResult, Decision

        out: list[ClassificationResult] = []
        for scores in self.classify_batch(texts, origins):
            attack_scores = {k: v for k, v in scores.items() if k != "clean"}
            top_category = max(attack_scores, key=lambda k: attack_scores[k])
            max_attack = attack_scores[top_category]

            if max_attack < self.allow_threshold:
                decision = Decision.ALLOW
            elif max_attack > self.block_threshold:
                decision = Decision.BLOCK
            else:
                decision = Decision.FLAG

            out.append(
                ClassificationResult(
                    scores=scores,
                    decision=decision,
                    top_category=top_category,
                    confidence=max_attack,
                )
            )
        return out

    def __repr__(self) -> str:
        return (
            f"KatanaV11Classifier(model={self.model_path.name}, "
            f"backend={self.backend}, device={self._device}, "
            f"default_origin={self.default_origin})"
        )


# -----------------------------------------------------------------------
# Re-export ZvecEmbedder as DeBERTaEmbedder for backward compatibility
# -----------------------------------------------------------------------

DeBERTaEmbedder = ZvecEmbedder
"""Backward-compatibility alias — use ZvecEmbedder for the zvec embedder."""


# -----------------------------------------------------------------------
# ATTACK_LABELS placeholder (imported from fusion at runtime)
# -----------------------------------------------------------------------


def _get_attack_labels() -> list[str]:
    """Late import of ATTACK_LABELS to avoid circular import at module level."""
    try:
        from hermes_katana.scabbard.fusion import ATTACK_LABELS

        return ATTACK_LABELS
    except Exception:
        # Graceful fallback — should never reach here in normal operation
        return [
            "clean",
            "content_injection",
            "semantic_manipulation",
            "behavioral_control",
            "exfiltration_attempt",
            "jailbreak",
            "cognitive_state_attack",
            "encoding_evasion",
            "persona_jailbreak",
            "unknown_anomaly",
        ]


_ATTACK_LABELS: list[str] = []


def _init_attack_labels() -> None:
    global _ATTACK_LABELS
    _ATTACK_LABELS[:] = _get_attack_labels()


# Initialise on module load
_init_attack_labels()
