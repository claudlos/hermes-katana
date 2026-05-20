"""Scabbard configuration with deployment profiles.

Three profiles trade off latency vs detection capability:

- **minimal**: normalizer + TF-IDF n-grams only (~1 ms, zero ML deps)
- **standard**: + zvec embeddings + centroids + fusion (~5 ms)
- **full**: + perplexity analysis + LLM judge (~25 ms + judge)
- **cascade**: explicit per-tier thresholds for the ScabbardCascadeRouter

Embedding / Classification models
---------------------------------
The standard/full profiles support three backends:

1. **DeBERTaClassifier** (recommended): DeBERTa-v2-base fine-tuned on 8 attack
   classes.  Takes priority over the zvec pipeline when
   ``deberta_model_cls`` is set.  Returns a full ClassificationResult
   (scores + decision) directly — skips the feature-extractor + fusion stages.

2. **zvec** (default): sentence-transformers/all-MiniLM-L6-v2 + learned 128-dim
   projector.  Fast (~5K rec/s on CPU), small (87MB), trained on attack data.
   Output: 128-dim L2-normalized.

3. **deberta** (legacy embedder): microsoft/deberta-v3-large fine-tuned
   classifier used only as an embedder (768-dim [CLS] token) for the
   zvec pipeline.  Slower and larger than zvec; retained for backward
   compatibility.

Set ``deberta_model_cls`` to use the DeBERTaClassifier directly.
Set ``deberta_model`` (legacy) to use DeBERTa as an embedder for zvec pipeline.
Centroids are auto-detected from the embedder output dim (128 for zvec,
768 for DeBERTa).
"""

from __future__ import annotations
import importlib.util

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_katana.artifacts import (
    artifact_status,
    minilm_onnx_spec,
    resolve_minilm_onnx,
    resolve_v15_large,
)

# -----------------------------------------------------------------------
# Default model paths (relative to project root)
# -----------------------------------------------------------------------

_TRAINING_MODELS = Path(__file__).parent.parent.parent.parent / "training" / "models"

# zvec (contrastive MiniLM — default)
_ZVEC_BASE = _TRAINING_MODELS / "zvec_quantized-20260408T061203Z-3-001" / "zvec_quantized"
_ZVEC_BACKBONE = _ZVEC_BASE / "backbone_fp32"
_ZVEC_PROJECTOR = _ZVEC_BASE / "projector_fp32.pt"
_ZVEC_TOKENIZER = _ZVEC_BASE / "tokenizer"

# Centroids (auto-selected by embedder output dim)
_CENTROIDS_128 = _TRAINING_MODELS / "attack_centroids_128d.npz"
_CENTROIDS_768 = _TRAINING_MODELS / "attack_centroids.npz"

# TF-IDF + Fusion
_TFIDF = _TRAINING_MODELS / "tfidf_vectorizer.pkl"
_FUSION = _TRAINING_MODELS / "fusion_xgb.json"

# Katana model checkpoints — DeBERTa-v3-large 9-class origin-aware.
# Each one lives at training/checkpoints/<name>/best/ under the repo root.
_CHECKPOINTS = Path(__file__).parent.parent.parent.parent / "training" / "checkpoints"
_KATANA_V11_BEST = _CHECKPOINTS / "katana_v11" / "best"  # baseline (data_v5_1)
_KATANA_V12_BEST = _CHECKPOINTS / "katana_v12" / "best"  # data_v6 (3.5% origin aug)
_KATANA_V14_BEST = _CHECKPOINTS / "katana_v14" / "best"  # PRODUCTION (data_v7, LR fix)
_KATANA_V15_BEST = _CHECKPOINTS / "katana_v15" / "best"  # candidate (data_v8; explicit only)
_KATANA_V15_MINILM_BEST = _CHECKPOINTS / "katana_v15_distill_minilm" / "best"
_KATANA_V15_MINILM_TAG = "katana_v15_distill_minilm"


def _newest_available_katana_checkpoint() -> Path:
    """Return the highest-version katana checkpoint that exists locally.

    Resolution order: v15 > v14 > v12 > v11 > v11 path even if missing
    (so callers get a clear FileNotFoundError on misconfiguration).

    This helper is for diagnostics and candidate discovery only. Production
    resolution intentionally uses _production_katana_checkpoint() so dropping a
    candidate checkpoint into training/checkpoints never silently promotes it.
    """
    for cand in (_KATANA_V15_BEST, _KATANA_V14_BEST, _KATANA_V12_BEST, _KATANA_V11_BEST):
        if (cand / "model.safetensors").is_file():
            return cand
    return _KATANA_V11_BEST


def _production_katana_checkpoint() -> Path:
    """Return the blessed production checkpoint path.

    v15 is deliberately excluded until it is promoted through the threshold,
    evasion, and shadow-rollout gates.
    """
    for cand in (_KATANA_V14_BEST, _KATANA_V12_BEST, _KATANA_V11_BEST):
        if (cand / "model.safetensors").is_file():
            return cand
    return _KATANA_V14_BEST


# Map "best" checkpoint dir -> stable model_version tag used in metrics/audit rows.
_VERSION_TAGS = {
    _KATANA_V11_BEST: "katana_v11",
    _KATANA_V12_BEST: "katana_v12",
    _KATANA_V14_BEST: "katana_v14",
    _KATANA_V15_BEST: "katana_v15",
    _KATANA_V15_MINILM_BEST: "katana_v15_distill_minilm",
}


def _model_version_for_checkpoint(checkpoint_path: Path) -> str:
    """Return the metrics/audit model_version tag for a checkpoint.

    Accepts either the ``best/`` dir or its ONNX/INT8 sibling; resolves
    by walking up to the parent that matches a known katana_v* directory.
    Falls back to the directory name if the lookup fails.
    """
    p = Path(checkpoint_path).resolve()
    for known_best, tag in _VERSION_TAGS.items():
        known_root = known_best.resolve().parent
        try:
            p.relative_to(known_root)
        except ValueError:
            continue
        return tag
    return p.parent.name or "katana-unknown"


def _katana_v15_minilm_onnx_artifact_path() -> Path:
    """Return a verified MiniLM ONNX artifact path, downloading only if opted in."""
    return resolve_minilm_onnx(download=None)


def _katana_v15_large_artifact_path() -> Path:
    """Return a verified large v15 artifact path, downloading only if opted in."""
    return resolve_v15_large(download=None)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _path_has_entries(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _default_centroid_path(*, use_deberta_embedder: bool = False) -> Optional[str]:
    """Pick a centroid artifact compatible with the active embedder."""
    if use_deberta_embedder:
        return str(_CENTROIDS_768) if _CENTROIDS_768.exists() else None
    return str(_CENTROIDS_128) if _CENTROIDS_128.exists() else None


def _minimal_runtime_ready() -> bool:
    """Return whether the minimal profile can classify locally."""
    return _module_available("numpy")


def _standard_runtime_ready(*, use_deberta_embedder: bool = False) -> bool:
    """Return whether the standard runtime has the artifacts and deps it expects."""
    required_modules = ("numpy", "joblib", "xgboost", "torch", "transformers")
    if not all(_module_available(name) for name in required_modules):
        return False
    if not _TFIDF.is_file() or not _FUSION.is_file():
        return False
    centroid_path = _default_centroid_path(use_deberta_embedder=use_deberta_embedder)
    if centroid_path is None:
        return False
    if use_deberta_embedder:
        return True
    return _path_has_entries(_ZVEC_BACKBONE) and _ZVEC_PROJECTOR.is_file() and _path_has_entries(_ZVEC_TOKENIZER)


@dataclass(frozen=True)
class ScabbardConfig:
    """Immutable configuration for the Scabbard pipeline.

    Default ``standard()`` factory uses zvec (MiniLM + projector, 128-dim).
    To use DeBERTa instead, pass ``deberta_model="/path/to/deberta"`` — the
    centroid path is auto-selected based on embedder output dimension.
    """

    # Profile: minimal | standard | full
    profile: str = "standard"

    # KatanaV11Classifier: trained DeBERTa-v3-large 9-class origin-aware
    # classifier (the v1.0 baseline). When set, takes priority over both the
    # legacy DeBERTaClassifier and the zvec+fusion pipeline.
    katana_v11_path: Optional[str] = None
    katana_v11_default_origin: str = "user_input"
    # Backend: "torch" | "onnx" | "onnx_int8". ONNX backends are CPU-only and
    # require onnxruntime installed.
    katana_v11_backend: str = "torch"
    # Optional device for torch backends ("cuda" or "cpu"). None means auto.
    # ONNX backends ignore this and run on CPU in the current runtime.
    katana_v11_device: Optional[str] = None
    # Free-form identifier recorded on every ClassificationResult.metadata so
    # audit/metrics rows can be filtered by model version. Recommend semver
    # or a date-stamped tag (e.g., "katana_v11-20260507").
    model_version: str = "katana_v11-default"
    # Per-call timeout for the classifier in seconds. 0 disables. When the
    # classifier exceeds this, the middleware falls back to ``timeout_decision``
    # (default ALLOW — fail-open in production where latency budgets matter
    # more than catching every attack; switch to DENY for fail-closed).
    classifier_timeout_seconds: float = 0.0
    classifier_timeout_decision: str = "allow"  # "allow" | "deny"

    # Shadow classifier: when set, runs a second KatanaV11Classifier alongside
    # the primary on every call. Logs disagreements to
    # ``hermes_katana.middleware.shadow`` at INFO level. Does NOT affect
    # actual decisions — used for staged rollouts of new model versions
    # before promoting them to primary.
    shadow_v11_path: Optional[str] = None
    shadow_v11_backend: str = "torch"
    shadow_v11_device: Optional[str] = None
    shadow_v11_default_origin: str = "user_input"
    shadow_model_version: str = "shadow-unknown"

    # DeBERTaClassifier: trained DeBERTa-v2-base 8-class classifier (legacy).
    # When set (and katana_v11_path is None), takes priority over the zvec
    # +fusion pipeline.
    deberta_model_cls: Optional[str] = None

    # Legacy embedder backends (kept for backward compatibility):
    # deberta_model: DeBERTa-v3-large as a zvec embedder (768-dim [CLS] output)
    deberta_model: Optional[str] = None
    # zvec-specific paths (used when neither deberta_model_cls nor deberta_model is set)
    zvec_backbone_path: Optional[str] = None
    zvec_projector_path: Optional[str] = None
    zvec_tokenizer_path: Optional[str] = None

    # Feature model paths
    perplexity_model: Optional[str] = None
    fusion_model: Optional[str] = str(_FUSION) if _FUSION.exists() else None
    centroid_path: Optional[str] = _default_centroid_path()
    tfidf_path: Optional[str] = str(_TFIDF) if _TFIDF.exists() else None
    homoglyph_path: Optional[str] = None

    # Decision thresholds. Defaults updated 2026-05-08 from block=0.70 to
    # block=0.50 after a principled threshold sweep on confirmed_only_v1 +
    # hard_negatives + splits/test (see results/threshold_tune_v14_*).
    # The sweep showed:
    #   - hard_negatives FPR is FLAT at 0.10% across the entire 0.05-0.95
    #     range (v14 is rock-solid on adversarial benigns).
    #   - confirmed_only_v1 recall climbs from 0.9780 (@0.70) to 0.9902
    #     (@0.50) with only +0.0017 FPR (0.70%->0.87%).
    #   - The 5/30 confirmed attacks that scored in [0.50, 0.70] in the
    #     2026-05-08 live test (codex+minimax) all became real exploits;
    #     the runtime correctly catches them at block=0.50.
    # 0.50 is preferred over the F1-maximizer 0.25 because it preserves a
    # defensible mental model ("model says it's more likely an attack than
    # not") and reduces user-override friction in borderline cases.
    allow_threshold: float = 0.3
    block_threshold: float = 0.5

    # LLM judge (full profile only)
    judge_endpoint: Optional[str] = None
    judge_model: Optional[str] = None
    judge_timeout: float = 0.5
    judge_max_tokens: int = 100

    # Cascade router thresholds (used by ScabbardCascadeRouter)
    # Tier 1 allow-confidence above this -> skip Tier 2 & 3 (default 0.9)
    cascade_tier1_allow_threshold: float = 0.9
    # Tier 1 attack-score above this -> BLOCK immediately (default 0.8)
    cascade_tier1_block_threshold: float = 0.8
    # Tier 2 confidence above this -> skip Tier 3 (default 0.85)
    cascade_tier2_confidence_threshold: float = 0.85

    # ProtectAI Tier 1.5 gate
    protectai_enabled: bool = True
    protectai_injection_threshold: float = 0.7
    protectai_safe_threshold: float = 0.9

    # ---------- factory helpers ----------

    def __post_init__(self) -> None:
        if not 0.0 <= self.allow_threshold <= 1.0:
            raise ValueError("allow_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.block_threshold <= 1.0:
            raise ValueError("block_threshold must be between 0.0 and 1.0")
        if self.block_threshold < self.allow_threshold:
            raise ValueError("block_threshold must be greater than or equal to allow_threshold")
        if self.judge_timeout <= 0.0:
            raise ValueError("judge_timeout must be greater than 0.0")

    @classmethod
    def minimal(cls) -> "ScabbardConfig":
        """Normalizer + n-grams only.  Zero ML dependencies, <1 ms."""
        return cls(profile="minimal")

    @classmethod
    def minimal_runtime_ready(cls) -> bool:
        """Return whether the local environment can run the minimal profile."""
        return _minimal_runtime_ready()

    @classmethod
    def standard_runtime_ready(cls, *, use_deberta_embedder: bool = False) -> bool:
        """Return whether the local environment can run the standard profile."""
        return _standard_runtime_ready(use_deberta_embedder=use_deberta_embedder)

    @classmethod
    def katana_v11_available(cls) -> bool:
        """Return True if the v1.0 katana_v11 checkpoint is present locally."""
        return (_KATANA_V11_BEST / "model.safetensors").is_file()

    @classmethod
    def katana_v14_available(cls) -> bool:
        """Return True if the v1.2 production katana_v14 checkpoint is present locally."""
        return (_KATANA_V14_BEST / "model.safetensors").is_file()

    @classmethod
    def katana_v15_available(cls) -> bool:
        """Return True if the v15 candidate checkpoint is present locally."""
        return (_KATANA_V15_BEST / "model.safetensors").is_file()

    @classmethod
    def katana_v15_minilm_available(cls, *, backend: str = "onnx") -> bool:
        """Return True if the distilled v15 MiniLM artifact is present locally."""
        if backend == "onnx":
            return artifact_status(minilm_onnx_spec()).present
        if backend == "torch":
            return (_KATANA_V15_MINILM_BEST / "model.safetensors").is_file()
        return False

    @classmethod
    def katana_default_available(cls) -> bool:
        """Return True if the blessed production checkpoint chain is loadable."""
        return (_production_katana_checkpoint() / "model.safetensors").is_file()

    @classmethod
    def production(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.5,
        backend: str = "torch",
        device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Production profile pinned to the blessed production checkpoint.

        Resolves to v14 if present (v1.2; data_v7 + LR fix; macro F1 0.9179
        on confirmed_only_v1, 0/50 homoglyph bypasses, 5/5 origin sweep);
        falls back to v12 then v11 if v14 isn't downloaded yet. v15 remains an
        explicit candidate path until promoted. Pass ``model_path`` explicitly
        to pin to a specific checkpoint.

        Default thresholds (allow=0.3, block=0.5) were chosen by sweep on
        confirmed_only_v1 + hard_negatives + test split. See ScabbardConfig
        class docstring for the data points behind the choice; in short,
        block=0.5 catches +12 attacks per 1000 vs block=0.7 with negligible
        hard-negatives FPR change.

        ``backend`` selects the runtime: ``"torch"`` (default; GPU-aware) /
        ``"onnx"`` (CPU; ~2x faster than torch on CPU) / ``"onnx_int8"`` when
        the selected production checkpoint has passed parity tests.

        ONNX/INT8 variants are expected beside the selected checkpoint root
        as ``onnx/`` and ``onnx_int8/``. To regenerate them for a new Katana
        version, run ``training/export_v11_to_onnx.py`` with ``--src`` pointed
        at that version's ``best/`` checkpoint.
        """
        if model_path is None:
            base = _production_katana_checkpoint()
            if backend == "onnx":
                model_path = str(base.parent / "onnx")
            elif backend == "onnx_int8":
                model_path = str(base.parent / "onnx_int8")
            else:
                model_path = str(base)
        return cls(
            profile="standard",
            katana_v11_path=model_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version=_model_version_for_checkpoint(Path(model_path)),
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def katana_v11(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.7,
        backend: str = "torch",
        device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Pin to the v1.0 baseline katana_v11 checkpoint.

        Use this for ablation studies or to reproduce v1.0 paper results.
        For production, prefer :meth:`production` (resolves to v14).

        ``backend`` selects torch / onnx / onnx_int8.
        """
        if backend == "onnx" and model_path is None:
            model_path = str(_KATANA_V11_BEST.parent / "onnx")
        elif backend == "onnx_int8" and model_path is None:
            model_path = str(_KATANA_V11_BEST.parent / "onnx_int8")
        resolved_path = model_path or str(_KATANA_V11_BEST)
        return cls(
            profile="standard",
            katana_v11_path=resolved_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version=_model_version_for_checkpoint(Path(resolved_path)),
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def katana_v14(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.5,
        backend: str = "torch",
        device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Pin to the v1.2 production katana_v14 checkpoint (DeBERTa-v3-large
        on data_v7, LR 3e-5 / warmup 0.10).

        Same field shape as :meth:`katana_v11` for runtime compatibility —
        the underlying classifier code is version-agnostic and just loads
        whatever checkpoint is at the given path. Default thresholds match
        the production tuned values (block=0.5).
        """
        if backend == "onnx" and model_path is None:
            model_path = str(_KATANA_V14_BEST.parent / "onnx")
        elif backend == "onnx_int8" and model_path is None:
            model_path = str(_KATANA_V14_BEST.parent / "onnx_int8")
        resolved_path = model_path or str(_KATANA_V14_BEST)
        return cls(
            profile="standard",
            katana_v11_path=resolved_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version=_model_version_for_checkpoint(Path(resolved_path)),
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def katana_v15(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.5,
        backend: str = "torch",
        device: Optional[str] = None,
        allow_unverified_int8: bool = False,
    ) -> "ScabbardConfig":
        """Pin to the v15 candidate checkpoint trained on frozen data_v8.

        Use this for shadow rollout/eval until threshold and evasion sweeps
        justify promoting it through :meth:`production`.
        """
        if backend == "onnx_int8" and not allow_unverified_int8:
            raise ValueError(
                "katana_v15 INT8 is not promoted: Colab parity was 54/100 "
                "against torch. Use backend='torch' or backend='onnx', or pass "
                "allow_unverified_int8=True only for quantization debugging."
            )
        if backend == "onnx" and model_path is None:
            model_path = str(_KATANA_V15_BEST.parent / "onnx")
        elif backend == "onnx_int8" and model_path is None:
            model_path = str(_KATANA_V15_BEST.parent / "onnx_int8")
        resolved_path = model_path or str(_KATANA_V15_BEST)
        return cls(
            profile="standard",
            katana_v11_path=resolved_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version=_model_version_for_checkpoint(Path(resolved_path)),
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def katana_v15_large(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.5,
        backend: str = "torch",
        device: Optional[str] = None,
        allow_unverified_int8: bool = False,
    ) -> "ScabbardConfig":
        """Pin to the large v15 DeBERTa teacher/candidate model.

        This is a clearer alias for :meth:`katana_v15` when comparing the
        full-size DeBERTa runtime against distilled CPU students.
        """
        if backend == "onnx_int8" and not allow_unverified_int8:
            raise ValueError(
                "katana_v15 INT8 is not promoted: Colab parity was 54/100 "
                "against torch. Use backend='torch' or backend='onnx', or pass "
                "allow_unverified_int8=True only for quantization debugging."
            )
        if model_path is None:
            if backend == "torch":
                model_path = str(_katana_v15_large_artifact_path())
            elif backend == "onnx":
                model_path = str(_KATANA_V15_BEST.parent / "onnx")
            elif backend == "onnx_int8":
                model_path = str(_KATANA_V15_BEST.parent / "onnx_int8")
        resolved_path = model_path or str(_KATANA_V15_BEST)
        return cls(
            profile="standard",
            katana_v11_path=resolved_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version="katana_v15",
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def katana_v15_minilm(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.5,
        backend: str = "onnx",
        device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Pin to the distilled v15 MiniLM-L6 student.

        Default backend is fp32 ONNX because this is the shippable 88 MB CPU
        artifact. ``backend='torch'`` loads the source checkpoint for parity
        checks and development. INT8 is intentionally unsupported here because
        fp32 already fits the deployment budget.
        """
        if backend == "onnx_int8":
            raise ValueError("katana_v15_distill_minilm ships as fp32 ONNX; INT8 is not configured")
        if backend == "onnx" and model_path is None:
            model_path = str(_katana_v15_minilm_onnx_artifact_path())
        resolved_path = model_path or str(_KATANA_V15_MINILM_BEST)
        return cls(
            profile="standard",
            katana_v11_path=resolved_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version=_KATANA_V15_MINILM_TAG,
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def production_with_v15_shadow(
        cls,
        *,
        production_model_path: Optional[str] = None,
        v15_model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.5,
        backend: str = "torch",
        shadow_backend: str = "onnx",
        shadow_device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Production v14 primary with v15 as a non-decision shadow model.

        Shadow results are logged by middleware for disagreement review and do
        not affect the live decision. v15 INT8 is blocked here because its A100
        export failed parity; use v15 torch/fp32 ONNX for shadow runs.
        """
        if shadow_backend == "onnx_int8":
            raise ValueError("v15 INT8 is not approved for shadow rollout; use torch or onnx.")

        primary = cls.production(
            model_path=production_model_path,
            default_origin=default_origin,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            backend=backend,
        )
        shadow = cls.katana_v15(
            model_path=v15_model_path,
            default_origin=default_origin,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            backend=shadow_backend,
            device=shadow_device,
        )
        return cls(
            profile="standard",
            katana_v11_path=primary.katana_v11_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=primary.katana_v11_device,
            model_version=primary.model_version,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            shadow_v11_path=shadow.katana_v11_path,
            shadow_v11_backend=shadow_backend,
            shadow_v11_device=shadow_device,
            shadow_v11_default_origin=default_origin,
            shadow_model_version=shadow.model_version,
            centroid_path=None,
            tfidf_path=None,
            fusion_model=None,
        )

    @classmethod
    def default_runtime_profile(cls) -> str:
        """Choose the safest default runtime profile for the current checkout."""
        if cls.katana_default_available():
            return "production"
        if cls.standard_runtime_ready():
            return "standard"
        if cls.minimal_runtime_ready():
            return "minimal"
        return "minimal"

    @classmethod
    def runtime_default(cls) -> "ScabbardConfig":
        """Return the safest default config for the current runtime state."""
        profile = cls.default_runtime_profile()
        if profile == "production":
            return cls.production()
        if profile == "standard":
            return cls.standard()
        return cls.minimal()

    @classmethod
    def standard(
        cls,
        *,
        deberta_model_cls: Optional[str] = None,
        deberta_model: Optional[str] = None,
        centroid_path: Optional[str] = None,
        tfidf_path: Optional[str] = None,
        fusion_model: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Standard profile with zvec (default) or DeBERTaClassifier or DeBERTa embeddings.

        - ``deberta_model_cls``: path to a trained DeBERTaClassifier.
          Takes priority; skips zvec+fusion entirely.
        - ``deberta_model`` (legacy): path to DeBERTa-v3-large to use as
          an embedder for the zvec pipeline (768-dim [CLS]).
        - Default: zvec embedder (MiniLM-L6-v2 + 128-dim projector).

        Centroids are auto-selected to match embedder output dimension.
        """
        use_deberta_embedder = deberta_model is not None
        auto_centroid = (
            centroid_path
            if centroid_path is not None
            else (None if deberta_model_cls else _default_centroid_path(use_deberta_embedder=use_deberta_embedder))
        )
        return cls(
            profile="standard",
            deberta_model_cls=deberta_model_cls,
            deberta_model=deberta_model,
            centroid_path=auto_centroid,
            tfidf_path=tfidf_path,
            fusion_model=fusion_model,
        )

    @classmethod
    def full(
        cls,
        *,
        deberta_model_cls: Optional[str] = None,
        deberta_model: Optional[str] = None,
        centroid_path: Optional[str] = None,
        tfidf_path: Optional[str] = None,
        fusion_model: Optional[str] = None,
        perplexity_model: Optional[str] = None,
        judge_endpoint: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_timeout: float = 0.5,
        cascade_tier1_allow_threshold: float = 0.9,
        cascade_tier1_block_threshold: float = 0.8,
        cascade_tier2_confidence_threshold: float = 0.85,
    ) -> "ScabbardConfig":
        """Full profile with perplexity analysis, LLM judge, and cascade router."""
        use_deberta_embedder = deberta_model is not None
        auto_centroid = (
            centroid_path
            if centroid_path is not None
            else (None if deberta_model_cls else _default_centroid_path(use_deberta_embedder=use_deberta_embedder))
        )
        return cls(
            profile="full",
            deberta_model_cls=deberta_model_cls,
            deberta_model=deberta_model,
            centroid_path=auto_centroid,
            tfidf_path=tfidf_path,
            fusion_model=fusion_model,
            perplexity_model=perplexity_model,
            judge_endpoint=judge_endpoint,
            judge_model=judge_model,
            judge_timeout=judge_timeout,
            cascade_tier1_allow_threshold=cascade_tier1_allow_threshold,
            cascade_tier1_block_threshold=cascade_tier1_block_threshold,
            cascade_tier2_confidence_threshold=cascade_tier2_confidence_threshold,
        )

    @classmethod
    def cascade(
        cls,
        *,
        judge_endpoint: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_timeout: float = 0.5,
        deberta_model_cls: Optional[str] = None,
        deberta_model: Optional[str] = None,
        tier1_allow_threshold: float = 0.9,
        tier1_block_threshold: float = 0.8,
        tier2_confidence_threshold: float = 0.85,
    ) -> "ScabbardConfig":
        """Cascade profile optimised for the 3-tier ScabbardCascadeRouter."""
        return cls(
            profile="full",
            deberta_model_cls=deberta_model_cls,
            deberta_model=deberta_model,
            judge_endpoint=judge_endpoint,
            judge_model=judge_model,
            judge_timeout=judge_timeout,
            cascade_tier1_allow_threshold=tier1_allow_threshold,
            cascade_tier1_block_threshold=tier1_block_threshold,
            cascade_tier2_confidence_threshold=tier2_confidence_threshold,
        )
