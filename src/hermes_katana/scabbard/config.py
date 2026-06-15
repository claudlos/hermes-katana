"""Scabbard configuration with deployment profiles.

Three profiles trade off latency vs detection capability:

- **minimal**: normalizer + TF-IDF n-grams only (~1 ms, zero ML deps)
- **standard**: + zvec embeddings + fusion (~5 ms); centroid features are experimental opt-in
- **full**: + perplexity analysis + LLM judge (~25 ms + judge)
- **cascade**: explicit per-tier thresholds for the ScabbardCascadeRouter

Embedding / Classification models
---------------------------------
The standard/full profiles support three backends:

1. **DeBERTaClassifier** (recommended): DeBERTa-v2-base fine-tuned on 8 attack
   classes.  Takes priority over the zvec pipeline when
   ``deberta_model_cls`` is set.  Returns a full ClassificationResult
   (scores + decision) directly — skips the feature-extractor + fusion stages.

2. **zvec** (experimental): sentence-transformers/all-MiniLM-L6-v2 +
   learned 128-dim projector. Fast (~5K rec/s on CPU), small (87MB),
   trained on attack data. Output: 128-dim L2-normalized.

3. **deberta** (legacy embedder): microsoft/deberta-v3-large fine-tuned
   classifier used only as an embedder (768-dim [CLS] token) for the
   zvec pipeline.  Slower and larger than zvec; retained for backward
   compatibility.

Set ``deberta_model_cls`` to use the DeBERTaClassifier directly.
Set ``deberta_model`` (legacy) to use DeBERTa as an embedder for zvec pipeline.
Centroid artifacts are experimental and not part of the default runtime path.
Pass ``centroid_path`` explicitly, or set
``HERMES_KATANA_ENABLE_EXPERIMENTAL_CENTROIDS=1`` to let standard/full profile
factories auto-discover local centroid files for research runs.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_katana.artifacts import (
    artifact_status,
    minilm_onnx_spec,
    minilm_torch_spec,
    resolve_minilm_onnx,
    resolve_minilm_torch,
    resolve_v15_large,
    resolve_v17_minilm,
    v17_minilm_spec,
)

logger = logging.getLogger(__name__)

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
_KATANA_V17_MINILM_TAG = "katana_v17_minilm"


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

    v17 origin-aware MiniLM is preferred when its artifact is present; v15 is
    deliberately excluded until it is promoted through the threshold, evasion,
    and shadow-rollout gates. v14/v12/v11 are kept as fallbacks for repos that
    have not yet downloaded the v17 artifact.
    """
    # Resolve the v17 path defensively: _katana_v17_minilm_artifact_path() raises
    # ArtifactNotFoundError when v17 has not been downloaded, but this function is
    # called from readiness diagnostics (katana_default_available ->
    # default_runtime_profile -> katana_status), which must never crash just
    # because the optional v17 artifact is absent.
    v17_minilm_path: Optional[Path] = None
    try:
        candidate = _katana_v17_minilm_artifact_path()
        if (candidate / "model.safetensors").is_file():
            return candidate
        v17_minilm_path = candidate
    except Exception:
        v17_minilm_path = None
    for cand in (_KATANA_V14_BEST, _KATANA_V12_BEST, _KATANA_V11_BEST):
        if (cand / "model.safetensors").is_file():
            return cand
    return v17_minilm_path if v17_minilm_path is not None else _KATANA_V14_BEST


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


def _katana_v15_minilm_torch_artifact_path() -> Path:
    """Return a verified MiniLM PyTorch artifact path, downloading only if opted in."""
    return resolve_minilm_torch(download=None)


def _katana_v15_large_artifact_path() -> Path:
    """Return a verified large v15 artifact path, downloading only if opted in."""
    return resolve_v15_large(download=None)


def _katana_v17_minilm_artifact_path() -> Path:
    """Return a verified v17 MiniLM artifact path, downloading only if opted in."""
    return resolve_v17_minilm(download=None)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _path_has_entries(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _experimental_centroids_enabled() -> bool:
    """Return whether experimental centroid auto-discovery is enabled."""
    value = os.environ.get("HERMES_KATANA_ENABLE_EXPERIMENTAL_CENTROIDS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _require_trained_scabbard() -> bool:
    """Return whether the deployment refuses to run on the rule-based fallback."""
    value = os.environ.get("HERMES_KATANA_REQUIRE_TRAINED_SCABBARD", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_centroid_path(*, use_deberta_embedder: bool = False) -> Optional[str]:
    """Pick a centroid artifact only when experimental auto-discovery is enabled."""
    if not _experimental_centroids_enabled():
        return None
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
    # classifier exceeds this, the middleware falls back to
    # ``classifier_timeout_decision``:
    #   "deny"     — BLOCK (fail closed)
    #   "escalate" — FLAG, routed through the escalation policy (default for
    #                enforcement profiles that enable the timeout)
    #   "allow"    — ALLOW (fail open; the result is still stamped degraded so
    #                the timeout is visible in extras/audit)
    classifier_timeout_seconds: float = 0.0
    classifier_timeout_decision: str = "allow"  # "allow" | "deny" | "escalate"

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

    # Feature model paths. Centroid auto-discovery is experimental and disabled
    # unless HERMES_KATANA_ENABLE_EXPERIMENTAL_CENTROIDS=1 is set.
    perplexity_model: Optional[str] = None
    fusion_model: Optional[str] = str(_FUSION) if _FUSION.exists() else None
    centroid_path: Optional[str] = _default_centroid_path()
    tfidf_path: Optional[str] = str(_TFIDF) if _TFIDF.exists() else None
    homoglyph_path: Optional[str] = None

    # Decision thresholds. Set to block=0.60 on 2026-06-14 as a middle
    # ground when v17 origin-aware MiniLM became the default model: the
    # v14-era sweep (block=0.50 catching +12 attacks/1000 with FPR unchanged
    # at 0.10%) was measured on the curated eval set, which is light on
    # security-domain benign content. On the v17 model, block=0.50 caused
    # 38/154 (24.7%) FPs in the security-notes subset of
    # tests/smoke/false_positive_gate.py (all confident false positives on
    # benign text that *talks about* security). block=0.70 was too
    # conservative and would have lost those +12 attacks/1000 from the
    # v14-era sweep entirely. block=0.60 is a deliberate middle ground;
    # if the v17 model is recalibrated, this can be re-tuned.
    allow_threshold: float = 0.3
    block_threshold: float = 0.7

    # Cosine-similarity false-positive softener. When enabled, a BLOCK on
    # non-degraded, non-tainted text is softened if the text is cosine-close to
    # a vetted benign exemplar (policies/scabbard_benign_exemplars.yaml) via the
    # torch-free ONNX encoder. The threshold sits above the adversarial-corpus
    # attack ceiling, so it never softens an attack (see the evasion gate +
    # tests/smoke/test_similarity_allowlist_safety.py). Fails closed (no soften)
    # if the encoder artifact or onnxruntime is missing.
    similarity_allowlist_enabled: bool = True
    # When True, record a truncated preview (<=200 chars) of the exact classified
    # text for softened AND denied Scabbard blocks, so live false positives can be
    # reviewed and allowlisted by call_id. Off by default: it stores tool-argument
    # plaintext in the audit trail.
    audit_blocked_text: bool = False

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
        if self.classifier_timeout_decision not in ("allow", "deny", "escalate"):
            # An unrecognized value must not silently fail open at timeout time.
            raise ValueError("classifier_timeout_decision must be one of: allow, deny, escalate")

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
            return artifact_status(minilm_torch_spec()).present
        return False

    @classmethod
    def katana_v17_minilm_available(cls) -> bool:
        """Return True if the v17 origin-aware MiniLM artifact is present locally."""
        return artifact_status(v17_minilm_spec()).present

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
        block_threshold: float = 0.7,
        backend: str = "torch",
        device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Production profile pinned to the blessed production checkpoint.

        Resolves to v14 if present (v1.2; data_v7 + LR fix; macro F1 0.9179
        on confirmed_only_v1, 0/50 homoglyph bypasses, 5/5 origin sweep);
        falls back to v12 then v11 if v14 isn't downloaded yet. v15 remains an
        explicit candidate path until promoted. Pass ``model_path`` explicitly
        to pin to a specific checkpoint.

        Default thresholds (allow=0.3, block=0.7) are the chosen default
        on the v17 model. See the ScabbardConfig class docstring for the trade-off
        between 0.5 (catches +12 attacks/1000 vs 0.7, but caused 24.7% FPs
        on security-domain benign text on the v17 model) and 0.7 (cleaner
        FPR but loses the +12 attacks). The empirical measurement on
        tests/smoke/false_positive_gate.py is: 0.5 -> 38 FPs, 0.6 -> 37
        FPs, 0.7 -> 34 FPs (all 100% attack recall). 0.7 is the chosen
        default; the 34 remaining FPs cluster at 1.0 confidence and are
        not threshold-tunable. Re-tuning the v17 model itself is the
        path to reducing FPs further.


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
        block_threshold: float = 0.7,
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
        block_threshold: float = 0.7,
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
        block_threshold: float = 0.7,
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
        block_threshold: float = 0.7,
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
        elif backend == "torch" and model_path is None:
            model_path = str(_katana_v15_minilm_torch_artifact_path())
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
    def katana_v17_minilm(
        cls,
        *,
        model_path: Optional[str] = None,
        default_origin: str = "user_input",
        allow_threshold: float = 0.3,
        block_threshold: float = 0.7,
        backend: str = "torch",
        device: Optional[str] = None,
    ) -> "ScabbardConfig":
        """Pin to the v3.1 origin-aware distilled MiniLM-L6 research artifact."""
        if backend != "torch":
            raise ValueError("katana_v17_minilm ships as a PyTorch safetensors artifact; use backend='torch'")
        resolved_path = model_path or str(_katana_v17_minilm_artifact_path())
        return cls(
            profile="standard",
            katana_v11_path=resolved_path,
            katana_v11_default_origin=default_origin,
            katana_v11_backend=backend,
            katana_v11_device=device,
            allow_threshold=allow_threshold,
            block_threshold=block_threshold,
            model_version=_KATANA_V17_MINILM_TAG,
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
        block_threshold: float = 0.7,
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
        """Return the safest default config for the current runtime state.

        When no trained artifacts are present this degrades to the rule-based
        ``minimal`` profile, which does NOT reliably block prompt injections.
        That degradation is logged at WARNING, and deployments can refuse it
        entirely by setting ``HERMES_KATANA_REQUIRE_TRAINED_SCABBARD=1``
        (fail closed: raises instead of silently weakening enforcement).
        """
        profile = cls.default_runtime_profile()
        if profile == "production":
            # Capability-aware backend. The production checkpoint (v17/v14
            # MiniLM/DeBERTa) is a PyTorch model. In a torch-free deployment the
            # torch classifier fails to load and Scabbard runs DEGRADED -- every
            # call fails closed to BLOCK and cannot be softened, which manifests
            # as the security tool blocking its own benign tool calls. If torch
            # is unavailable but the v15 ONNX MiniLM (run via onnxruntime) is
            # present, use it instead of degrading.
            if not _module_available("torch") and cls.katana_v15_minilm_available(backend="onnx"):
                logger.warning(
                    "PyTorch is unavailable; using the v15 ONNX MiniLM Scabbard "
                    "backend (onnxruntime) instead of the torch production checkpoint "
                    "to avoid degraded fail-closed enforcement."
                )
                return cls.katana_v15_minilm(backend="onnx")
            return cls.production()
        if profile == "standard":
            return cls.standard()
        if _require_trained_scabbard():
            raise RuntimeError(
                "HERMES_KATANA_REQUIRE_TRAINED_SCABBARD is set but no trained Scabbard "
                "artifacts were found; refusing to fall back to the rule-based 'minimal' "
                "profile. Run `katana artifacts setup` to install detection models, or "
                "unset the variable to accept degraded rule-based enforcement."
            )
        logger.warning(
            "No trained Scabbard artifacts found; falling back to the rule-based "
            "'minimal' profile. This fallback does NOT reliably block prompt "
            "injections. Run `katana artifacts setup` to install detection models, "
            "or set HERMES_KATANA_REQUIRE_TRAINED_SCABBARD=1 to fail closed instead."
        )
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

        Centroids are experimental. They are used only when ``centroid_path`` is
        supplied or experimental auto-discovery is enabled via environment.
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
