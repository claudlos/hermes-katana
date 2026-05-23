"""Adversarial semantic recall scanner using zvec vector index.

Supports two embedding models:
- Default: all-MiniLM-L6-v2 (384-dim, original zvec index)
- Contrastive: contrastive_zvec_v1 (128-dim, 3x smaller index, 2-3x faster queries)

Uses 145k translated adversarial prompts across 16 languages to
semantically match suspicious input against known attack patterns.

Usage:
    from hermes_katana.scanner.semantic_recall import detect_semantic
    findings = detect_semantic("ignore all your instructions")
"""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

__all__ = [
    "SemanticCategory",
    "SemanticFinding",
    "SemanticSeverity",
    "detect_semantic",
    "scan_semantic",
    "semantic_backend_status",
]


class SemanticCategory(str, Enum):
    SEMANTIC_MATCH = "semantic_match"


class SemanticSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class SemanticFinding:
    category: SemanticCategory
    severity: SemanticSeverity
    confidence: float
    matched_text: str
    position: tuple[int, int]
    description: str
    score: float = 0.0
    label: str = ""
    lang: str = ""
    source: str = ""


# Fast-path keywords (same as standalone)
_ATTACK_KW_RE = re.compile(
    "|".join(
        re.escape(kw)
        for kw in sorted(
            [
                "ignore all previous",
                "ignore prior",
                "ignore all instructions",
                "bypass safety",
                "bypass filter",
                "jailbreak me",
                "sql injection",
                "xss payload",
                "buffer overflow",
                "system prompt leak",
                "reveal your instructions",
                "you are now DAN",
                "act as DAN",
                "do anything now",
                "evil mode",
                "unfiltered mode",
                "developer mode enabled",
                "make a pipe bomb",
                "generate malware",
                "phishing email",
                "override all safety",
                "disregard all rules",
                "disregard all safety",
            ],
            key=len,
            reverse=True,
        )
    ),
    re.IGNORECASE,
)

THRESHOLD_HIGH = 0.70
THRESHOLD_ATTACK = 0.55
THRESHOLD_SUSPICIOUS = 0.55  # raised from 0.35 — standard model needs higher bar
THRESHOLD_STANDALONE_VECTOR = 0.95
_CORROBORATING_ATTACK_RE = re.compile(
    r"\b("
    r"ignore|disregard|override|bypass|jailbreak|developer\s+mode|system\s+prompt|"
    r"previous\s+instructions?|reveal|leak|exfiltrate|dump|print|send|upload|"
    r"api\s*keys?|secret|token|credential|password|private\s+key|environment\s+variables?|env\s+vars?"
    r")\b",
    re.I,
)
_STRONG_ATTACK_INTENT_RE = re.compile(
    r"("
    r"\b(?:ignore|disregard)\b.{0,40}\b(?:previous|prior|system|developer|instructions?)\b|"
    r"\boverride\b.{0,40}\b(?:safety|policy|system|developer|instructions?|guardrails?)\b|"
    r"\bbypass\b.{0,40}\b(?:safety|filter|policy|guardrails?|restrictions?)\b|"
    r"\b(?:jailbreak|developer\s+mode|system\s+prompt|previous\s+instructions?|reveal|leak|exfiltrate|dump|print|send|upload)\b"
    r")",
    re.I,
)
_DEFENSIVE_CONTEXT_RE = re.compile(
    r"\b("
    r"audit|finding|checklist|best\s+practices?|defen[cs]e|protection|hardening|"
    r"recommendation|remediation|validate|verify|reject|use\s+[A-Z0-9_-]+|"
    r"scan\s+results?|threat\s+model|vulnerabilit(?:y|ies)|upgrade|pattern|example|notes?|"
    r"function|method|class|parameter|default\s+behavior|code\s+snippet"
    r")\b",
    re.I,
)
_DEFENSIVE_SAFE_FAST_PATH_TERMS = {"sql injection", "xss payload", "buffer overflow"}

# Paths — support either full contrastive_zvec_v1 artifacts or the existing
# quantized zvec layout already used by Scabbard.
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = SCRIPT_DIR.parent.parent.parent / "training"
CONTRASTIVE_MODEL_DIR = TRAINING_DIR / "models" / "contrastive_zvec_v1"
SEMANTIC_INDEX_DIR = TRAINING_DIR / "data" / "contrastive_zvec_index"
ORIGINAL_INDEX_DIR = TRAINING_DIR / "data" / "translations_clean" / "zvec_semantic_index"
QUANTIZED_MODEL_DIR = TRAINING_DIR / "models" / "zvec_quantized-20260408T061203Z-3-001" / "zvec_quantized"


def _contrastive_backbone_dir() -> Path:
    return CONTRASTIVE_MODEL_DIR / "best" / "backbone"


def _contrastive_projector_path() -> Path:
    return CONTRASTIVE_MODEL_DIR / "best" / "projector.pt"


def _quantized_backbone_dir() -> Path:
    return QUANTIZED_MODEL_DIR / "backbone_fp32"


def _quantized_projector_path() -> Path:
    return QUANTIZED_MODEL_DIR / "projector_fp32.pt"


def _quantized_tokenizer_dir() -> Path:
    return QUANTIZED_MODEL_DIR / "tokenizer"


def _path_has_entries(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def _load_projector_checkpoint(path: Path) -> tuple[dict, int]:
    """Load and normalize supported projector checkpoint schemas.

    Deployed zvec/contrastive artifacts have historically used
    ``{"projector": state_dict, "embed": dim, "temp": ...}``, while newer
    training code may emit ``{"projector_state": state_dict, "embed_dim": dim}``.
    Normalize both forms here so readiness checks and runtime loading agree.
    """
    from hermes_katana.ml_artifacts import safe_torch_load

    state = safe_torch_load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Projector checkpoint {path} is not a dict")

    if "projector_state" in state and "embed_dim" in state:
        projector_state = state["projector_state"]
        embed_dim = state["embed_dim"]
    elif "projector" in state:
        projector_state = state["projector"]
        embed_dim = state.get("embed", 128)
    else:
        raise ValueError(f"Projector checkpoint {path} has unsupported schema")

    if not isinstance(projector_state, dict):
        raise ValueError(f"Projector checkpoint {path} has invalid projector weights")
    if not isinstance(embed_dim, int):
        raise ValueError(f"Projector checkpoint {path} has invalid embedding dimension")
    if projector_state and all(isinstance(key, str) for key in projector_state):
        keys = set(projector_state)
        if not any(key.startswith("net.") for key in keys) and any(key.startswith(("0.", "2.")) for key in keys):
            projector_state = {f"net.{key}": value for key, value in projector_state.items()}
    return projector_state, embed_dim


def _projector_checkpoint_loadable(path: Path) -> bool:
    try:
        _load_projector_checkpoint(path)
    except Exception:
        return False
    return True


def _contrastive_model_ready() -> bool:
    projector = _contrastive_projector_path()
    return (
        _path_has_entries(_contrastive_backbone_dir())
        and projector.is_file()
        and _projector_checkpoint_loadable(projector)
    )


def _quantized_model_ready() -> bool:
    return (
        _path_has_entries(_quantized_backbone_dir())
        and _quantized_projector_path().is_file()
        and _path_has_entries(_quantized_tokenizer_dir())
    )


def semantic_backend_status() -> dict[str, str | bool]:
    """Expose semantic backend readiness for debugging/eval reporting."""
    index_ready = _path_has_entries(SEMANTIC_INDEX_DIR)
    contrastive_model_ready = _contrastive_model_ready()
    quantized_model_ready = _quantized_model_ready()

    if index_ready and contrastive_model_ready:
        backend = "contrastive"
        reason = "contrastive artifacts available"
        active_index_dir = SEMANTIC_INDEX_DIR
        model_layout = "contrastive_zvec_v1"
    elif index_ready and quantized_model_ready:
        backend = "zvec_quantized"
        reason = "using existing zvec_quantized model with rebuilt semantic index"
        active_index_dir = SEMANTIC_INDEX_DIR
        model_layout = "zvec_quantized"
    else:
        backend = "minilm_fallback"
        model_layout = "fallback"
        missing = []
        if not index_ready:
            missing.append("semantic index")
        if not contrastive_model_ready and not quantized_model_ready:
            missing.append("compatible semantic model")
        elif not contrastive_model_ready and quantized_model_ready:
            missing.append("contrastive model (using quantized model once index exists)")
        reason = "missing " + ", ".join(missing) if missing else "semantic artifacts unavailable"
        active_index_dir = ORIGINAL_INDEX_DIR

    return {
        "backend": backend,
        "model_layout": model_layout,
        "contrastive_ready": bool(index_ready and contrastive_model_ready),
        "quantized_ready": bool(index_ready and quantized_model_ready),
        "reason": reason,
        "active_index_dir": str(active_index_dir),
        "semantic_index_dir": str(SEMANTIC_INDEX_DIR),
        "contrastive_model_dir": str(CONTRASTIVE_MODEL_DIR),
        "quantized_model_dir": str(QUANTIZED_MODEL_DIR),
        "original_index_dir": str(ORIGINAL_INDEX_DIR),
    }


_BACKEND_STATUS = semantic_backend_status()
_ZVEC_INDEX_DIR = _BACKEND_STATUS["active_index_dir"]
_USE_CONTRASTIVE = _BACKEND_STATUS["backend"] == "contrastive"
_USE_QUANTIZED_ZVEC = _BACKEND_STATUS["backend"] == "zvec_quantized"


def _classify(score: float) -> SemanticSeverity:
    if score >= THRESHOLD_HIGH:
        return SemanticSeverity.CRITICAL
    if score >= THRESHOLD_ATTACK:
        return SemanticSeverity.HIGH
    if score >= THRESHOLD_SUSPICIOUS:
        return SemanticSeverity.MEDIUM
    return SemanticSeverity.LOW


class _Lazy:
    """Process-level singleton."""

    collection = None
    model = None
    projector = None
    embedder = None
    _zvec = None
    _lock = threading.Lock()

    @classmethod
    def ensure(cls):
        if cls.collection is not None and (cls.model is not None or cls.embedder is not None):
            return

        with cls._lock:
            if cls.collection is None:
                if cls._zvec is None:
                    import zvec as _z

                    cls._zvec = _z
                cls.collection = cls._zvec.open(path=_ZVEC_INDEX_DIR)

            if cls.model is None and cls.embedder is None:
                if _USE_CONTRASTIVE:
                    from sentence_transformers import SentenceTransformer

                    # Load contrastive MiniLM + projection head.
                    # Force device="cpu" — SentenceTransformer's default
                    # device autodetect calls torch.cuda.is_available(),
                    # which can stall indefinitely if the host's NVIDIA
                    # driver/nvml is hung. Inference here is fast enough
                    # on CPU and we don't currently rely on GPU for it.
                    cls.model = SentenceTransformer(
                        str(_contrastive_backbone_dir()),
                        device="cpu",
                    )

                    # Load projector weights
                    import torch

                    projector_weights, embed_dim = _load_projector_checkpoint(_contrastive_projector_path())

                    class _ProjectionHead(torch.nn.Module):
                        def __init__(self, hidden_size=384, out=128):
                            super().__init__()
                            self.net = torch.nn.Sequential(
                                torch.nn.Linear(hidden_size, hidden_size),
                                torch.nn.GELU(),
                                torch.nn.Linear(hidden_size, out),
                            )

                        def forward(self, x):
                            return torch.nn.functional.normalize(self.net(x), p=2, dim=-1)

                    cls.projector = _ProjectionHead(out=embed_dim)
                    cls.projector.load_state_dict(projector_weights)
                    cls.projector.eval()
                    print(f"[semantic_recall] Loaded contrastive model (embed_dim={embed_dim})")
                elif _USE_QUANTIZED_ZVEC:
                    from hermes_katana.scabbard.embedder import ZvecEmbedder

                    cls.embedder = ZvecEmbedder(
                        model_path=str(_quantized_backbone_dir()),
                        projector_path=str(_quantized_projector_path()),
                        tokenizer_path=str(_quantized_tokenizer_dir()),
                        device="cpu",
                    )
                    cls.projector = None
                    print("[semantic_recall] Loaded zvec_quantized model (backbone_fp32 + projector_fp32)")
                else:
                    from sentence_transformers import SentenceTransformer

                    # Fallback to standard all-MiniLM-L6-v2
                    cls.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                    cls.projector = None
                    print(f"[semantic_recall] Loaded standard MiniLM-L6-v2 (fallback: {_BACKEND_STATUS['reason']})")


def _encode_text(text: str) -> List[float]:
    """Encode text using the active model (contrastive, zvec_quantized, or fallback)."""
    _Lazy.ensure()
    if _Lazy.embedder is not None:
        emb = _Lazy.embedder.encode(text)
        return [float(value) for value in emb.tolist()]

    import torch

    model = _Lazy.model
    if model is None:
        raise RuntimeError("Semantic recall model was not initialized")

    with torch.inference_mode():
        emb = model.encode([text])[0]
        if _Lazy.projector is not None:
            emb_t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            emb = _Lazy.projector(emb_t).squeeze(0).detach().cpu().numpy()
    return [float(value) for value in emb.tolist()]


def detect_semantic(
    text: str,
    topk: int = 5,
    min_score: float = THRESHOLD_SUSPICIOUS,
) -> List[SemanticFinding]:
    """Scan text for semantic similarity to known adversarial prompts."""
    findings: List[SemanticFinding] = []

    # Fast path
    kw = _ATTACK_KW_RE.search(text[:500])
    defensive_context = bool(_DEFENSIVE_CONTEXT_RE.search(text[:1000]))
    strong_attack_intent = bool(_STRONG_ATTACK_INTENT_RE.search(text[:1000]))
    kw_text = kw.group(0).lower() if kw else ""
    kw_is_defensive_safe = defensive_context and kw_text in _DEFENSIVE_SAFE_FAST_PATH_TERMS
    corroborated = bool(
        strong_attack_intent
        or (kw and not kw_is_defensive_safe)
        or (_CORROBORATING_ATTACK_RE.search(text[:1000]) and not defensive_context)
    )
    if kw and not kw_is_defensive_safe:
        t_lower = text.lower()
        start = t_lower.find(kw.group(0))
        if start < 0:
            start = 0
        findings.append(
            SemanticFinding(
                category=SemanticCategory.SEMANTIC_MATCH,
                severity=SemanticSeverity.CRITICAL,
                confidence=1.0,
                matched_text=kw.group(0),
                position=(start, start + len(kw.group(0))),
                description=f"Fast-path keyword match: {kw.group(0)}",
                score=1.0,
            )
        )

    if _BACKEND_STATUS["backend"] == "minilm_fallback":
        return findings

    # Vector search
    q_emb = _encode_text(text)
    collection = _Lazy.collection
    zvec_mod = _Lazy._zvec
    if collection is None or zvec_mod is None:
        raise RuntimeError("Semantic recall index was not initialized")

    docs = collection.query(
        zvec_mod.VectorQuery("embedding", vector=q_emb),  # type: ignore[attr-defined]
        topk=topk,
    )

    for doc in docs:
        s = float(getattr(doc, "score", 0.0) or 0.0)
        if s < min_score:
            continue
        if s < THRESHOLD_STANDALONE_VECTOR and not corroborated:
            continue
        f = doc.fields
        label = f.get("label", "")
        lang = f.get("lang", "")
        source = f.get("source", "")
        txt = f.get("text_en_short", "")[:100]

        findings.append(
            SemanticFinding(
                category=SemanticCategory.SEMANTIC_MATCH,
                severity=_classify(s),
                confidence=s,
                matched_text=txt,
                position=(0, len(text)),
                description=f"Semantic match: {txt[:80]} (label={label}, lang={lang}, src={source})",
                score=s,
                label=label,
                lang=lang,
                source=source,
            )
        )

    findings.sort(key=lambda x: x.confidence, reverse=True)
    return findings


def scan_semantic(text: str, topk: int = 5) -> list[dict]:
    """Return list of dicts for eval-suite compatibility."""
    findings = detect_semantic(text, topk=topk)
    return [
        {
            "scanner": "semantic_recall",
            "verdict": "deny" if f.confidence >= THRESHOLD_ATTACK else "warn",
            "confidence": f.confidence,
            "label": f.label,
            "lang": f.lang,
            "source": f.source,
            "description": f.description,
        }
        for f in findings
    ]
