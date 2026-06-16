"""Cosine-similarity false-positive softener for the Scabbard classifier.

Hash-based :mod:`hermes_katana.scabbard.known_fps` only matches a benign text
*verbatim*; a model that rephrases its output each retry slips past it. This
module adds a *generalizing* softener: on a Scabbard BLOCK, embed the classified
text with a small ONNX sentence-encoder and soften the verdict only if the text
is cosine-close to a vetted benign exemplar
(:mod:`policies/scabbard_benign_exemplars.yaml`).

Design constraints (all deliberate):

* **Torch-free.** Uses ``onnxruntime`` + a HuggingFace tokenizer, the only ML
  stack present in the production gateway venv (no torch / sentence-transformers).
  The embedder is a general-purpose all-MiniLM-L6-v2 ONNX export, NOT the Scabbard
  zvec embedder — the zvec projector is trained to *cluster attacks by type*, so
  benign-vs-malicious security text sits close in that space and it is unsafe as
  an allowlist signal. A general semantic encoder keeps attacks far from benign
  documentation (empirically: max attack→exemplar cosine 0.46 vs benign FPs 0.6+).

* **Fail toward blocking.** If onnxruntime, the embedder artifact, or the
  tokenizer is unavailable, or anything raises, the softener is a no-op and the
  BLOCK stands. A broken softener never weakens the gate.

* **Threshold above the attack ceiling.** ``similarity_threshold`` (read from the
  exemplar YAML) must exceed the highest attack→exemplar similarity on the
  adversarial corpus. The evasion gate (``tests/smoke/evasion_gate.py``) and
  ``tests/smoke/test_similarity_allowlist_safety.py`` are the arbiters: every
  adversarial case must still BLOCK, and rephrased attacks must stay below
  threshold.

* **Never softens tainted content.** Callers pass ``origin``; untrusted tiers
  (tool output, retrieved/web content) are never similarity-softened — only
  assistant/user-authored arguments, which is where the structural FP lives.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

_ENV_EXEMPLARS = "KATANA_BENIGN_EXEMPLARS_PATH"
_ENV_EMBEDDER = "KATANA_SIM_EMBEDDER_DIR"
_ENV_DISABLE = "KATANA_SIM_ALLOWLIST_DISABLE"
_DEFAULT_RELATIVE = Path("policies") / "scabbard_benign_exemplars.yaml"
_DEFAULT_EMBEDDER_SUBDIR = "onnx_embedder_allMiniLM"
_DEFAULT_THRESHOLD = 0.62
_MAX_TOKENS = 256

# Origin tiers that must never be similarity-softened (untrusted provenance).
_UNTRUSTED_ORIGINS = frozenset({"tool_output", "tool_result", "retrieved", "rag", "web", "untrusted", "external"})


def is_untrusted_origin(origin: Optional[str]) -> bool:
    """Return True if ``origin`` denotes an untrusted provenance tier."""
    return bool(origin) and origin.lower() in _UNTRUSTED_ORIGINS


def _default_exemplars_path() -> Path:
    override = os.environ.get(_ENV_EXEMPLARS)
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    for ancestor in (here.parent.parent.parent.parent, here.parent.parent.parent):
        candidate = ancestor / _DEFAULT_RELATIVE
        if candidate.is_file():
            return candidate
    return here.parent.parent.parent / _DEFAULT_RELATIVE


def _default_embedder_dir() -> Optional[Path]:
    override = os.environ.get(_ENV_EMBEDDER)
    if override:
        return Path(override).expanduser()
    try:
        from hermes_katana.artifacts import default_artifact_cache_dir

        return default_artifact_cache_dir() / _DEFAULT_EMBEDDER_SUBDIR
    except Exception:
        return None


class _OnnxTextEmbedder:
    """Lazy, torch-free sentence embedder (mean-pooled, L2-normalized)."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._session = None
        self._tokenizer = None
        self._input_names: List[str] = []
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> bool:
        if self._session is not None and self._tokenizer is not None:
            return True
        with self._lock:
            if self._session is not None and self._tokenizer is not None:
                return True
            model_path = self._model_dir / "onnx" / "model.onnx"
            if not model_path.is_file():
                # Some exports place model.onnx at the top level.
                alt = self._model_dir / "model.onnx"
                model_path = alt if alt.is_file() else model_path
            if not model_path.is_file():
                return False
            import onnxruntime as ort  # noqa: PLC0415
            from transformers import AutoTokenizer  # noqa: PLC0415

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_dir))
            self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self._input_names = [i.name for i in self._session.get_inputs()]
        return True

    def encode(self, texts: List[str]) -> "np.ndarray":
        import numpy as np  # noqa: PLC0415

        enc = self._tokenizer(  # type: ignore[misc]
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_TOKENS,
            return_tensors="np",
        )
        feed = {n: enc[n].astype(np.int64) for n in self._input_names if n in enc}
        last_hidden = self._session.run(None, feed)[0]  # (B, T, H)
        mask = enc["attention_mask"].astype(np.float32)[..., None]
        summed = (last_hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        vecs = summed / counts
        norms = np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9, None)
        return (vecs / norms).astype(np.float32)


class SimilarityAllowlist:
    """Cosine-similarity benign exemplar allowlist. Fails closed (no soften)."""

    def __init__(self) -> None:
        self._threshold: float = _DEFAULT_THRESHOLD
        self._exemplars: List[str] = []
        self._exemplar_vecs: "Optional[np.ndarray]" = None
        self._embedder: Optional[_OnnxTextEmbedder] = None
        self._ready: Optional[bool] = None  # tri-state: None=unattempted
        self._lock = threading.Lock()
        self._load_exemplars()

    def _load_exemplars(self) -> None:
        path = _default_exemplars_path()
        if not path.is_file():
            return
        try:
            import yaml  # noqa: PLC0415

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return
        self._exemplars = [str(e) for e in (data.get("exemplars") or []) if str(e).strip()]
        thr = data.get("similarity_threshold")
        if isinstance(thr, (int, float)):
            self._threshold = float(thr)

    @property
    def threshold(self) -> float:
        return self._threshold

    def enabled(self) -> bool:
        if os.environ.get(_ENV_DISABLE):
            return False
        return bool(self._exemplars)

    def _ensure_ready(self) -> bool:
        """Load the embedder and embed exemplars once. Returns False on any failure."""
        if self._ready is not None:
            return self._ready
        with self._lock:
            if self._ready is not None:
                return self._ready
            self._ready = False
            try:
                if not self.enabled():
                    return False
                model_dir = _default_embedder_dir()
                if model_dir is None or not model_dir.is_dir():
                    return False
                embedder = _OnnxTextEmbedder(model_dir)
                if not embedder._ensure_loaded():
                    return False
                self._exemplar_vecs = embedder.encode(self._exemplars)
                self._embedder = embedder
                self._ready = True
            except Exception:
                self._ready = False
            return self._ready

    def match(self, text: str, origin: Optional[str] = None) -> Tuple[bool, float]:
        """Return ``(is_benign_fp, similarity)``.

        ``True`` only when the embedder is ready, ``origin`` is not untrusted,
        and the text's cosine similarity to the nearest vetted benign exemplar
        is ``>= similarity_threshold``. Any failure returns ``(False, 0.0)`` so
        the BLOCK stands.
        """
        if not text or not text.strip():
            return (False, 0.0)
        if is_untrusted_origin(origin):
            return (False, 0.0)
        if not self._ensure_ready():
            return (False, 0.0)
        try:
            import numpy as np  # noqa: PLC0415

            vec = self._embedder.encode([text])[0]  # type: ignore[union-attr]
            sims = self._exemplar_vecs @ vec  # type: ignore[operator]
            score = float(np.max(sims))
        except Exception:
            return (False, 0.0)
        return (score >= self._threshold, score)


@lru_cache(maxsize=1)
def _allowlist() -> SimilarityAllowlist:
    return SimilarityAllowlist()


def similarity_match(text: str, origin: Optional[str] = None) -> Tuple[bool, float]:
    """Module-level convenience: ``(is_benign_fp, similarity)`` for ``text``."""
    return _allowlist().match(text, origin=origin)


def reload_similarity_allowlist() -> int:
    """Force re-read of exemplars and re-embed. Returns the exemplar count."""
    _allowlist.cache_clear()
    al = _allowlist()
    return len(al._exemplars)


__all__ = [
    "SimilarityAllowlist",
    "is_untrusted_origin",
    "reload_similarity_allowlist",
    "similarity_match",
]
