"""RAG-style retrieval fusion: FAISS index over attack seed phrases.

Builds a FAISS index from the labelled attack seed phrase corpus and retrieves
top-K nearest neighbours for each input, feeding similarity scores back as
additional features for the fusion classifier.

Lazy imports are used throughout so this module can be imported without
installing faiss or sentence-transformers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Lazy third-party loaders
# ---------------------------------------------------------------------------


def _faiss() -> Any:
    """Lazy import for faiss."""
    try:
        import faiss

        return faiss
    except ImportError as exc:
        raise ImportError("faiss is required for retrieval.  Install with: pip install faiss-cpu") from exc


def _st() -> Any:
    """Lazy import for sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for retrieval.  Install with: pip install sentence-transformers"
        ) from exc


def _np() -> Any:
    """Lazy import for numpy."""
    try:
        import numpy as np

        return np
    except ImportError as exc:
        raise ImportError("numpy is required for retrieval.  Install with: pip install numpy") from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED_PHRASES_PATH = Path(__file__).parent / "data" / "attack_seed_phrases.json"
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim embeddings
DEFAULT_K = 5
INDEX_CACHE_PATH = Path(__file__).parent / "data" / "attack_seed_faiss.index"
PHRASES_CACHE_PATH = Path(__file__).parent / "data" / "attack_seed_phrases_indexed.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Result of a top-K retrieval query."""

    phrases: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    category: str = ""


@dataclass
class RetrievalFeatures:
    """Per-category retrieval similarity features.

    Attributes
    ----------
    max_similarities : list[float]
        For each category, the cosine similarity of the closest matching seed phrase.
    mean_similarities : list[float]
        For each category, the mean cosine similarity across top-K matches.
    topk_scores : list[float]
        Flattened list: for each category, the top-K scores concatenated.
        Length = num_categories * k.
    """

    max_similarities: list[float] = field(default_factory=list)
    mean_similarities: list[float] = field(default_factory=list)
    topk_scores: list[float] = field(default_factory=list)

    def to_array(self) -> Any:
        """Concatenate all retrieval features into a flat vector."""
        np = _np()
        parts = list(self.max_similarities)
        parts.extend(self.mean_similarities)
        parts.extend(self.topk_scores)
        return np.array(parts, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return len(self.to_array())


# ---------------------------------------------------------------------------
# RetrievalIndex
# ---------------------------------------------------------------------------


class RetrievalIndex:
    """FAISS index over attack seed phrases.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model name.  Defaults to ``all-MiniLM-L6-v2``.
    k : int
        Number of top neighbours to retrieve per query.  Defaults to 5.
    index_path : str | Path, optional
        Path to a pre-built FAISS index.  If not provided, the default cache
        path is used.
    phrases_path : str | Path, optional
        Path to the seed phrases JSON.  If not provided, the packaged data file
        is used.
    """

    CATEGORIES: list[str] = [
        "content_injection",
        "semantic_manipulation",
        "behavioral_control",
        "exfiltration_attempt",
        "jailbreak",
        "cognitive_state_attack",
    ]

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        k: int = DEFAULT_K,
        index_path: Optional[str | Path] = None,
        phrases_path: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self.k = k
        self._index: Any = None
        self._model: Any = None
        self._phrases: list[str] = []
        self._phrase_categories: list[str] = []
        self._category_to_indices: dict[str, list[int]] = {}

        self._index_path = Path(index_path) if index_path else INDEX_CACHE_PATH
        self._phrases_path = Path(phrases_path) if phrases_path else phrases_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, phrases_path: Optional[str | Path] = None) -> None:
        """Build the FAISS index from the seed phrases JSON.

        Parameters
        ----------
        phrases_path : str | Path, optional
            Path to ``attack_seed_phrases.json``.  Falls back to the packaged
            data file.
        """
        faiss_lib = _faiss()
        SentenceTransformer = _st()
        np = _np()

        path = Path(phrases_path) if phrases_path else self._phrases_path or SEED_PHRASES_PATH

        with open(path, encoding="utf-8") as fh:
            data: dict[str, list[str]] = json.load(fh)

        all_phrases: list[str] = []
        phrase_categories: list[str] = []
        category_to_indices: dict[str, list[int]] = {cat: [] for cat in self.CATEGORIES}

        for cat in self.CATEGORIES:
            phrases = data.get(cat, [])
            for phrase in phrases:
                idx = len(all_phrases)
                all_phrases.append(phrase)
                phrase_categories.append(cat)
                category_to_indices[cat].append(idx)

        self._phrases = all_phrases
        self._phrase_categories = phrase_categories
        self._category_to_indices = category_to_indices

        model = SentenceTransformer(self.model_name)
        embeddings = model.encode(all_phrases, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss_lib.IndexFlatIP(dim)
        faiss_lib.normalize_L2(embeddings)
        self._index.add(embeddings)

    def load(self, index_path: Optional[str | Path] = None) -> bool:
        """Load a pre-built FAISS index and phrase mapping.

        Returns ``True`` if the index was loaded, ``False`` if the file was not
        found.
        """
        faiss_lib = _faiss()
        _np()

        path = Path(index_path) if index_path else self._index_path
        if not path.exists():
            return False

        self._index = faiss_lib.read_index(str(path))

        phrases_cache = path.with_suffix(".phrases.json")
        if phrases_cache.exists():
            with open(phrases_cache, encoding="utf-8") as fh:
                cache = json.load(fh)
            self._phrases = cache.get("phrases", [])
            self._phrase_categories = cache.get("categories", [])
            self._category_to_indices = cache.get("cat_to_idx", {})
        else:
            # Rebuild mapping from the original phrases JSON
            self._rebuild_category_mapping()

        return True

    def save(self, index_path: Optional[str | Path] = None) -> None:
        """Persist the FAISS index and phrase mapping to disk."""
        faiss_lib = _faiss()

        path = Path(index_path) if index_path else self._index_path
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_lib.write_index(self._index, str(path))

        phrases_cache = path.with_suffix(".phrases.json")
        with open(phrases_cache, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "phrases": self._phrases,
                    "categories": self._phrase_categories,
                    "cat_to_idx": self._category_to_indices,
                },
                fh,
            )

    def retrieve(self, text: str, k: Optional[int] = None) -> list[RetrievalResult]:
        """Retrieve top-K nearest seed phrases for ``text``.

        Parameters
        ----------
        text : str
            Input text to encode and search.
        k : int, optional
            Override the default number of neighbours.

        Returns
        -------
        list[RetrievalResult]
            One result per category, ordered by ``CATEGORIES``, containing the
            top-K phrases and their cosine similarities.
        """
        if self._index is None:
            raise RuntimeError("Index is not built or loaded.  Call build() or load() first.")

        SentenceTransformer = _st()
        np = _np()
        k = k if k is not None else self.k

        model = SentenceTransformer(self.model_name)
        vec = model.encode([text], convert_to_numpy=True).astype(np.float32)
        np.linalg.norm(vec)
        _np().linalg.norm(vec)
        _np().linalg.norm(vec.squeeze())
        vec_norm = vec / np.linalg.norm(vec)

        scores, indices = self._index.search(vec_norm, k * len(self.CATEGORIES))

        results: list[RetrievalResult] = []
        for cat in self.CATEGORIES:
            cat_indices = self._category_to_indices.get(cat, [])
            if not cat_indices:
                results.append(RetrievalResult(category=cat))
                continue

            cat_scores: list[tuple[int, float]] = []
            for idx in indices[0]:
                if idx in cat_indices:
                    pos = list(indices[0]).index(idx)
                    cat_scores.append((idx, float(scores[0][pos])))
                if len(cat_scores) >= k:
                    break

            phrases = [self._phrases[idx] for idx, _ in cat_scores]
            scos = [score for _, score in cat_scores]
            results.append(RetrievalResult(phrases=phrases, scores=scos, category=cat))

        return results

    def compute_features(self, text: str, k: Optional[int] = None) -> RetrievalFeatures:
        """Compute retrieval features for fusion classifier.

        Parameters
        ----------
        text : str
            Input text.
        k : int, optional
            Override default neighbour count.

        Returns
        -------
        RetrievalFeatures
            ``max_similarities`` (1 per category), ``mean_similarities`` (1 per
            category), and flattened ``topk_scores``.
        """
        k = k if k is not None else self.k
        results = self.retrieve(text, k=k)

        max_sims: list[float] = []
        mean_sims: list[float] = []
        topk_flat: list[float] = []

        for res in results:
            if res.scores:
                max_sims.append(max(res.scores))
                mean_sims.append(sum(res.scores) / len(res.scores))
                topk_flat.extend(res.scores)
            else:
                max_sims.append(0.0)
                mean_sims.append(0.0)
                topk_flat.extend([0.0] * k)

        return RetrievalFeatures(
            max_similarities=max_sims,
            mean_similarities=mean_sims,
            topk_scores=topk_flat,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rebuild_category_mapping(self) -> None:
        """Rebuild category→indices mapping from the phrases list."""
        self._category_to_indices = {cat: [] for cat in self.CATEGORIES}
        for idx, phrase in enumerate(self._phrases):
            for cat in self.CATEGORIES:
                if phrase in self._get_seed_data().get(cat, []):
                    self._category_to_indices[cat].append(idx)

    def _get_seed_data(self) -> dict[str, list[str]]:
        with open(SEED_PHRASES_PATH, encoding="utf-8") as fh:
            return cast(dict[str, list[str]], json.load(fh))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def load_or_build_index(
    model_name: str = DEFAULT_MODEL,
    k: int = DEFAULT_K,
) -> RetrievalIndex:
    """Attempt to load a cached index; fall back to building one.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model name.
    k : int
        Number of neighbours per retrieval.

    Returns
    -------
    RetrievalIndex
        Loaded or freshly-built index.
    """
    idx = RetrievalIndex(model_name=model_name, k=k)
    if idx.load():
        return idx
    idx.build()
    try:
        idx.save()
    except OSError:
        pass  # Read-only filesystem — not fatal
    return idx
