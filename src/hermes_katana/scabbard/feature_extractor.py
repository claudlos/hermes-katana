"""Stage 2: Multi-Signal Feature Extraction.

Extracts heterogeneous detection signals from normalised text.
Each signal catches a different class of attack that the others miss.

All ML dependencies (numpy, torch, transformers) are lazily imported so the
module can be imported in a ``minimal`` deployment without any ML libraries.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    pass


def _np() -> Any:
    """Lazy import for numpy."""
    try:
        import numpy

        return numpy
    except ImportError as exc:
        raise ImportError("numpy is required for feature extraction.  Install with: pip install numpy") from exc


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeatureVector:
    """Combined feature vector from all signals."""

    text_embedding: Any = None  # 768-dim ndarray
    context_embedding: Any = None  # 768-dim ndarray
    intent_divergence: float = 0.0
    centroid_distances: Any = None  # 8-dim ndarray
    perplexity_features: Any = None  # 3-dim ndarray
    ngram_features: Any = None  # 20-dim ndarray
    encoding_flags: Any = None  # 5-dim ndarray
    retrieval_features: Any = None  # variable-dim ndarray

    def to_array(self) -> Any:
        """Concatenate all signals into a flat vector for the fusion classifier."""
        np = _np()
        parts: list[Any] = []
        if self.text_embedding is not None:
            parts.append(self.text_embedding)
        if self.context_embedding is not None:
            parts.append(self.context_embedding)
        parts.append(np.array([self.intent_divergence]))
        if self.centroid_distances is not None:
            parts.append(self.centroid_distances)
        if self.perplexity_features is not None:
            parts.append(self.perplexity_features)
        if self.ngram_features is not None:
            parts.append(self.ngram_features)
        if self.encoding_flags is not None:
            parts.append(self.encoding_flags)
        if self.retrieval_features is not None:
            parts.append(self.retrieval_features)
        return np.concatenate(parts)

    @property
    def dimension(self) -> int:
        return len(self.to_array())


# ---------------------------------------------------------------------------
# Signal A: Intent Divergence
# ---------------------------------------------------------------------------


class IntentDivergenceDetector:
    """Compare semantic intent of content vs agent context/system prompt.

    Low cosine similarity between content embedding and context embedding
    indicates the content is trying to redirect the agent to a different task.
    """

    @staticmethod
    def cosine_similarity(a: Any, b: Any) -> float:
        """Compute cosine similarity between two vectors."""
        if a is None or b is None:
            return 0.0
        np = _np()
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def compute(self, text_embedding: Any, context_embedding: Any) -> float:
        """Return similarity score.  Range: [-1, 1].  Low (<0.3) = suspicious."""
        return self.cosine_similarity(text_embedding, context_embedding)


# ---------------------------------------------------------------------------
# Signal B: Centroid Detector
# ---------------------------------------------------------------------------


class CentroidDetector:
    """Distance from text embedding to pre-computed attack-category centroids.

    Each centroid is the mean embedding of labelled attack examples for one
    category.  High similarity to an attack centroid signals that attack type.
    """

    # Must match keys in attack_centroids_128d.npz
    CATEGORIES: list[str] = [
        "content_injection",
        "semantic_manipulation",
        "cognitive_state_attack",
        "behavioral_control",
        "exfiltration_attempt",
        "jailbreak",
        "encoding_evasion",
        "persona_jailbreak",
    ]

    def __init__(self, centroids: Optional[dict[str, Any]] = None) -> None:
        self.centroids: dict[str, Any] = centroids or {}

    @classmethod
    def load(cls, path: str) -> CentroidDetector:
        """Load pre-computed centroids from ``.npz`` file."""
        np = _np()
        data = np.load(path)
        centroids = {}
        for name in cls.CATEGORIES:
            if name not in data:
                continue
            centroid = np.asarray(data[name], dtype=np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[name] = centroid
        return cls(centroids)

    def compute_distances(self, text_embedding: Any) -> Any:
        """Return cosine similarities to attack centroids, one per category."""
        np = _np()
        distances = np.zeros(len(self.CATEGORIES))
        for i, cat in enumerate(self.CATEGORIES):
            if cat in self.centroids:
                centroid = self.centroids[cat]
                dot = np.dot(text_embedding, centroid)
                norm_t = np.linalg.norm(text_embedding)
                norm_c = np.linalg.norm(centroid)
                if norm_t > 0 and norm_c > 0:
                    distances[i] = dot / (norm_t * norm_c)
        return distances


# ---------------------------------------------------------------------------
# Signal C: Perplexity Analyzer
# ---------------------------------------------------------------------------


class PerplexityAnalyzer:
    """Detect injection seams via windowed perplexity analysis.

    Legitimate content has smooth perplexity across windows.  An injection
    creates a sharp shift at the boundary where the injected instructions
    begin.
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        window_size: int = 50,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size

    def _heuristic_perplexity(self, text: str) -> float:
        """Fast heuristic approximation when no model is loaded."""
        if not text:
            return 0.0
        char_counts: dict[str, int] = {}
        for ch in text.lower():
            char_counts[ch] = char_counts.get(ch, 0) + 1
        total = len(text)
        entropy = -sum((count / total) * math.log2(count / total) for count in char_counts.values() if count > 0)
        words = text.split()
        if not words:
            return entropy * 10
        avg_word_len = sum(len(w) for w in words) / len(words)
        unique_ratio = len({w.lower() for w in words}) / len(words)
        return entropy * avg_word_len * unique_ratio * 10

    def compute_features(self, text: str) -> Any:
        """Return 3-dim array: ``[mean_perplexity, max_spike, variance]``."""
        np = _np()
        if not text or len(text) < 20:
            return np.zeros(3)

        words = text.split()
        if len(words) < self.window_size:
            ppl = self._heuristic_perplexity(text)
            return np.array([ppl, 0.0, 0.0])

        window_ppls: list[float] = []
        for i in range(0, len(words), self.window_size // 2):
            window = " ".join(words[i : i + self.window_size])
            if len(window) < 10:
                continue
            ppl = self._heuristic_perplexity(window)
            window_ppls.append(ppl)

        if not window_ppls:
            return np.zeros(3)

        ppls = np.array(window_ppls)
        mean_ppl = float(np.mean(ppls))
        max_spike = 0.0
        if len(ppls) > 1:
            diffs = np.abs(np.diff(ppls))
            max_spike = float(np.max(diffs))
        variance = float(np.var(ppls))
        return np.array([mean_ppl, max_spike, variance])


# ---------------------------------------------------------------------------
# Signal D: N-gram Feature Extractor
# ---------------------------------------------------------------------------


class NgramFeatureExtractor:
    """TF-IDF features over known injection n-grams.

    Fast, interpretable, catches known attack templates.
    """

    INJECTION_NGRAMS: list[str] = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard previous",
        "forget previous instructions",
        "system prompt",
        "reveal system prompt",
        "print system prompt",
        "you are now",
        "new instructions",
        "override instructions",
        "ignore the above",
        "do not follow",
        "instead of",
        "pretend you are",
        "act as if",
        "jailbreak",
        "dan mode",
        "developer mode",
        "ignore safety",
        "bypass filters",
        "ignore restrictions",
        "execute code",
        "run command",
        "access file",
        "send data to",
        "exfiltrate",
        "transmit to",
        "base64 decode",
        "eval(",
        "exec(",
    ]

    def __init__(self, vectorizer: Any = None) -> None:
        self.vectorizer = vectorizer
        self._patterns = [(ngram, re.compile(re.escape(ngram), re.IGNORECASE)) for ngram in self.INJECTION_NGRAMS]

    def compute_features(self, text: str) -> Any:
        """Return 20-dim feature array."""
        np = _np()
        if self.vectorizer is not None:
            features = self.vectorizer.transform([text]).toarray()[0]
            if len(features) > 20:
                indices = np.argsort(np.abs(features))[-20:]
                features = features[indices]
            return features

        # Fallback: binary n-gram matching
        text_lower = text.lower()
        features = np.zeros(min(len(self._patterns), 20))
        for i, (_ngram, pattern) in enumerate(self._patterns[:20]):
            if pattern.search(text_lower):
                features[i] = 1.0
        return features


# ---------------------------------------------------------------------------
# Encoding flags helper
# ---------------------------------------------------------------------------

_FLAG_KEYS: list[str] = [
    "base64_encoded",
    "hex_encoded",
    "homoglyphs",
    "invisible_chars",
    "whitespace_anomaly",
]


def flags_to_array(flags: dict[str, bool]) -> Any:
    """Convert normaliser flags dict to a fixed-size numeric array."""
    np = _np()
    return np.array([float(flags.get(k, False)) for k in _FLAG_KEYS])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Orchestrate all feature extraction signals."""

    def __init__(
        self,
        embedder: Any = None,
        centroid_detector: Optional[CentroidDetector] = None,
        perplexity_analyzer: Optional[PerplexityAnalyzer] = None,
        ngram_extractor: Optional[NgramFeatureExtractor] = None,
        retrieval_index: Any = None,
    ) -> None:
        self.embedder = embedder
        self.intent_detector = IntentDivergenceDetector()
        self.centroid_detector = centroid_detector or CentroidDetector()
        self.perplexity_analyzer = perplexity_analyzer or PerplexityAnalyzer()
        self.ngram_extractor = ngram_extractor or NgramFeatureExtractor()
        self.retrieval_index = retrieval_index
        self._context_cache: dict[str, Any] = {}
        # Embedding dimension: probe embedder, else fall back to centroid dim
        # to ensure zero-vector fallback matches centroid detector's expected dim
        if embedder is not None:
            self._embedding_dim: int = getattr(embedder, "embedding_dim", 768)
        elif self.centroid_detector.centroids:
            # Probe first centroid to get its dimension
            first = next(iter(self.centroid_detector.centroids.values()))
            self._embedding_dim = int(first.shape[0])
        else:
            self._embedding_dim = 768

    def _embed(self, text: str) -> Any:
        """Get embedding for text.  Uses embedder if available, else zero vector."""
        np = _np()
        if self.embedder is not None:
            return self.embedder.encode(text)
        return np.zeros(self._embedding_dim)

    def _get_context_embedding(self, context: str) -> Any:
        """Get or compute cached context embedding."""
        np = _np()
        if not context:
            return np.zeros(self._embedding_dim)
        if context not in self._context_cache:
            self._context_cache[context] = self._embed(context)
        return self._context_cache[context]

    def extract(
        self,
        text: str,
        context: str = "",
        flags: Optional[dict[str, bool]] = None,
    ) -> FeatureVector:
        """Extract all feature signals from normalised text."""
        np = _np()
        text_emb = self._embed(text)
        ctx_emb = self._get_context_embedding(context)
        divergence = self.intent_detector.compute(text_emb, ctx_emb)
        centroid_dists = self.centroid_detector.compute_distances(text_emb)
        ppl_features = self.perplexity_analyzer.compute_features(text)
        ngram_features = self.ngram_extractor.compute_features(text)
        encoding_flags = flags_to_array(flags) if flags else np.zeros(5)
        retrieval_features = None
        if self.retrieval_index is not None:
            try:
                computed = self.retrieval_index.compute_features(text)
                retrieval_features = computed.to_array() if hasattr(computed, "to_array") else computed
                retrieval_features = np.asarray(retrieval_features, dtype=np.float32)
            except Exception:
                retrieval_features = None

        return FeatureVector(
            text_embedding=text_emb,
            context_embedding=ctx_emb,
            intent_divergence=divergence,
            centroid_distances=centroid_dists,
            perplexity_features=ppl_features,
            ngram_features=ngram_features,
            encoding_flags=encoding_flags,
            retrieval_features=retrieval_features,
        )
