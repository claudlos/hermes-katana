"""Scabbard: multi-signal prompt injection and agent trap classifier.

Stage 1 -- Normalizer:      Unicode NFKC, homoglyph folding, invisible char
                             stripping, encoding detection (base64/hex/URL),
                             HTML entity decoding.

Stage 2 -- FeatureExtractor: Intent divergence, centroid distances, perplexity
                             analysis, n-gram TF-IDF, encoding flags.

Stage 3 -- FusionClassifier: XGBoost / sklearn MLP / rule-based fallback over
                             the heterogeneous feature vector.

Public API: :class:`ScabbardClassifier`, :class:`ScabbardConfig`,
            and :class:`ScabbardCascadeRouter`.
"""

from hermes_katana.scabbard.cascade import CascadeResult, ScabbardCascadeRouter
from hermes_katana.scabbard.config import ScabbardConfig
from hermes_katana.scabbard.fusion import (
    ATTACK_LABELS,
    ClassificationResult,
    Decision,
    FusionClassifier,
)
from hermes_katana.scabbard.normalizer import NormalizedResult, normalize
from hermes_katana.scabbard.known_fps import is_known_fp, reload_known_fps, text_hash
from hermes_katana.scabbard.similarity_allowlist import (
    SimilarityAllowlist,
    reload_similarity_allowlist,
    similarity_match,
)
from hermes_katana.scabbard.routing import (
    RouteKind,
    RouteMode,
    RoutedText,
    ScabbardRouteDecision,
    extract_scabbard_arg_texts,
    extract_scabbard_output_texts,
    has_scabbard_adversarial_signal,
    normalize_route_mode,
    should_scabbard_scan_arg,
)
from hermes_katana.scabbard.scabbard import ScabbardClassifier

__all__ = [
    "ATTACK_LABELS",
    "CascadeResult",
    "ClassificationResult",
    "Decision",
    "FusionClassifier",
    "NormalizedResult",
    "RouteKind",
    "RouteMode",
    "RoutedText",
    "ScabbardCascadeRouter",
    "ScabbardClassifier",
    "ScabbardConfig",
    "ScabbardRouteDecision",
    "extract_scabbard_arg_texts",
    "extract_scabbard_output_texts",
    "has_scabbard_adversarial_signal",
    "is_known_fp",
    "reload_known_fps",
    "text_hash",
    "SimilarityAllowlist",
    "reload_similarity_allowlist",
    "similarity_match",
    "normalize",
    "normalize_route_mode",
    "should_scabbard_scan_arg",
]
