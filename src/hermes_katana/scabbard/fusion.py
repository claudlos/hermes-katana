"""Stage 3: Fusion Classifier.

Takes the multi-signal feature vector and produces multi-label attack
probabilities.  Uses XGBoost or CatBoost (primary) or sklearn MLP (fallback).

The key insight: no single signal catches everything.  The fusion layer
learns which signal combinations indicate real attacks vs false positives.

All ML dependencies are lazily imported so the rule-based fallback works
without any ML libraries (``minimal`` profile).

New in M23:
- L2-normalized 256-dim projection head output (replaces raw [CLS] token)
- Mahalanobis distance for centroid features (not Euclidean)
- CatBoost as alternative to XGBoost (handles categorical encoding flags natively)
- Stacking with cross-validated OOF predictions
- Feature importance analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


def _np() -> Any:
    """Lazy import for numpy."""
    try:
        import numpy

        return numpy
    except ImportError as exc:
        raise ImportError("numpy is required for fusion classification.  Install with: pip install numpy") from exc


class Decision(str, Enum):
    """Scabbard verdict."""

    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


ATTACK_LABELS: list[str] = [
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


# ---------------------------------------------------------------------------
# Projection Head (M23: L2-normalized 256-dim)
# ---------------------------------------------------------------------------


class ProjectionHead:
    """L2-normalized 256-dim projection head for embedding compression.

    Replaces raw 768-dim [CLS] token with a learnable projection that:
    - Compresses to 256 dimensions (fewer parameters, less overfitting)
    - L2-normalizes output (cosine similarity becomes dot product)
    - Is usable as a drop-in feature extractor
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 256) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._weights: Optional[Any] = None
        self._bias: Optional[Any] = None

    def _init_weights(self) -> None:
        """Lazy initialization of projection weights (Xavier uniform)."""
        np = _np()
        # Xavier uniform for stable training
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self._weights = np.random.uniform(-scale, scale, (self.input_dim, self.output_dim)).astype(np.float32)
        self._bias = np.zeros(self.output_dim, dtype=np.float32)

    def project(self, embedding: Any) -> Any:
        """Project and L2-normalize an embedding.

        Args:
            embedding: 1D array of shape (input_dim,)

        Returns:
            L2-normalized output_dim projection
        """
        np = _np()
        if self._weights is None:
            self._init_weights()

        # Linear projection
        projected = np.dot(embedding, self._weights) + self._bias

        # L2 normalize
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm
        else:
            projected = np.zeros(self.output_dim, dtype=np.float32)

        return projected

    def project_batch(self, embeddings: Any) -> Any:
        """Project and L2-normalize a batch of embeddings.

        Args:
            embeddings: 2D array of shape (batch, input_dim)

        Returns:
            L2-normalized array of shape (batch, output_dim)
        """
        np = _np()
        if self._weights is None:
            self._init_weights()

        projected = np.dot(embeddings, self._weights) + self._bias

        # L2 normalize each row
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        projected = projected / norms

        return projected


# ---------------------------------------------------------------------------
# Mahalanobis Centroid Features (M23)
# ---------------------------------------------------------------------------


class MahalanobisCentroidDetector:
    """Centroid detector using Mahalanobis distance instead of cosine similarity.

    Mahalanobis distance accounts for feature covariance, making it more robust
    than Euclidean when features have different scales or correlations.
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

    def __init__(self, centroids: Optional[dict[str, Any]] = None, covariance: Optional[dict[str, Any]] = None) -> None:
        self.centroids: dict[str, Any] = centroids or {}
        self.covariance: dict[str, Any] = covariance or {}
        self._inv_cov: dict[str, Any] = {}
        self._fitted: bool = False

    def _compute_inverse_covariance(self) -> None:
        """Pre-compute inverse covariance matrices for each category."""
        if self._fitted:
            return
        np = _np()
        for cat in self.CATEGORIES:
            if cat in self.covariance and cat in self.centroids:
                cov = self.covariance[cat]
                try:
                    # Regularized inverse: add small identity to ensure invertibility
                    reg = 1e-6 * np.eye(cov.shape[0])
                    self._inv_cov[cat] = np.linalg.inv(cov + reg)
                except np.linalg.LinAlgError:
                    # Fallback: use pseudo-inverse
                    self._inv_cov[cat] = np.linalg.pinv(cov)
        self._fitted = True

    def compute_distances(self, text_embedding: Any) -> Any:
        """Return one Mahalanobis distance per attack centroid.

        Args:
            text_embedding: 1D array of shape (input_dim,)

        Returns:
            Array of Mahalanobis distances (lower = closer to centroid = more suspicious).
        """
        np = _np()
        self._compute_inverse_covariance()

        distances = np.zeros(len(self.CATEGORIES))
        for i, cat in enumerate(self.CATEGORIES):
            if cat not in self.centroids:
                distances[i] = 0.0
                continue

            centroid = self.centroids[cat]
            diff = text_embedding - centroid

            if cat in self._inv_cov:
                # Mahalanobis: sqrt((x-mu)^T * Sigma^-1 * (x-mu))
                diff_col = diff.reshape(-1, 1)
                mahal = np.dot(np.dot(diff.T, self._inv_cov[cat]), diff_col)
                distances[i] = float(np.sqrt(max(0.0, mahal)))
            else:
                # Fallback: normalized Euclidean distance
                norm_diff = np.linalg.norm(diff)
                norm_centroid = np.linalg.norm(centroid)
                if norm_centroid > 0:
                    distances[i] = norm_diff / norm_centroid
                else:
                    distances[i] = norm_diff

        return distances


# ---------------------------------------------------------------------------
# Stacking Ensemble (M23)
# ---------------------------------------------------------------------------


class StackingEnsemble:
    """Stacking ensemble with cross-validated OOF predictions.

    Level 0: XGBoost, CatBoost, sklearn MLP
    Level 1: Logistic Regression meta-learner

    Uses stratified K-fold cross-validation to generate out-of-fold predictions
    that train the meta-learner without data leakage.
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_folds = n_folds
        self.random_state = random_state
        self.level0_models: list[Any] = []
        self.meta_model: Any = None
        self._fitted: bool = False

    def _get_base_models(self) -> list[Any]:
        """Instantiate level-0 base models."""
        models = []

        # XGBoost
        try:
            import xgboost as xgb

            models.append(
                (
                    "xgb",
                    xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        use_label_encoder=False,
                        eval_metric="logloss",
                    ),
                )
            )
        except ImportError:
            pass

        # CatBoost
        try:
            from catboost import CatBoostClassifier

            models.append(
                (
                    "catboost",
                    CatBoostClassifier(
                        iterations=100,
                        depth=4,
                        learning_rate=0.1,
                        random_state=self.random_state,
                        verbose=False,
                    ),
                )
            )
        except ImportError:
            pass

        # Sklearn MLP
        try:
            from sklearn.neural_network import MLPClassifier

            models.append(
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=300,
                        random_state=self.random_state,
                        early_stopping=True,
                    ),
                )
            )
        except ImportError:
            pass

        return models

    def fit(self, X: Any, y: Any) -> StackingEnsemble:
        """Fit the stacking ensemble using cross-validated OOF predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Label array of shape (n_samples,)

        Returns:
            self
        """
        np = _np()
        n_samples = X.shape[0]
        n_labels = len(ATTACK_LABELS)

        self.level0_models = self._get_base_models()
        if not self.level0_models:
            self._fitted = False
            return self

        # Initialize OOF prediction matrix
        n_models = len(self.level0_models)
        oof_predictions = np.zeros((n_samples, n_labels * n_models))

        # Stratified K-fold
        from sklearn.model_selection import StratifiedKFold

        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            for j, (name, model) in enumerate(self.level0_models):
                # Clone model for this fold
                try:
                    from sklearn.base import clone

                    fold_model = clone(model)
                except Exception:
                    continue

                # Fit on training fold
                try:
                    fold_model.fit(X_train, y_train)
                    probs = fold_model.predict_proba(X_val)

                    # Handle multi-class vs binary
                    if probs.shape[1] != n_labels:
                        # Map to all labels
                        full_probs = np.zeros((len(val_idx), n_labels))
                        for idx, label_idx in enumerate(fold_model.classes_):
                            if label_idx < n_labels:
                                full_probs[:, label_idx] = probs[:, idx]
                        probs = full_probs

                    oof_predictions[val_idx, j * n_labels : (j + 1) * n_labels] = probs
                except Exception:
                    continue

        # Train meta-model on OOF predictions
        try:
            from sklearn.linear_model import LogisticRegression

            self.meta_model = LogisticRegression(max_iter=500, random_state=self.random_state)
            self.meta_model.fit(oof_predictions, y)
            self._fitted = True
        except Exception:
            self._fitted = False

        return self

    def predict_proba(self, X: Any) -> Any:
        """Predict using the stacking ensemble.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Probability matrix of shape (n_samples, n_labels)
        """
        np = _np()
        if not self._fitted or not self.level0_models:
            return np.zeros((X.shape[0], len(ATTACK_LABELS)))

        n_labels = len(ATTACK_LABELS)
        n_models = len(self.level0_models)
        stacked = np.zeros((X.shape[0], n_labels * n_models))

        for j, (name, model) in enumerate(self.level0_models):
            try:
                probs = model.predict_proba(X)
                if probs.shape[1] != n_labels:
                    full_probs = np.zeros((X.shape[0], n_labels))
                    for idx, label_idx in enumerate(model.classes_):
                        if label_idx < n_labels:
                            full_probs[:, label_idx] = probs[:, idx]
                    probs = full_probs
                stacked[:, j * n_labels : (j + 1) * n_labels] = probs
            except Exception:
                continue

        return self.meta_model.predict_proba(stacked)

    def get_feature_importance(self) -> dict[str, float]:
        """Get aggregated feature importance from base models.

        Returns:
            Dict mapping feature names to importance scores
        """
        importance: dict[str, float] = {}

        for name, model in self.level0_models:
            if hasattr(model, "feature_importances_"):
                # XGBoost / CatBoost
                imp = model.feature_importances_
                importance[name] = float(_np().mean(imp))
            elif hasattr(model, "coef_"):
                # Logistic regression / MLP
                imp = _np().abs(model.coef_)
                importance[name] = float(_np().mean(imp))

        return importance


# ---------------------------------------------------------------------------
# Feature Importance Analyzer (M23)
# ---------------------------------------------------------------------------


class FeatureImportanceAnalyzer:
    """Analyze and rank feature importance for the fusion classifier.

    Supports multiple methods:
    - Permutation importance (model-agnostic)
    - Built-in feature importances (XGBoost/CatBoost)
    - Correlation analysis with targets
    """

    def __init__(self, n_permutations: int = 10, random_state: int = 42) -> None:
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.feature_names: list[str] = []
        self._importance_scores: Optional[dict[str, float]] = None

    def _get_default_feature_names(self, n_features: int) -> list[str]:
        """Generate default feature names based on feature vector layout.

        Feature vector layout (from FeatureVector.to_array()):

        ========== ===================================
        Indices    Content
        ========== ===================================
        0:256      text_projection (L2-normed 256-dim)
        256:512    context_projection (L2-normed 256-dim)
        512        intent_divergence
        513:521    mahalanobis_centroid_distances (8)
        521:524    perplexity_features (3)
        524:544    ngram_features (20)
        544:549    encoding_flags (5)
        ========== ===================================
        """
        names = []
        names.extend([f"text_proj_{i}" for i in range(256)])
        names.extend([f"context_proj_{i}" for i in range(256)])
        names.append("intent_divergence")
        names.extend([f"centroid_{cat}" for cat in MahalanobisCentroidDetector.CATEGORIES])
        names.extend(["ppl_mean", "ppl_max_spike", "ppl_variance"])
        names.extend([f"ngram_{i}" for i in range(20)])
        names.extend(["flag_base64", "flag_hex", "flag_homoglyph", "flag_invisible", "flag_whitespace"])
        return names[:n_features]

    def analyze(
        self,
        model: Any,
        X: Any,
        y: Any,
        method: str = "permutation",
    ) -> dict[str, float]:
        """Analyze feature importance.

        Args:
            model: Fitted classifier
            X: Feature matrix
            y: Labels
            method: 'permutation', 'builtin', or 'correlation'

        Returns:
            Dict mapping feature names to importance scores
        """
        _np()
        n_features = X.shape[1]
        self.feature_names = self._get_default_feature_names(n_features)

        if method == "permutation":
            scores = self._permutation_importance(model, X, y)
        elif method == "builtin":
            scores = self._builtin_importance(model)
        elif method == "correlation":
            scores = self._correlation_importance(X, y)
        else:
            scores = {}

        self._importance_scores = scores
        return scores

    def _permutation_importance(self, model: Any, X: Any, y: Any) -> dict[str, float]:
        """Compute permutation importance (model-agnostic)."""
        np = _np()
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        try:
            model.fit(X_train, y_train)
            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=self.n_permutations,
                random_state=self.random_state,
                n_jobs=-1,
            )
            importances = result.importances_mean
        except Exception:
            importances = np.zeros(len(self.feature_names))

        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def _builtin_importance(self, model: Any) -> dict[str, float]:
        """Use model built-in feature importances."""
        np = _np()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            return {}

        return {name: float(imp) for name, imp in zip(self.feature_names, importances)}

    def _correlation_importance(self, X: Any, y: Any) -> dict[str, float]:
        """Correlation-based importance: absolute Pearson correlation with target."""
        np = _np()
        scores = {}
        for i, name in enumerate(self.feature_names):
            if i < X.shape[1]:
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                scores[name] = float(corr) if not np.isnan(corr) else 0.0
            else:
                scores[name] = 0.0
        return scores

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self._importance_scores is None:
            return []
        sorted_items = sorted(self._importance_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ClassificationResult:
    """Output of the fusion classifier."""

    scores: dict[str, float]
    decision: Decision
    top_category: str
    confidence: float
    feature_importance: Optional[dict[str, float]] = None
    # Set when this result was produced by a weaker path than the deployment
    # configured (trained classifier missing/raised, or classify timed out).
    # Consumers can fail closed on it instead of trusting a silent downgrade.
    degraded: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "scores": self.scores,
            "decision": self.decision.value,
            "top_category": self.top_category,
            "confidence": self.confidence,
        }
        if self.degraded:
            result["degraded"] = self.degraded
        return result

    def to_risk_report(self, source: str = "", content_type: str = "") -> dict[str, Any]:
        """Format as a risk report for the LLM judge (Stage 4)."""
        flags = [
            {"type": label, "score": score} for label, score in self.scores.items() if label != "clean" and score > 0.2
        ]
        return {
            "source": source,
            "content_type": content_type,
            "flags": flags,
            "decision": self.decision.value,
            "top_category": self.top_category,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Fusion Classifier
# ---------------------------------------------------------------------------


class FusionClassifier:
    """Multi-label fusion classifier over heterogeneous feature signals.

    Supports three backends:

    - **XGBoost**: fast, interpretable, handles mixed features well.
    - **CatBoost**: alternative to XGBoost, handles categorical encoding
      flags natively, often more robust.
    - **sklearn MLP**: can be trained end-to-end; fallback if neither
      XGBoost nor CatBoost is available.
    - **Stacking Ensemble (M23)**: combines XGBoost, CatBoost, and MLP
      with a logistic regression meta-learner.

    A rule-based fallback is provided for deployments without trained models.

    New in M23:
    - L2-normalized 256-dim projection head for text/context embeddings
    - Mahalanobis distance for centroid features
    - CatBoost as alternative backend
    - Stacking ensemble with OOF predictions
    - Feature importance analysis
    """

    def __init__(
        self,
        model: Any = None,
        thresholds: Optional[dict[str, float]] = None,
        backend: str = "xgboost",
        use_projection_head: bool = True,
        use_mahalanobis: bool = True,
        centroid_detector: Optional[MahalanobisCentroidDetector] = None,
        projection_head: Optional[ProjectionHead] = None,
        feature_importance: Optional[dict[str, float]] = None,
    ) -> None:
        self.model = model
        self.thresholds = thresholds or {"allow": 0.3, "block": 0.7}
        self.backend = backend
        self.use_projection_head = use_projection_head
        self.use_mahalanobis = use_mahalanobis
        self.centroid_detector = centroid_detector
        self.projection_head = projection_head or ProjectionHead()
        self.feature_importance = feature_importance
        self._stacking_model: Optional[StackingEnsemble] = None

    @classmethod
    def load(cls, path: str) -> FusionClassifier:
        """Load trained model from file."""
        fpath = Path(path)
        if fpath.suffix == ".json":
            try:
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(fpath))
                return cls(model=model, backend="xgboost")
            except (ImportError, OSError, Exception):
                pass
        elif fpath.suffix in (".pkl", ".joblib"):
            try:
                from hermes_katana.ml_artifacts import UnsafeArtifactError, safe_joblib_load, safe_pickle_load

                if fpath.suffix == ".joblib":
                    model = safe_joblib_load(fpath)
                else:
                    model = safe_pickle_load(fpath)
                return cls(model=model)
            except UnsafeArtifactError as exc:
                logger.warning("Refusing to load untrusted fusion model %s: %s", fpath, exc)
                return cls()
            except ImportError:
                logger.warning("Could not import ML artifact loader dependencies for %s", fpath)
                return cls()
        return cls()

    def set_centroid_detector(self, detector: MahalanobisCentroidDetector) -> None:
        """Set the Mahalanobis centroid detector for distance computation."""
        self.centroid_detector = detector

    # ------------------------------------------------------------------
    # Projection head methods (M23)
    # ------------------------------------------------------------------

    def project_text_embedding(self, embedding: Any) -> Any:
        """Project text embedding through L2-normalized projection head.

        Args:
            embedding: input_dim raw text embedding

        Returns:
            output_dim L2-normalized projection
        """
        if self.use_projection_head:
            return self.projection_head.project(embedding)
        return embedding

    def project_context_embedding(self, embedding: Any) -> Any:
        """Project context embedding through L2-normalized projection head.

        Args:
            embedding: input_dim raw context embedding

        Returns:
            output_dim L2-normalized projection
        """
        if self.use_projection_head:
            return self.projection_head.project(embedding)
        return embedding

    # ------------------------------------------------------------------
    # Mahalanobis centroid distances (M23)
    # ------------------------------------------------------------------

    def compute_mahalanobis_distances(self, text_embedding: Any) -> Any:
        """Compute Mahalanobis distances to attack centroids.

        Args:
            text_embedding: input_dim text embedding

        Returns:
            One distance per centroid category.
        """
        if self.use_mahalanobis and self.centroid_detector is not None:
            return self.centroid_detector.compute_distances(text_embedding)
        # Fallback to standard centroid detector
        return self._euclidean_centroid_distances(text_embedding)

    def _euclidean_centroid_distances(self, text_embedding: Any) -> Any:
        """Fallback Euclidean centroid distances."""
        np = _np()
        # Simple fallback using cosine similarity (1 - cosine = distance)
        if self.centroid_detector is not None:
            return 1.0 - self.centroid_detector.compute_distances(text_embedding)
        return np.zeros(len(MahalanobisCentroidDetector.CATEGORIES))

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_classify(self, features: Any) -> dict[str, float]:
        """Rule-based classification when no trained model is available.

        Supports current 8-slot and legacy 6-slot feature vector layouts:
        - Old: 768 + 768 + 1 + centroid slots + 3 + 20 + 5 = 1571 or 1573
        - M23: 256 + 256 + 1 + centroid slots + 3 + 20 + 5 = 547 or 549
        - zvec: 128 + 128 + 1 + centroid slots + 3 + 20 + 5 = 291 or 293

        Centroid distances are cosine similarity (HIGH = close to centroid =
        strong attack signal).  This is the opposite of Mahalanobis distance.
        """
        np = _np()
        scores: dict[str, float] = {label: 0.0 for label in ATTACK_LABELS}
        scores["clean"] = 0.5

        n_features = len(features)

        layout = self._feature_layout(n_features)
        intent_idx = layout["intent_idx"]
        centroid_start = layout["centroid_start"]
        centroid_count = layout["centroid_count"]
        perplexity_start = layout["perplexity_start"]
        ngram_start = layout["ngram_start"]
        ngram_end = layout["ngram_end"]
        flags_start = layout["flags_start"]

        # --- Intent Divergence ---
        # Low cosine similarity between text and context = suspicious
        if n_features > intent_idx:
            divergence = features[intent_idx]
            if divergence != 0.0 and divergence < 0.3:
                scores["semantic_manipulation"] += 0.3
                scores["behavioral_control"] += 0.2
                scores["clean"] -= 0.2

        # --- Centroid Cosine Similarities ---
        # HIGH cosine similarity = close to centroid = attack-like
        # Cosine sim range: [-1, 1], values > 0.7 indicate strong attack signal
        centroid_cats = MahalanobisCentroidDetector.CATEGORIES[:centroid_count]
        for i, cat in enumerate(centroid_cats):
            if centroid_start + i < n_features:
                sim = features[centroid_start + i]
                if sim > 0.90:  # very close to attack centroid
                    scores[cat] += 0.45
                    scores["clean"] -= 0.25
                elif sim > 0.80:  # moderately close
                    scores[cat] += 0.20
                    scores["clean"] -= 0.10

        # --- Perplexity ---
        if perplexity_start + 1 < n_features:
            ppl_spike = features[perplexity_start + 1]  # max_spike
            if ppl_spike > 5.0:
                scores["content_injection"] += 0.2
                scores["jailbreak"] += 0.1
                scores["clean"] -= 0.1

        # --- N-gram (injection keywords) ---
        if ngram_end <= n_features:
            ngram_sum = float(np.sum(features[ngram_start:ngram_end]))
            if ngram_sum >= 3:
                scores["jailbreak"] += 0.5
                scores["behavioral_control"] += 0.3
                scores["clean"] -= 0.5
            elif ngram_sum >= 2:
                scores["jailbreak"] += 0.4
                scores["behavioral_control"] += 0.2
                scores["clean"] -= 0.3
            elif ngram_sum >= 1:
                scores["content_injection"] += 0.2
                scores["jailbreak"] += 0.2
                scores["clean"] -= 0.2

        # --- Encoding Flags ---
        if flags_start + 5 <= n_features:
            flag_count = float(np.sum(features[flags_start : flags_start + 5]))
            if flag_count >= 2:
                scores["content_injection"] += 0.3
                scores["exfiltration_attempt"] += 0.2
                scores["clean"] -= 0.3
            elif flag_count >= 1:
                scores["content_injection"] += 0.1
                scores["clean"] -= 0.1

        # Clamp all scores to [0, 1]
        for label in scores:
            scores[label] = max(0.0, min(1.0, scores[label]))

        max_attack = max(v for k, v in scores.items() if k != "clean")
        scores["clean"] = max(0.0, 1.0 - max_attack)
        return scores

    # ------------------------------------------------------------------
    # Stacking ensemble (M23)
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_layout(n_features: int) -> dict[str, int]:
        """Infer feature positions for current 8-slot and legacy 6-slot vectors."""
        centroid_total = len(MahalanobisCentroidDetector.CATEGORIES)
        static_tail = 1 + 3 + 20 + 5
        layouts = (
            (128, centroid_total),
            (128, 6),
            (256, centroid_total),
            (256, 6),
            (768, centroid_total),
            (768, 6),
        )
        for emb_dim, centroid_count in layouts:
            expected = emb_dim + emb_dim + static_tail + centroid_count
            if n_features == expected:
                intent_idx = emb_dim * 2
                centroid_start = intent_idx + 1
                return {
                    "embedding_dim": emb_dim,
                    "centroid_count": centroid_count,
                    "intent_idx": intent_idx,
                    "centroid_start": centroid_start,
                    "perplexity_start": centroid_start + centroid_count,
                    "ngram_start": centroid_start + centroid_count + 3,
                    "ngram_end": centroid_start + centroid_count + 3 + 20,
                    "flags_start": centroid_start + centroid_count + 3 + 20,
                }

        # Best-effort fallback for partial or future vectors: preserve the old
        # broad zvec/M23/legacy split, but use the current centroid count.
        if n_features < 512:
            emb_dim = 128
        elif n_features < 1024:
            emb_dim = 256
        else:
            emb_dim = 768
        intent_idx = emb_dim * 2
        centroid_start = intent_idx + 1
        return {
            "embedding_dim": emb_dim,
            "centroid_count": centroid_total,
            "intent_idx": intent_idx,
            "centroid_start": centroid_start,
            "perplexity_start": centroid_start + centroid_total,
            "ngram_start": centroid_start + centroid_total + 3,
            "ngram_end": centroid_start + centroid_total + 3 + 20,
            "flags_start": centroid_start + centroid_total + 3 + 20,
        }

    def fit_stacking(self, X: Any, y: Any, n_folds: int = 5) -> None:
        """Fit a stacking ensemble with cross-validated OOF predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels (multi-label or multi-class)
            n_folds: Number of cross-validation folds
        """
        self._stacking_model = StackingEnsemble(n_folds=n_folds)
        self._stacking_model.fit(X, y)
        self.model = self._stacking_model

    def get_feature_importance_scores(self) -> Optional[dict[str, float]]:
        """Get feature importance scores from fitted model.

        Returns:
            Dict mapping feature names to importance scores
        """
        if self._stacking_model is not None:
            return self._stacking_model.get_feature_importance()
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            return dict(zip(self._get_feature_names(), self.model.feature_importances_))
        return self.feature_importance

    def _get_feature_names(self) -> list[str]:
        """Get feature names for importance analysis."""
        names = []
        names.extend([f"text_proj_{i}" for i in range(256)])
        names.extend([f"context_proj_{i}" for i in range(256)])
        names.append("intent_divergence")
        names.extend([f"centroid_{cat}" for cat in MahalanobisCentroidDetector.CATEGORIES])
        names.extend(["ppl_mean", "ppl_max_spike", "ppl_variance"])
        names.extend([f"ngram_{i}" for i in range(20)])
        names.extend(["flag_base64", "flag_hex", "flag_homoglyph", "flag_invisible", "flag_whitespace"])
        return names

    # ------------------------------------------------------------------
    # Main classify
    # ------------------------------------------------------------------

    def classify(self, features: Any) -> ClassificationResult:
        """Classify a feature vector.

        Args:
            features: Combined feature vector from
                :class:`~hermes_katana.scabbard.feature_extractor.FeatureExtractor`.

        Returns:
            :class:`ClassificationResult` with scores, decision, and metadata.
        """
        if self.model is not None:
            try:
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(features.reshape(1, -1))
                    if isinstance(probs, list):
                        scores = {
                            ATTACK_LABELS[i]: float(p[0][1] if p.shape[1] > 1 else p[0][0]) for i, p in enumerate(probs)
                        }
                    else:
                        scores = {
                            ATTACK_LABELS[i]: float(probs[0][i]) for i in range(min(len(ATTACK_LABELS), probs.shape[1]))
                        }
                        # Greeting bias damping: conversational greetings (e.g. "Hello how are you")
                        # are falsely flagged because the zvec embedder clusters conversational
                        # language near attack centroids. Dampen when content_injection centroid
                        # (idx 0) > 0.90 AND cognitive_state_attack centroid (idx 2) is notably
                        # lower (< 0.845). Real attacks have ALL centroids high (> 0.88);
                        # greetings have one centroid conspicuously lower.
                        n_feat = len(features)
                        layout = self._feature_layout(n_feat)
                        centroid_start = layout["centroid_start"]
                        centroid_count = layout["centroid_count"]
                        cs = features[centroid_start : centroid_start + centroid_count]
                        if len(cs) >= 3:
                            if cs[0] > 0.90 and cs[2] < 0.845:
                                # Probable greeting FP — dampen attack scores
                                for k in scores:
                                    scores[k] = scores[k] * 0.30
                else:
                    pred = self.model.predict(features.reshape(1, -1))
                    scores = {label: 0.0 for label in ATTACK_LABELS}
                    scores["clean"] = 1.0 - float(pred[0])
                    scores["unknown_anomaly"] = float(pred[0])

                # Get feature importance if available
                importance = self.get_feature_importance_scores()
            except Exception:  # noqa: BLE001
                scores = self._rule_based_classify(features)
                importance = None
        else:
            scores = self._rule_based_classify(features)
            importance = None

        attack_scores = {k: v for k, v in scores.items() if k != "clean"}
        top_category = max(attack_scores, key=lambda k: attack_scores[k])
        max_attack_score = attack_scores[top_category]

        if max_attack_score < self.thresholds["allow"]:
            decision = Decision.ALLOW
        elif max_attack_score > self.thresholds["block"]:
            decision = Decision.BLOCK
        else:
            decision = Decision.FLAG

        return ClassificationResult(
            scores=scores,
            decision=decision,
            top_category=top_category,
            confidence=max_attack_score,
            feature_importance=importance,
        )
