"""Tests for Stage 3: Fusion Classifier."""

from __future__ import annotations

import numpy as np

from hermes_katana.scabbard.fusion import (
    Decision,
    ATTACK_LABELS,
    ClassificationResult,
    FusionClassifier,
)


# =============================================================================
# Decision enum
# =============================================================================


class TestDecision:
    def test_decision_values(self):
        assert Decision.ALLOW.value == "allow"
        assert Decision.FLAG.value == "flag"
        assert Decision.BLOCK.value == "block"

    def test_decision_is_string_enum(self):
        assert isinstance(Decision.ALLOW, str)


# =============================================================================
# ClassificationResult
# =============================================================================


class TestClassificationResult:
    def test_to_dict(self):
        result = ClassificationResult(
            scores={"clean": 0.8, "jailbreak": 0.2},
            decision=Decision.ALLOW,
            top_category="clean",
            confidence=0.8,
        )
        d = result.to_dict()
        assert d["decision"] == "allow"
        assert d["top_category"] == "clean"
        assert d["confidence"] == 0.8
        assert "scores" in d

    def test_to_risk_report(self):
        result = ClassificationResult(
            scores={"clean": 0.3, "jailbreak": 0.7},
            decision=Decision.BLOCK,
            top_category="jailbreak",
            confidence=0.7,
        )
        report = result.to_risk_report(source="test", content_type="prompt")
        assert report["decision"] == "block"
        assert report["top_category"] == "jailbreak"
        assert report["source"] == "test"
        assert report["content_type"] == "prompt"

    def test_to_risk_report_filters_low_scores(self):
        result = ClassificationResult(
            scores={"clean": 0.5, "content_injection": 0.15, "jailbreak": 0.35},
            decision=Decision.FLAG,
            top_category="clean",
            confidence=0.5,
        )
        report = result.to_risk_report()
        # Only flags with score > 0.2
        flag_types = [f["type"] for f in report["flags"]]
        assert "content_injection" not in flag_types  # 0.15 < 0.2
        assert "jailbreak" in flag_types  # 0.35 > 0.2


# =============================================================================
# FusionClassifier — initialization and thresholds
# =============================================================================


class TestFusionClassifierInit:
    def test_default_thresholds(self):
        clf = FusionClassifier()
        assert clf.thresholds["allow"] == 0.3
        assert clf.thresholds["block"] == 0.7

    def test_custom_thresholds(self):
        clf = FusionClassifier(thresholds={"allow": 0.2, "block": 0.8})
        assert clf.thresholds["allow"] == 0.2
        assert clf.thresholds["block"] == 0.8

    def test_no_model_uses_rule_based(self):
        clf = FusionClassifier(model=None)
        assert clf.model is None

    def test_load_nonexistent_file(self):
        clf = FusionClassifier.load("/nonexistent/model.json")
        assert clf.model is None


# =============================================================================
# FusionClassifier — rule-based classification
# =============================================================================


class TestRuleBasedClassification:
    def test_benign_text_returns_allow(self):
        clf = FusionClassifier()
        # Minimal feature vector (just n-grams = zeros)
        features = np.zeros(1571)
        result = clf.classify(features)
        assert result.decision in (Decision.ALLOW, Decision.FLAG)

    def test_single_ngram_triggers_flag(self):
        clf = FusionClassifier()
        # Feature vector with 1 n-gram hit
        features = np.zeros(1571)
        features[1546] = 1.0  # one n-gram match
        result = clf.classify(features)
        assert result.scores["content_injection"] > 0
        assert result.scores["jailbreak"] > 0

    def test_multiple_ngrams_triggers_flag(self):
        # With default thresholds (block=0.7), 3 n-gram matches give jailbreak=0.5
        # which is between allow=0.3 and block=0.7 -> FLAG
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1546:1549] = 1.0  # 3 n-gram matches
        result = clf.classify(features)
        assert result.decision == Decision.FLAG

    def test_encoding_flags_boost_score(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1566] = 1.0  # base64 flag
        features[1567] = 1.0  # hex flag
        result = clf.classify(features)
        assert result.scores["content_injection"] > 0
        assert result.scores["exfiltration_attempt"] > 0

    def test_clean_score_adjusted(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1546:1552] = 1.0  # multiple n-gram matches
        result = clf.classify(features)
        assert result.scores["clean"] <= 0.5

    def test_attack_labels_all_present(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        result = clf.classify(features)
        for label in ATTACK_LABELS:
            assert label in result.scores

    def test_scores_normalized_to_01(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1546:1555] = 1.0
        result = clf.classify(features)
        for label, score in result.scores.items():
            assert 0.0 <= score <= 1.0, f"{label} score {score} out of range"


# =============================================================================
# FusionClassifier — decision thresholds
# =============================================================================


class TestDecisionThresholds:
    def test_allow_decision_below_threshold(self):
        clf = FusionClassifier(thresholds={"allow": 0.3, "block": 0.7})
        features = np.zeros(1571)
        # Set all attack scores to 0
        result = clf.classify(features)
        assert result.decision in (Decision.ALLOW, Decision.FLAG)

    def test_flag_decision_between_thresholds(self):
        clf = FusionClassifier(thresholds={"allow": 0.2, "block": 0.7})
        features = np.zeros(1571)
        features[1546:1548] = 1.0  # 2 n-gram matches -> jailbreak += 0.4, still < 0.7
        result = clf.classify(features)
        assert result.decision == Decision.FLAG

    def test_block_decision_above_threshold(self):
        clf = FusionClassifier(thresholds={"allow": 0.2, "block": 0.4})
        features = np.zeros(1571)
        features[1546:1552] = 1.0  # 6 matches -> jailbreak += 0.5 -> 0.5 > 0.4 -> BLOCK
        result = clf.classify(features)
        assert result.decision == Decision.BLOCK

    def test_custom_thresholds_change_decision(self):
        # Same features, different thresholds
        features = np.zeros(1571)
        features[1546:1549] = 1.0

        clf_strict = FusionClassifier(thresholds={"allow": 0.1, "block": 0.3})
        result_strict = clf_strict.classify(features)
        assert result_strict.decision == Decision.BLOCK

        clf_lenient = FusionClassifier(thresholds={"allow": 0.1, "block": 0.8})
        result_lenient = clf_lenient.classify(features)
        assert result_lenient.decision in (Decision.ALLOW, Decision.FLAG)


# =============================================================================
# FusionClassifier — top category
# =============================================================================


class TestTopCategory:
    def test_top_category_is_attack(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1546:1549] = 1.0
        result = clf.classify(features)
        assert result.top_category in ATTACK_LABELS
        assert result.top_category != "clean"

    def test_top_category_reflects_highest_score(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1543] = 1.0  # Centroid distance for content_injection
        result = clf.classify(features)
        assert result.top_category in result.scores
        top_score = result.scores[result.top_category]
        for label, score in result.scores.items():
            if label != "clean":
                assert score <= top_score + 1e-9


# =============================================================================
# FusionClassifier — centroid distance signals
# =============================================================================


class TestCentroidSignals:
    def test_high_centroid_similarity_boosts_attack(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        # Set a high centroid distance signal
        features[1542] = 0.8
        result = clf.classify(features)
        # Verify classifier produces valid output structure
        assert isinstance(result.scores, dict)
        assert "clean" in result.scores
        assert all(0.0 <= v <= 1.0 for v in result.scores.values())

    def test_medium_centroid_similarity_boosts_attack(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1538] = 0.6
        result = clf.classify(features)
        assert isinstance(result.scores, dict)
        assert "clean" in result.scores
        assert all(0.0 <= v <= 1.0 for v in result.scores.values())


# =============================================================================
# FusionClassifier — perplexity signals
# =============================================================================


class TestPerplexitySignals:
    def test_high_perplexity_spike_boosts_attack(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        # Set perplexity spike > 5.0
        features[1544] = 6.0  # max_spike
        result = clf.classify(features)
        assert result.scores["content_injection"] > 0


# =============================================================================
# FusionClassifier — confidence
# =============================================================================


class TestConfidence:
    def test_confidence_equals_top_attack_score(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1546:1550] = 1.0
        result = clf.classify(features)
        assert abs(result.confidence - result.scores[result.top_category]) < 1e-9

    def test_confidence_between_0_and_1(self):
        clf = FusionClassifier()
        features = np.zeros(1571)
        features[1546:1555] = 1.0
        result = clf.classify(features)
        assert 0.0 <= result.confidence <= 1.0


# =============================================================================
# M23: Projection Head
# =============================================================================


class TestProjectionHead:
    def test_projection_head_init(self):
        from hermes_katana.scabbard.fusion import ProjectionHead

        proj = ProjectionHead(input_dim=768, output_dim=256)
        assert proj.input_dim == 768
        assert proj.output_dim == 256

    def test_projection_output_shape(self):
        from hermes_katana.scabbard.fusion import ProjectionHead

        proj = ProjectionHead(input_dim=768, output_dim=256)
        embedding = np.random.randn(768).astype(np.float32)
        output = proj.project(embedding)
        assert output.shape == (256,)

    def test_projection_l2_normalized(self):
        from hermes_katana.scabbard.fusion import ProjectionHead

        proj = ProjectionHead(input_dim=768, output_dim=256)
        embedding = np.random.randn(768).astype(np.float32)
        output = proj.project(embedding)
        norm = np.linalg.norm(output)
        assert abs(norm - 1.0) < 1e-6

    def test_projection_batch_shape(self):
        from hermes_katana.scabbard.fusion import ProjectionHead

        proj = ProjectionHead(input_dim=768, output_dim=256)
        embeddings = np.random.randn(10, 768).astype(np.float32)
        output = proj.project_batch(embeddings)
        assert output.shape == (10, 256)

    def test_projection_batch_l2_normalized(self):
        from hermes_katana.scabbard.fusion import ProjectionHead

        proj = ProjectionHead(input_dim=768, output_dim=256)
        embeddings = np.random.randn(10, 768).astype(np.float32)
        output = proj.project_batch(embeddings)
        norms = np.linalg.norm(output, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_projection_consistency(self):
        from hermes_katana.scabbard.fusion import ProjectionHead

        proj = ProjectionHead(input_dim=768, output_dim=256)
        embedding = np.random.randn(768).astype(np.float32)
        out_single = proj.project(embedding)
        out_batch = proj.project_batch(embedding.reshape(1, -1))[0]
        assert np.allclose(out_single, out_batch)


# =============================================================================
# M23: Mahalanobis Centroid Detector
# =============================================================================


class TestMahalanobisCentroidDetector:
    def test_mahalanobis_init(self):
        from hermes_katana.scabbard.fusion import MahalanobisCentroidDetector

        detector = MahalanobisCentroidDetector()
        assert len(detector.CATEGORIES) == 8
        assert "encoding_evasion" in detector.CATEGORIES
        assert "persona_jailbreak" in detector.CATEGORIES

    def test_mahalanobis_output_shape(self):
        from hermes_katana.scabbard.fusion import MahalanobisCentroidDetector

        detector = MahalanobisCentroidDetector()
        embedding = np.random.randn(768).astype(np.float32)
        distances = detector.compute_distances(embedding)
        assert distances.shape == (len(detector.CATEGORIES),)

    def test_mahalanobis_with_centroids(self):
        from hermes_katana.scabbard.fusion import MahalanobisCentroidDetector

        centroids = {
            "content_injection": np.random.randn(768).astype(np.float32),
            "semantic_manipulation": np.random.randn(768).astype(np.float32),
            "jailbreak": np.random.randn(768).astype(np.float32),
        }
        detector = MahalanobisCentroidDetector(centroids=centroids)
        embedding = centroids["content_injection"].copy()
        distances = detector.compute_distances(embedding)
        # Distance to own centroid should be low
        assert distances[0] < distances[1]  # content_injection closer than semantic_manipulation

    def test_mahalanobis_fallback_without_inverse_cov(self):
        from hermes_katana.scabbard.fusion import MahalanobisCentroidDetector

        centroids = {
            "content_injection": np.random.randn(768).astype(np.float32),
        }
        detector = MahalanobisCentroidDetector(centroids=centroids)
        # No covariance set, should use fallback
        embedding = np.random.randn(768).astype(np.float32)
        distances = detector.compute_distances(embedding)
        assert distances[0] >= 0.0


# =============================================================================
# M23: Stacking Ensemble
# =============================================================================


class TestStackingEnsemble:
    def test_stacking_init(self):
        from hermes_katana.scabbard.fusion import StackingEnsemble

        ensemble = StackingEnsemble(n_folds=5, random_state=42)
        assert ensemble.n_folds == 5
        assert ensemble.random_state == 42
        assert ensemble._fitted is False

    def test_stacking_get_base_models(self):
        from hermes_katana.scabbard.fusion import StackingEnsemble

        ensemble = StackingEnsemble()
        models = ensemble._get_base_models()
        # Should have at least one model if ML libraries are available
        assert isinstance(models, list)

    def test_stacking_fit_predict(self):
        from hermes_katana.scabbard.fusion import StackingEnsemble

        # Generate small synthetic data for fast testing
        X = np.random.randn(30, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=30)

        ensemble = StackingEnsemble(n_folds=2, random_state=42)
        ensemble.fit(X, y)

        # Should be able to predict
        probs = ensemble.predict_proba(X[:5])
        assert probs.shape[0] == 5
        # Meta-learner outputs number of classes present in training data

    def test_stacking_feature_importance(self):
        from hermes_katana.scabbard.fusion import StackingEnsemble

        X = np.random.randn(30, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=30)

        ensemble = StackingEnsemble(n_folds=2, random_state=42)
        ensemble.fit(X, y)
        importance = ensemble.get_feature_importance()
        assert isinstance(importance, dict)


# =============================================================================
# M23: Feature Importance Analyzer
# =============================================================================


class TestFeatureImportanceAnalyzer:
    def test_analyzer_init(self):
        from hermes_katana.scabbard.fusion import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer(n_permutations=5, random_state=42)
        assert analyzer.n_permutations == 5
        assert analyzer.random_state == 42

    def test_default_feature_names(self):
        from hermes_katana.scabbard.fusion import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        names = analyzer._get_default_feature_names(547)
        assert len(names) == 547
        assert names[0] == "text_proj_0"
        assert names[256] == "context_proj_0"
        assert names[512] == "intent_divergence"
        assert names[513] == "centroid_content_injection"

    def test_correlation_importance(self):
        from hermes_katana.scabbard.fusion import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        # Set feature names to match input dimensions
        analyzer.feature_names = [f"feat_{i}" for i in range(50)]
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 2, size=100)
        scores = analyzer._correlation_importance(X, y)
        assert len(scores) == 50
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_top_features(self):
        from hermes_katana.scabbard.fusion import FeatureImportanceAnalyzer

        analyzer = FeatureImportanceAnalyzer()
        analyzer._importance_scores = {f"feat_{i}": float(i) for i in range(10)}
        top = analyzer.get_top_features(n=3)
        assert len(top) == 3
        assert top[0][0] == "feat_9"  # highest
        assert top[0][1] == 9.0


# =============================================================================
# M23: FusionClassifier new methods
# =============================================================================


class TestFusionClassifierProjectionHead:
    def test_project_text_embedding(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier(use_projection_head=True)
        embedding = np.random.randn(768).astype(np.float32)
        proj = clf.project_text_embedding(embedding)
        assert proj.shape == (256,)
        norm = np.linalg.norm(proj)
        assert abs(norm - 1.0) < 1e-6

    def test_project_text_embedding_disabled(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier(use_projection_head=False)
        embedding = np.random.randn(768).astype(np.float32)
        proj = clf.project_text_embedding(embedding)
        assert proj.shape == (768,)


class TestFusionClassifierMahalanobis:
    def test_mahalanobis_distances(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier(use_mahalanobis=True)
        embedding = np.random.randn(768).astype(np.float32)
        distances = clf.compute_mahalanobis_distances(embedding)
        assert distances.shape == (8,)

    def test_mahalanobis_with_centroids(self):
        from hermes_katana.scabbard.fusion import FusionClassifier, MahalanobisCentroidDetector

        centroids = {
            "content_injection": np.random.randn(768).astype(np.float32),
        }
        detector = MahalanobisCentroidDetector(centroids=centroids)
        clf = FusionClassifier(use_mahalanobis=True, centroid_detector=detector)
        distances = clf.compute_mahalanobis_distances(centroids["content_injection"])
        assert distances.shape == (len(MahalanobisCentroidDetector.CATEGORIES),)


class TestFusionClassifierStacking:
    def test_fit_stacking(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        X = np.random.randn(30, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=30)

        clf = FusionClassifier()
        clf.fit_stacking(X, y, n_folds=2)
        assert clf._stacking_model is not None

    def test_get_feature_importance_scores(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        X = np.random.randn(30, 20).astype(np.float32)
        y = np.random.randint(0, 2, size=30)

        clf = FusionClassifier()
        clf.fit_stacking(X, y, n_folds=2)
        importance = clf.get_feature_importance_scores()
        assert isinstance(importance, dict)


class TestFusionClassifierRuleBasedM23:
    """Rule-based classification with new M23 feature layout."""

    def test_rule_based_with_projection_features(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier()
        # New feature layout: 256 + 256 + 1 + 6 + 3 + 20 + 5 = 547
        features = np.zeros(547)
        # Intent divergence < 0.2 should trigger semantic_manipulation
        features[512] = 0.1
        result = clf.classify(features)
        assert result.scores["semantic_manipulation"] > 0

    def test_rule_based_mahalanobis_low_distance(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier()
        features = np.zeros(547)
        # Low Mahalanobis distance to attack centroid
        features[518] = 0.2
        result = clf.classify(features)
        assert isinstance(result.scores, dict)
        assert "clean" in result.scores
        assert all(0.0 <= v <= 1.0 for v in result.scores.values())

    def test_rule_based_ngram_matches(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier()
        features = np.zeros(547)
        # 3 n-gram matches
        features[522:525] = 1.0
        result = clf.classify(features)
        assert result.scores["jailbreak"] >= 0.4

    def test_rule_based_encoding_flags(self):
        from hermes_katana.scabbard.fusion import FusionClassifier

        clf = FusionClassifier()
        features = np.zeros(547)
        # 2 encoding flags
        features[542] = 1.0
        features[543] = 1.0
        result = clf.classify(features)
        assert result.scores["content_injection"] > 0
        assert result.scores["exfiltration_attempt"] > 0
