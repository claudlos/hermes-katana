"""Tests for scabbard/fusion.py — fusion classifier."""

import numpy as np


class TestFusionClassifier:
    """Tests for the FusionClassifier with rule-based fallback."""

    def test_rule_based_classify_291d_zvec(self):
        """Rule-based must handle 291-dim zvec feature vectors."""
        from hermes_katana.scabbard.fusion import FusionClassifier

        fusion = FusionClassifier()
        # 128 + 128 + 1 + 6 + 3 + 20 + 5 = 291
        features = np.random.randn(291).astype(np.float32)
        result = fusion.classify(features)
        assert hasattr(result, "decision")
        assert result.decision.value in ("allow", "flag", "block")
        assert all(0.0 <= s <= 1.0 for s in result.scores.values())

    def test_rule_based_centroid_fires_high_sim(self):
        """High cosine similarity to centroids must increase attack score."""
        from hermes_katana.scabbard.fusion import FusionClassifier

        fusion = FusionClassifier()
        # 291-dim vector: centroid similarities = 0.95 (very attack-like)
        features = np.zeros(291, dtype=np.float32)
        features[256:262] = 0.95  # centroid sims
        result = fusion.classify(features)
        attack_scores = {k: v for k, v in result.scores.items() if k != "clean"}
        assert max(attack_scores.values()) > 0.3

    def test_scores_clamped_to_01(self):
        """All scores must be in [0, 1]."""
        from hermes_katana.scabbard.fusion import FusionClassifier

        fusion = FusionClassifier()
        features = np.random.randn(291).astype(np.float32) * 10
        result = fusion.classify(features)
        assert all(0.0 <= s <= 1.0 for s in result.scores.values())
