"""Tests for centroid loading and cosine similarity."""

from pathlib import Path

import numpy as np
import pytest


def _centroid_path() -> Path:
    models_dir = Path(__file__).resolve().parents[2] / "training" / "models"
    for candidate in (
        models_dir / "attack_centroids_128d.npz",
        models_dir / "attack_centroids.npz",
    ):
        if candidate.exists():
            return candidate
    pytest.skip("no centroid artifact is available under training/models")


class TestCentroidDetector:
    """Tests for centroid detector with the available centroid artifact."""

    def test_load_centroids(self):
        """Must load attack centroids from the checked-in npz file."""
        from hermes_katana.scabbard.feature_extractor import CentroidDetector

        cd = CentroidDetector.load(str(_centroid_path()))
        assert len(cd.centroids) in (6, len(CentroidDetector.CATEGORIES))
        expected_dim = None
        for name, vec in cd.centroids.items():
            expected_dim = expected_dim or int(vec.shape[0])
            assert vec.shape == (expected_dim,), f"{name}: expected ({expected_dim},), got {vec.shape}"
            norm = float(np.linalg.norm(vec))
            assert 0.999 < norm < 1.001, f"{name}: not unit norm ({norm})"

    def test_compute_distances(self):
        """compute_distances must return one cosine similarity per runtime category."""
        from hermes_katana.scabbard.feature_extractor import CentroidDetector

        cd = CentroidDetector.load(str(_centroid_path()))
        embedding_dim = next(iter(cd.centroids.values())).shape[0]
        dummy_emb = np.random.randn(embedding_dim).astype(np.float32)
        dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)
        dists = cd.compute_distances(dummy_emb)
        assert dists.shape == (len(CentroidDetector.CATEGORIES),), f"Unexpected shape: {dists.shape}"
        assert all(-1.0 <= d <= 1.0 for d in dists)
