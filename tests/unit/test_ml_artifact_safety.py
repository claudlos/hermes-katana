"""Regression tests for safe ML artifact loading gates."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from hermes_katana.ml_artifacts import UnsafeArtifactError, artifact_sha256, require_trusted_artifact
from hermes_katana.scabbard.fusion import FusionClassifier
from hermes_katana.scanner.ensemble import EnsembleClassifier


def test_untrusted_pickle_artifact_is_rejected_by_default(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"not a trusted model")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    with pytest.raises(UnsafeArtifactError):
        require_trusted_artifact(artifact)


def test_untrusted_joblib_model_is_not_loaded_by_default(tmp_path, monkeypatch):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"not a trusted model")
    calls = []

    fake_joblib = SimpleNamespace(load=lambda path: calls.append(path) or object())
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    classifier = FusionClassifier.load(str(artifact))

    assert classifier.model is None
    assert calls == []


def test_hash_verified_artifact_is_trusted(tmp_path, monkeypatch):
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"trusted content")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    digest = "c057a979afc2b228aa3dd57b07b444d015a8439d8483cf445b3fb532a7abbc87"

    assert require_trusted_artifact(artifact, expected_sha256=digest) == artifact


def test_colocated_sha256_sidecar_is_not_a_trust_root(tmp_path, monkeypatch):
    """A pickle-adjacent .sha256 can be attacker-controlled and must not authorize loading."""
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"attacker controlled pickle bytes")
    artifact.with_suffix(".pkl.sha256").write_text(artifact_sha256(artifact), encoding="utf-8")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    import hermes_katana.ml_artifacts as ml_artifacts

    calls: list[str | None] = []

    def fake_safe_pickle_load(path, *, expected_sha256=None):
        calls.append(expected_sha256)
        raise UnsafeArtifactError("blocked before pickle load")

    monkeypatch.setattr(ml_artifacts, "safe_pickle_load", fake_safe_pickle_load)

    classifier = EnsembleClassifier()
    classifier.load(artifact)

    assert calls == [None]
    assert classifier._pipeline is None
    assert classifier._trained is False


def test_hash_compare_uses_hmac_compare_digest(tmp_path, monkeypatch):
    """The hash check must use a constant-time compare to resist timing attacks."""
    import hmac

    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"trusted content")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    calls: list[tuple[str, str]] = []
    real_compare = hmac.compare_digest

    def spy(a, b):
        calls.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr("hermes_katana.ml_artifacts.hmac.compare_digest", spy)
    digest = "c057a979afc2b228aa3dd57b07b444d015a8439d8483cf445b3fb532a7abbc87"
    require_trusted_artifact(artifact, expected_sha256=digest)
    assert calls, "hmac.compare_digest must be invoked for hash verification"


def test_hash_compare_handles_uppercase_and_whitespace(tmp_path, monkeypatch):
    """Hash compare normalizes case and trims whitespace before constant-time compare."""
    artifact = tmp_path / "model.pkl"
    artifact.write_bytes(b"trusted content")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    digest = "  C057A979AFC2B228AA3DD57B07B444D015A8439D8483CF445B3FB532A7ABBC87  "
    assert require_trusted_artifact(artifact, expected_sha256=digest) == artifact


def test_safe_torch_load_with_weights_only_false_requires_trust(tmp_path, monkeypatch):
    """torch.load(weights_only=False) is a pickle gadget — must require trust."""
    from hermes_katana.ml_artifacts import safe_torch_load

    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"untrusted")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    with pytest.raises(UnsafeArtifactError):
        # No expected_sha256, no env opt-in, weights_only=False → must reject.
        safe_torch_load(artifact, weights_only=False)


def test_safe_torch_load_weights_only_true_skips_trust_when_no_hash(tmp_path, monkeypatch):
    """weights_only=True is safe (no pickle); trust gate is opt-in via hash."""
    from hermes_katana import ml_artifacts

    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"weights")
    monkeypatch.delenv("HERMES_KATANA_TRUST_ML_ARTIFACTS", raising=False)

    fake_torch = SimpleNamespace(
        load=lambda path, weights_only=True, **kw: {"loaded": str(path), "weights_only": weights_only}
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Should not raise — weights_only=True is treated as safe.
    out = ml_artifacts.safe_torch_load(artifact, weights_only=True)
    assert out["weights_only"] is True
