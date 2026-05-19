"""Batch 3 production profile defaults for the middleware chain."""

from __future__ import annotations

import pytest

from hermes_katana import hermes_plugin
from hermes_katana.middleware.integration import create_default_chain


def _middleware_by_name(chain):
    return {mw.name: mw for mw in chain.list_middleware()}


def test_fast_cpu_profile_uses_cpu_first_scabbard_without_redundant_ml_gates():
    chain = create_default_chain({"profile": "fast_cpu"})

    middleware = _middleware_by_name(chain)
    scabbard = middleware["katana.scabbard"]

    assert middleware["katana.protectai"].enabled is False
    assert middleware["katana.sentinel"].enabled is False
    assert middleware["katana.scan"].enabled is True
    assert scabbard.enabled is True
    assert scabbard._route_mode == "balanced"
    assert scabbard._config.katana_v11_backend == "onnx"
    assert scabbard._config.katana_v11_device is None
    assert scabbard._config.protectai_enabled is False
    assert scabbard._config.model_version == "katana_v15_distill_minilm"


def test_balanced_profile_keeps_scabbard_primary_and_disables_overlapping_ml_by_default():
    chain = create_default_chain({"profile": "balanced"})

    middleware = _middleware_by_name(chain)
    scabbard = middleware["katana.scabbard"]

    assert scabbard.enabled is True
    assert scabbard._route_mode == "balanced"
    assert middleware["katana.scan"]._route_aware is True
    assert middleware["katana.protectai"].enabled is False
    assert middleware["katana.sentinel"].enabled is False


def test_paranoid_profile_enables_overlapping_ml_gates_and_fails_closed_outputs():
    chain = create_default_chain({"profile": "paranoid"})

    middleware = _middleware_by_name(chain)

    assert middleware["katana.protectai"].enabled is True
    assert middleware["katana.sentinel"].enabled is True
    assert middleware["katana.scabbard"]._enforce_output_blocks is True
    assert middleware["katana.scan"]._enforce_output_findings is True
    assert middleware["katana.scan"]._route_aware is False


def test_profile_defaults_are_applied_before_explicit_overrides():
    chain = create_default_chain({"profile": "fast_cpu", "protectai.enabled": True, "scabbard.route_mode": "strict"})

    middleware = _middleware_by_name(chain)

    assert middleware["katana.protectai"].enabled is True
    assert middleware["katana.scabbard"]._route_mode == "strict"


def test_plugin_runtime_passes_katana_profile_to_chain(monkeypatch):
    captured = {}
    import hermes_katana.middleware.integration as integration_mod

    monkeypatch.setattr(hermes_plugin, "_open_vault", lambda: None)
    monkeypatch.setattr(hermes_plugin, "_open_audit", lambda _config: None)
    monkeypatch.setattr(hermes_plugin, "_collect_vault_values", lambda _vault: set())
    monkeypatch.setattr(integration_mod, "create_default_chain", lambda cfg: captured.setdefault("config", cfg))

    hermes_plugin._initialize_runtime({"katana_profile": "fast_cpu", "audit_enabled": False})

    assert captured["config"]["profile"] == "fast_cpu"


def test_unknown_production_profile_fails_closed():
    with pytest.raises(ValueError, match="profile"):
        create_default_chain({"profile": "turbo_unsafe"})
