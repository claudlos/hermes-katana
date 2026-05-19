"""Batch 3 production profile defaults for the middleware chain."""

from __future__ import annotations

import json

import pytest

from hermes_katana import hermes_plugin
from hermes_katana.middleware.chain import CallContext, DispatchDecision
from hermes_katana.middleware.integration import KatanaScanMiddleware
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


def test_katana_status_includes_readiness_diagnostics_for_fast_cpu():
    class Context:
        config = {"katana_profile": "fast_cpu", "audit_enabled": False}

        def __init__(self):
            self.hooks = {}
            self.tools = {}

        def register_hook(self, name, callback):
            self.hooks[name] = callback

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

    try:
        hermes_plugin.setup(Context())
        payload = json.loads(hermes_plugin._handle_katana_status())
    finally:
        hermes_plugin._chain = None
        hermes_plugin._audit_trail = None
        hermes_plugin._tracker = None
        hermes_plugin._vault = None
        hermes_plugin._config = {}
        hermes_plugin._initialized = False
        hermes_plugin._initialization_error = None
        hermes_plugin._clear_stash()

    diagnostics = payload["diagnostics"]
    assert diagnostics["active_profile"] == "fast_cpu"
    assert "katana.scabbard" in diagnostics["scanners"]["active"]
    assert "katana.protectai" in diagnostics["scanners"]["inactive"]
    assert "katana.sentinel" in diagnostics["scanners"]["inactive"]
    assert diagnostics["ml"]["scabbard_backend"] == "onnx"
    assert diagnostics["ml"]["scabbard_device"] == "cpu"
    assert diagnostics["ml"]["model_version"] == "katana_v15_distill_minilm"
    assert isinstance(diagnostics["unavailable_optional_scanners"], dict)


def test_scan_middleware_reuses_scabbard_routes_without_recalculating(monkeypatch):
    import hermes_katana.scabbard.routing as routing_mod
    import hermes_katana.scanner as scanner_mod

    def fail_if_recalculated(*args, **kwargs):
        raise AssertionError("should reuse scabbard_routes from context")

    calls = []

    def fake_scan_input(text, **kwargs):
        calls.append(text)

        class Result:
            has_findings = False
            risk_score = 0.0
            summary = "clean"

        return Result()

    monkeypatch.setattr(routing_mod, "should_scabbard_scan_arg", fail_if_recalculated)
    monkeypatch.setattr(scanner_mod, "scan_input", fake_scan_input)

    ctx = CallContext(tool_name="delegate_task", args={"prompt": "Summarize this document."})
    ctx.extras["scabbard_routes"] = [
        {"arg": "prompt", "scan": True, "reason": "natural_language_field", "kind": "natural_language"}
    ]

    decision = KatanaScanMiddleware(route_aware=True).pre_dispatch(ctx)

    assert decision == DispatchDecision.ALLOW
    assert calls == ["Summarize this document."]


def test_fast_cpu_latency_metadata_excludes_disabled_heavy_ml_gates():
    chain = create_default_chain({"profile": "fast_cpu"})

    ctx = chain.execute("read_file", {"path": "/tmp/example.txt", "limit": 10})

    assert ctx.extras["active_profile"] == "fast_cpu"
    latency = ctx.extras["middleware_latency_ms"]
    assert "katana.scabbard" in latency
    assert "katana.scan" in latency
    assert "katana.protectai" not in latency
    assert "katana.sentinel" not in latency
    assert ctx.extras["middleware_total_ms"] >= 0.0
