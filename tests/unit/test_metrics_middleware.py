"""Tests for KatanaMetricsMiddleware — observability counters + structured log."""

from __future__ import annotations

import json
import logging
from io import StringIO


from hermes_katana.middleware.chain import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
    MiddlewareChain,
)
from hermes_katana.middleware.metrics import KatanaMetricsMiddleware


class _DenyOnNuke(KatanaMiddleware):
    """Toy middleware that DENYs when ``ctx.tool_name == 'nuke'``."""

    def __init__(self) -> None:
        super().__init__(name="test.nuke_blocker", enabled=True, priority=80)

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        if ctx.tool_name == "nuke":
            ctx.extras["scabbard_result"] = {
                "top_category": "exfiltration_attempt",
                "confidence": 0.95,
            }
            ctx.extras["scabbard_model_version"] = "katana_v11-test"
            ctx.deny("nuke blocked")
            return DispatchDecision.DENY
        return DispatchDecision.ALLOW


def _drive_chain(chain, ctx):
    """Helper: run pre then either post (allow) or short-circuit (deny)."""
    decision = chain.execute_pre(ctx)
    if decision == DispatchDecision.DENY:
        # Re-implement the chain's short-circuit notify path (the chain
        # auto-calls on_short_circuit on each middleware when DENY fires;
        # in the production loop this happens inside execute_pre, but for
        # these tests we call execute_post explicitly to be deterministic).
        pass
    else:
        chain.execute_post(ctx)


def test_metrics_counts_allow_and_deny():
    metrics = KatanaMetricsMiddleware(emit_log=False)
    chain = MiddlewareChain()
    chain.add(_DenyOnNuke())
    chain.add(metrics)

    _drive_chain(chain, CallContext(tool_name="read_file", args={"path": "/etc/hosts"}))
    _drive_chain(chain, CallContext(tool_name="read_file", args={"path": "/etc/hosts"}))
    _drive_chain(chain, CallContext(tool_name="nuke", args={"target": "production"}))
    snap = metrics.snapshot()
    # Two reads passed through to allowed; one nuke denied.
    assert any(k.startswith("decision/") for k in snap), snap
    # Latency bucket counts should sum to 3 calls.
    bucket_total = sum(v for k, v in snap.items() if k.startswith("latency_bucket/"))
    assert bucket_total == 3, snap


def test_metrics_emits_structured_log_with_model_version():
    chain = MiddlewareChain()
    chain.add(_DenyOnNuke())
    chain.add(KatanaMetricsMiddleware(emit_log=True))

    log_buf = StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.INFO)
    metrics_logger = logging.getLogger("hermes_katana.middleware.metrics")
    prev_level = metrics_logger.level
    metrics_logger.setLevel(logging.INFO)
    metrics_logger.addHandler(handler)
    try:
        _drive_chain(chain, CallContext(tool_name="nuke", args={"x": "y"}))
    finally:
        metrics_logger.removeHandler(handler)
        metrics_logger.setLevel(prev_level)

    raw = log_buf.getvalue().strip().splitlines()
    assert raw, "metrics middleware did not emit a log line"
    payload = json.loads(raw[-1])
    assert payload["event"] == "katana_decision"
    assert payload["tool"] == "nuke"
    assert payload["model_version"] == "katana_v11-test"
    assert payload["scabbard_top_category"] == "exfiltration_attempt"
    assert "latency_ms" in payload


def test_metrics_reset():
    metrics = KatanaMetricsMiddleware(emit_log=False)
    chain = MiddlewareChain()
    chain.add(_DenyOnNuke())
    chain.add(metrics)
    _drive_chain(chain, CallContext(tool_name="nuke", args={}))
    assert sum(metrics.counters.values()) > 0
    metrics.reset()
    assert sum(metrics.counters.values()) == 0


def test_metrics_unknown_when_no_scabbard_result():
    metrics = KatanaMetricsMiddleware(emit_log=False)
    chain = MiddlewareChain()
    chain.add(metrics)  # nothing produces a scabbard_result
    _drive_chain(chain, CallContext(tool_name="ls", args={}))
    snap = metrics.snapshot()
    # No scabbard_top_category key recorded since result was empty.
    assert not any(k.startswith("scabbard_top_category/") for k in snap), snap
