"""Observability middleware: per-decision counters with structured logging.

Mounts at the *very end* of the chain (priority=10) so it sees every middleware's
decision recorded in ``ctx.extras``. Emits one structured log line per call
with: tool name, final decision, per-middleware decision, classifier
version, latency. Designed for ingestion by Prometheus / Datadog / Honeycomb
scrapers — the log line is a single JSON object so any structured-log pipeline
can parse it without custom regex.

Counters are kept in memory for in-process introspection (``mw.counters``)
and reset on demand. Production deployments would route them to a metrics
backend; this middleware deliberately ships zero infrastructure-specific
code so users can wire whatever they want.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter

from hermes_katana.middleware.chain import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
)

logger = logging.getLogger(__name__)


class KatanaMetricsMiddleware(KatanaMiddleware):
    """Emit per-decision metrics + structured log for ops dashboards.

    Counters live at ``self.counters`` and are keyed by:

      * ``("decision", final_decision)`` — total ALLOW/ESCALATE/DENY counts
      * ``("decision", final_decision, model_version)`` — per-model
      * ``("scabbard_top_category", category, decision)`` — per attack class
      * ``("latency_bucket", bucket)`` — coarse latency histogram

    A JSON log line is emitted on every call at INFO level under the logger
    ``hermes_katana.metrics``. Set that logger's handler to your structured
    sink to get observability for free.

    Example log line::

        {"event": "katana_decision", "tool": "Read", "final_decision": "deny",
         "model_version": "katana_v11-best", "scabbard_top_category": "exfiltration_attempt",
         "scabbard_confidence": 0.97, "latency_ms": 12.4,
         "scabbard_origins_used": {"path": "user_input"}}

    Mounting::

        chain.add(KatanaMetricsMiddleware(name="katana.metrics"))
    """

    LATENCY_BUCKETS_MS: tuple[float, ...] = (1, 5, 10, 25, 50, 100, 250, 500, 1000)

    def __init__(
        self,
        *,
        name: str = "katana.metrics",
        enabled: bool = True,
        emit_log: bool = True,
        priority: int = 10,
    ) -> None:
        super().__init__(name=name, enabled=enabled, priority=priority)
        self.emit_log = emit_log
        self._lock = threading.Lock()
        self.counters: Counter[tuple] = Counter()
        self._call_starts: dict[int, float] = {}

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()

    def snapshot(self) -> dict[str, int]:
        """Return a copy of counters keyed as 'a/b/c' strings (Prom-friendly)."""
        with self._lock:
            return {"/".join(str(p) for p in k): v for k, v in self.counters.items()}

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        # Record start time keyed by ctx id (stable for the duration of the call).
        self._call_starts[id(ctx)] = time.perf_counter()
        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Called on ALLOW path after the tool ran."""
        self._record(ctx, fallback_decision="allowed")

    def on_short_circuit(self, ctx: CallContext) -> None:
        """Called on DENY path when an earlier middleware short-circuits.

        Without this hook, denials would never be counted because the chain
        bypasses ``post_dispatch`` after a DENY decision.
        """
        self._record(ctx, fallback_decision="denied")

    def _record(self, ctx: CallContext, *, fallback_decision: str) -> None:
        start = self._call_starts.pop(id(ctx), None)
        latency_ms = (time.perf_counter() - start) * 1000.0 if start else 0.0
        bucket = "above"
        for b in self.LATENCY_BUCKETS_MS:
            if latency_ms <= b:
                bucket = f"<={int(b)}ms"
                break

        # Pull whichever final-decision marker the chain has set.
        final_decision = getattr(ctx, "final_decision", None) or ctx.extras.get("final_decision") or fallback_decision
        scabbard_result = ctx.extras.get("scabbard_result") or {}
        top_category = scabbard_result.get("top_category", "")
        confidence = float(scabbard_result.get("confidence", 0.0) or 0.0)
        model_version = ctx.extras.get("scabbard_model_version", "unknown")
        origins_used = ctx.extras.get("scabbard_arg_origins", {})

        with self._lock:
            self.counters[("decision", final_decision)] += 1
            self.counters[("decision", final_decision, model_version)] += 1
            if top_category:
                self.counters[("scabbard_top_category", top_category, final_decision)] += 1
            self.counters[("latency_bucket", bucket)] += 1

        if self.emit_log:
            payload = {
                "event": "katana_decision",
                "tool": ctx.tool_name,
                "final_decision": final_decision,
                "model_version": model_version,
                "scabbard_top_category": top_category,
                "scabbard_confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 2),
                "latency_bucket": bucket,
                "scabbard_origins_used": origins_used or None,
            }
            logger.info("%s", json.dumps(payload, default=str, sort_keys=True))
