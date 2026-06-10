"""Taint middleware extracted from the main Hermes integration module."""

from __future__ import annotations

import logging
from typing import Any

from hermes_katana.middleware.chain import CallContext, DispatchDecision, KatanaMiddleware

logger = logging.getLogger(__name__)


class KatanaTaintMiddleware(KatanaMiddleware):
    """Taint-tracking middleware for the Hermes dispatch pipeline."""

    def __init__(
        self,
        tracker: Any | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.taint", enabled=enabled, priority=100)
        self._tracker = tracker

    @property
    def tracker(self) -> Any:
        """Lazy-load the tracker to avoid circular imports at module level."""
        if self._tracker is None:
            from hermes_katana.taint import TaintTracker

            self._tracker = TaintTracker.get_instance()
        return self._tracker

    @staticmethod
    def _find_tainted(value: Any, _tainted_types: tuple | None = None) -> list[Any]:
        """Recursively find all tainted values in a nested structure."""
        from hermes_katana.taint import TaintedValue
        from hermes_katana.taint.value import TaintedStr

        if _tainted_types is None:
            _tainted_types = (TaintedStr, TaintedValue)

        found: list[Any] = []
        if isinstance(value, _tainted_types):
            found.append(value)
        elif isinstance(value, dict):
            for nested in value.keys():
                found.extend(KatanaTaintMiddleware._find_tainted(nested, _tainted_types))
            for nested in value.values():
                found.extend(KatanaTaintMiddleware._find_tainted(nested, _tainted_types))
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.extend(KatanaTaintMiddleware._find_tainted(item, _tainted_types))
        return found

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Check taint flows for all tool arguments."""
        from hermes_katana.taint import FlowDecision, source_risk_level

        tracker = self.tracker
        tainted_fields: dict[str, Any] = {}
        worst_flow = None

        for arg_name, arg_val in ctx.args.items():
            tainted_vals = self._find_tainted(arg_val)
            if not tainted_vals:
                continue

            for tainted_val in tainted_vals:
                sources = tainted_val.sources
                labels = [source.label.name for source in sources]
                origins = [source.origin for source in sources]

                tainted_fields[arg_name] = {
                    "is_tainted": True,
                    "source": origins[0] if origins else "unknown",
                    "labels": labels,
                    "readers": [reader.name for reader in tainted_val.readers] if tainted_val.readers else [],
                    # 0-10 risk gradient from the per-label risk table (NOT the
                    # enum ordinal — see taint.labels.source_risk_level). This is
                    # what the policy engine's taint_level_gte/lte rules read.
                    "level": max((source_risk_level(source) for source in sources), default=0),
                }

                flow_decision = tracker.check_flow(tainted_val, ctx.tool_name, ctx.args)

                if flow_decision == FlowDecision.DENY:
                    worst_flow = FlowDecision.DENY
                    ctx.deny(
                        f"Taint flow violation: field '{arg_name}' "
                        f"(sources={origins}) cannot flow to tool '{ctx.tool_name}'"
                    )
                elif flow_decision == FlowDecision.ASK_USER and worst_flow != FlowDecision.DENY:
                    worst_flow = FlowDecision.ASK_USER
                    ctx.escalate(
                        f"Taint escalation: field '{arg_name}' requires human approval for tool '{ctx.tool_name}'"
                    )

        if tainted_fields:
            ctx.taint_context["tainted_fields"] = tainted_fields
            ctx.extras["taint_checked"] = True

        if worst_flow == FlowDecision.DENY:
            return DispatchDecision.DENY
        if worst_flow == FlowDecision.ASK_USER:
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Wrap tool output in a tainted value for downstream propagation."""
        if ctx.tool_output is None or ctx.is_denied:
            return

        try:
            from hermes_katana.taint import Source, TaintLabel, TrustLevel

            source = Source(
                label=TaintLabel.TOOL_OUTPUT,
                origin=f"tool:{ctx.tool_name}",
                trust_level=TrustLevel.CONDITIONAL,
                metadata={"call_id": ctx.call_id, "tool": ctx.tool_name},
            )
            tainted_output = self.tracker.register(ctx.tool_output, source)
            ctx.tool_output = tainted_output
            ctx.extras["tainted_output"] = tainted_output
        except Exception:
            logger.debug("Could not taint-wrap output for %s", ctx.tool_name, exc_info=True)
