"""
Hermes integration middleware for HermesKatana.

Concrete middleware implementations that wire HermesKatana's subsystems
(taint tracking, scanning, policy engine, audit trail) into the Hermes
tool-dispatch pipeline via the :class:`MiddlewareChain`.

Middleware stack (default order, highest priority first)
--------------------------------------------------------
1. **KatanaTaintMiddleware** (pri=100) — wraps tool outputs in TaintedValues,
   checks taint flows before tool calls.
2. **KatanaScanMiddleware** (pri=80) — runs the multi-layer scanner on
   inputs and outputs to detect injections, secrets, and dangerous content.
3. **KatanaPolicyMiddleware** (pri=60) — evaluates the declarative policy
   engine to produce ALLOW / DENY / ESCALATE / LOG_ONLY decisions.
4. **KatanaAuditMiddleware** (pri=20) — logs every decision to the
   structured audit trail for post-incident analysis.

Usage::

    from hermes_katana.middleware.integration import create_default_chain

    chain = create_default_chain(config)
    ctx = chain.execute("terminal", {"command": "ls"}, taint_ctx)
"""

from __future__ import annotations

__all__ = [
    "KatanaTaintMiddleware",
    "KatanaScanMiddleware",
    "KatanaPolicyMiddleware",
    "KatanaAuditMiddleware",
    "create_default_chain",
]


import hashlib
import json
import logging
import time
from typing import Any

from hermes_katana.middleware.chain import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
    MiddlewareChain,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Taint middleware
# ---------------------------------------------------------------------------


class KatanaTaintMiddleware(KatanaMiddleware):
    """Taint-tracking middleware for the Hermes dispatch pipeline.

    **Pre-dispatch**: inspects all tool arguments for :class:`TaintedValue`
    wrappers and checks whether the tainted data may flow to the target
    tool.  If the flow analyzer returns DENY, the call is blocked.

    **Post-dispatch**: wraps tool outputs in :class:`TaintedValue` with
    a ``tool_output`` source label so downstream taint propagation works.

    Args:
        tracker: The :class:`TaintTracker` singleton (or scoped instance).
        enabled: Whether this middleware is active.
    """

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
    def _find_tainted(value: Any, _tainted_types: tuple | None = None) -> list:
        """Recursively find all tainted values in a nested structure."""
        from hermes_katana.taint import TaintedValue
        from hermes_katana.taint.value import TaintedStr

        if _tainted_types is None:
            _tainted_types = (TaintedStr, TaintedValue)

        found: list = []
        if isinstance(value, _tainted_types):
            found.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                found.extend(KatanaTaintMiddleware._find_tainted(v, _tainted_types))
        elif isinstance(value, (list, tuple)):
            for item in value:
                found.extend(KatanaTaintMiddleware._find_tainted(item, _tainted_types))
        return found

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Check taint flows for all tool arguments.

        Scans each argument recursively; if any nested value is tainted,
        checks its flow against the target tool.  The most restrictive
        decision wins.

        Also populates ``ctx.taint_context`` with structured taint metadata
        for downstream policy evaluation.
        """
        from hermes_katana.taint import FlowDecision

        tracker = self.tracker
        tainted_fields: dict[str, Any] = {}
        worst_flow = None

        for arg_name, arg_val in ctx.args.items():
            # Recursively find tainted values in nested structures
            tainted_vals = self._find_tainted(arg_val)
            if not tainted_vals:
                continue

            for tainted_val in tainted_vals:
                sources = tainted_val.sources
                labels = [s.label.name for s in sources]
                origins = [s.origin for s in sources]

                tainted_fields[arg_name] = {
                    "is_tainted": True,
                    "source": origins[0] if origins else "unknown",
                    "labels": labels,
                    "readers": [r.name for r in tainted_val.readers] if tainted_val.readers else [],
                    "level": max((s.label.value if hasattr(s.label, "value") else 5) for s in sources)
                    if sources
                    else 0,
                }

                # Check flow
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

        # Merge taint context into the call context
        if tainted_fields:
            ctx.taint_context["tainted_fields"] = tainted_fields
            ctx.extras["taint_checked"] = True

        if worst_flow == FlowDecision.DENY:
            return DispatchDecision.DENY
        if worst_flow == FlowDecision.ASK_USER:
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Wrap tool output in a TaintedValue for downstream propagation.

        The output is tagged with a ``TOOL_OUTPUT`` source so it can be
        tracked through subsequent operations.
        """
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
            ctx.extras["tainted_output"] = tainted_output
        except Exception:
            # Don't fail the call if taint wrapping has issues
            logger.debug("Could not taint-wrap output for %s", ctx.tool_name, exc_info=True)


# ---------------------------------------------------------------------------
# 2. Scanner middleware
# ---------------------------------------------------------------------------


class KatanaScanMiddleware(KatanaMiddleware):
    """Multi-layer scanner middleware for the Hermes dispatch pipeline.

    **Pre-dispatch**: runs the scanner on all string arguments to detect
    prompt injections, secrets, dangerous content, and Unicode attacks.

    **Post-dispatch**: scans tool outputs for content attacks and secret
    leakage.

    Args:
        vault_values:  Optional set of known secret values to detect leakage.
        block_threshold: Risk score threshold for blocking (default: 0.7).
        warn_threshold:  Risk score threshold for escalation (default: 0.4).
        enabled:        Whether this middleware is active.
    """

    def __init__(
        self,
        vault_values: set[str] | None = None,
        *,
        block_threshold: float = 0.7,
        warn_threshold: float = 0.4,
        check_injection: bool = True,
        check_secrets: bool = True,
        check_unicode: bool = True,
        check_content: bool = True,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.scan", enabled=enabled, priority=80)
        self._vault_values = vault_values or set()
        self._block_threshold = block_threshold
        self._warn_threshold = warn_threshold
        self._check_injection = check_injection
        self._check_secrets = check_secrets
        self._check_unicode = check_unicode
        self._check_content = check_content

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Scan all string arguments for attacks.

        Uses ``scan_input()`` for general text and ``scan_command()`` for
        command-type arguments.
        """
        from hermes_katana.scanner import scan_input, scan_command

        worst_score = 0.0
        all_results = []

        for arg_name, arg_val in ctx.args.items():
            text = str(arg_val) if arg_val is not None else ""
            if not text:
                continue

            # Use command scanner for command-like arguments
            if arg_name in ("command", "cmd", "shell_command", "script"):
                result = scan_command(
                    text,
                    check_secrets=self._check_secrets,
                    vault_values=self._vault_values,
                )
            else:
                result = scan_input(
                    text,
                    vault_values=self._vault_values,
                    check_injection=self._check_injection,
                    check_secrets=self._check_secrets,
                    check_unicode=self._check_unicode,
                    check_content=self._check_content,
                )

            all_results.append(result)
            worst_score = max(worst_score, result.risk_score)

            if result.has_findings:
                logger.debug(
                    "Scanner findings for %s.%s: %s",
                    ctx.tool_name,
                    arg_name,
                    result.summary,
                )

        ctx.scan_results = all_results
        ctx.extras["scan_risk_score"] = worst_score

        if worst_score >= self._block_threshold:
            findings_summary = "; ".join(r.summary for r in all_results if r.has_findings)
            ctx.deny(f"Scanner blocked: {findings_summary}")
            return DispatchDecision.DENY

        if worst_score >= self._warn_threshold:
            ctx.escalate(f"Scanner warning: risk_score={worst_score:.2f}")
            return DispatchDecision.ESCALATE

        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Scan tool output for content attacks and secret leakage."""
        if ctx.tool_output is None or ctx.is_denied:
            return

        try:
            from hermes_katana.scanner import scan_output

            output_text = str(ctx.tool_output)
            if output_text:
                result = scan_output(
                    output_text,
                    vault_values=self._vault_values,
                    check_injection=self._check_injection,
                    check_secrets=self._check_secrets,
                    check_unicode=self._check_unicode,
                    check_content=self._check_content,
                )
                ctx.extras["output_scan_result"] = result
                if result.has_findings:
                    logger.warning(
                        "Post-dispatch scan findings for %s: %s",
                        ctx.tool_name,
                        result.summary,
                    )
        except Exception:
            logger.debug("Post-dispatch scan failed for %s", ctx.tool_name, exc_info=True)


# ---------------------------------------------------------------------------
# 3. Policy middleware
# ---------------------------------------------------------------------------


class KatanaPolicyMiddleware(KatanaMiddleware):
    """Declarative policy evaluation middleware.

    Evaluates the :class:`PolicyEngine` against the current tool call
    and taint context.  Maps policy results to dispatch decisions:

    - ``PolicyResult.ALLOW``    → ``DispatchDecision.ALLOW``
    - ``PolicyResult.DENY``     → ``DispatchDecision.DENY``
    - ``PolicyResult.ESCALATE`` → ``DispatchDecision.ESCALATE``
    - ``PolicyResult.LOG_ONLY`` → ``DispatchDecision.ALLOW`` (with log)

    Args:
        engine:  The :class:`PolicyEngine` instance (or None for lazy init).
        preset:  Built-in preset name if ``engine`` is None (default: ``balanced``).
        enabled: Whether this middleware is active.
    """

    def __init__(
        self,
        engine: Any | None = None,
        *,
        preset: str = "balanced",
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.policy", enabled=enabled, priority=60)
        self._engine = engine
        self._preset = preset

    @property
    def engine(self) -> Any:
        """Lazy-load the policy engine."""
        if self._engine is None:
            from hermes_katana.policy import PolicyEngine

            self._engine = PolicyEngine.with_defaults(self._preset)
        return self._engine

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Evaluate policy engine for the current tool call.

        Uses the taint context built by upstream middleware (especially
        KatanaTaintMiddleware) for condition evaluation.
        """
        from hermes_katana.policy import PolicyResult

        result = self.engine.evaluate(
            tool_name=ctx.tool_name,
            args=ctx.args,
            taint_context=ctx.taint_context,
        )

        ctx.policy_result = result
        ctx.extras["policy_action"] = result.action.value
        ctx.extras["policy_reason"] = result.reason

        if result.matched_policy:
            ctx.extras["policy_name"] = result.matched_policy.name

        # Deny-by-default for unknown tools with tainted args
        if result.matched_policy is None and ctx.taint_context.get("tainted_fields"):
            ctx.deny(f"Unknown tool '{ctx.tool_name}' with tainted arguments — deny by default (no matching policy)")
            return DispatchDecision.DENY

        # Escalate unknown tools with clean args (fail-closed)
        if result.matched_policy is None:
            ctx.escalate(f"Unknown tool '{ctx.tool_name}' — escalate by default (no matching policy)")
            return DispatchDecision.ESCALATE

        if result.action == PolicyResult.DENY:
            ctx.deny(f"Policy denied: {result.reason}")
            return DispatchDecision.DENY

        if result.action == PolicyResult.ESCALATE:
            ctx.escalate(f"Policy escalation: {result.reason}")
            return DispatchDecision.ESCALATE

        if result.action == PolicyResult.LOG_ONLY:
            logger.info(
                "Policy LOG_ONLY for %s: %s",
                ctx.tool_name,
                result.reason,
            )

        return DispatchDecision.ALLOW


# ---------------------------------------------------------------------------
# 4. Audit middleware
# ---------------------------------------------------------------------------


class KatanaAuditMiddleware(KatanaMiddleware):
    """Audit trail middleware — logs every dispatch decision.

    Records structured audit events for both pre-dispatch decisions and
    post-dispatch results.  Integrates with the ``hermes_katana.audit``
    module when available, and falls back to Python logging otherwise.

    Args:
        audit_trail: Optional audit trail instance (lazy-loaded if None).
        log_allow:   Whether to log ALLOW decisions (default: True).
        enabled:     Whether this middleware is active.
    """

    def __init__(
        self,
        audit_trail: Any | None = None,
        *,
        log_allow: bool = True,
        enabled: bool = True,
    ) -> None:
        # GAP 4.4: Audit runs BEFORE policy (higher priority) so denied calls
        # are always logged even on short-circuit.
        super().__init__(name="katana.audit", enabled=enabled, priority=65)
        self._audit_trail = audit_trail
        self._log_allow = log_allow

    @property
    def audit_trail(self) -> Any | None:
        """Lazy-load the audit trail if available."""
        if self._audit_trail is None:
            try:
                from hermes_katana.audit import AuditTrail

                self._audit_trail = AuditTrail()
            except ImportError:
                # Audit module not yet available — will use logging fallback
                pass
        return self._audit_trail

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Log the pre-dispatch decision.

        This middleware never blocks calls — it only observes and records.
        """
        # Don't log allowed calls if configured to skip them
        if ctx.decision == DispatchDecision.ALLOW and not self._log_allow:
            return DispatchDecision.ALLOW

        event = {
            "type": "tool_dispatch",
            "phase": "pre",
            "call_id": ctx.call_id,
            "tool_name": ctx.tool_name,
            "decision": ctx.decision.value,
            "deny_reasons": ctx.deny_reasons,
            "escalate_reasons": ctx.escalate_reasons,
            "scan_risk_score": ctx.extras.get("scan_risk_score", 0.0),
            "policy_action": ctx.extras.get("policy_action"),
            "policy_name": ctx.extras.get("policy_name"),
            "has_taint": bool(ctx.taint_context.get("tainted_fields")),
            "timestamp": time.time(),
        }

        self._record_event(event, ctx)
        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        """Log the post-dispatch result including tool output metadata."""
        if ctx.is_denied and ctx.extras.get("katana.audit_short_circuit_logged"):
            return

        event = {
            "type": "tool_dispatch",
            "phase": "post",
            "call_id": ctx.call_id,
            "tool_name": ctx.tool_name,
            "decision": ctx.decision.value,
            "tool_duration_ms": ctx.tool_duration_ms,
            "middleware_ms": ctx.total_middleware_ms,
            "had_error": ctx.tool_error is not None,
            "output_scan_findings": bool(ctx.extras.get("output_scan_result")),
            "timestamp": time.time(),
        }

        self._record_event(event, ctx)

    def on_short_circuit(self, ctx: CallContext) -> None:
        """Log a denied pre-dispatch call that short-circuited the chain."""
        event = {
            "type": "tool_dispatch",
            "phase": "short_circuit",
            "call_id": ctx.call_id,
            "tool_name": ctx.tool_name,
            "decision": ctx.decision.value,
            "deny_reasons": ctx.deny_reasons,
            "escalate_reasons": ctx.escalate_reasons,
            "scan_risk_score": ctx.extras.get("scan_risk_score", 0.0),
            "policy_action": ctx.extras.get("policy_action"),
            "policy_name": ctx.extras.get("policy_name"),
            "short_circuit_middleware": ctx.extras.get("short_circuit_middleware"),
            "timestamp": time.time(),
        }
        self._record_event(event, ctx)
        ctx.extras["katana.audit_short_circuit_logged"] = True

    def _record_event(self, event: dict[str, Any], ctx: CallContext) -> None:
        """Write an event to the audit trail or fall back to logging."""
        trail = self.audit_trail
        if trail is not None:
            try:
                from hermes_katana.audit import AuditEntry, AuditEventType

                args_hash = hashlib.sha256(
                    f"{ctx.call_id}|{event.get('phase', '')}|{event.get('tool_name', '')}".encode("utf-8")
                ).hexdigest()[:16]
                entry = AuditEntry(
                    event_type=AuditEventType.TOOL_CALL,
                    tool_name=str(event.get("tool_name", "")),
                    args_hash=args_hash,
                    decision=str(event.get("decision", "")),
                    details=json.dumps(event, sort_keys=True, default=str),
                )
                trail.log(entry)
                return
            except Exception:
                logger.debug("Audit trail record failed, falling back to logging", exc_info=True)

        # Fallback: structured log
        level = logging.WARNING if ctx.is_denied else logging.INFO
        logger.log(
            level,
            "AUDIT [%s] %s %s → %s (risk=%.2f, %s)",
            event.get("phase", "?"),
            event.get("call_id", "?"),
            event.get("tool_name", "?"),
            event.get("decision", "?"),
            event.get("scan_risk_score", 0.0),
            event.get("policy_action", "none"),
        )


# ---------------------------------------------------------------------------
# Factory: create the default middleware chain
# ---------------------------------------------------------------------------


def create_default_chain(
    config: dict[str, Any] | None = None,
) -> "MiddlewareChain":
    """Build the default Katana middleware chain.

    Creates and wires the four standard middleware in the recommended
    order.  Configuration overrides can disable individual middleware
    or adjust thresholds.

    Args:
        config: Optional configuration dict with keys:

            - ``taint.enabled`` (bool, default True)
            - ``scan.enabled`` (bool, default True)
            - ``scan.block_threshold`` (float, default 0.7)
            - ``scan.warn_threshold`` (float, default 0.4)
            - ``scan.vault_values`` (set[str], default empty)
            - ``policy.enabled`` (bool, default True)
            - ``policy.preset`` (str, default "balanced")
            - ``policy.engine`` (PolicyEngine instance, optional)
            - ``audit.enabled`` (bool, default True)
            - ``audit.log_allow`` (bool, default True)
            - ``audit.trail`` (AuditTrail instance, optional)

    Returns:
        A fully-configured :class:`MiddlewareChain`.

    Example::

        chain = create_default_chain({
            "policy.preset": "paranoid",
            "scan.block_threshold": 0.5,
            "audit.log_allow": False,
        })
    """
    from hermes_katana.middleware.chain import MiddlewareChain

    cfg = config or {}
    chain = MiddlewareChain()

    # 1. Taint tracking (highest priority)
    taint_mw = KatanaTaintMiddleware(
        tracker=cfg.get("taint.tracker"),
        enabled=cfg.get("taint.enabled", True),
    )
    chain.add(taint_mw)

    # 2. Scanner
    scan_mw = KatanaScanMiddleware(
        vault_values=cfg.get("scan.vault_values"),
        block_threshold=cfg.get("scan.block_threshold", 0.7),
        warn_threshold=cfg.get("scan.warn_threshold", 0.4),
        check_injection=cfg.get("scan.check_injection", True),
        check_secrets=cfg.get("scan.check_secrets", True),
        check_unicode=cfg.get("scan.check_unicode", True),
        check_content=cfg.get("scan.check_content", True),
        enabled=cfg.get("scan.enabled", True),
    )
    chain.add(scan_mw)

    # 3. Policy engine
    policy_mw = KatanaPolicyMiddleware(
        engine=cfg.get("policy.engine"),
        preset=cfg.get("policy.preset", "balanced"),
        enabled=cfg.get("policy.enabled", True),
    )
    chain.add(policy_mw)

    # 4. Audit trail (lowest priority — observes everything)
    audit_mw = KatanaAuditMiddleware(
        audit_trail=cfg.get("audit.trail"),
        log_allow=cfg.get("audit.log_allow", True),
        enabled=cfg.get("audit.enabled", True),
    )
    chain.add(audit_mw)

    logger.info(
        "Default middleware chain created: %s",
        [m.name for m in chain.list_middleware()],
    )
    return chain
