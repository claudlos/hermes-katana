"""Native Hermes agent plugin for HermesKatana.

Registers with the Hermes plugin system via pip entry point
(``hermes_agent.plugins``) and hooks into the tool dispatch pipeline
using ``pre_tool_call`` and ``post_tool_call`` hooks.

This replaces the source-patching approach with a zero-modification
integration: install hermes-katana, and the plugin activates automatically.

Plugin hooks
------------
- ``pre_tool_call``:  Run the full middleware chain (taint, scan, policy).
  Raises :class:`KatanaSecurityError` on DENY,
  :class:`EscalationRequired` on ESCALATE.
- ``post_tool_call``: Scan tool results, register taint on outputs,
  and log the completed call to the audit trail.
- ``on_session_start``: Initialize audit session entry and taint scope.
- ``on_session_end``: Close audit session and flush state.

Configuration
-------------
Plugin config lives under ``katana:`` in Hermes ``config.yaml``::

    plugins:
      katana:
        policy_preset: balanced     # paranoid | balanced | permissive
        scan_block_threshold: 0.7
        taint_enabled: true
        audit_enabled: true
        audit_log_allow: true
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plugin metadata (required by Hermes plugin system)
# ---------------------------------------------------------------------------

plugin_name = "katana"
plugin_version = "0.2.0"

# ---------------------------------------------------------------------------
# Module-level state (initialized in setup())
# ---------------------------------------------------------------------------

_chain: Any = None
_audit_trail: Any = None
_tracker: Any = None
_vault: Any = None
_config: dict[str, Any] = {}
_initialized: bool = False


def register(context: Any) -> None:
    """Initialize the Katana plugin.

    Called by the Hermes plugin manager with a PluginContext that provides
    ``register_hook``, ``register_tool``, and ``config``.

    The function is named ``register`` to match the Hermes plugin contract.
    A ``setup`` alias is provided for backward compatibility with tests.

    Args:
        context: Hermes PluginContext instance.
    """
    global _chain, _audit_trail, _tracker, _vault, _config, _initialized

    _config = getattr(context, "config", {}) or {}
    if not isinstance(_config, dict):
        _config = {}

    try:
        _chain, _audit_trail, _tracker, _vault = _initialize_runtime(_config)
        _initialized = True
    except Exception:
        logger.warning(
            "HermesKatana plugin initialization failed; running in passthrough mode",
            exc_info=True,
        )
        _initialized = False
        return

    # Register hooks
    context.register_hook("pre_tool_call", _on_pre_tool_call)
    context.register_hook("post_tool_call", _on_post_tool_call)
    context.register_hook("on_session_start", _on_session_start)
    context.register_hook("on_session_end", _on_session_end)

    # Register the katana_status tool.
    # The Hermes register_tool API requires a ``toolset`` parameter.
    # Schema must be the inner dict only — register_tool wraps it in
    # {"type": "function", "function": schema} automatically.
    _katana_schema = {
        "name": "katana_status",
        "description": (
            "Show HermesKatana security status: active policy, "
            "middleware chain, scan stats, and taint tracker state."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
    try:
        context.register_tool(
            name="katana_status",
            toolset="katana",
            schema=_katana_schema,
            handler=_handle_katana_status,
            description="HermesKatana security status",
        )
    except TypeError:
        try:
            context.register_tool(
                name="katana_status",
                schema=_katana_schema,
                handler=_handle_katana_status,
            )
        except Exception:
            logger.debug("Could not register katana_status tool", exc_info=True)

    logger.info(
        "HermesKatana plugin v%s initialized (policy=%s, taint=%s, audit=%s)",
        plugin_version,
        _config.get("policy_preset", "balanced"),
        _config.get("taint_enabled", True),
        _config.get("audit_enabled", True),
    )


# ---------------------------------------------------------------------------
# Runtime initialization
# ---------------------------------------------------------------------------


def _initialize_runtime(config: dict[str, Any]) -> tuple:
    """Build the middleware chain and supporting subsystems.

    Returns:
        (chain, audit_trail, tracker, vault) tuple.
    """
    from hermes_katana.middleware.integration import create_default_chain
    from hermes_katana.taint import TaintTracker

    tracker = TaintTracker.get_instance()
    vault = _open_vault()
    audit_trail = _open_audit(config)

    chain_config = {
        "taint.enabled": config.get("taint_enabled", True),
        "taint.tracker": tracker,
        "scan.enabled": config.get("scan_enabled", True),
        "scan.block_threshold": config.get("scan_block_threshold", 0.7),
        "scan.warn_threshold": config.get("scan_warn_threshold", 0.4),
        "scan.check_injection": config.get("scan_check_injection", True),
        "scan.check_secrets": config.get("scan_check_secrets", True),
        "scan.check_unicode": config.get("scan_check_unicode", True),
        "scan.check_content": config.get("scan_check_content", True),
        "scan.vault_values": _collect_vault_values(vault),
        "policy.enabled": config.get("policy_enabled", True),
        "policy.preset": config.get("policy_preset", "balanced"),
        "audit.enabled": config.get("audit_enabled", True),
        "audit.log_allow": config.get("audit_log_allow", True),
        "audit.trail": audit_trail,
    }

    chain = create_default_chain(chain_config)
    return chain, audit_trail, tracker, vault


def _open_vault():
    """Open the vault, returning None if unavailable."""
    try:
        from hermes_katana.vault import Vault
        return Vault(auto_create=False)
    except Exception:
        logger.debug("Vault not available for plugin", exc_info=True)
        return None


def _open_audit(config: dict[str, Any]):
    """Open the audit trail, returning None if disabled."""
    if not config.get("audit_enabled", True):
        return None
    try:
        from hermes_katana.audit import AuditTrail
        return AuditTrail()
    except Exception:
        logger.debug("Audit trail not available for plugin", exc_info=True)
        return None


def _collect_vault_values(vault: Any) -> set[str]:
    """Collect vault values for secret-leak scanning."""
    if vault is None:
        return set()
    try:
        return {
            value
            for key in vault.list_keys()
            if (value := vault.get(key))
        }
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Hook: pre_tool_call
# ---------------------------------------------------------------------------


def _on_pre_tool_call(
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    task_id: str = "",
    **kwargs: Any,
) -> None:
    """Middleware chain evaluation before tool execution.

    Runs taint check, scanning, and policy evaluation. Raises on DENY
    or ESCALATE so Hermes can handle the decision.

    Args:
        tool_name: Name of the tool being called.
        args: Tool arguments dict.
        task_id: Current task/session identifier.
    """
    if not _initialized or _chain is None:
        return

    from hermes_katana.exceptions import EscalationRequired, KatanaSecurityError
    from hermes_katana.middleware.chain import CallContext, DispatchDecision

    ctx = CallContext(
        tool_name=tool_name,
        args=args or {},
        extras={"task_id": task_id},
    )

    try:
        decision = _chain.execute_pre(ctx)
    except Exception:
        logger.error("Middleware chain error in pre_tool_call", exc_info=True)
        return  # Fail open — don't block on internal errors

    if decision == DispatchDecision.DENY:
        reasons = ctx.deny_reasons or ["Blocked by HermesKatana security policy"]
        raise KatanaSecurityError(
            f"Tool call '{tool_name}' denied: {'; '.join(reasons)}",
            tool_name=tool_name,
            reasons=reasons,
            call_id=ctx.call_id,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
        )

    if decision == DispatchDecision.ESCALATE:
        reasons = ctx.escalate_reasons or ["Requires human approval"]
        raise EscalationRequired(
            f"Tool call '{tool_name}' requires approval: {'; '.join(reasons)}",
            tool_name=tool_name,
            reasons=reasons,
            call_id=ctx.call_id,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
            escalation_context={
                "tool_name": tool_name,
                "args_keys": list((args or {}).keys()),
                "reasons": reasons,
                "taint_context": ctx.taint_context,
            },
        )

    # Stash context for post_tool_call to pick up
    _stash_context(ctx.call_id, ctx)


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    result: str = "",
    task_id: str = "",
    **kwargs: Any,
) -> None:
    """Post-execution scanning, taint registration, and audit logging.

    Args:
        tool_name: Name of the tool that was called.
        args: Tool arguments dict.
        result: The tool's return value (string).
        task_id: Current task/session identifier.
    """
    if not _initialized or _chain is None:
        return

    from hermes_katana.middleware.chain import CallContext

    # Recover pre-dispatch context or create a fresh one
    ctx = _pop_context(tool_name, task_id)
    if ctx is None:
        ctx = CallContext(
            tool_name=tool_name,
            args=args or {},
            extras={"task_id": task_id},
        )

    ctx.tool_output = result
    ctx.tool_error = None

    try:
        _chain.execute_post(ctx)
    except Exception:
        logger.error("Middleware chain error in post_tool_call", exc_info=True)

    # Register taint on tool output
    if _tracker is not None and isinstance(result, str) and result:
        try:
            from hermes_katana.taint.registrar import taint_tool_output
            taint_tool_output(result, tool_name)
        except Exception:
            logger.debug("Taint registration failed for %s output", tool_name, exc_info=True)


# ---------------------------------------------------------------------------
# Hook: session lifecycle
# ---------------------------------------------------------------------------


def _on_session_start(
    session_id: str = "",
    task_id: str = "",
    **kwargs: Any,
) -> None:
    """Log session start to audit trail."""
    if not _initialized:
        return

    if _audit_trail is not None:
        try:
            from hermes_katana.audit import AuditEntry, AuditEventType
            entry = AuditEntry(
                event_type=AuditEventType.SESSION_START,
                details=json.dumps({
                    "session_id": session_id,
                    "task_id": task_id,
                    "plugin_version": plugin_version,
                    "timestamp": time.time(),
                }),
            )
            _audit_trail.log(entry)
        except Exception:
            logger.debug("Failed to log session start", exc_info=True)

    # Initialize a fresh taint scope for this session
    if _tracker is not None:
        try:
            _tracker.scoped()
        except Exception:
            pass


def _on_session_end(
    session_id: str = "",
    task_id: str = "",
    **kwargs: Any,
) -> None:
    """Log session end to audit trail and flush state."""
    if not _initialized:
        return

    if _audit_trail is not None:
        try:
            from hermes_katana.audit import AuditEntry, AuditEventType
            entry = AuditEntry(
                event_type=AuditEventType.SESSION_END,
                details=json.dumps({
                    "session_id": session_id,
                    "task_id": task_id,
                    "timestamp": time.time(),
                }),
            )
            _audit_trail.log(entry)
        except Exception:
            logger.debug("Failed to log session end", exc_info=True)

    _clear_stash()


# ---------------------------------------------------------------------------
# Tool: katana_status
# ---------------------------------------------------------------------------


def _handle_katana_status(**kwargs: Any) -> str:
    """Return current HermesKatana security status as a JSON string."""
    status: dict[str, Any] = {
        "plugin_version": plugin_version,
        "initialized": _initialized,
    }

    if not _initialized:
        status["message"] = "HermesKatana plugin not initialized"
        return json.dumps(status, indent=2)

    # Middleware chain
    if _chain is not None:
        try:
            mw_list = _chain.list_middleware()
            status["middleware"] = [
                {
                    "name": m.name,
                    "enabled": m.enabled,
                    "priority": m.priority,
                }
                for m in mw_list
            ]
        except Exception:
            status["middleware"] = "error reading chain"

    # Policy
    status["policy_preset"] = _config.get("policy_preset", "balanced")

    # Taint tracker
    if _tracker is not None:
        try:
            stats = _tracker.stats
            status["taint_tracker"] = {
                "registered_values": stats.values_registered,
                "flow_checks": stats.flow_checks,
                "flow_denials": stats.flow_denied,
            }
        except Exception:
            status["taint_tracker"] = "available"

    # Vault
    if _vault is not None:
        try:
            keys = _vault.list_keys()
            status["vault"] = {"secret_count": len(keys), "locked": _vault.is_locked()}
        except Exception:
            status["vault"] = "available"
    else:
        status["vault"] = "not configured"

    # Audit
    if _audit_trail is not None:
        status["audit"] = "active"
    else:
        status["audit"] = "disabled"

    return json.dumps(status, indent=2)


# ---------------------------------------------------------------------------
# Context stash (pass pre-dispatch context to post-dispatch)
# ---------------------------------------------------------------------------

_context_stash: dict[str, Any] = {}


def _stash_context(call_id: str, ctx: Any) -> None:
    """Store a pre-dispatch context for post-dispatch recovery."""
    _context_stash[call_id] = ctx
    # Prevent unbounded growth: evict old entries
    if len(_context_stash) > 100:
        oldest = list(_context_stash.keys())[: len(_context_stash) - 50]
        for k in oldest:
            _context_stash.pop(k, None)


def _pop_context(tool_name: str, task_id: str) -> Any:
    """Find and remove the most recent pre-dispatch context for this tool.

    Falls back to the most recent context if an exact match isn't found.
    """
    # Try to find by tool name (most recent first)
    for call_id in reversed(list(_context_stash.keys())):
        ctx = _context_stash.get(call_id)
        if ctx is not None and getattr(ctx, "tool_name", "") == tool_name:
            _context_stash.pop(call_id, None)
            return ctx
    return None


def _clear_stash() -> None:
    """Clear all stashed contexts (on session end)."""
    _context_stash.clear()


# Backward-compatibility alias used by tests and alternative plugin loaders.
setup = register
