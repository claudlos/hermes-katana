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
        scabbard_profile: katana_v15_minilm   # minimal | standard | full | katana_v15_minilm | katana_v15_large
        scabbard_backend: onnx                # onnx for MiniLM, torch for v15 large
        scabbard_device: cuda                 # optional torch device for v15 large
        scabbard_route_mode: balanced         # off | content_only | balanced | paranoid
        scabbard_scan_outputs: true           # scan routed tool-output content
        scabbard_audit_routes: true           # record scan/skip route reasons
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any, Optional

from hermes_katana.security_logging import log_security_event, summarize_tool_call

__all__ = [
    "register",
    "setup",
    "plugin_name",
    "plugin_version",
]

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
_initialization_error: str | None = None


def register(context: Any) -> None:
    """Initialize the Katana plugin.

    Called by the Hermes plugin manager with a PluginContext that provides
    ``register_hook``, ``register_tool``, and ``config``.

    The function is named ``register`` to match the Hermes plugin contract.
    A ``setup`` alias is provided for backward compatibility with tests.

    Args:
        context: Hermes PluginContext instance.
    """
    global _chain, _audit_trail, _tracker, _vault, _config, _initialized, _initialization_error

    _config = getattr(context, "config", {}) or {}
    if not isinstance(_config, dict):
        _config = {}

    _chain = None
    _audit_trail = None
    _tracker = None
    _vault = None
    _initialized = False
    _initialization_error = None

    try:
        _chain, _audit_trail, _tracker, _vault = _initialize_runtime(_config)
        _initialized = True
    except Exception as exc:
        from hermes_katana.cli._support import hermetic_ml_ready_required

        _initialization_error = str(exc) or exc.__class__.__name__
        log_security_event(
            logger,
            logging.WARNING,
            "plugin_startup_failed",
            reason=_initialization_error,
            fail_closed=True,
            hermetic_ml_required=hermetic_ml_ready_required(_config),
            policy_preset=_config.get("policy_preset", "balanced"),
        )

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
            "Show HermesKatana security status: active policy, middleware chain, scan stats, and taint tracker state."
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

    log_security_event(
        logger,
        logging.INFO,
        "plugin_initialized",
        plugin_version=plugin_version,
        initialized=_initialized,
        fail_closed=not _initialized,
        policy_preset=_config.get("policy_preset", "balanced"),
        taint_enabled=_config.get("taint_enabled", True),
        audit_enabled=_config.get("audit_enabled", True),
    )


# ---------------------------------------------------------------------------
# Runtime initialization
# ---------------------------------------------------------------------------


def _initialize_runtime(config: dict[str, Any]) -> tuple:
    """Build the middleware chain and supporting subsystems.

    Returns:
        (chain, audit_trail, tracker, vault) tuple.
    """
    from hermes_katana.cli._support import enforce_hermetic_ml_readiness
    from hermes_katana.middleware.integration import create_default_chain
    from hermes_katana.taint import TaintTracker

    enforce_hermetic_ml_readiness(config)

    tracker = TaintTracker.get_instance()
    vault = _open_vault()
    audit_trail = _open_audit(config)

    _katana_profile = config.get("katana_profile", config.get("profile"))
    chain_config = {
        "taint.enabled": config.get("taint_enabled", True),
        "taint.tracker": tracker,
        "scan.enabled": config.get("scan_enabled", True),
        "scan.check_injection": config.get("scan_check_injection", True),
        "scan.check_secrets": config.get("scan_check_secrets", True),
        "scan.check_unicode": config.get("scan_check_unicode", True),
        "scan.check_content": config.get("scan_check_content", True),
        "scan.vault_values": _collect_vault_values(vault),
        "mcp.enabled": config.get("mcp_enabled", True),
        "mcp.block_on_critical": config.get("mcp_block_on_critical", True),
        "multiturn.enabled": config.get("multiturn_enabled", True),
        "multiturn.block_threshold": config.get("multiturn_block_threshold", 0.75),
        "multiturn.warn_threshold": config.get("multiturn_warn_threshold", 0.45),
        "rag_injection.enabled": config.get("rag_injection_enabled", True),
        "rag_injection.block_threshold": config.get("rag_injection_block_threshold", 0.90),
        "rag_injection.warn_threshold": config.get("rag_injection_warn_threshold", 0.60),
        "policy.enabled": config.get("policy_enabled", True),
        "audit.enabled": config.get("audit_enabled", True),
        "audit.log_allow": config.get("audit_log_allow", True),
        "audit.trail": audit_trail,
    }
    if _katana_profile is not None:
        chain_config["profile"] = _katana_profile
    for external_key, chain_key, default in (
        ("scan_block_threshold", "scan.block_threshold", 0.7),
        ("scan_warn_threshold", "scan.warn_threshold", 0.4),
        ("policy_preset", "policy.preset", "balanced"),
    ):
        if external_key in config or _katana_profile is None:
            chain_config[chain_key] = config.get(external_key, default)

    from hermes_katana.scabbard import ScabbardConfig

    _scabbard_profile = config.get("scabbard_profile")
    _scabbard_backend = config.get("scabbard_backend")
    _scabbard_device = config.get("scabbard_device")
    _scabbard_model_path = config.get("scabbard_model_path")
    if _scabbard_profile == "minimal":
        scabbard_cfg = ScabbardConfig.minimal()
    elif _scabbard_profile == "full":
        scabbard_cfg = ScabbardConfig.full()
    elif _scabbard_profile == "standard":
        scabbard_cfg = ScabbardConfig.standard()
    elif _scabbard_profile in {"katana_v15_minilm", "v15_minilm", "minilm"}:
        scabbard_cfg = ScabbardConfig.katana_v15_minilm(
            model_path=_scabbard_model_path,
            backend=_scabbard_backend or "onnx",
            device=_scabbard_device,
        )
    elif _scabbard_profile in {"katana_v15_large", "v15_large", "v15"}:
        scabbard_cfg = ScabbardConfig.katana_v15_large(
            model_path=_scabbard_model_path,
            backend=_scabbard_backend or "torch",
            device=_scabbard_device,
        )
    elif _katana_profile is None:
        scabbard_cfg = ScabbardConfig.runtime_default()
    else:
        scabbard_cfg = None

    if scabbard_cfg is not None:
        chain_config["scabbard.config"] = scabbard_cfg
        chain_config["scabbard.profile"] = scabbard_cfg.profile
    chain_config["scabbard.enabled"] = config.get("scabbard_enabled", True)
    for external_key, chain_key, default in (
        ("scabbard_route_mode", "scabbard.route_mode", "balanced"),
        ("scabbard_scan_outputs", "scabbard.scan_outputs", True),
        ("scabbard_audit_routes", "scabbard.audit_routes", True),
    ):
        if external_key in config or _katana_profile is None:
            chain_config[chain_key] = config.get(external_key, default)

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
        return {value for key in vault.list_keys() if (value := vault.get(key))}
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
    if _initialization_error is not None:
        from hermes_katana.exceptions import KatanaSecurityError

        log_security_event(
            logger,
            logging.WARNING,
            "tool_call_blocked_runtime_uninitialized",
            reason=_initialization_error,
            fail_closed=True,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id),
        )
        raise KatanaSecurityError(
            f"HermesKatana initialization failed; blocking tool call '{tool_name}' until the runtime is fixed",
            tool_name=tool_name,
            reasons=[_initialization_error],
        )

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
    except Exception as exc:
        log_security_event(
            logger,
            logging.ERROR,
            "tool_call_pre_dispatch_failed",
            error=str(exc) or exc.__class__.__name__,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
            **summarize_tool_call(tool_name, args or {}, task_id=task_id, call_id=ctx.call_id),
        )
        raise KatanaSecurityError(
            f"Tool call '{tool_name}' denied because HermesKatana pre-dispatch failed",
            tool_name=tool_name,
            reasons=[str(exc) or exc.__class__.__name__],
            call_id=ctx.call_id,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
        ) from exc

    if decision == DispatchDecision.DENY:
        reasons = ctx.deny_reasons or ["Blocked by HermesKatana security policy"]
        log_security_event(
            logger,
            logging.WARNING,
            "tool_call_denied",
            reasons=reasons,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
            taint_context=ctx.taint_context,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id, call_id=ctx.call_id),
        )
        raise KatanaSecurityError(
            f"Tool call '{tool_name}' denied: {'; '.join(reasons)}",
            tool_name=tool_name,
            reasons=reasons,
            call_id=ctx.call_id,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
        )

    if decision == DispatchDecision.ESCALATE:
        reasons = ctx.escalate_reasons or ["Requires human approval"]
        log_security_event(
            logger,
            logging.WARNING,
            "tool_call_escalated",
            reasons=reasons,
            scan_score=ctx.extras.get("scan_risk_score", 0.0),
            taint_context=ctx.taint_context,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id, call_id=ctx.call_id),
        )
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
    except Exception as exc:
        log_security_event(
            logger,
            logging.ERROR,
            "tool_call_post_dispatch_failed",
            error=str(exc) or exc.__class__.__name__,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id, call_id=ctx.call_id),
        )

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
                details=json.dumps(
                    {
                        "session_id": session_id,
                        "task_id": task_id,
                        "plugin_version": plugin_version,
                        "timestamp": time.time(),
                    }
                ),
            )
            _audit_trail.log(entry)
        except Exception:
            logger.debug("Failed to log session start", exc_info=True)

    # Initialize a fresh taint scope for this session
    if _tracker is not None:
        try:
            _tracker.clear()
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
                details=json.dumps(
                    {
                        "session_id": session_id,
                        "task_id": task_id,
                        "timestamp": time.time(),
                    }
                ),
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
    from hermes_katana.cli._support import collect_ml_runtime_status

    status: dict[str, Any] = {
        "plugin_version": plugin_version,
        "initialized": _initialized,
    }

    if not _initialized:
        status["message"] = "HermesKatana plugin not initialized"
        if _initialization_error is not None:
            status["initialization_error"] = _initialization_error
        return json.dumps(status, indent=2)

    # Middleware chain
    if _chain is not None:
        try:
            from hermes_katana.middleware.integration import collect_chain_diagnostics

            mw_list = _chain.list_middleware()
            status["middleware"] = [
                {
                    "name": m.name,
                    "enabled": m.enabled,
                    "priority": m.priority,
                }
                for m in mw_list
            ]
            status["diagnostics"] = collect_chain_diagnostics(_chain)
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

    status["ml_runtime"] = collect_ml_runtime_status()

    return json.dumps(status, indent=2)


# ---------------------------------------------------------------------------
# Context stash (pass pre-dispatch context to post-dispatch)
# ---------------------------------------------------------------------------

_context_stash: dict[str, Any] = {}
_context_stash_lock = threading.RLock()


def _current_runtime_key() -> tuple[int, int | None]:
    """Return a best-effort runtime key for the current thread/task."""
    task_key: int | None = None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    if task is not None:
        task_key = id(task)
    return threading.get_ident(), task_key


def _context_task_id(ctx: Any) -> str:
    """Extract the task id stored in a stashed context."""
    extras = getattr(ctx, "extras", None)
    if isinstance(extras, dict):
        value = extras.get("task_id", "")
        if isinstance(value, str):
            return value
    value = getattr(ctx, "task_id", "")
    return value if isinstance(value, str) else ""


def _context_runtime_key(ctx: Any) -> tuple[int, int | None] | None:
    """Extract the runtime key stored in a stashed context."""
    extras = getattr(ctx, "extras", None)
    if isinstance(extras, dict):
        value = extras.get("_runtime_key")
        if isinstance(value, tuple) and len(value) == 2:
            return value
    return getattr(ctx, "_katana_runtime_key", None)


def _stash_context(call_id: str, ctx: Any) -> None:
    """Store a pre-dispatch context for post-dispatch recovery."""
    runtime_key = _current_runtime_key()
    extras = getattr(ctx, "extras", None)
    if isinstance(extras, dict):
        extras.setdefault("_runtime_key", runtime_key)
    else:
        setattr(ctx, "_katana_runtime_key", runtime_key)

    with _context_stash_lock:
        _context_stash[call_id] = ctx
        # Prevent unbounded growth: evict old entries
        if len(_context_stash) > 100:
            oldest = list(_context_stash.keys())[: len(_context_stash) - 50]
            for k in oldest:
                _context_stash.pop(k, None)


def _pop_context(tool_name: str, task_id: str) -> Any:
    """Find and remove the pre-dispatch context for this tool and task_id.

    Matches on both tool_name and task_id to avoid context mix-up under
    concurrent calls. Falls back to tool_name-only match if task_id is
    not available.
    """
    runtime_key = _current_runtime_key()

    with _context_stash_lock:
        ordered = list(_context_stash.keys())

        # First pass: exact tool + task id + runtime.
        for call_id in reversed(ordered):
            ctx = _context_stash.get(call_id)
            if (
                ctx is not None
                and getattr(ctx, "tool_name", "") == tool_name
                and task_id
                and _context_task_id(ctx) == task_id
                and _context_runtime_key(ctx) == runtime_key
            ):
                _context_stash.pop(call_id, None)
                return ctx

        # Second pass: exact tool + task id, even if runtime moved.
        for call_id in reversed(ordered):
            ctx = _context_stash.get(call_id)
            if (
                ctx is not None
                and getattr(ctx, "tool_name", "") == tool_name
                and task_id
                and _context_task_id(ctx) == task_id
            ):
                _context_stash.pop(call_id, None)
                return ctx

        # Third pass: tool + runtime key for call sites that do not provide task ids.
        for call_id in reversed(ordered):
            ctx = _context_stash.get(call_id)
            if (
                ctx is not None
                and getattr(ctx, "tool_name", "") == tool_name
                and _context_runtime_key(ctx) == runtime_key
            ):
                _context_stash.pop(call_id, None)
                return ctx

        # Last resort: tool-name-only recovery.
        for call_id in reversed(ordered):
            ctx = _context_stash.get(call_id)
            if ctx is not None and getattr(ctx, "tool_name", "") == tool_name:
                _context_stash.pop(call_id, None)
                return ctx
    return None


def _clear_stash() -> None:
    """Clear all stashed contexts (on session end)."""
    with _context_stash_lock:
        _context_stash.clear()


# Backward-compatibility alias used by tests and alternative plugin loaders.
setup = register
