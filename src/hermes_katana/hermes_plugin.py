"""Native Hermes agent plugin for HermesKatana.

Registers with the Hermes plugin system via pip entry point
(``hermes_agent.plugins``) and hooks into the tool dispatch pipeline
using ``pre_tool_call``, ``post_tool_call``, and ``transform_tool_result``
hooks.

Modern Hermes loads entry-point plugins only when they are listed in
``plugins.enabled``. When enabled, this plugin provides a zero-modification
integration path; the source patch installer remains available for checkout
level enforcement.

Plugin hooks
------------
- ``pre_tool_call``: Run the full middleware chain (taint, scan, policy)
  and return a Hermes block directive on DENY/ESCALATE.
- ``post_tool_call``: Observe completed calls and prepare transformed output.
- ``transform_tool_result``: Return the final scanned/redacted tool output.
- ``on_session_start``: Initialize audit session entry and taint scope.
- ``on_session_end``: Close audit session and flush state.

Configuration
-------------
Plugin config lives under ``katana:`` in Hermes ``config.yaml``::

    plugins:
      katana:
        policy_preset: balanced     # max | balanced | permissive
        scan_block_threshold: 0.7
        taint_enabled: true
        audit_enabled: true
        audit_log_allow: true
        scabbard_profile: katana_v17_minilm   # minimal | standard | full | katana_v17_minilm | katana_v15_minilm | katana_v15_large
        scabbard_backend: torch                # torch for v17_minilm (default), onnx for v15_minilm, torch for v15 large
        scabbard_device: cuda                 # optional torch device for v15 large
        scabbard_route_mode: balanced         # off | content_only | balanced | max
        scabbard_scan_outputs: true           # scan routed tool-output content
        scabbard_audit_routes: true           # record scan/skip route reasons
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Optional

from hermes_katana._version import __version__
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
plugin_version = __version__

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
_result_transform_stash: dict[tuple[str, str, str, str, str], str] = {}
_result_transform_stash_lock = threading.RLock()


def register(context: Any) -> None:
    """Initialize the Katana plugin.

    Called by the Hermes plugin manager with a PluginContext that provides
    ``register_hook`` and ``register_tool``. Older Hermes versions may also
    provide ``config``.

    The function is named ``register`` to match the Hermes plugin contract.
    A ``setup`` alias is provided for backward compatibility with tests.

    Args:
        context: Hermes PluginContext instance.
    """
    global _chain, _audit_trail, _tracker, _vault, _config, _initialized, _initialization_error

    _config = _load_context_config(context)

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
    context.register_hook("transform_tool_result", _on_transform_tool_result)
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


def _load_context_config(context: Any) -> dict[str, Any]:
    """Load plugin config from old and current Hermes plugin contracts."""
    direct = getattr(context, "config", None)
    if isinstance(direct, dict) and direct:
        return dict(direct)

    config: dict[str, Any] = {}
    try:
        from hermes_cli.config import load_config

        raw = load_config() or {}
    except Exception:
        raw = {}

    if isinstance(raw, dict):
        plugins = raw.get("plugins")
        if isinstance(plugins, dict):
            direct_plugin = plugins.get("katana")
            if isinstance(direct_plugin, dict):
                config.update(direct_plugin)

            entries = plugins.get("entries")
            if isinstance(entries, dict):
                for plugin_id in _context_plugin_ids(context):
                    entry = entries.get(plugin_id)
                    if isinstance(entry, dict):
                        config.update(_normalize_plugin_entry_config(entry))

    if isinstance(direct, dict):
        config.update(direct)
    return config


def _context_plugin_ids(context: Any) -> tuple[str, ...]:
    """Return plausible Hermes config keys for this plugin."""
    ids = ["katana", plugin_name]
    manifest = getattr(context, "manifest", None)
    for attr in ("key", "name"):
        value = getattr(manifest, attr, None)
        if isinstance(value, str) and value:
            ids.append(value)
    return tuple(dict.fromkeys(ids))


def _normalize_plugin_entry_config(entry: dict[str, Any]) -> dict[str, Any]:
    """Extract Katana settings from ``plugins.entries.<id>`` config."""
    nested = entry.get("config")
    if isinstance(nested, dict):
        return dict(nested)
    return {key: value for key, value in entry.items() if key != "llm"}


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
    elif _scabbard_profile in {"katana_v17_minilm", "v17_minilm", "minilm_v17"}:
        scabbard_cfg = ScabbardConfig.katana_v17_minilm(
            model_path=_scabbard_model_path,
            backend=_scabbard_backend or "torch",
            device=_scabbard_device,
        )
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
) -> dict[str, str] | None:
    """Middleware chain evaluation before tool execution.

    Runs taint check, scanning, and policy evaluation. Modern Hermes
    blocks tools when a pre hook returns ``{"action": "block", ...}``;
    raising from hooks is treated as observational failure and is swallowed.

    Args:
        tool_name: Name of the tool being called.
        args: Tool arguments dict.
        task_id: Current task/session identifier.
    """
    if _source_patch_active():
        # The source-patch dispatch hook already enforces in model_tools.py;
        # running the native hook too would scan/deny twice. Defer to it.
        return None
    if _initialization_error is not None:
        log_security_event(
            logger,
            logging.WARNING,
            "tool_call_blocked_runtime_uninitialized",
            reason=_initialization_error,
            fail_closed=True,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id),
        )
        return _block_directive(
            f"HermesKatana initialization failed; blocking tool call '{tool_name}' until the runtime is fixed"
        )

    if not _initialized or _chain is None:
        return None

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
        return _block_directive(f"Tool call '{tool_name}' denied because HermesKatana pre-dispatch failed")

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
        return _block_directive(f"Tool call '{tool_name}' denied: {'; '.join(reasons)}")

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
        from hermes_katana.escalation import resolve_escalation

        approved = resolve_escalation(
            _config.get("escalate_action", "block"),
            tool_name=tool_name,
            reasons=reasons,
            args=args,
            task_id=task_id,
            call_id=ctx.call_id,
        )
        if not approved:
            return _block_directive(f"Tool call '{tool_name}' requires approval: {'; '.join(reasons)}")
        # Human approved (or auto_approve): fall through to ALLOW.

    # Stash context for post_tool_call to pick up
    _stash_context(ctx.call_id, ctx)
    return None


def _block_directive(message: str) -> dict[str, str]:
    """Return the Hermes pre-tool-call blocking contract."""
    return {"action": "block", "message": message}


def _source_patch_active() -> bool:
    """True when Katana source patches are enforcing in this process.

    The source-patch dispatch hook sets ``KATANA_SOURCE_PATCHED=1`` when it
    bootstraps. When that path is active, the native plugin must not also
    enforce, or every tool call would be scanned/denied twice (and outputs
    redacted twice). Checked at hook time so it is robust to load ordering
    between plugin discovery and dispatcher bootstrap.
    """
    return os.environ.get("KATANA_SOURCE_PATCHED", "").strip().lower() in {"1", "true", "yes", "on"}


def _on_post_tool_call(
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    result: str = "",
    task_id: str = "",
    **kwargs: Any,
) -> str | None:
    """Post-execution scanning, taint registration, and audit logging.

    Args:
        tool_name: Name of the tool that was called.
        args: Tool arguments dict.
        result: The tool's return value (string).
        task_id: Current task/session identifier.
    """
    if _source_patch_active() or not _initialized or _chain is None:
        return None

    transformed = _process_tool_result(
        tool_name=tool_name,
        args=args,
        result=result,
        task_id=task_id,
        **kwargs,
    )
    if isinstance(transformed, str):
        _stash_transformed_result(
            tool_name=tool_name,
            task_id=task_id,
            session_id=str(kwargs.get("session_id", "")),
            tool_call_id=str(kwargs.get("tool_call_id", "")),
            original=result,
            transformed=transformed,
        )
        return transformed
    return None


def _on_transform_tool_result(
    tool_name: str = "",
    args: Optional[dict[str, Any]] = None,
    result: str = "",
    task_id: str = "",
    **kwargs: Any,
) -> str | None:
    """Return a scanned/redacted tool result for modern Hermes."""
    if _source_patch_active() or not _initialized or _chain is None:
        return None

    cached = _pop_transformed_result(
        tool_name=tool_name,
        task_id=task_id,
        session_id=str(kwargs.get("session_id", "")),
        tool_call_id=str(kwargs.get("tool_call_id", "")),
        original=result,
    )
    if cached is not None:
        return cached

    return _process_tool_result(
        tool_name=tool_name,
        args=args,
        result=result,
        task_id=task_id,
        **kwargs,
    )


def _process_tool_result(
    *,
    tool_name: str,
    args: Optional[dict[str, Any]],
    result: str,
    task_id: str,
    **kwargs: Any,
) -> str:
    """Run Katana post-dispatch middleware and return the final result."""
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
        error_name = str(exc) or exc.__class__.__name__
        log_security_event(
            logger,
            logging.ERROR,
            "tool_call_post_dispatch_failed",
            error=error_name,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id, call_id=ctx.call_id),
        )
        return json.dumps(
            {
                "error": f"Tool call '{tool_name}' result blocked because HermesKatana post-dispatch failed: {error_name}"
            },
            ensure_ascii=False,
        )

    final_result = ctx.tool_output if isinstance(ctx.tool_output, str) else result

    # Register taint on tool output
    if _tracker is not None and isinstance(final_result, str) and final_result:
        try:
            from hermes_katana.taint.registrar import taint_tool_output

            taint_tool_output(final_result, tool_name)
        except Exception:
            logger.debug("Taint registration failed for %s output", tool_name, exc_info=True)

    return final_result


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


def _handle_katana_status(args: dict | None = None, **kwargs: Any) -> str:
    """Return current HermesKatana security status as a JSON string.

    The Hermes tool registry dispatches handlers as ``handler(args, **kwargs)``
    (tools/registry.py), so ``args`` must be accepted as the first positional
    argument. The previous ``**kwargs``-only signature raised a live TypeError
    ("takes 0 positional arguments but 1 was given") when katana_status was
    invoked through the registry. ``args`` is unused here (status takes no
    parameters) but the positional slot is required by the dispatch contract.
    """
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
    with _result_transform_stash_lock:
        _result_transform_stash.clear()


def _result_stash_key(
    *,
    tool_name: str,
    task_id: str,
    session_id: str,
    tool_call_id: str,
    original: Any,
) -> tuple[str, str, str, str, str]:
    """Build a stable key linking post and transform hooks for one call.

    When Hermes supplies a ``tool_call_id`` it already uniquely identifies the
    call, so the ``(id, session, task, tool)`` tuple is a stable link between the
    post and transform hooks and we skip hashing the (potentially large) output.
    Only when there is no id do we fall back to a content digest, which avoids
    mismatching concurrent results for the same tool. The two key spaces are kept
    disjoint (id present -> first element set, digest empty; id absent -> first
    element empty, digest set) so they can never collide.

    The id-less fallback is best-effort: two id-less calls with identical
    ``(task, tool, session, content)`` map to the same key. That is benign in
    practice -- identical content through the (deterministic) post-dispatch chain
    yields an identical transformed result, so an overwrite replaces a value with
    the same value. Modern Hermes always supplies ``tool_call_id``, so the keyed
    path is the norm and this fallback is rarely exercised.
    """
    if tool_call_id:
        return tool_call_id, session_id or "", task_id or "", tool_name or "", ""
    if isinstance(original, str):
        raw = original.encode("utf-8", errors="surrogatepass")
    else:
        raw = repr(original).encode("utf-8", errors="surrogatepass")
    digest = hashlib.sha256(raw).hexdigest()
    return "", session_id or "", task_id or "", tool_name or "", digest


def _stash_transformed_result(
    *,
    tool_name: str,
    task_id: str,
    session_id: str,
    tool_call_id: str,
    original: Any,
    transformed: str,
) -> None:
    """Store a post-hook result so transform_tool_result can return it."""
    key = _result_stash_key(
        tool_name=tool_name,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        original=original,
    )
    with _result_transform_stash_lock:
        _result_transform_stash[key] = transformed
        if len(_result_transform_stash) > 100:
            oldest = list(_result_transform_stash.keys())[: len(_result_transform_stash) - 50]
            for stale_key in oldest:
                _result_transform_stash.pop(stale_key, None)


def _pop_transformed_result(
    *,
    tool_name: str,
    task_id: str,
    session_id: str,
    tool_call_id: str,
    original: Any,
) -> str | None:
    """Return and clear the transformed result for this hook pair."""
    key = _result_stash_key(
        tool_name=tool_name,
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
        original=original,
    )
    with _result_transform_stash_lock:
        return _result_transform_stash.pop(key, None)


# Backward-compatibility alias used by tests and alternative plugin loaders.
setup = register
