"""Centralized resolution of ``DispatchDecision.ESCALATE`` outcomes.

A middleware raises ESCALATE when a tool call should pause for *human approval*
rather than be auto-allowed or hard-denied.  How that pause resolves is governed
by a single setting, ``escalate_action``:

- ``block`` (default): fail-closed.  The escalated call is denied.  This is the
  safe choice for non-interactive contexts (CLI batch runs, the gateway, and the
  proving ground) where there is no human present to answer a prompt.
- ``acp_prompt``: ask the human through Hermes' own interactive approval
  callback -- the same generic ``request_permission`` bridge that Zed/ACP binds
  via :func:`tools.terminal_tool.set_approval_callback`.  If no interactive
  approver is bound (e.g. a headless run), this falls back to *block*.
- ``auto_approve``: allow the call without prompting, emitting a loud,
  production-unsafe warning.  Intended only for trusted automation.

The legacy ``KATANA_AUTO_APPROVE_ESCALATIONS`` environment variable still works:
when truthy it forces ``auto_approve`` regardless of the configured action, for
backward compatibility with the previous escalation handler.

Both integration paths share this module:

- the native Hermes plugin (:mod:`hermes_katana.hermes_plugin`), and
- the source-patch dispatch hook injected into ``model_tools.py``.

Keeping the logic here means the two paths cannot drift apart, and the decision
is unit-testable without a live Hermes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

ESCALATE_BLOCK = "block"
ESCALATE_ACP_PROMPT = "acp_prompt"
ESCALATE_AUTO_APPROVE = "auto_approve"

VALID_ESCALATE_ACTIONS: tuple[str, ...] = (
    ESCALATE_BLOCK,
    ESCALATE_ACP_PROMPT,
    ESCALATE_AUTO_APPROVE,
)

DEFAULT_ESCALATE_ACTION = ESCALATE_BLOCK

_TRUTHY = {"1", "true", "yes", "on"}

# Approval-callback return values that count as "the human allowed this".
# Hermes' ACP bridge maps its permission outcomes to these strings
# (``once`` / ``session`` / ``always`` for allow, ``deny`` for reject); we also
# accept a few looser synonyms in case a different approver is registered.
# Anything not in this set -- including ``deny`` and unknown values -- blocks.
_ALLOW_OUTCOMES = {
    "once",
    "session",
    "always",
    "allow",
    "allow_once",
    "allow_session",
    "allow_always",
    "approve",
    "approved",
    "yes",
    "true",
}


def normalize_escalate_action(value: Any) -> str:
    """Coerce a raw config value into a known escalate action.

    Unknown or empty values normalize to the fail-closed default (``block``).
    """
    action = str(value if value is not None else "").strip().lower()
    return action if action in VALID_ESCALATE_ACTIONS else DEFAULT_ESCALATE_ACTION


def _env_forces_auto_approve() -> bool:
    return os.environ.get("KATANA_AUTO_APPROVE_ESCALATIONS", "").strip().lower() in _TRUTHY


def resolve_escalation(
    action: Any,
    *,
    tool_name: str,
    reasons: Optional[Sequence[str]] = None,
    args: Optional[dict[str, Any]] = None,
    task_id: str = "",
    call_id: str = "",
) -> bool:
    """Resolve an ESCALATE decision into allow (True) or block (False).

    Always fails closed: any unexpected error, an unknown action, or a missing
    interactive approver results in a block.

    Args:
        action: The configured ``escalate_action`` (raw value is normalized).
        tool_name: Name of the escalated tool.
        reasons: Human-readable escalation reasons (shown in the approval card).
        args: The tool call arguments (used to summarize the command).
        task_id: Current task/session identifier (for audit logging).
        call_id: Middleware call identifier (for audit logging).

    Returns:
        ``True`` if the call may proceed, ``False`` if it must be blocked.
    """
    reason_list = [str(r) for r in (reasons or []) if str(r)]
    resolved = ESCALATE_AUTO_APPROVE if _env_forces_auto_approve() else normalize_escalate_action(action)

    if resolved == ESCALATE_AUTO_APPROVE:
        _log_event(
            logging.WARNING,
            "escalation_auto_approved",
            tool_name=tool_name,
            reasons=reason_list,
            args=args,
            task_id=task_id,
            call_id=call_id,
            production_safe=False,
            reason="escalate_action=auto_approve",
        )
        return True

    if resolved == ESCALATE_ACP_PROMPT:
        approved = _prompt_for_approval(
            tool_name=tool_name,
            reason_list=reason_list,
            args=args,
        )
        _log_event(
            logging.WARNING,
            "escalation_prompt_allowed" if approved else "escalation_prompt_blocked",
            tool_name=tool_name,
            reasons=reason_list,
            args=args,
            task_id=task_id,
            call_id=call_id,
            fail_closed=not approved,
        )
        return approved

    # block (default / fail-closed)
    _log_event(
        logging.WARNING,
        "escalation_blocked",
        tool_name=tool_name,
        reasons=reason_list,
        args=args,
        task_id=task_id,
        call_id=call_id,
        fail_closed=True,
        reason="escalate_action=block",
    )
    return False


def _prompt_for_approval(
    *,
    tool_name: str,
    reason_list: list[str],
    args: Optional[dict[str, Any]],
) -> bool:
    """Ask the bound interactive approver to allow/deny this escalated call.

    Reuses Hermes' generic approval callback (the same one ACP/Zed binds), so
    the human sees a real permission card.  Fails closed when no approver is
    bound or the approver errors/times out.
    """
    callback = _get_interactive_approver()
    if callback is None:
        logger.warning(
            "escalate_action=acp_prompt but no interactive approver is bound; "
            "blocking tool '%s' (this is expected for headless runs)",
            tool_name,
        )
        return False

    description = "HermesKatana approval required for tool '%s'" % tool_name
    if reason_list:
        description += ": " + "; ".join(reason_list)
    command = _summarize_command(tool_name, args)

    try:
        outcome = callback(command=command, description=description, allow_permanent=False)
    except TypeError:
        # Fall back to a positional call for approvers with a looser signature.
        try:
            outcome = callback(command, description)
        except Exception:
            logger.warning("Interactive approver raised; blocking '%s'", tool_name, exc_info=True)
            return False
    except Exception:
        logger.warning("Interactive approver raised; blocking '%s'", tool_name, exc_info=True)
        return False

    return _outcome_is_allow(outcome)


def _get_interactive_approver():
    """Return Hermes' bound approval callback, or ``None`` if unavailable."""
    try:
        from tools import terminal_tool  # type: ignore[import-not-found]
    except Exception:
        return None

    getter = getattr(terminal_tool, "_get_approval_callback", None)
    if not callable(getter):
        return None
    try:
        callback = getter()
    except Exception:
        return None
    return callback if callable(callback) else None


def _outcome_is_allow(outcome: Any) -> bool:
    """Interpret an approval-callback return value as allow (True) or block."""
    if isinstance(outcome, bool):
        return outcome
    return str(outcome if outcome is not None else "").strip().lower() in _ALLOW_OUTCOMES


def _summarize_command(tool_name: str, args: Optional[dict[str, Any]]) -> str:
    """Build a short command string for the approval card."""
    if isinstance(args, dict):
        for key in ("command", "cmd", "code", "query", "path"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return tool_name


def _log_event(
    level: int,
    event: str,
    *,
    tool_name: str,
    reasons: list[str],
    args: Optional[dict[str, Any]],
    task_id: str,
    call_id: str,
    **extra: Any,
) -> None:
    """Emit a structured security log event; never raises."""
    try:
        from hermes_katana.security_logging import log_security_event, summarize_tool_call

        log_security_event(
            logger,
            level,
            event,
            escalation_reasons=reasons,
            **extra,
            **summarize_tool_call(tool_name, args or {}, task_id=task_id, call_id=call_id),
        )
    except Exception:
        logger.log(level, "%s tool=%s reasons=%s", event, tool_name, reasons)
