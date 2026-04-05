"""HermesKatana exception hierarchy.

Custom exceptions used across the toolkit to signal security decisions
to calling code (Hermes plugin hooks, CLI, tests).
"""

from __future__ import annotations

from typing import Any, Optional


class KatanaSecurityError(Exception):
    """Base exception for all security-related denials.

    Attributes:
        tool_name: The tool call that was blocked.
        reasons: One or more human-readable reasons for the denial.
        call_id: The middleware call-context ID (for audit correlation).
        scan_score: Aggregate scanner risk score, if applicable.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        reasons: Optional[list[str]] = None,
        call_id: str = "",
        scan_score: float = 0.0,
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.reasons = reasons or []
        self.call_id = call_id
        self.scan_score = scan_score


class EscalationRequired(KatanaSecurityError):
    """Raised when a tool call requires human approval.

    The Hermes plugin converts this into the agent's native approval
    flow (clarify, confirm_tool_use, etc.) so the user can accept or
    reject the call.

    Attributes:
        escalation_context: Structured context dict for the approval UI.
    """

    def __init__(
        self,
        message: str,
        *,
        tool_name: str = "",
        reasons: Optional[list[str]] = None,
        call_id: str = "",
        scan_score: float = 0.0,
        escalation_context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            tool_name=tool_name,
            reasons=reasons,
            call_id=call_id,
            scan_score=scan_score,
        )
        self.escalation_context = escalation_context or {}


class TaintFlowDenied(KatanaSecurityError):
    """Raised when taint analysis blocks a data flow.

    Attributes:
        source_labels: The taint labels on the blocked data.
        target_tool: The tool that was denied access.
    """

    def __init__(
        self,
        message: str,
        *,
        source_labels: Optional[list[str]] = None,
        target_tool: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, tool_name=target_tool, **kwargs)
        self.source_labels = source_labels or []
        self.target_tool = target_tool


class ScanBlocked(KatanaSecurityError):
    """Raised when scanner findings exceed the blocking threshold.

    Attributes:
        findings_summary: Short description of what was detected.
    """

    def __init__(
        self,
        message: str,
        *,
        findings_summary: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.findings_summary = findings_summary


class PolicyDenied(KatanaSecurityError):
    """Raised when the policy engine denies a tool call.

    Attributes:
        policy_name: Name of the policy rule that triggered.
        policy_action: The policy result (deny/escalate).
    """

    def __init__(
        self,
        message: str,
        *,
        policy_name: str = "",
        policy_action: str = "deny",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.policy_name = policy_name
        self.policy_action = policy_action
