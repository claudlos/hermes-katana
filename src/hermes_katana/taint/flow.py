"""Data-flow analysis engine for taint-aware access control.

Determines whether a :class:`TaintedValue` may flow into a given tool
invocation based on configurable :class:`FlowRule` policies.  This is the
enforcement layer that prevents prompt-injection payloads embedded in
untrusted data from reaching security-critical sinks (terminal, message
dispatch, memory writes, etc.).

Default rules implement the CaMeL principle: data originating from
``TOOL_OUTPUT``, ``WEB_CONTENT``, or ``MCP`` sources may *not* flow into
``terminal``, ``send_message``, or ``memory_write`` without explicit user
approval.
"""

from __future__ import annotations

import json as _json
import logging
import fnmatch
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto, unique
from typing import Any, Optional, Sequence

from hermes_katana.taint.labels import TaintLabel, TrustLevel
from hermes_katana.taint.value import TaintedValue, collect_sources

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Flow decisions
# ---------------------------------------------------------------------------

@unique
class FlowDecision(Enum):
    """Outcome of a data-flow policy check."""

    ALLOW = auto()
    """The data may flow freely to the target tool."""

    DENY = auto()
    """The data MUST NOT reach the target tool — block the call."""

    ASK_USER = auto()
    """Escalate to the human operator for an explicit allow/deny decision."""

    QUARANTINE = auto()
    """Allow the call but sandbox / log it for later review."""


# ---------------------------------------------------------------------------
# Flow rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class FlowRule:
    """A single data-flow policy rule.

    Parameters
    ----------
    source_labels:
        Taint labels that this rule matches.  If *any* source on the value
        carries a matching label, the rule fires.
    target_tools:
        Tool names (or glob patterns) this rule applies to.  Use ``{"*"}``
        as a wildcard that matches everything.
    decision:
        What to do when the rule fires.
    reason:
        Human-readable explanation (logged and shown to the user on ASK).
    priority:
        Higher-priority rules are evaluated first.  Default is 0.
    """

    source_labels: frozenset[TaintLabel]
    target_tools: frozenset[str]
    decision: FlowDecision
    reason: str = ""
    priority: int = 0

    def matches_labels(self, labels: frozenset[TaintLabel]) -> bool:
        """``True`` if any of the value's labels intersect this rule's set."""
        return bool(self.source_labels & labels)

    def matches_tool(self, tool_name: str) -> bool:
        """``True`` if *tool_name* is covered by this rule."""
        if "*" in self.target_tools:
            return True
        for pattern in self.target_tools:
            if fnmatch.fnmatch(tool_name, pattern):
                return True
        return False


# ---------------------------------------------------------------------------
# Flow analysis result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FlowAnalysis:
    """Detailed result of a flow-policy evaluation.

    Attributes
    ----------
    decision:
        The overall verdict.
    matched_rules:
        Rules that fired (in priority order).
    labels_present:
        Taint labels found on the value.
    tool_name:
        The target tool being evaluated.
    reasoning:
        Human-readable explanation of the decision.
    timestamp:
        When the analysis was performed.
    """

    decision: FlowDecision
    matched_rules: list[FlowRule] = field(default_factory=list)
    labels_present: frozenset[TaintLabel] = field(default_factory=frozenset)
    tool_name: str = ""
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)

    def is_blocked(self) -> bool:
        """``True`` if the decision prevents execution."""
        return self.decision in (FlowDecision.DENY, FlowDecision.ASK_USER)


# ---------------------------------------------------------------------------
# Default security rules
# ---------------------------------------------------------------------------

# Tools considered security-critical sinks
CRITICAL_SINKS: frozenset[str] = frozenset({
    "terminal",
    "bash",
    "shell",
    "execute",
    "run_command",
    "send_message",
    "send_email",
    "memory",
    "memory_write",
    "memory_update",
    "memory_delete",
    "file_write",
    "write_file",
    "patch",
    "mcp_patch",
    "subprocess",
    "os.system",
    "exec",
    "eval",
    "http_request",
    "fetch",
    "api_call",
    "browser_type",
    "browser_click",
    "browser_press",
    "browser_navigate",
    "text_to_speech",
    "cronjob",
    "skill_manage",
})

# Labels considered untrusted by default
UNTRUSTED_LABELS: frozenset[TaintLabel] = frozenset({
    TaintLabel.WEB_CONTENT,
    TaintLabel.MCP,
    TaintLabel.MCP_TOOL_DESCRIPTION,  # Highest-risk MCP label
    TaintLabel.MCP_TOOL_RESULT,
    TaintLabel.MCP_RESOURCE,
    TaintLabel.MCP_PROMPT,
    TaintLabel.UNKNOWN,
})

# Labels that need approval for critical sinks
CONDITIONAL_LABELS: frozenset[TaintLabel] = frozenset({
    TaintLabel.TOOL_OUTPUT,
    TaintLabel.FILE_CONTENT,
    TaintLabel.MEMORY,
    TaintLabel.AGENT_DELEGATED,  # Sub-agent output may carry injections
    TaintLabel.CROSS_SESSION,    # Previous-session memory unverifiable
})


def default_rules() -> list[FlowRule]:
    """Return the default set of data-flow security rules.

    These implement the core CaMeL-inspired policy:

    1. **DENY**: Untrusted data (web, MCP, unknown) → critical sinks.
    2. **ASK**:  Conditional data (tool output, files, memory) → critical sinks.
    3. **ALLOW**: Trusted data (user, system) → anything.
    4. **ALLOW**: Any data → non-critical sinks (read-only tools).
    """
    return [
        # Rule 1: Hard deny for clearly untrusted → critical
        FlowRule(
            source_labels=UNTRUSTED_LABELS,
            target_tools=CRITICAL_SINKS,
            decision=FlowDecision.DENY,
            reason=(
                "Data from untrusted sources (web content, MCP servers, or "
                "unknown origins) cannot flow to security-critical tools "
                "without sanitization."
            ),
            priority=100,
        ),
        # Rule 2: Ask user for conditional → critical
        FlowRule(
            source_labels=CONDITIONAL_LABELS,
            target_tools=CRITICAL_SINKS,
            decision=FlowDecision.ASK_USER,
            reason=(
                "Data from conditionally-trusted sources (tool output, files, "
                "memory) requires user approval before flowing to "
                "security-critical tools."
            ),
            priority=50,
        ),
        # Rule 3: Allow trusted data everywhere
        FlowRule(
            source_labels=frozenset({TaintLabel.USER, TaintLabel.SYSTEM}),
            target_tools=frozenset({"*"}),
            decision=FlowDecision.ALLOW,
            reason="Trusted data (user input, system prompt) is allowed everywhere.",
            priority=10,
        ),
        # Rule 4: Agent-generated content — quarantine for critical sinks
        FlowRule(
            source_labels=frozenset({TaintLabel.AGENT}),
            target_tools=CRITICAL_SINKS,
            decision=FlowDecision.QUARANTINE,
            reason=(
                "Agent-generated content flowing to critical sinks is allowed "
                "but logged for review."
            ),
            priority=25,
        ),
        # Rule 5 (research doc 01): MCP tool descriptions → skill_manage always denied
        # Skill mutation from poisoned tool descriptions is a high-privilege attack.
        FlowRule(
            source_labels=frozenset({
                TaintLabel.MCP,
                TaintLabel.MCP_TOOL_DESCRIPTION,
                TaintLabel.MCP_TOOL_RESULT,
            }),
            target_tools=frozenset({"skill_manage", "skill_view", "skills_list"}),
            decision=FlowDecision.DENY,
            reason=(
                "MCP-tainted data cannot flow to skill management tools. "
                "Skill mutation via a poisoned MCP server would persist across sessions."
            ),
            priority=110,  # Higher than base UNTRUSTED_LABELS rule
        ),
        # Rule 6 (research doc 01): Any tainted data → delegate_task requires approval
        # Sub-agents amplify injections; require explicit user approval before delegation.
        FlowRule(
            source_labels=UNTRUSTED_LABELS | frozenset({TaintLabel.TOOL_OUTPUT}),
            target_tools=frozenset({"delegate_task"}),
            decision=FlowDecision.ASK_USER,
            reason=(
                "Untrusted or tool-sourced data cannot flow to delegate_task without "
                "user approval. Sub-agents can amplify injections across the pipeline."
            ),
            priority=95,
        ),
        # Rule 7 (research doc 01): MEMORY-tainted data → send_message denied
        # Memory poisoning + exfiltration is a common stored-injection chain.
        FlowRule(
            source_labels=frozenset({TaintLabel.MEMORY, TaintLabel.CROSS_SESSION}),
            target_tools=frozenset({"send_message", "send_email", "text_to_speech"}),
            decision=FlowDecision.DENY,
            reason=(
                "Memory-sourced data cannot flow to messaging tools. "
                "This prevents stored injection → exfiltration attack chains."
            ),
            priority=105,
        ),
    ]


# ---------------------------------------------------------------------------
# FlowAnalyzer
# ---------------------------------------------------------------------------


class FlowAnalyzer:
    """Evaluates data-flow policies for tainted values targeting tools.

    Parameters
    ----------
    rules:
        List of :class:`FlowRule` to enforce.  If ``None``, uses
        :func:`default_rules`.
    default_decision:
        Fallback when no rule matches.  Defaults to ``ALLOW`` (permissive).
    strict_mode:
        If ``True``, the default decision becomes ``ASK_USER`` instead of
        ``ALLOW`` for extra caution.
    """

    def __init__(
        self,
        rules: Optional[Sequence[FlowRule]] = None,
        default_decision: FlowDecision = FlowDecision.ASK_USER,
        strict_mode: bool = False,
    ) -> None:
        self._rules: list[FlowRule] = sorted(
            rules if rules is not None else default_rules(),
            key=lambda r: -r.priority,
        )
        self._default = FlowDecision.ASK_USER if strict_mode else default_decision
        self._strict = strict_mode
        self._lock = threading.Lock()
        self._history: list[FlowAnalysis] = []
        self._MAX_HISTORY = 1000

    @property
    def rules(self) -> list[FlowRule]:
        """Current rules, sorted by priority (highest first)."""
        return list(self._rules)

    @property
    def history(self) -> list[FlowAnalysis]:
        """All analyses performed so far."""
        return list(self._history)

    def add_rule(self, rule: FlowRule) -> None:
        """Add a rule and re-sort by priority."""
        with self._lock:
            self._rules.append(rule)
            self._rules.sort(key=lambda r: -r.priority)

    def remove_rule(self, rule: FlowRule) -> bool:
        """Remove a rule. Returns ``True`` if it was found and removed."""
        with self._lock:
            try:
                self._rules.remove(rule)
                return True
            except ValueError:
                return False

    def analyze(
        self,
        value: TaintedValue[Any],
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
    ) -> FlowAnalysis:
        """Determine whether *value* may flow to *tool_name*.

        Parameters
        ----------
        value:
            The tainted value being passed as input.
        tool_name:
            Name of the target tool / function.
        args:
            Optional dict of other arguments for context-aware rules.

        Returns
        -------
        FlowAnalysis
            Detailed result with decision, reasoning, and matched rules.
        """
        # Collect all sources, including from nested structures
        all_sources = collect_sources(value) | value.sources
        labels = frozenset(s.label for s in all_sources)

        matched: list[FlowRule] = []
        reasoning_parts: list[str] = []

        with self._lock:
            rules_snapshot = list(self._rules)

        # Evaluate rules in priority order
        for rule in rules_snapshot:
            if rule.matches_tool(tool_name) and rule.matches_labels(labels):
                matched.append(rule)
                intersecting = rule.source_labels & labels
                label_names = ", ".join(sorted(lbl.name for lbl in intersecting))
                reasoning_parts.append(
                    f"[P{rule.priority}] {rule.decision.name}: "
                    f"labels {{{label_names}}} → {tool_name}. {rule.reason}"
                )

        # Determine final decision (highest-priority matched rule wins)
        if matched:
            decision = matched[0].decision
        else:
            decision = self._default
            reasoning_parts.append(
                f"No rule matched for labels "
                f"{{{', '.join(sorted(lbl.name for lbl in labels))}}} → {tool_name}. "
                f"Applying default: {decision.name}."
            )

        # Special case: if all sources are trusted, downgrade escalation to allow
        # (but never override an explicit DENY from a matched rule)
        if all_sources and all(s.trust_level is TrustLevel.TRUSTED for s in all_sources):
            if decision == FlowDecision.ASK_USER:
                decision = FlowDecision.ALLOW
                reasoning_parts.append(
                    "Override: all sources are TRUSTED — downgrading escalation to allow."
                )

        # Special case: no sources at all — treat as clean
        if not all_sources:
            decision = FlowDecision.ALLOW
            reasoning_parts.append("No taint sources present — allowing flow.")

        # Build analysis result
        reasoning = "\n".join(reasoning_parts)
        result = FlowAnalysis(
            decision=decision,
            matched_rules=matched,
            labels_present=labels,
            tool_name=tool_name,
            reasoning=reasoning,
        )

        with self._lock:
            self._history.append(result)
            if len(self._history) > self._MAX_HISTORY:
                self._flush_history_to_disk(self._history[:-self._MAX_HISTORY // 2])
                self._history = self._history[-self._MAX_HISTORY // 2:]
        logger.debug(
            "Flow analysis: %s → %s = %s",
            sorted(lbl.name for lbl in labels),
            tool_name,
            decision.name,
        )

        return result

    def check(
        self,
        value: TaintedValue[Any],
        tool_name: str,
        args: Optional[dict[str, Any]] = None,
    ) -> FlowDecision:
        """Convenience wrapper returning just the decision enum."""
        return self.analyze(value, tool_name, args).decision

    def clear_history(self) -> None:
        """Discard all recorded analyses."""
        self._history.clear()

    def _flush_history_to_disk(self, entries: list[FlowAnalysis]) -> None:
        """Flush oldest history entries to an audit trail JSONL file.

        Called automatically before truncating history at _MAX_HISTORY.
        Entries are appended to ~/.config/hermes-katana/flow_audit.jsonl.
        """
        from pathlib import Path as _P
        try:
            audit_dir = _P.home() / ".config" / "hermes-katana"
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "flow_audit.jsonl"
            with open(audit_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    record = {
                        "decision": entry.decision.name,
                        "tool_name": entry.tool_name,
                        "labels": sorted(lbl.name for lbl in entry.labels_present),
                        "reasoning": entry.reasoning,
                        "timestamp": entry.timestamp,
                    }
                    f.write(_json.dumps(record, default=str) + "\n")
                f.flush()
        except Exception:
            logger.debug("Failed to flush flow history to disk", exc_info=True)

    def __repr__(self) -> str:
        return (
            f"FlowAnalyzer(rules={len(self._rules)}, "
            f"strict={self._strict}, "
            f"history={len(self._history)})"
        )
