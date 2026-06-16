"""
Behavioral analysis scanner for HermesKatana.

Provides both stateless text-pattern analysis and stateful tracking across
tool invocations.  The :class:`BehavioralTracker` maintains a sliding window
of tool calls and detects:

- Tool-invocation spikes (sudden burst of file/network tools)
- Anomalous tool sequences (read_file → write_file → curl in quick succession)
- Output length anomalies (unusually long/short for a tool)
- Repeated failed attempts (brute-force / retry patterns)
- Conversation/persona shift signals (text-pattern based)

The module-level :func:`detect_behavioral` and :func:`behavioral_risk_score`
functions operate on plain text and use pre-compiled regex patterns, mirroring
the interface of other scanner modules.
"""

from __future__ import annotations

import re
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Optional

__all__ = [
    "BehavioralCategory",
    "BehavioralSeverity",
    "BehavioralFinding",
    "ToolCall",
    "BehavioralTracker",
    "detect_behavioral",
    "behavioral_risk_score",
]

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class BehavioralCategory(str, Enum):
    """High-level category of a behavioral finding."""

    TOOL_SPIKE = "tool_spike"
    """Sudden burst of tool invocations in a short window."""

    ANOMALOUS_SEQUENCE = "anomalous_sequence"
    """Dangerous tool sequence (e.g. read → write → exfil)."""

    OUTPUT_LENGTH_ANOMALY = "output_length_anomaly"
    """Tool output is significantly longer/shorter than its historical baseline."""

    REPEATED_FAILURE = "repeated_failure"
    """Same tool failing multiple times — possible brute-force or probe."""

    PERSONA_SHIFT = "persona_shift"
    """Text signals a role/identity change that may indicate jailbreak."""

    CONVERSATION_DRIFT = "conversation_drift"
    """Conversation topic/style changed sharply — possible prompt injection."""


class BehavioralSeverity(str, Enum):
    """Severity of a behavioral finding."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BehavioralFinding:
    """A single behavioral detection finding.

    Attributes:
        pattern_name: Short identifier for the pattern that fired.
        category: High-level category.
        severity: How dangerous this finding is.
        matched_text: Excerpt of text or description that triggered the finding.
        position: (start, end) character offsets in the source text, or (-1,-1)
                  when the finding is derived from session-level state rather
                  than a specific text span.
        description: Human-readable explanation.
        confidence: Detection confidence in [0.0, 1.0].
    """

    pattern_name: str
    category: BehavioralCategory
    severity: BehavioralSeverity
    matched_text: str
    position: tuple[int, int]
    description: str
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Stateless text-pattern detection
# ---------------------------------------------------------------------------

_PatternEntry = tuple[str, re.Pattern[str], BehavioralCategory, BehavioralSeverity, str, float]
_TEXT_PATTERNS: list[_PatternEntry] = []


def _register(
    name: str,
    pattern: str,
    category: BehavioralCategory,
    severity: BehavioralSeverity,
    description: str,
    confidence: float = 1.0,
    flags: int = re.IGNORECASE,
) -> None:
    _TEXT_PATTERNS.append((name, re.compile(pattern, flags), category, severity, description, confidence))


# --- Persona-shift / jailbreak signals ---
_register(
    "persona_new_identity",
    r"\b(you are now|act as|pretend (to be|you('re| are))|roleplay as|your new (persona|identity|name) is)\b",
    BehavioralCategory.PERSONA_SHIFT,
    BehavioralSeverity.HIGH,
    "Instruction to assume a different identity — possible jailbreak attempt.",
    confidence=0.9,
)
_register(
    "persona_no_restrictions",
    r"\b(no (restrictions?|limits?|rules?|guidelines?|filters?)|ignore (all )?rules|disregard (your |all )?(previous )?instructions?)\b",
    BehavioralCategory.PERSONA_SHIFT,
    BehavioralSeverity.CRITICAL,
    "Explicit instruction to drop safety constraints.",
    confidence=0.95,
)
_register(
    "persona_developer_mode",
    r"\b(developer mode|DAN mode|jailbreak(ed)?|unrestricted mode|god mode|evil (mode|twin)|shadow self)\b",
    BehavioralCategory.PERSONA_SHIFT,
    BehavioralSeverity.HIGH,
    "Reference to a known jailbreak persona or mode.",
    confidence=0.9,
)
_register(
    "persona_override_system",
    r"\b(override (the )?(system|safety|security) (prompt|instructions?|rules?)|new (system|master) prompt)\b",
    BehavioralCategory.PERSONA_SHIFT,
    BehavioralSeverity.CRITICAL,
    "Attempt to override system-level instructions.",
    confidence=0.95,
)

# --- Conversation drift / topic injection ---
_register(
    "drift_ignore_previous",
    r"\b(ignore (all )?(previous|prior|above) (instructions?|context|messages?|prompts?)|forget (everything|what you (were|have been) told))\b",
    BehavioralCategory.CONVERSATION_DRIFT,
    BehavioralSeverity.HIGH,
    "Instruction to discard prior conversation context.",
    confidence=0.92,
)
_register(
    "drift_hidden_prompt",
    r"(<\s*hidden\s*>|<!--.*?(inject|payload|override).*?-->|\[\s*INST\s*\]|\[SYS\]|###\s*SYSTEM\s*###)",
    BehavioralCategory.CONVERSATION_DRIFT,
    BehavioralSeverity.HIGH,
    "Hidden or bracketed instruction block detected.",
    confidence=0.88,
    flags=re.IGNORECASE | re.DOTALL,
)
_register(
    "drift_topic_switch",
    r"\b(disregard (the )?(topic|task|assignment)|switch (topics?|roles?|tasks?)|new (task|objective|mission|goal) for you)\b",
    BehavioralCategory.CONVERSATION_DRIFT,
    BehavioralSeverity.MEDIUM,
    "Abrupt topic or task switch that may indicate prompt injection.",
    confidence=0.75,
)

# --- Anomalous tool-sequence signals in text ---
_register(
    "seq_exfil_chain",
    r"\b(read.{0,30}(then|and|,).{0,30}(send|upload|post|curl|wget|exfiltrat))\b",
    BehavioralCategory.ANOMALOUS_SEQUENCE,
    BehavioralSeverity.HIGH,
    "Read-then-exfiltrate pattern detected in text.",
    confidence=0.82,
    flags=re.IGNORECASE | re.DOTALL,
)
_register(
    "seq_write_execute",
    r"\b(write.{0,30}(then|and|,).{0,30}(exec(ute)?|run|chmod|sh |bash |python ))\b",
    BehavioralCategory.ANOMALOUS_SEQUENCE,
    BehavioralSeverity.HIGH,
    "Write-then-execute pattern detected in text.",
    confidence=0.82,
    flags=re.IGNORECASE | re.DOTALL,
)

# --- Repeated-failure / brute-force signals in text ---
_register(
    "brute_retry",
    r"\b(try again|retry|attempt \d+|failed attempt|wrong password|access denied.{0,50}try)\b",
    BehavioralCategory.REPEATED_FAILURE,
    BehavioralSeverity.MEDIUM,
    "Possible brute-force or repeated-failure signal in text.",
    confidence=0.7,
)


def detect_behavioral(text: str) -> list[BehavioralFinding]:
    """Scan *text* for behavioral patterns using pre-compiled regex.

    This is a **stateless** function — it does not require a tracker and
    can be called on any text independently.

    Args:
        text: The input text to analyse.

    Returns:
        A list of :class:`BehavioralFinding` instances (empty = no detections).
    """
    findings: list[BehavioralFinding] = []
    for name, regex, category, severity, description, confidence in _TEXT_PATTERNS:
        for match in regex.finditer(text):
            snippet = match.group(0)
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            findings.append(
                BehavioralFinding(
                    pattern_name=name,
                    category=category,
                    severity=severity,
                    matched_text=snippet,
                    position=(match.start(), match.end()),
                    description=description,
                    confidence=confidence,
                )
            )
    return findings


def behavioral_risk_score(text: str) -> float:
    """Return a 0–1 risk score derived from behavioural text findings.

    Weights: CRITICAL=0.5, HIGH=0.3, MEDIUM=0.15, LOW=0.05.  Capped at 1.0.
    """
    findings = detect_behavioral(text)
    if not findings:
        return 0.0
    score = 0.0
    for f in findings:
        if f.severity == BehavioralSeverity.CRITICAL:
            score += 0.5
        elif f.severity == BehavioralSeverity.HIGH:
            score += 0.3
        elif f.severity == BehavioralSeverity.MEDIUM:
            score += 0.15
        else:
            score += 0.05
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Stateful tracker
# ---------------------------------------------------------------------------

# Tool categories for spike / sequence analysis
_FILE_TOOLS = frozenset({"read_file", "write_file", "create_file", "delete_file", "list_files", "glob", "grep"})
_NETWORK_TOOLS = frozenset({"curl", "wget", "http_request", "fetch", "web_fetch", "web_search", "requests"})
_EXEC_TOOLS = frozenset({"bash", "terminal", "shell", "exec", "run_command", "python"})
_ALL_SENSITIVE = _FILE_TOOLS | _NETWORK_TOOLS | _EXEC_TOOLS

# Dangerous three-step sequences: (tool_a, tool_b, tool_c)
_DANGEROUS_TRIPLES: list[tuple[frozenset[str], frozenset[str], frozenset[str], str]] = [
    (
        _FILE_TOOLS,
        _FILE_TOOLS,
        _NETWORK_TOOLS,
        "read→write→network exfiltration chain",
    ),
    (
        _FILE_TOOLS,
        _NETWORK_TOOLS,
        _NETWORK_TOOLS,
        "file read followed by repeated network calls — possible exfiltration",
    ),
    (
        _EXEC_TOOLS,
        _FILE_TOOLS,
        _NETWORK_TOOLS,
        "exec→file→network — possible payload deployment",
    ),
    (
        _FILE_TOOLS,
        _EXEC_TOOLS,
        _NETWORK_TOOLS,
        "file→exec→network — write-execute-exfil chain",
    ),
]


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    timestamp: float
    output_length: int
    had_error: bool
    duration_ms: float = 0.0
    args_summary: str = ""


@dataclass
class BehavioralTracker:
    """Stateful tracker that detects anomalies across multiple tool calls.

    Maintains a sliding window of recent tool calls and computes per-tool
    output-length statistics to surface anomalies.

    Args:
        window_seconds: Time window (seconds) for spike / sequence analysis.
        spike_threshold: Number of sensitive tool calls in the window that
                         triggers a spike alert.
        failure_threshold: Number of consecutive failures on the same tool
                           before raising a :attr:`BehavioralCategory.REPEATED_FAILURE`.
        length_z_threshold: Z-score threshold for output-length anomaly detection
                            (requires at least ``length_min_samples`` prior calls).
        length_min_samples: Minimum historical samples before length anomaly fires.
    """

    window_seconds: float = 60.0
    # Raised from 5 to 10: an active coding/research agent legitimately issues
    # several file/exec calls within a minute, so 5 produced a continuous stream
    # of false TOOL_SPIKE findings during normal work. The spike is observe-only
    # (KatanaBehavioralMiddleware does not block on it unless block_on_sequence is
    # set), so this only affects the anomaly signal's sensitivity. Tune per
    # deployment via BehavioralTracker(spike_threshold=...).
    spike_threshold: int = 10
    failure_threshold: int = 3
    length_z_threshold: float = 3.0
    length_min_samples: int = 5

    _history: Deque[ToolCall] = field(default_factory=lambda: deque(maxlen=500))
    _tool_lengths: dict[str, list[int]] = field(default_factory=dict)
    _consecutive_failures: dict[str, int] = field(default_factory=dict)
    # Highest sensitive-call count already reported as a spike in the current
    # episode; used to suppress re-emitting the same finding on every subsequent
    # call while the window stays saturated. Reset when the count falls back
    # below threshold.
    _last_spike_count: int = 0

    def record_tool_call(
        self,
        tool_name: str,
        output: Optional[str] = None,
        *,
        had_error: bool = False,
        duration_ms: float = 0.0,
        args_summary: str = "",
    ) -> list[BehavioralFinding]:
        """Record a completed tool call and return any new behavioral findings.

        This is the primary entry point for the middleware post-dispatch hook.

        Args:
            tool_name: Name of the tool that was called.
            output: String output of the tool (may be None if errored).
            had_error: Whether the tool raised an exception.
            duration_ms: Execution time in milliseconds.
            args_summary: Short summary of tool arguments (for logging).

        Returns:
            List of :class:`BehavioralFinding` instances representing
            anomalies detected after this call.
        """
        now = time.monotonic()
        output_length = len(output) if output else 0

        call = ToolCall(
            tool_name=tool_name,
            timestamp=now,
            output_length=output_length,
            had_error=had_error,
            duration_ms=duration_ms,
            args_summary=args_summary,
        )
        self._history.append(call)

        findings: list[BehavioralFinding] = []

        # Update failure counter
        if had_error:
            self._consecutive_failures[tool_name] = self._consecutive_failures.get(tool_name, 0) + 1
        else:
            self._consecutive_failures[tool_name] = 0

        # Update length history (only for successful calls)
        if not had_error and output is not None:
            self._tool_lengths.setdefault(tool_name, []).append(output_length)

        # --- Detection checks ---
        findings.extend(self._check_spike(now))
        findings.extend(self._check_sequence())
        findings.extend(self._check_repeated_failure(tool_name))
        if not had_error and output is not None:
            findings.extend(self._check_output_length(tool_name, output_length))

        return findings

    # ------------------------------------------------------------------
    # Internal detectors
    # ------------------------------------------------------------------

    def _recent_calls(self, now: float) -> list[ToolCall]:
        """Return calls within the sliding window."""
        cutoff = now - self.window_seconds
        return [c for c in self._history if c.timestamp >= cutoff]

    def _check_spike(self, now: float) -> list[BehavioralFinding]:
        """Detect a sudden spike in sensitive tool usage."""
        recent = self._recent_calls(now)
        sensitive_count = sum(1 for c in recent if c.tool_name.lower() in _ALL_SENSITIVE)
        if sensitive_count >= self.spike_threshold:
            # De-duplicate: only emit when the spike reaches a NEW high for this
            # episode. Otherwise a saturated window re-emits the identical finding
            # on every call (the live logs showed the same spike logged dozens of
            # times). Re-arms once the window drains below threshold.
            if sensitive_count <= self._last_spike_count:
                return []
            self._last_spike_count = sensitive_count
            return [
                BehavioralFinding(
                    pattern_name="tool_spike",
                    category=BehavioralCategory.TOOL_SPIKE,
                    severity=BehavioralSeverity.HIGH,
                    matched_text=f"{sensitive_count} sensitive tool calls in {self.window_seconds:.0f}s window",
                    position=(-1, -1),
                    description=(
                        f"Spike detected: {sensitive_count} file/network/exec tool calls "
                        f"within the last {self.window_seconds:.0f} seconds."
                    ),
                    confidence=0.85,
                )
            ]
        self._last_spike_count = 0
        return []

    def _check_sequence(self) -> list[BehavioralFinding]:
        """Detect dangerous three-tool sequences in recent history."""
        if len(self._history) < 3:
            return []
        tail = list(self._history)[-3:]
        names = [c.tool_name.lower() for c in tail]
        findings: list[BehavioralFinding] = []
        for set_a, set_b, set_c, description in _DANGEROUS_TRIPLES:
            if names[0] in set_a and names[1] in set_b and names[2] in set_c:
                findings.append(
                    BehavioralFinding(
                        pattern_name="anomalous_sequence",
                        category=BehavioralCategory.ANOMALOUS_SEQUENCE,
                        severity=BehavioralSeverity.HIGH,
                        matched_text=" → ".join(names),
                        position=(-1, -1),
                        description=description,
                        confidence=0.88,
                    )
                )
        return findings

    def _check_repeated_failure(self, tool_name: str) -> list[BehavioralFinding]:
        """Detect repeated failures on the same tool."""
        count = self._consecutive_failures.get(tool_name, 0)
        if count >= self.failure_threshold:
            return [
                BehavioralFinding(
                    pattern_name="repeated_failure",
                    category=BehavioralCategory.REPEATED_FAILURE,
                    severity=BehavioralSeverity.MEDIUM,
                    matched_text=f"{count} consecutive failures on '{tool_name}'",
                    position=(-1, -1),
                    description=(
                        f"Tool '{tool_name}' has failed {count} consecutive times — "
                        "possible brute-force probe or misconfiguration."
                    ),
                    confidence=0.8,
                )
            ]
        return []

    def _check_output_length(self, tool_name: str, output_length: int) -> list[BehavioralFinding]:
        """Detect output length anomalies using rolling mean/stdev."""
        history = self._tool_lengths.get(tool_name, [])
        # Need enough samples; last entry is the current call
        if len(history) < self.length_min_samples + 1:
            return []

        prior = history[:-1]  # exclude the most recent point
        mean = statistics.mean(prior)
        stdev = statistics.stdev(prior) if len(prior) >= 2 else 0.0
        if stdev < 1.0:
            return []

        z = (output_length - mean) / stdev
        if abs(z) >= self.length_z_threshold:
            direction = "much longer" if z > 0 else "much shorter"
            severity = BehavioralSeverity.HIGH if abs(z) >= 5.0 else BehavioralSeverity.MEDIUM
            return [
                BehavioralFinding(
                    pattern_name="output_length_anomaly",
                    category=BehavioralCategory.OUTPUT_LENGTH_ANOMALY,
                    severity=severity,
                    matched_text=f"{output_length} chars (z={z:.1f}, mean={mean:.0f}, stdev={stdev:.0f})",
                    position=(-1, -1),
                    description=(
                        f"Output from '{tool_name}' is {direction} than usual "
                        f"(z-score {z:.1f}, threshold ±{self.length_z_threshold})."
                    ),
                    confidence=0.75,
                )
            ]
        return []

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._history.clear()
        self._tool_lengths.clear()
        self._consecutive_failures.clear()

    @property
    def call_count(self) -> int:
        """Total number of tool calls recorded."""
        return len(self._history)
