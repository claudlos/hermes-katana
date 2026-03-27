"""Runtime metrics collector for HermesKatana.

Lightweight, thread-safe counters and histograms for tool calls,
scanner hits, taint flows, and policy evaluations.  No external
dependencies — pure stdlib.

Exposes metrics via:
- ``get_summary()`` for programmatic access
- ``export_prometheus()`` for Prometheus text format
- ``katana metrics`` CLI command (wired separately)

Usage::

    from hermes_katana.metrics import MetricsCollector

    metrics = MetricsCollector.get_instance()
    metrics.record_tool_call("terminal", "allow", latency_ms=12.5)
    metrics.record_scan_hit("injection", "instruction_override", "high")

    summary = metrics.get_summary()
    print(summary.tool_calls)  # Counter by (tool, decision)
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricsSummary:
    """Snapshot of collected metrics.

    All counters are copies (mutation-safe).

    Attributes:
        tool_calls: Counter keyed by (tool_name, decision).
        scan_hits: Counter keyed by (scanner_name, category, severity).
        taint_flows: Counter keyed by (source_label, target_tool, decision).
        policy_evals: Counter keyed by (result,).
        latency_samples: List of (tool_name, latency_ms) recent samples.
        uptime_seconds: Seconds since the collector was created/reset.
        started_at: Unix timestamp of collector start/reset.
    """

    tool_calls: dict[tuple, int] = field(default_factory=dict)
    scan_hits: dict[tuple, int] = field(default_factory=dict)
    taint_flows: dict[tuple, int] = field(default_factory=dict)
    policy_evals: dict[tuple, int] = field(default_factory=dict)
    latency_samples: list[tuple[str, float]] = field(default_factory=list)
    uptime_seconds: float = 0.0
    started_at: float = 0.0

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    @property
    def total_scan_hits(self) -> int:
        return sum(self.scan_hits.values())

    @property
    def total_denials(self) -> int:
        return sum(v for k, v in self.tool_calls.items() if len(k) >= 2 and k[1] == "deny")

    @property
    def total_escalations(self) -> int:
        return sum(v for k, v in self.tool_calls.items() if len(k) >= 2 and k[1] == "escalate")


class MetricsCollector:
    """Thread-safe runtime metrics collector (singleton).

    All recording methods are O(1) and lock-free for the common case
    (Python's GIL + Counter operations).  The explicit lock is only
    taken for reset and summary export.
    """

    _instance: Optional[MetricsCollector] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tool_calls: Counter[tuple] = Counter()
        self._scan_hits: Counter[tuple] = Counter()
        self._taint_flows: Counter[tuple] = Counter()
        self._policy_evals: Counter[tuple] = Counter()
        self._latency_samples: list[tuple[str, float]] = []
        self._max_latency_samples = 1000
        self._started_at = time.time()

    @classmethod
    def get_instance(cls) -> MetricsCollector:
        """Return the global singleton."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the global singleton (for testing)."""
        with cls._instance_lock:
            cls._instance = None

    def record_tool_call(
        self,
        tool_name: str,
        decision: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a tool call with its decision and latency.

        Args:
            tool_name: Name of the tool called.
            decision: 'allow', 'deny', or 'escalate'.
            latency_ms: Middleware evaluation time in milliseconds.
        """
        self._tool_calls[(tool_name, decision)] += 1
        if latency_ms > 0:
            with self._lock:
                self._latency_samples.append((tool_name, latency_ms))
                if len(self._latency_samples) > self._max_latency_samples:
                    self._latency_samples = self._latency_samples[-self._max_latency_samples:]

    def record_scan_hit(
        self,
        scanner_name: str,
        pattern_category: str,
        severity: str,
    ) -> None:
        """Record a scanner finding.

        Args:
            scanner_name: 'injection', 'secrets', 'commands', 'content', 'unicode'.
            pattern_category: The finding category.
            severity: 'critical', 'high', 'medium', 'low'.
        """
        self._scan_hits[(scanner_name, pattern_category, severity)] += 1

    def record_taint_flow(
        self,
        source_label: str,
        target_tool: str,
        decision: str,
    ) -> None:
        """Record a taint flow check result.

        Args:
            source_label: The taint label (e.g., 'WEB_CONTENT').
            target_tool: The target tool name.
            decision: 'allow', 'deny', 'ask_user', 'quarantine'.
        """
        self._taint_flows[(source_label, target_tool, decision)] += 1

    def record_policy_eval(
        self,
        preset: str,
        tool_name: str,
        result: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a policy evaluation result.

        Args:
            preset: The policy preset name.
            tool_name: The tool being evaluated.
            result: 'allow', 'deny', 'escalate', 'log_only'.
            latency_ms: Evaluation time in milliseconds.
        """
        self._policy_evals[(result,)] += 1

    def get_summary(self) -> MetricsSummary:
        """Return a snapshot of all collected metrics."""
        with self._lock:
            return MetricsSummary(
                tool_calls=dict(self._tool_calls),
                scan_hits=dict(self._scan_hits),
                taint_flows=dict(self._taint_flows),
                policy_evals=dict(self._policy_evals),
                latency_samples=list(self._latency_samples[-100:]),
                uptime_seconds=time.time() - self._started_at,
                started_at=self._started_at,
            )

    def reset(self) -> None:
        """Reset all counters and start fresh."""
        with self._lock:
            self._tool_calls.clear()
            self._scan_hits.clear()
            self._taint_flows.clear()
            self._policy_evals.clear()
            self._latency_samples.clear()
            self._started_at = time.time()

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format.

        Returns:
            Multi-line string in Prometheus text format.
        """
        lines: list[str] = []

        lines.append("# HELP katana_tool_calls_total Tool calls by tool and decision")
        lines.append("# TYPE katana_tool_calls_total counter")
        for (tool, decision), count in sorted(self._tool_calls.items()):
            lines.append(
                f'katana_tool_calls_total{{tool="{tool}",decision="{decision}"}} {count}'
            )

        lines.append("# HELP katana_scan_hits_total Scanner findings by type")
        lines.append("# TYPE katana_scan_hits_total counter")
        for key, count in sorted(self._scan_hits.items()):
            scanner, category, severity = key
            lines.append(
                f'katana_scan_hits_total{{scanner="{scanner}",'
                f'category="{category}",severity="{severity}"}} {count}'
            )

        lines.append("# HELP katana_taint_flows_total Taint flow checks by result")
        lines.append("# TYPE katana_taint_flows_total counter")
        for key, count in sorted(self._taint_flows.items()):
            source, target, decision = key
            lines.append(
                f'katana_taint_flows_total{{source="{source}",'
                f'target="{target}",decision="{decision}"}} {count}'
            )

        lines.append("# HELP katana_policy_evals_total Policy evaluations by result")
        lines.append("# TYPE katana_policy_evals_total counter")
        for (result,), count in sorted(self._policy_evals.items()):
            lines.append(
                f'katana_policy_evals_total{{result="{result}"}} {count}'
            )

        lines.append(
            f"# HELP katana_uptime_seconds Collector uptime\n"
            f"# TYPE katana_uptime_seconds gauge\n"
            f"katana_uptime_seconds {time.time() - self._started_at:.1f}"
        )

        return "\n".join(lines) + "\n"
