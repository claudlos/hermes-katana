"""Taint tracker — central registry for all tainted values in a session.

The :class:`TaintTracker` is a context-aware singleton that maintains a
registry of every tainted value created during an execution scope.  It
provides:

- **Registration**: wrap raw values with taint metadata on ingestion.
- **Propagation**: automatically derive taint when values are combined.
- **Provenance**: reconstruct the full chain of sources for any value.
- **Flow control**: check whether a value may flow to a target tool.
- **Scoped tracking**: use as a context manager for isolated sessions.

Thread-safety: the tracker uses a threading lock for all mutations.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, TypeVar

from hermes_katana.taint.flow import FlowAnalysis, FlowAnalyzer, FlowDecision
from hermes_katana.taint.labels import Reader, Source, TaintLabel
from hermes_katana.taint.value import (
    TaintedDict,
    TaintedList,
    TaintedStr,
    TaintedValue,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Tracker statistics
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrackerStats:
    """Diagnostic counters for the taint tracker."""

    values_registered: int = 0
    values_propagated: int = 0
    flow_checks: int = 0
    flow_denied: int = 0
    flow_asked: int = 0
    flow_allowed: int = 0
    flow_quarantined: int = 0
    session_start: float = field(default_factory=time.time)

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.session_start

    def record_flow(self, decision: FlowDecision) -> None:
        self.flow_checks += 1
        if decision is FlowDecision.ALLOW:
            self.flow_allowed += 1
        elif decision is FlowDecision.DENY:
            self.flow_denied += 1
        elif decision is FlowDecision.ASK_USER:
            self.flow_asked += 1
        elif decision is FlowDecision.QUARANTINE:
            self.flow_quarantined += 1


# ---------------------------------------------------------------------------
# TaintTracker singleton
# ---------------------------------------------------------------------------

class TaintTracker:
    """Central taint registry and flow-control engine.

    Usage::

        tracker = TaintTracker.get_instance()
        tv = tracker.register("some web data", Source.web("https://example.com"))
        decision = tracker.check_flow(tv, "terminal")
        # decision == FlowDecision.DENY

    Or as a context manager for scoped tracking::

        with TaintTracker.scoped() as tracker:
            tv = tracker.register(data, source)
            ...
        # tracker is cleared on exit

    Parameters
    ----------
    analyzer:
        Optional custom :class:`FlowAnalyzer`.  If ``None``, uses default
        rules.
    """

    _instance: Optional[TaintTracker] = None
    _lock: threading.Lock = threading.Lock()

    _MAX_REGISTRY_SIZE = 10_000

    def __init__(self, analyzer: Optional[FlowAnalyzer] = None) -> None:
        self._analyzer = analyzer or FlowAnalyzer()
        self._registry: dict[int, TaintedValue[Any]] = {}
        self._stats = TrackerStats()
        self._session_id: str = f"session-{int(time.time() * 1000)}"
        self._mutex = threading.Lock()

    # -- Singleton access -----------------------------------------------------

    @classmethod
    def get_instance(cls, analyzer: Optional[FlowAnalyzer] = None) -> TaintTracker:
        """Return the global singleton, creating it if needed."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(analyzer=analyzer)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the global singleton (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
            cls._instance = None

    @classmethod
    def _reset_instance(cls) -> None:
        """Reset singleton in forked child process (fork safety).

        Called via os.register_at_fork(after_in_child=...) to ensure
        each child process gets a fresh tracker instead of sharing
        corrupted state with the parent.
        """
        cls._instance = None
        cls._lock = threading.Lock()

    @classmethod
    @contextmanager
    def scoped(cls, analyzer: Optional[FlowAnalyzer] = None) -> Iterator[TaintTracker]:
        """Context manager that provides an isolated tracker, cleared on exit.

        Does **not** affect the global singleton — creates a fresh instance.
        """
        tracker = cls(analyzer=analyzer)
        try:
            yield tracker
        finally:
            tracker.clear()

    # -- Registration ---------------------------------------------------------

    def register(
        self,
        value: T,
        source: Source,
        readers: Optional[frozenset[Reader]] = None,
    ) -> TaintedValue[T]:
        """Wrap a raw value with taint metadata and register it.

        Parameters
        ----------
        value:
            The raw Python value to taint.
        source:
            Provenance information.
        readers:
            Optional reader restrictions.

        Returns
        -------
        TaintedValue[T]
            The tainted wrapper (or a specialised subclass for str/list/dict).
        """
        sources = frozenset({source})
        rdr = readers or frozenset()

        # Choose the right wrapper type
        if isinstance(value, str):
            tv: TaintedValue[Any] = TaintedStr(
                value=value,
                sources=sources,
                readers=rdr,
            )
        elif isinstance(value, list):
            tv = TaintedList(
                value=value,
                sources=sources,
                readers=rdr,
            )
        elif isinstance(value, dict):
            tv = TaintedDict(
                value=value,
                sources=sources,
                readers=rdr,
            )
        else:
            tv = TaintedValue(
                value=value,
                sources=sources,
                readers=rdr,
            )

        with self._mutex:
            self._registry[id(tv)] = tv
            self._stats.values_registered += 1
            self._evict_if_needed()

        logger.debug(
            "Registered tainted value: label=%s origin=%s type=%s",
            source.label.name,
            source.origin,
            type(value).__name__,
        )
        return tv  # type: ignore[return-value]

    def register_multi(
        self,
        value: T,
        sources: frozenset[Source],
        readers: Optional[frozenset[Reader]] = None,
    ) -> TaintedValue[T]:
        """Register a value with multiple sources at once."""
        rdr = readers or frozenset()

        if isinstance(value, str):
            tv: TaintedValue[Any] = TaintedStr(
                value=value, sources=sources, readers=rdr,
            )
        elif isinstance(value, list):
            tv = TaintedList(
                value=value, sources=sources, readers=rdr,
            )
        elif isinstance(value, dict):
            tv = TaintedDict(
                value=value, sources=sources, readers=rdr,
            )
        else:
            tv = TaintedValue(
                value=value, sources=sources, readers=rdr,
            )

        with self._mutex:
            self._registry[id(tv)] = tv
            self._stats.values_registered += 1
            self._evict_if_needed()

        return tv  # type: ignore[return-value]

    # -- Propagation ----------------------------------------------------------

    def propagate(
        self,
        result: T,
        *inputs: TaintedValue[Any],
    ) -> TaintedValue[T]:
        """Create a tainted value whose metadata is derived from *inputs*.

        This is the core taint-propagation operation: when a computation
        combines multiple tainted inputs into a new result, call this to
        ensure the result carries the union of all input taint.

        Parameters
        ----------
        result:
            The computed raw value.
        *inputs:
            The tainted values that contributed to *result*.

        Returns
        -------
        TaintedValue[T]
            A new tainted wrapper with merged metadata.
        """
        all_sources: set[Source] = set()
        all_readers: set[Reader] = set()
        deps: list[TaintedValue[Any]] = []

        for inp in inputs:
            all_sources.update(inp.sources)
            all_readers.update(inp.readers)
            deps.append(inp)

        if isinstance(result, str):
            tv: TaintedValue[Any] = TaintedStr(
                value=result,
                sources=frozenset(all_sources),
                readers=frozenset(all_readers),
                dependencies=tuple(deps),
            )
        elif isinstance(result, list):
            tv = TaintedList(
                value=result,
                sources=frozenset(all_sources),
                readers=frozenset(all_readers),
                dependencies=tuple(deps),
            )
        elif isinstance(result, dict):
            tv = TaintedDict(
                value=result,
                sources=frozenset(all_sources),
                readers=frozenset(all_readers),
                dependencies=tuple(deps),
            )
        else:
            tv = TaintedValue(
                value=result,
                sources=frozenset(all_sources),
                readers=frozenset(all_readers),
                dependencies=tuple(deps),
            )

        with self._mutex:
            self._registry[id(tv)] = tv
            self._stats.values_propagated += 1
            self._evict_if_needed()

        return tv  # type: ignore[return-value]

    # -- Provenance -----------------------------------------------------------

    def get_taint_chain(self, value: TaintedValue[Any]) -> list[Source]:
        """Reconstruct the full provenance chain for *value*.

        Walks the dependency graph depth-first and returns all sources
        encountered, ordered by timestamp (oldest first).
        """
        seen: set[int] = set()
        sources: list[Source] = []

        def _walk(tv: TaintedValue[Any]) -> None:
            tid = id(tv)
            if tid in seen:
                return
            seen.add(tid)
            for src in tv.sources:
                if src not in sources:
                    sources.append(src)
            for dep in tv.dependencies:
                _walk(dep)

        _walk(value)
        sources.sort(key=lambda s: s.timestamp)
        return sources

    def get_labels(self, value: TaintedValue[Any]) -> frozenset[TaintLabel]:
        """Return all taint labels in the full dependency chain."""
        chain = self.get_taint_chain(value)
        return frozenset(s.label for s in chain)

    # -- Flow control ---------------------------------------------------------

    def check_flow(
        self,
        value: TaintedValue[Any],
        target_tool: str,
        args: Optional[dict[str, Any]] = None,
    ) -> FlowDecision:
        """Check whether *value* may flow to *target_tool*.

        Returns
        -------
        FlowDecision
            ALLOW, DENY, ASK_USER, or QUARANTINE.
        """
        decision = self._analyzer.check(value, target_tool, args)
        with self._mutex:
            self._stats.record_flow(decision)
        return decision

    def analyze_flow(
        self,
        value: TaintedValue[Any],
        target_tool: str,
        args: Optional[dict[str, Any]] = None,
    ) -> FlowAnalysis:
        """Like :meth:`check_flow` but returns full analysis details."""
        analysis = self._analyzer.analyze(value, target_tool, args)
        with self._mutex:
            self._stats.record_flow(analysis.decision)
        return analysis

    def check_args_flow(
        self,
        tool_name: str,
        **kwargs: Any,
    ) -> FlowDecision:
        """Check all keyword arguments for taint violations against *tool_name*.

        Scans each kwarg recursively; if any is a :class:`TaintedValue`,
        checks its flow.  Returns the *most restrictive* decision.
        """
        worst = FlowDecision.ALLOW
        priority = {
            FlowDecision.ALLOW: 0,
            FlowDecision.QUARANTINE: 1,
            FlowDecision.ASK_USER: 2,
            FlowDecision.DENY: 3,
        }

        def _walk(val: Any) -> None:
            nonlocal worst
            if isinstance(val, (TaintedStr, TaintedValue)):
                decision = self.check_flow(val, tool_name)
                if priority[decision] > priority[worst]:
                    worst = decision
            if isinstance(val, dict):
                for v in val.values():
                    _walk(v)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    _walk(item)

        for _key, val in kwargs.items():
            _walk(val)
            if worst == FlowDecision.DENY:
                break  # Short-circuit on DENY
        return worst

    # -- Internal helpers -----------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest entries when registry exceeds size cap.

        Must be called while holding ``self._mutex``.
        Drops the oldest half of entries (by insertion order) to amortize
        the eviction cost.
        """
        if len(self._registry) <= self._MAX_REGISTRY_SIZE:
            return
        # dict preserves insertion order in Python 3.7+; drop the oldest half
        keys = list(self._registry.keys())
        evict_count = len(keys) // 2
        for k in keys[:evict_count]:
            del self._registry[k]
        logger.debug(
            "Evicted %d entries from taint registry (size was %d, now %d)",
            evict_count, len(keys), len(self._registry),
        )

    # -- Session management ---------------------------------------------------

    def clear(self) -> None:
        """Clear all tracked values and reset statistics."""
        with self._mutex:
            self._registry.clear()
            self._stats = TrackerStats()
            self._session_id = f"session-{int(time.time() * 1000)}"
        self._analyzer.clear_history()
        logger.debug("Taint tracker cleared")

    @property
    def stats(self) -> TrackerStats:
        """Current diagnostic counters."""
        return self._stats

    @property
    def session_id(self) -> str:
        """Unique identifier for the current tracking session."""
        return self._session_id

    @property
    def analyzer(self) -> FlowAnalyzer:
        """The underlying flow analyzer."""
        return self._analyzer

    @property
    def tracked_count(self) -> int:
        """Number of values currently tracked (strong refs)."""
        return len(self._registry)

    def __repr__(self) -> str:
        return (
            f"TaintTracker(session={self._session_id!r}, "
            f"tracked={self.tracked_count}, "
            f"registered={self._stats.values_registered}, "
            f"checks={self._stats.flow_checks})"
        )

    # -- Context manager for scoped inline tracking ---------------------------

    def __enter__(self) -> TaintTracker:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.clear()


# Register fork handler to reset singleton in child processes
try:
    os.register_at_fork(after_in_child=lambda: TaintTracker._reset_instance())
except AttributeError:
    # os.register_at_fork not available on all platforms (e.g. Windows)
    pass
