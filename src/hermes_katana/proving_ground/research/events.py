"""Typed event stream for the research agent.

Every step of the agent — what it tried, what it observed, what hypothesis
it's considering, what claim was downgraded — lands on the event stream as
a timestamped, typed event. This is the single source of truth for:

  - Replay ("what did the agent do last Tuesday?")
  - Verification (the Verifier reads Observations + Results, not chat)
  - Loop detection (the DoomLoopDetector fingerprints recent Actions)
  - Compaction (hypothesis-preserving compaction keeps certain kinds)
  - Audit ("how did the agent justify this Result?")

The stream is append-only and JSONL-serializable.

Types intentionally small; every event carries a kind tag so downstream
code can dispatch without isinstance ladders when desired.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


# ---------------------------------------------------------------------------
# Base event
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Common fields. `kind` is the discriminant; payload lives in subclasses."""

    kind: str
    event_id: str = field(default_factory=lambda: _new_id("ev"))
    ts: float = field(default_factory=_now)
    run_id: str | None = None
    hypothesis_id: str | None = None  # which hypothesis this relates to
    parent_event_id: str | None = None  # causal parent in the agent loop

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Event types — keep these small and orthogonal
# ---------------------------------------------------------------------------


@dataclass
class Action(Event):
    """Agent proposed / executed an action (tool call, fleet launch, ...)."""

    kind: str = "action"
    tool: str = ""
    args: dict = field(default_factory=dict)
    rationale: str = ""  # why the agent chose this action
    status: str = "issued"  # issued | executed | rejected


@dataclass
class Observation(Event):
    """Raw result of an Action, or a data-gathering pass (query, fetch)."""

    kind: str = "observation"
    source: str = ""  # "tool:query", "fleet:launch", "verifier", ...
    summary: str = ""  # short human-readable
    data: dict = field(default_factory=dict)


@dataclass
class Hypothesis(Event):
    """Agent-proposed hypothesis node (registered via research.registry separately)."""

    kind: str = "hypothesis"
    title: str = ""
    statement: str = ""
    predicted_direction: str = ""  # greater | less | two-sided | equivalence
    parent_hypothesis_id: str | None = None  # DAG structure


@dataclass
class Result(Event):
    """A claim that passed the rigor contract. Carries full statistical context."""

    kind: str = "result"
    primary_outcome: str = ""
    value: float = 0.0
    n_samples: int = 0
    ci: tuple[float, float] | None = None
    test_kind: str | None = None
    p_value: float | None = None
    effect_size: dict | None = None
    baseline_run_id: str | None = None
    comparison_run_id: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ClaimDowngrade(Event):
    """A claim that FAILED the rigor contract. The reason is material."""

    kind: str = "downgrade"
    attempted_outcome: str = ""
    reasons: list[str] = field(default_factory=list)
    rule_set: str = ""


@dataclass
class BudgetTick(Event):
    """Budget state snapshot. Emitted per tick or on charge()."""

    kind: str = "budget_tick"
    state: dict = field(default_factory=dict)  # BudgetLedger.snapshot()


@dataclass
class DoomLoopFired(Event):
    """Doom-loop detector tripped; agent should drop-a-gear."""

    kind: str = "doom_loop"
    window: int = 0
    fingerprint: str = ""
    recent_actions: list[str] = field(default_factory=list)


@dataclass
class GateRejected(Event):
    """Lint-gate rejected an action pre-execution."""

    kind: str = "gate_rejected"
    tool: str = ""
    reason: str = ""
    args: dict = field(default_factory=dict)


@dataclass
class HumanGate(Event):
    """Agent paused for human approval on a scoped action."""

    kind: str = "human_gate"
    action_id: str = ""
    reason: str = ""
    resolved_as: str | None = None  # "approved" | "denied" | None while pending


# ---------------------------------------------------------------------------
# EventStream — append-only, JSONL-persisted
# ---------------------------------------------------------------------------


class EventStream:
    """In-memory + JSONL-backed event stream.

    Agents write via `append(evt)`; readers iterate via `__iter__` or slice
    via `tail(n)`. Persist-on-append keeps crash-recovery simple.
    """

    def __init__(self, jsonl_path: Path | None = None):
        self._events: list[Event] = []
        self.jsonl_path = jsonl_path
        if jsonl_path:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if jsonl_path.exists():
                self._load()

    def _load(self) -> None:
        assert self.jsonl_path is not None
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                self._events.append(_decode_event(d))
            except Exception:
                continue

    def append(self, evt: Event) -> Event:
        self._events.append(evt)
        if self.jsonl_path:
            with self.jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(evt.to_dict(), default=str) + "\n")
        return evt

    def tail(self, n: int) -> list[Event]:
        return self._events[-n:]

    def by_kind(self, kind: str) -> list[Event]:
        return [e for e in self._events if e.kind == kind]

    def __iter__(self):
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)


# ---------------------------------------------------------------------------
# Decoding (for replay)
# ---------------------------------------------------------------------------

_KIND_TO_CLS: dict[str, type[Event]] = {
    "action": Action,
    "observation": Observation,
    "hypothesis": Hypothesis,
    "result": Result,
    "downgrade": ClaimDowngrade,
    "budget_tick": BudgetTick,
    "doom_loop": DoomLoopFired,
    "gate_rejected": GateRejected,
    "human_gate": HumanGate,
}


def _decode_event(d: dict) -> Event:
    cls = _KIND_TO_CLS.get(d.get("kind", ""), Event)
    # Filter to known fields to avoid TypeError on schema evolution
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# Hypothesis-preserving compaction
# ---------------------------------------------------------------------------

KINDS_KEPT_ALWAYS = {"hypothesis", "result", "downgrade", "doom_loop"}


def compacted_view(
    stream: EventStream,
    keep_recent: int = 50,
) -> list[Event]:
    """Return a compacted copy suitable for sending to an LLM as context.

    Rule: keep the last `keep_recent` events verbatim, plus ALL of the
    load-bearing kinds (hypothesis, result, downgrade, doom_loop) from
    earlier history. The rest is dropped. This prevents the "compaction
    ate my hypothesis tree" failure observed in LangGraph long-horizon
    runs.
    """
    events = list(stream)
    if len(events) <= keep_recent:
        return events
    recent = events[-keep_recent:]
    older = events[:-keep_recent]
    kept_older = [e for e in older if e.kind in KINDS_KEPT_ALWAYS]
    return kept_older + recent
