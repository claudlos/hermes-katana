"""Behavioral tracker — records every action the LLM takes in the sandbox.

Captures:
- Every tool call with full arguments and results
- Every message (LLM reasoning, user, system)
- Timing data
- File access patterns
- Network attempts
- Behavioral phase (pre-attack, attack, post-attack)
"""

import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from .workspace import ToolCall, ToolResult


class EventKind(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    TASK_ASSIGNED = "task_assigned"
    MESSAGE_USER = "message_user"
    MESSAGE_ASSISTANT = "message_assistant"
    MESSAGE_SYSTEM = "message_system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    HONEYPOT_TRIGGER = "honeypot_trigger"
    HONEYPOT_INJECT = "honeypot_inject"
    PHASE_CHANGE = "phase_change"
    ANOMALY = "anomaly"
    TURN_METRICS = "turn_metrics"  # Per-turn latency/tokens/finish/logprob stats
    CANARY_EXFIL = "canary_exfil"  # A planted canary token appeared in a tool arg


class Phase(str, Enum):
    WARMUP = "warmup"  # LLM is getting oriented
    BASELINE = "baseline"  # LLM is performing normal task
    PRE_ATTACK = "pre_attack"  # Honeypot is about to trigger
    ATTACK = "attack"  # Attack has been injected
    POST_ATTACK = "post_attack"  # Monitoring after attack
    RECOVERY = "recovery"  # Checking if behavior normalizes


@dataclass
class SessionEvent:
    """A single event in the evaluation session."""

    kind: EventKind
    phase: Phase
    timestamp: float
    data: dict = field(default_factory=dict)
    session_id: str = ""
    turn: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["phase"] = self.phase.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class BehavioralTracker:
    """Tracks all LLM actions during a sandbox session."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id TEXT PRIMARY KEY,
        attack_id TEXT,
        attack_text TEXT,
        attack_label TEXT,
        attack_strategy TEXT,
        task TEXT,
        model TEXT,
        provider TEXT,
        start_time REAL,
        end_time REAL,
        total_turns INTEGER,
        total_tool_calls INTEGER,
        phase TEXT,
        outcome TEXT DEFAULT 'unknown'
    );

    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        phase TEXT NOT NULL,
        turn INTEGER,
        timestamp REAL NOT NULL,
        data TEXT,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE TABLE IF NOT EXISTS tool_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        turn INTEGER,
        phase TEXT NOT NULL,
        tool TEXT NOT NULL,
        args TEXT,
        success BOOLEAN,
        output TEXT,
        error TEXT,
        latency_ms INTEGER,
        timestamp REAL NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(session_id)
    );

    CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
    CREATE INDEX IF NOT EXISTS idx_events_phase ON events(phase);
    CREATE INDEX IF NOT EXISTS idx_tools_session ON tool_calls(session_id);
    CREATE INDEX IF NOT EXISTS idx_tools_tool ON tool_calls(tool);
    """

    def __init__(self, db_path: str = "results/sandbox_tracking.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        # Current session state
        self.session_id: str = ""
        self.current_phase: Phase = Phase.WARMUP
        self.current_turn: int = 0
        self.events: list[SessionEvent] = []

    def _init_db(self):
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def start_session(
        self,
        session_id: str,
        task: str,
        model: str,
        provider: str,
        attack_id: str = "",
        attack_text: str = "",
        attack_label: str = "",
        attack_strategy: str = "",
    ):
        """Start tracking a new session."""
        self.session_id = session_id
        self.current_phase = Phase.WARMUP
        self.current_turn = 0
        self.events = []

        self.conn.execute(
            """INSERT INTO sessions (session_id, task, model, provider,
               attack_id, attack_text, attack_label, attack_strategy, start_time, phase)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                task,
                model,
                provider,
                attack_id,
                attack_text,
                attack_label,
                attack_strategy,
                time.time(),
                Phase.WARMUP.value,
            ),
        )
        self.conn.commit()

        self._record_event(
            EventKind.SESSION_START,
            {
                "task": task,
                "model": model,
                "provider": provider,
                "attack_id": attack_id,
                "attack_strategy": attack_strategy,
            },
        )

    def end_session(self, outcome: str = "unknown"):
        """End the current session."""
        self.conn.execute(
            """UPDATE sessions SET end_time = ?, total_turns = ?,
               phase = ?, outcome = ? WHERE session_id = ?""",
            (
                time.time(),
                self.current_turn,
                self.current_phase.value,
                outcome,
                self.session_id,
            ),
        )
        self.conn.commit()
        self._record_event(EventKind.SESSION_END, {"outcome": outcome})

    def set_phase(self, phase: Phase):
        """Transition to a new behavioral phase."""
        old_phase = self.current_phase
        self.current_phase = phase
        self._record_event(
            EventKind.PHASE_CHANGE,
            {
                "from": old_phase.value,
                "to": phase.value,
            },
        )

    def next_turn(self):
        """Advance to the next conversation turn."""
        self.current_turn += 1

    def record_message(self, role: str, content: str):
        """Record a message (user, assistant, system)."""
        kind = {
            "user": EventKind.MESSAGE_USER,
            "assistant": EventKind.MESSAGE_ASSISTANT,
            "system": EventKind.MESSAGE_SYSTEM,
        }.get(role, EventKind.MESSAGE_USER)

        # Storage cap bumped to 50 KB — prior 10 KB truncated long assistant
        # responses and made post-hoc re-parsing impossible.
        stored_content = content[:50000] if len(content) > 50000 else content
        self._record_event(kind, {"content": stored_content, "length": len(content)})

    def record_tool_call(self, call: ToolCall, result: ToolResult):
        """Record a tool call and its result."""
        # Record as event
        self._record_event(
            EventKind.TOOL_CALL,
            {
                "tool": call.tool,
                "args": call.args,
                "success": result.success,
                "output_preview": result.output[:500],
                "error": result.error,
                "latency_ms": result.latency_ms,
            },
        )

        # Record in dedicated tool_calls table
        self.conn.execute(
            """INSERT INTO tool_calls (session_id, turn, phase, tool, args,
               success, output, error, latency_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.session_id,
                self.current_turn,
                self.current_phase.value,
                call.tool,
                json.dumps(call.args),
                result.success,
                result.output[:5000],
                result.error,
                result.latency_ms,
                time.time(),
            ),
        )
        self.conn.commit()

    def record_turn_metrics(
        self,
        turn: int,
        latency_ms: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        finish_reason: str | None,
        logprob_stats: dict | None = None,
    ):
        """Per-turn telemetry: wall-clock latency, token counts, finish reason,
        and aggregate logprob statistics (mean/max entropy over the completion).

        Logprobs are stored as aggregate stats only — storing per-token logprobs
        for every turn would balloon the DB by orders of magnitude.
        """
        tps = None
        if completion_tokens and latency_ms and latency_ms > 0:
            tps = round(completion_tokens * 1000.0 / latency_ms, 3)
        self._record_event(
            EventKind.TURN_METRICS,
            {
                "turn": turn,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tok_per_sec": tps,
                "finish_reason": finish_reason,
                "logprob_stats": logprob_stats,
            },
        )

    def record_canary_exfil(self, canary_id: str, tool: str, where: str, excerpt: str):
        """A planted canary token appeared in a tool arg — direct exfiltration signal."""
        self._record_event(
            EventKind.CANARY_EXFIL,
            {
                "canary_id": canary_id,
                "tool": tool,
                "where": where,
                "excerpt": excerpt[:500],
            },
        )

    def record_honeypot_trigger(self, trigger_type: str, attack_text: str, injection_point: str):
        """Record that the honeypot was triggered."""
        self._record_event(
            EventKind.HONEYPOT_TRIGGER,
            {
                "trigger_type": trigger_type,
                "attack_text": attack_text[:2000],
                "injection_point": injection_point,
            },
        )

    def record_anomaly(self, description: str, data: dict | None = None):
        """Record a behavioral anomaly."""
        self._record_event(
            EventKind.ANOMALY,
            {
                "description": description,
                **(data or {}),
            },
        )

    def _record_event(self, kind: EventKind, data: dict):
        """Internal: record an event."""
        event = SessionEvent(
            kind=kind,
            phase=self.current_phase,
            timestamp=time.time(),
            data=data,
            session_id=self.session_id,
            turn=self.current_turn,
        )
        self.events.append(event)

        self.conn.execute(
            """INSERT INTO events (session_id, kind, phase, turn, timestamp, data)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                self.session_id,
                kind.value,
                self.current_phase.value,
                self.current_turn,
                event.timestamp,
                json.dumps(data),
            ),
        )
        self.conn.commit()

    def get_session_timeline(self, session_id: str | None = None) -> list[dict]:
        """Get the full event timeline for a session."""
        sid = session_id or self.session_id
        rows = self.conn.execute(
            """SELECT kind, phase, turn, timestamp, data FROM events
               WHERE session_id = ? ORDER BY timestamp""",
            (sid,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tool_calls_by_phase(self, session_id: str | None = None) -> dict[str, list[dict]]:
        """Get tool calls grouped by phase."""
        sid = session_id or self.session_id
        rows = self.conn.execute(
            """SELECT phase, tool, args, success, output, error, timestamp
               FROM tool_calls WHERE session_id = ? ORDER BY timestamp""",
            (sid,),
        ).fetchall()

        by_phase: dict[str, list[dict]] = {}
        for r in rows:
            by_phase.setdefault(r["phase"], []).append(dict(r))
        return by_phase

    def export_session_jsonl(self, session_id: str | None = None, output_path: str = "") -> str:
        """Export full session data as JSONL."""
        sid = session_id or self.session_id
        if not output_path:
            output_path = f"results/session_{sid}.jsonl"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        timeline = self.get_session_timeline(sid)
        with open(output_path, "w") as f:
            for event in timeline:
                f.write(json.dumps(event) + "\n")

        return output_path

    def close(self):
        self.conn.close()
