"""Unified observation schema — reconciles API-backend and agent-CLI outputs.

Before this module, downstream analysers had to branch on source:
    if "attack_run" in row:        # agent-CLI shape (run_agent_shard.py)
        ...
    elif "total_turns" in row:     # API-backend shape (run_shard.py)
        ...

That led to code duplication in cross_reference_confirm.py,
export_channel_weights.py, and every future script. ObservationRow is the
canonical shape both sources normalise onto.

Fields chosen to be the LCM of what's useful downstream:
  - identity: session_id, source, agent_or_model, platform
  - context: shard, task, channel, attack_id, attack_label, attack_text
  - severity: severity (0-100), components, top_signal, effective, collapsed
  - signals: canary_leaked, baseline_tools, post_tools, tool_drift
  - raw: original dict for debugging

Each field has a clear "missing" semantic (None or 0 or False) so analysers
don't have to handle absent keys.

Usage:
    from hermes_katana.proving_ground.sandbox.observation import load_all_observations
    obs = list(load_all_observations())
    for o in obs:
        ... same code works for both sources ...
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional


def _platform_of(model_or_agent_id: str) -> str:
    """Platform bucket from a model_id / agent_id string. Kept in sync with
    scripts/cross_reference_confirm.py — if you change one, change both (or
    move this into a shared helper)."""
    m = (model_or_agent_id or "").lower()
    if m.startswith("claude_cli"):
        return "claude_code_cli"
    if m.startswith("gemini_cli"):
        return "gemini_cli"
    if m.startswith("hermes_"):
        return "hermes_agent_cli"
    if "anthropic/" in m or m.startswith("or-claude") or "claude-haiku" in m or "claude-sonnet" in m:
        return "openrouter_api_anthropic"
    if "google/" in m or "gemini" in m:
        return "openrouter_api_google"
    if "openai/" in m or "gpt-" in m:
        return "openrouter_api_openai"
    if ":free" in m or "nvidia/" in m or "arcee-ai/" in m or "liquid/" in m:
        return "openrouter_api_free"
    if "minimax" in m:
        return "minimax_api"
    if m.startswith("qwen") or m.startswith("nemotron") or m.startswith("tinyllama") or m.startswith("bonsai"):
        return "local_llama_cpp"
    return "unknown"


@dataclass
class ObservationRow:
    """Canonical per-session observation. Every analyser consumes this."""

    source: str  # "api_session" | "cli_session"
    session_id: str
    shard: Optional[int]
    agent_or_model: str
    platform: str
    task: str
    channel: str  # file_content / code_comment / data_row / tool_output
    attack_id: str
    attack_label: str
    attack_text: str

    effective: bool
    collapsed: bool
    canary_leaked: bool

    severity: int  # 0-100
    severity_components: dict
    severity_top_signal: str

    # Source-specific signals — None when irrelevant to the other source.
    tool_drift: Optional[float] = None  # API only
    task_drift: Optional[float] = None  # API only
    exfil_risk: Optional[float] = None  # API only
    danger_rate: Optional[float] = None  # API only
    baseline_tools: Optional[int] = None
    post_tools: Optional[int] = None
    total_turns: Optional[int] = None  # API only
    duration_sec: Optional[float] = None  # CLI only (subprocess wall-clock)
    attack_mirror_rate: Optional[float] = None  # CLI only (semantic scoring)
    semantic_drift: Optional[float] = None  # CLI only

    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Keep the row JSON-friendly; raw can be large so we only include
        # a size marker by default. Callers that want raw explicitly can
        # access it directly.
        d.pop("raw", None)
        return d


def from_api_row(d: dict) -> ObservationRow:
    """Normalise a run_shard.py / battery_sessions.jsonl row."""
    model = d.get("model_id") or d.get("model") or ""
    return ObservationRow(
        source="api_session",
        session_id=d.get("session_id", ""),
        shard=d.get("shard"),
        agent_or_model=model,
        platform=_platform_of(model),
        task=d.get("task", ""),
        channel=d.get("channel", "file_content"),
        attack_id=d.get("attack_id", ""),
        attack_label=d.get("attack_label", ""),
        attack_text=d.get("attack_text", ""),
        effective=bool(d.get("effective")),
        collapsed=bool(d.get("collapse") or d.get("collapsed")),
        canary_leaked=bool(d.get("canary_leaked", False)),
        severity=int(d.get("severity", 0) or 0),
        severity_components=dict(d.get("severity_components") or {}),
        severity_top_signal=str(d.get("severity_top_signal", "")),
        tool_drift=d.get("tool_drift"),
        task_drift=d.get("task_drift"),
        exfil_risk=d.get("exfil_risk"),
        danger_rate=d.get("danger_rate"),
        baseline_tools=d.get("tools_before"),
        post_tools=d.get("tools_after"),
        total_turns=d.get("total_turns"),
        raw=d,
    )


def from_cli_row(d: dict) -> ObservationRow:
    """Normalise a run_agent_shard.py JSONL row."""
    agent = d.get("agent_id", "")
    attack_run = d.get("attack_run", {}) or {}
    baseline = d.get("baseline", {}) or {}
    semantic = d.get("semantic", {}) or {}
    return ObservationRow(
        source="cli_session",
        session_id=attack_run.get("agent_id", agent) + "_" + d.get("attack_id", "")[:8],
        shard=d.get("shard"),
        agent_or_model=agent,
        platform=_platform_of(agent),
        task=d.get("task", ""),
        channel=d.get("channel", "file_content"),
        attack_id=d.get("attack_id", ""),
        attack_label=d.get("attack_label", ""),
        attack_text="",  # CLI rows don't store attack_text; backfill from shards if needed
        effective=bool(d.get("effective")),
        collapsed=bool(d.get("collapsed") or attack_run.get("collapsed")),
        canary_leaked=bool(d.get("canary_leaked") or attack_run.get("canary_hits")),
        severity=int(d.get("severity", 0) or 0),
        severity_components=dict(d.get("severity_components") or {}),
        severity_top_signal=str(d.get("severity_top_signal", "")),
        baseline_tools=baseline.get("tool_call_count"),
        post_tools=attack_run.get("tool_call_count"),
        duration_sec=attack_run.get("duration_sec"),
        attack_mirror_rate=semantic.get("attack_mirror_rate"),
        semantic_drift=semantic.get("semantic_drift_baseline"),
        raw=d,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def iter_api_rows_from_db(db_path: str) -> Iterator[dict]:
    """Recover API-session rows from battery.db. Used when the
    battery_sessions.jsonl isn't available or is out of sync."""
    if not Path(db_path).exists():
        return
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # This is best-effort — the DB stores raw session metadata, not the
    # already-scored row. Prefer battery_sessions.jsonl when you have it.
    for r in conn.execute(
        "SELECT session_id, attack_id, attack_text, attack_label, task, model, total_turns FROM sessions"
    ):
        yield dict(r)
    conn.close()


def load_api_observations(
    sessions_jsonl: str = "results/battery_sessions.jsonl",
) -> Iterator[ObservationRow]:
    path = Path(sessions_jsonl)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            yield from_api_row(d)


def load_cli_observations(
    runs_dir: str = "results/agent_shard_runs",
) -> Iterator[ObservationRow]:
    root = Path(runs_dir)
    if not root.exists():
        return
    for f in sorted(root.glob("*.jsonl")):
        if "_broken_" in str(f):
            continue
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                yield from_cli_row(d)


def load_all_observations(
    sessions_jsonl: str = "results/battery_sessions.jsonl",
    agent_runs_dir: str = "results/agent_shard_runs",
) -> Iterator[ObservationRow]:
    """One iterator over every known observation. Downstream analysers
    just filter on o.source or o.platform as they need to."""
    yield from load_api_observations(sessions_jsonl)
    yield from load_cli_observations(agent_runs_dir)
