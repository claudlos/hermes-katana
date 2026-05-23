"""Workspace sweeper — remove stale session workspace dirs on a schedule.

Every session creates a fresh dir under `sessions/` with task files,
canary plants, and (for CLI agents) whatever files the agent wrote.
A crashed worker or a SIGKILL leaves these behind. Cron used to run
battery_monitor.sh every 5-30 min to clean them; this module moves the
job into the workers themselves so it works even without cron.

Rules:
  - Never touch a workspace whose dir is being written to within the last
    `fresh_seconds` (default 5 minutes) — it's probably an active session.
  - Prefer removing EMPTY stale dirs first (cheapest + safest).
  - Then remove OLD stale dirs (> older_seconds, default 4 hours).
  - Log counts so the caller can report cleanup activity.

Call from a worker at startup + periodically during long runs:

    from hermes_katana.proving_ground.sandbox.workspace_sweeper import sweep_sessions
    result = sweep_sessions()   # at start
    ... later, after every N sessions:
    if sessions_done % 50 == 0:
        sweep_sessions()
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SweepResult:
    scanned: int = 0
    deleted_empty: int = 0
    deleted_stale: int = 0
    skipped_fresh: int = 0
    skipped_nonempty_recent: int = 0
    bytes_freed: int = 0

    def summary(self) -> str:
        mb = self.bytes_freed / (1024 * 1024)
        return (
            f"sweep: scanned={self.scanned} deleted_empty={self.deleted_empty} "
            f"deleted_stale={self.deleted_stale} skipped={self.skipped_fresh + self.skipped_nonempty_recent} "
            f"freed={mb:.1f}MB"
        )


def _dir_size(path: Path, cap_bytes: int = 200 * 1024 * 1024) -> int:
    """Total byte size of a directory tree. Capped at cap_bytes to avoid
    pathological huge trees."""
    total = 0
    try:
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                    if total >= cap_bytes:
                        return total
                except OSError:
                    continue
    except OSError:
        pass
    return total


def sweep_sessions(
    sessions_root: str | Path = "sessions",
    fresh_seconds: int = 300,
    older_seconds: int = 14400,
    aggressive: bool = False,
) -> SweepResult:
    """Remove stale workspace dirs under `sessions_root`.

    Parameters
    ----------
    sessions_root : path to the sessions/ directory. Missing → no-op.
    fresh_seconds : any dir modified within this many seconds is untouched
                    (probably the active worker's current session).
    older_seconds : dirs older than this are removed regardless of size.
                    Default 4 h keeps the most recent runs on disk for
                    debugging while freeing older ones.
    aggressive    : if True, also remove dirs older than `fresh_seconds`
                    even if non-empty. Use only at orchestrator startup
                    when you KNOW no worker is active.

    Returns SweepResult with counts.
    """
    root = Path(sessions_root)
    result = SweepResult()
    if not root.exists() or not root.is_dir():
        return result

    now = time.time()

    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        result.scanned += 1
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            continue
        age = now - mtime

        # Skip anything modified very recently — likely an active session.
        if age < fresh_seconds and not aggressive:
            result.skipped_fresh += 1
            continue

        try:
            is_empty = next(entry.iterdir(), None) is None
        except OSError:
            is_empty = False

        if is_empty:
            try:
                entry.rmdir()
                result.deleted_empty += 1
            except OSError:
                continue
            continue

        if age >= older_seconds or aggressive:
            size = _dir_size(entry)
            try:
                shutil.rmtree(entry)
                result.deleted_stale += 1
                result.bytes_freed += size
            except OSError:
                continue
        else:
            result.skipped_nonempty_recent += 1

    return result


def sweep_shard_runs_status(
    agent_runs_root: str | Path = "results/agent_shard_runs",
    stale_seconds: int = 7200,
) -> SweepResult:
    """Remove *.status.json files for agent-CLI workers that haven't been
    touched in `stale_seconds` (default 2 h). These are heartbeat files;
    a stale one means the worker died without clean shutdown. Deleting
    them cleans up the monitor view; the JSONL output files are left
    alone (those are the actual session records).
    """
    root = Path(agent_runs_root)
    result = SweepResult()
    if not root.exists():
        return result
    now = time.time()
    for status_file in root.glob("*.status.json"):
        result.scanned += 1
        try:
            age = now - status_file.stat().st_mtime
        except OSError:
            continue
        if age >= stale_seconds:
            try:
                status_file.unlink()
                result.deleted_stale += 1
            except OSError:
                continue
    return result
