"""Runtime guardrails for long-running fleet campaigns.

This is intentionally small and operational: it watches active status files and
interrupts only the specific worker process group that violates a guardrail.
Fleet supervisors then continue with the remaining queued lanes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_DIR = ROOT / "results" / "agent_shard_runs"
FLEET_DIR = ROOT / "results" / "fleet_runs"
DEFAULT_AGENT_DENYLIST = ROOT / "results" / "policies" / "or_free_lane_quarantine_20260505.json"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def _supervisor_alive(run_id: str) -> bool:
    pid_file = FLEET_DIR / run_id / "supervisor.pid"
    if not pid_file.exists():
        return False
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
        return True
    except Exception:
        return False


def _active_shards() -> list[tuple[int, str]]:
    try:
        out = subprocess.check_output(["pgrep", "-af", "run_agent_shard.py"], text=True)
    except subprocess.CalledProcessError:
        return []

    rows: list[tuple[int, str]] = []
    for line in out.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), parts[1]))
        except ValueError:
            continue
    return rows


def _interrupt_process_group(pid: int, why: str) -> bool:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGINT)
    except ProcessLookupError:
        return False
    except Exception as exc:
        _log(f"WARN failed to pull pid={pid}: {exc}")
        return False
    _log(f"pulled pid={pid} pgid={pgid}: {why}")
    return True


def _find_worker(run_id: str, agent_id: str, channel: str) -> list[int]:
    needle_agent = f"--agent-id {agent_id}"
    needle_channel = f"--channel {channel}"
    needle_run = f"--run-id {run_id}"
    return [pid for pid, cmd in _active_shards() if needle_agent in cmd and needle_channel in cmd and needle_run in cmd]


def _load_agent_denylist(paths: list[Path]) -> dict[str, str]:
    denied: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            _log(f"WARN failed to read denylist {path}: {exc}")
            continue

        entries = payload.get("quarantined_agents", [])
        if isinstance(entries, dict):
            entries = [{"agent_id": k, "reason": v} for k, v in entries.items()]
        for entry in entries:
            if isinstance(entry, str):
                denied[entry] = str(path)
            elif isinstance(entry, dict) and entry.get("agent_id"):
                denied[str(entry["agent_id"])] = str(entry.get("reason") or path)
    return denied


def _pull_denied_agents(
    run_id: str,
    denied_agents: dict[str, str],
    pulled: set[tuple[str, str, str]],
) -> None:
    if not denied_agents:
        return
    for pid, cmd in _active_shards():
        if f"--run-id {run_id}" not in cmd:
            continue
        for agent_id, reason in denied_agents.items():
            if f"--agent-id {agent_id}" not in cmd:
                continue
            key = (run_id, agent_id, str(pid))
            if key in pulled:
                continue
            pulled.add(key)
            _interrupt_process_group(
                pid,
                f"agent denylist: {agent_id}; {reason}",
            )


def _pull_claude_workers(run_id: str, pulled: set[tuple[str, str, str]]) -> None:
    for pid, cmd in _active_shards():
        if f"--run-id {run_id}" not in cmd or "--agent-id claude_cli" not in cmd:
            continue
        key = (run_id, "claude_cli", str(pid))
        if key in pulled:
            continue
        pulled.add(key)
        _interrupt_process_group(
            pid,
            "Claude usage limit active; pausing corpus_v2 Claude lane",
        )


def _pull_bad_invalid_lanes(
    run_id: str,
    min_done: int,
    max_invalid_rate: float,
    agent_prefixes: tuple[str, ...],
    pulled: set[tuple[str, str, str]],
) -> None:
    for status_path in glob.glob(str(STATUS_DIR / f"*{run_id}*.status.json")):
        try:
            status = json.loads(Path(status_path).read_text())
        except Exception:
            continue

        agent_id = status.get("agent_id") or ""
        channel = status.get("channel") or ""
        done = int(status.get("done") or 0)
        invalid = int(status.get("invalid_runs") or 0)
        if not agent_id.startswith(agent_prefixes) or done < min_done:
            continue

        invalid_rate = invalid / done if done else 0.0
        key = (run_id, agent_id, channel)
        if invalid_rate <= max_invalid_rate or key in pulled:
            continue

        pulled.add(key)
        workers = _find_worker(run_id, agent_id, channel)
        if not workers:
            _log(
                "threshold crossed but no active worker: "
                f"{agent_id} {channel} invalid={invalid}/{done} "
                f"rate={invalid_rate:.1%}"
            )
            continue
        for pid in workers:
            _interrupt_process_group(
                pid,
                f"invalid_runs {invalid}/{done} = {invalid_rate:.1%}",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-run-id", required=True)
    parser.add_argument("--or-run-id", required=True)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--min-done", type=int, default=10)
    parser.add_argument("--max-or-invalid-rate", type=float, default=0.50)
    parser.add_argument(
        "--max-invalid-rate",
        type=float,
        default=None,
        help="Generic invalid-run threshold. Defaults to --max-or-invalid-rate.",
    )
    parser.add_argument(
        "--invalid-agent-prefix",
        action="append",
        default=[],
        help=(
            "Agent prefix to monitor for invalid-run pulling. Can be repeated. "
            "Defaults to hermes_or_. Add hermes_minimax_ to guard MiniMax 429 tails."
        ),
    )
    parser.add_argument(
        "--pause-claude-main",
        action="store_true",
        help=(
            "Opt in to pulling claude_cli* workers on --main-run-id. This used "
            "to be automatic; leave disabled for mixed fleets unless a Claude "
            "limit is actually active."
        ),
    )
    parser.add_argument(
        "--agent-denylist",
        action="append",
        type=Path,
        default=[],
        help="JSON policy file containing quarantined_agents to pull immediately.",
    )
    parser.add_argument(
        "--disable-default-denylist",
        action="store_true",
        help="Do not load the default OR-free quarantine policy.",
    )
    args = parser.parse_args()

    denylist_paths = list(args.agent_denylist)
    if not args.disable_default_denylist:
        denylist_paths.append(DEFAULT_AGENT_DENYLIST)
    denied_agents = _load_agent_denylist(denylist_paths)
    invalid_prefixes = tuple(args.invalid_agent_prefix or ["hermes_or_"])
    max_invalid_rate = args.max_invalid_rate if args.max_invalid_rate is not None else args.max_or_invalid_rate

    pulled: set[tuple[str, str, str]] = set()
    _log("guard started")
    if denied_agents:
        _log(f"loaded agent denylist: {', '.join(sorted(denied_agents))}")
    _log(
        "invalid-rate watch: "
        f"prefixes={', '.join(invalid_prefixes)} "
        f"min_done={args.min_done} max_invalid_rate={max_invalid_rate:.1%}"
    )
    while _supervisor_alive(args.main_run_id) or _supervisor_alive(args.or_run_id):
        if _supervisor_alive(args.main_run_id):
            if args.pause_claude_main:
                _pull_claude_workers(args.main_run_id, pulled)
            _pull_denied_agents(args.main_run_id, denied_agents, pulled)
        if _supervisor_alive(args.or_run_id):
            _pull_denied_agents(args.or_run_id, denied_agents, pulled)
            _pull_bad_invalid_lanes(
                args.or_run_id,
                args.min_done,
                max_invalid_rate,
                invalid_prefixes,
                pulled,
            )
        time.sleep(args.interval_sec)
    _log("guard exiting; fleet supervisors are no longer active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
