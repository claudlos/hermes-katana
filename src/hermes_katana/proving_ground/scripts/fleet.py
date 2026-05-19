"""Fleet launcher — parallel run_agent_shard.py workers from a JSON spec.

Usage:
    python scripts/fleet.py launch                    # auto-generated run_id
    python scripts/fleet.py launch --spec my.json     # explicit spec
    python scripts/fleet.py launch --run-id mycam12   # explicit run_id
    python scripts/fleet.py status                    # list active supervisors
    python scripts/fleet.py stop                      # SIGINT the sole active supervisor
    python scripts/fleet.py stop --run-id mycam12     # target a specific run

Spec format (JSON):
  {
    "max_concurrency": 8,
    "workers": [
      {"agent": "<agent_id>",
       "shards": [N, ...],
       "channels": [<one of file_content/code_comment/tool_output/data_row>, ...],
       "max_attacks": N,
       "instances": N,             # replicate this (agent,shard,channel) N times
       "multi_turn": false,        # optional
       "matched_pair": false}      # optional
    ],
    "controls": [ ...same shape, prepended with --control... ]
  }

Persistent state lives in `results/fleet_runs/<run_id>/`:

    supervisor.pid        — PID for `fleet.py stop`
    supervisor.log        — terse per-second status log
    run_meta.json         — spec + started_at + git HEAD snapshot
    jobs/<tag>.log        — per-worker stdout/stderr (one per Job)

Each run gets a short hex run_id auto-generated on launch unless overridden
by --run-id. The run_id is threaded down to run_agent_shard.py so every
JSONL result row carries it (enables `scripts/query.py --run-id <id>`).

Re-running is idempotent — run_agent_shard.py skips already-done attacks
in the output JSONL, so an interrupted fleet resumes cleanly.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLEET_RUNS = ROOT / "results" / "fleet_runs"
SCHEMA_VERSION = 1


def _new_run_id() -> str:
    # 8 hex chars = 4 bytes — short enough to type, ample collision-resistance
    # for the small number of concurrent campaigns a human manages.
    return secrets.token_hex(4)


@dataclass
class RunDirs:
    """Paths rooted at results/fleet_runs/<run_id>/."""

    run_id: str

    @property
    def base(self) -> Path:
        return FLEET_RUNS / self.run_id

    @property
    def pid(self) -> Path:
        return self.base / "supervisor.pid"

    @property
    def log(self) -> Path:
        return self.base / "supervisor.log"

    @property
    def meta(self) -> Path:
        return self.base / "run_meta.json"

    @property
    def jobs(self) -> Path:
        return self.base / "jobs"

    def ensure(self) -> None:
        self.jobs.mkdir(parents=True, exist_ok=True)


@dataclass
class Job:
    agent: str
    shard: int
    channel: str
    max_attacks: int
    run_id: str
    control: bool = False
    multi_turn: bool = False
    matched_pair: bool = False
    n_repeats: int = 1  # G2 (Q-E): within-cell repeats
    split: str = "all"  # G1 (Q-D): split filter
    trial_plan: Path | None = None
    task: str = "code_review"  # forced-read variants in WORKSPACE_TASKS
    proc: subprocess.Popen | None = field(default=None, repr=False)
    log_path: Path | None = field(default=None, repr=False)
    started_at: float = 0.0

    def tag(self) -> str:
        mode = "ctrl" if self.control else "atk"
        flags = []
        if self.multi_turn:
            flags.append("mt")
        if self.matched_pair:
            flags.append("mp")
        if self.n_repeats > 1:
            flags.append(f"r{self.n_repeats}")
        if self.split != "all":
            flags.append(f"sp:{self.split}")
        if self.task and self.task != "code_review":
            flags.append(f"t:{self.task}")
        flag_str = ("+" + ",".join(flags)) if flags else ""
        return f"{mode}:{self.agent}:s{self.shard:03d}:{self.channel}{flag_str}"

    def cmd(self) -> list[str]:
        argv = [
            str(ROOT / ".venv/bin/python"),
            str(ROOT / "run_agent_shard.py"),
            "--shard-id",
            str(self.shard),
            "--agent-id",
            self.agent,
            "--channel",
            self.channel,
            "--max-attacks",
            str(self.max_attacks),
            "--run-id",
            self.run_id,
            "--split",
            self.split,
            "--n-repeats",
            str(self.n_repeats),
            "--task",
            self.task,
        ]
        if self.control:
            argv.append("--control")
        if self.multi_turn:
            argv.append("--multi-turn")
        if self.matched_pair:
            argv.append("--matched-pair")
        if self.trial_plan is not None:
            argv.extend(["--trial-plan", str(self.trial_plan)])
        return argv


def _load_spec(path: Path) -> dict:
    return json.loads(path.read_text())


def _expand(spec: dict, run_id: str, trial_plan: Path | None = None) -> list[Job]:
    """Expand spec → job list with round-robin interleave across workers."""
    per_worker: list[list[Job]] = []
    for entry in spec.get("workers", []):
        bucket: list[Job] = []
        _expand_entry(entry, run_id=run_id, control=False, into=bucket, trial_plan=trial_plan)
        if bucket:
            per_worker.append(bucket)
    for entry in spec.get("controls", []):
        bucket = []
        _expand_entry(entry, run_id=run_id, control=True, into=bucket, trial_plan=trial_plan)
        if bucket:
            per_worker.append(bucket)

    jobs: list[Job] = []
    more = True
    idx = 0
    while more:
        more = False
        for bucket in per_worker:
            if idx < len(bucket):
                jobs.append(bucket[idx])
                more = True
        idx += 1
    return jobs


def _expand_entry(
    entry: dict,
    run_id: str,
    control: bool,
    into: list[Job],
    trial_plan: Path | None = None,
) -> None:
    agent = entry["agent"]
    shards = entry.get("shards") or [entry.get("shard", 1)]
    channels = entry.get("channels") or [entry.get("channel", "file_content")]
    instances = int(entry.get("instances", 1))
    if instances > 1:
        raise ValueError(
            "instances>1 would launch duplicate workers against the same "
            "output/baseline files. Use n_repeats for repeated measures or "
            "split shards/channels across distinct worker entries."
        )
    max_attacks = int(entry.get("max_attacks", 20))
    multi_turn = bool(entry.get("multi_turn"))
    matched_pair = bool(entry.get("matched_pair"))
    n_repeats = int(entry.get("n_repeats", 1))  # G2 (Q-E)
    split = str(entry.get("split", "all"))  # G1 (Q-D)
    # Forced-read task variants (one per channel): readme_summarize,
    # refactor_app, csv_summarize, triage_log. Default code_review for
    # legacy specs.
    tasks = entry.get("tasks") or [entry.get("task", "code_review")]
    for shard in shards:
        for ch in channels:
            for task_name in tasks:
                for _ in range(instances):
                    into.append(
                        Job(
                            agent=agent,
                            shard=int(shard),
                            channel=ch,
                            max_attacks=max_attacks,
                            run_id=run_id,
                            control=control,
                            multi_turn=multi_turn,
                            matched_pair=matched_pair,
                            n_repeats=n_repeats,
                            split=split,
                            trial_plan=trial_plan,
                            task=task_name,
                        )
                    )


# The supervisor log is only opened when a RunDirs is in scope; bind it via
# closure in launch() rather than a module global so multiple launches in the
# same process (e.g. a unit test) don't cross-talk.
def _make_logger(log_path: Path):
    def _log(msg: str) -> None:
        line = f"[{time.strftime('%F %T')}] {msg}\n"
        with log_path.open("a") as f:
            f.write(line)
        sys.stdout.write(line)
        sys.stdout.flush()

    return _log


def _safe_tag(tag: str) -> str:
    return tag.replace(":", "_").replace("/", "_").replace("+", "_").replace(",", "_")


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
    except Exception:
        return None


def _update_run_meta(dirs: RunDirs, **updates) -> None:
    """Merge lifecycle metadata into run_meta.json without losing launch fields."""
    meta = {}
    if dirs.meta.exists():
        try:
            meta = json.loads(dirs.meta.read_text())
        except Exception:
            meta = {}
    meta.update(updates)
    dirs.meta.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _check_prereg(spec: dict, allow_no_prereg: bool) -> tuple[Path | None, str | None]:
    """Q-D / preregistration gate (2026-05-02).

    A spec JSON should declare `"_prereg_id": "H-YYYYMMDD-slug"` matching a
    YAML in `research/hypotheses/H-{id}.yaml` whose `status` is NOT
    `resolved`. If no prereg id is declared, refuse launch unless the
    operator explicitly passed --allow-no-prereg (smoke / debug runs).

    Returns (path_or_None, error_message_or_None).
      ok:   (Path("research/hypotheses/H-...yaml"), None)
      skip: (None, None)        — escape hatch was set
      bad:  (None, "<reason>")  — refuse to launch
    """
    pid = spec.get("_prereg_id")
    if not pid:
        if allow_no_prereg:
            return (None, None)
        return (
            None,
            "spec missing `_prereg_id` and --allow-no-prereg not set. "
            "Author a hypothesis in research/hypotheses/H-YYYYMMDD-<slug>.yaml "
            "(see FACTORY.md §1) and reference it from the spec, or pass "
            "--allow-no-prereg for ad-hoc / smoke runs (those rows will not "
            "be eligible for hypothesis resolution).",
        )
    candidate = ROOT / "research" / "hypotheses" / f"{pid}.yaml"
    if not candidate.exists():
        return (
            None,
            f"prereg `{pid}` not found at {candidate}. Create the hypothesis YAML before launching.",
        )
    try:
        import yaml

        meta = yaml.safe_load(candidate.read_text()) or {}
    except Exception as e:
        return (None, f"prereg `{pid}` failed to parse: {e}")
    status = (meta.get("status") or "").lower()
    if status == "resolved":
        return (
            None,
            f"prereg `{pid}` is already `status: resolved`. Author a "
            f"new hypothesis (e.g. `{pid}-v2`) rather than collecting "
            f"more data against a closed prediction (HARKing risk).",
        )
    return (candidate, None)


def launch(
    spec_path: Path,
    run_id: str,
    allow_no_prereg: bool = False,
    trial_plan: Path | None = None,
    design_id: str | None = None,
) -> int:
    spec = _load_spec(spec_path)
    max_concurrency = int(spec.get("max_concurrency", 6))
    trial_plan = trial_plan or (Path(spec["_trial_plan"]) if spec.get("_trial_plan") else None)
    design_id = design_id or spec.get("_design_id")

    prereg_path, prereg_err = _check_prereg(spec, allow_no_prereg)
    if prereg_err:
        print(f"REFUSE LAUNCH: {prereg_err}", file=sys.stderr, flush=True)
        return 3
    if prereg_path is not None:
        print(f"prereg matched: {prereg_path}", flush=True)

    dirs = RunDirs(run_id)
    dirs.ensure()
    log = _make_logger(dirs.log)

    # Refuse to clobber an already-active run_id.
    if dirs.pid.exists():
        try:
            existing_pid = int(dirs.pid.read_text().strip())
            os.kill(existing_pid, 0)  # no-op signal — raises if process gone
            log(f"ERROR: run_id={run_id} already has active supervisor pid={existing_pid}")
            return 2
        except (ProcessLookupError, ValueError):
            dirs.pid.unlink(missing_ok=True)

    try:
        jobs = _expand(spec, run_id=run_id, trial_plan=trial_plan)
    except ValueError as e:
        print(f"REFUSE LAUNCH: invalid fleet spec: {e}", file=sys.stderr, flush=True)
        return 3
    total = len(jobs)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "spec_path": str(spec_path),
        "spec": spec,
        "total_jobs": total,
        "max_concurrency": max_concurrency,
        "started_at": int(time.time()),
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head": _git_head(),
        "python": sys.executable,
        "prereg_id": spec.get("_prereg_id"),
        "prereg_path": str(prereg_path) if prereg_path else None,
        "design_id": design_id,
        "trial_plan": str(trial_plan) if trial_plan else None,
    }
    dirs.meta.write_text(json.dumps(meta, indent=2, sort_keys=True))

    log(f"fleet launch — run_id={run_id} jobs={total} max_concurrency={max_concurrency}")

    dirs.pid.write_text(str(os.getpid()))

    stop = {"flag": False}

    def _handle_sigint(_s, _f):
        if stop["flag"]:
            log("second SIGINT — force exit")
            sys.exit(130)
        stop["flag"] = True
        log("SIGINT received — propagating SIGINT to children, draining")

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    running: list[Job] = []
    done_count = 0
    failed_count = 0

    def _launch_one(job: Job) -> None:
        job.log_path = dirs.jobs / f"{_safe_tag(job.tag())}.log"
        log_fh = job.log_path.open("w")
        job.proc = subprocess.Popen(
            job.cmd(),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            start_new_session=True,
        )
        job.started_at = time.time()
        log(f"launched {job.tag()} pid={job.proc.pid} log={job.log_path}")

    def _poll_running(items: list[Job]) -> tuple[list[Job], list[Job]]:
        still: list[Job] = []
        done: list[Job] = []
        for job in items:
            if job.proc is None:
                continue
            rc = job.proc.poll()
            if rc is None:
                still.append(job)
            else:
                elapsed = time.time() - job.started_at
                tag = job.tag()
                log(f"finished {tag} rc={rc} elapsed={elapsed:.0f}s")
                # Regression early-warning: a real claude_cli_haiku job runs
                # >=baseline (~22s) + 1+ attacks. A non-zero exit in <10s
                # almost always means the broken-runner regression returned
                # (CLAUDECODE / ANTHROPIC_API_KEY env leak) — surface loudly
                # so the operator stops the fleet instead of silently
                # burning quota on 460+ broken trials.
                if rc != 0 and elapsed < 10:
                    log(
                        f"WARN: {tag} exited rc={rc} after only {elapsed:.1f}s "
                        f"— likely runner regression; check log={job.log_path}"
                    )
                done.append(job)
        return still, done

    try:
        while jobs or running:
            while (not stop["flag"]) and jobs and len(running) < max_concurrency:
                job = jobs.pop(0)
                _launch_one(job)
                running.append(job)

            running, newly_done = _poll_running(running)
            for j in newly_done:
                done_count += 1
                rc = j.proc.returncode if j.proc else -1
                if rc != 0:
                    failed_count += 1

            if stop["flag"] and running:
                for j in running:
                    if j.proc and j.proc.poll() is None:
                        try:
                            j.proc.send_signal(signal.SIGINT)
                        except Exception:
                            pass

            if int(time.time()) % 30 == 0:
                log(f"status — running={len(running)} queued={len(jobs)} done={done_count} failed={failed_count}")

            if stop["flag"] and not running:
                break
            time.sleep(2)
    finally:
        deadline = time.time() + 20
        while running and time.time() < deadline:
            running, _ = _poll_running(running)
            time.sleep(1)
        for j in running:
            if j.proc and j.proc.poll() is None:
                try:
                    j.proc.kill()
                    log(f"SIGKILL {j.tag()} pid={j.proc.pid}")
                except Exception:
                    pass
        dirs.pid.unlink(missing_ok=True)

    exit_code = 130 if stop["flag"] else (1 if failed_count else 0)
    _update_run_meta(
        dirs,
        finished_at=int(time.time()),
        finished_at_iso=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        done_jobs=done_count,
        failed_jobs=failed_count,
        queued_jobs_remaining=len(jobs),
        exit_code=exit_code,
        interrupted=bool(stop["flag"]),
        design_id=design_id,
        trial_plan=str(trial_plan) if trial_plan else None,
    )
    log(f"fleet exit — run_id={run_id} total={total} done={done_count} failed={failed_count}")
    return exit_code


def _list_active_runs() -> list[tuple[str, int]]:
    """Return [(run_id, pid), ...] for supervisors whose PID is still alive."""
    if not FLEET_RUNS.exists():
        return []
    out: list[tuple[str, int]] = []
    for d in sorted(FLEET_RUNS.iterdir()):
        pid_file = d / "supervisor.pid"
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            out.append((d.name, pid))
        except (ProcessLookupError, ValueError, FileNotFoundError):
            # Stale pid file — clean it up opportunistically.
            pid_file.unlink(missing_ok=True)
    return out


def status(run_id: str | None) -> int:
    active = _list_active_runs()
    if not active:
        print("no fleet supervisor running")
        return 0
    if run_id:
        active = [(rid, pid) for rid, pid in active if rid == run_id]
        if not active:
            print(f"no active supervisor for run_id={run_id}")
            return 0
    print(f"{'run_id':<12} {'pid':>8}  last status")
    for rid, pid in active:
        log_path = RunDirs(rid).log
        last = ""
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            for ln in reversed(lines):
                if " status " in ln:
                    last = ln.strip()
                    break
            if not last and lines:
                last = lines[-1].strip()
        print(f"{rid:<12} {pid:>8}  {last[:120]}")
    return 0


def stop(run_id: str | None) -> int:
    active = _list_active_runs()
    if not active:
        print("no fleet supervisor running")
        return 0
    if run_id is None:
        if len(active) > 1:
            print("multiple active supervisors — pass --run-id <id>. Active:")
            for rid, pid in active:
                print(f"  {rid}  pid={pid}")
            return 2
        run_id, pid = active[0]
    else:
        match = [(rid, pid) for rid, pid in active if rid == run_id]
        if not match:
            print(f"no active supervisor for run_id={run_id}")
            return 1
        run_id, pid = match[0]
    try:
        os.kill(pid, signal.SIGINT)
        print(f"SIGINT sent to supervisor run_id={run_id} pid={pid}")
    except ProcessLookupError:
        print(f"supervisor {pid} already gone")
        RunDirs(run_id).pid.unlink(missing_ok=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    launch_p = sub.add_parser("launch", help="start a fleet from spec JSON")
    launch_p.add_argument("--spec", default=str(ROOT / "scripts" / "fleet_v11.json"))
    launch_p.add_argument(
        "--run-id",
        default=None,
        help="8-char hex by default; override for recognizable runs",
    )
    launch_p.add_argument(
        "--allow-no-prereg",
        action="store_true",
        help="Bypass the preregistration gate (Q-D, 2026-05-02). "
        "Use for smoke tests and debug runs only — rows "
        "produced will not be eligible for hypothesis "
        "resolution.",
    )
    launch_p.add_argument(
        "--trial-plan",
        type=Path,
        default=None,
        help="Planned-trial manifest to pass to each worker. Overrides spec _trial_plan.",
    )
    launch_p.add_argument(
        "--design-id",
        default=None,
        help="Design id recorded in run_meta.json. Overrides spec _design_id.",
    )

    status_p = sub.add_parser("status", help="show running fleet state")
    status_p.add_argument("--run-id", default=None)

    stop_p = sub.add_parser("stop", help="SIGINT the supervisor")
    stop_p.add_argument("--run-id", default=None, help="required if >1 supervisor is active")

    args = p.parse_args()
    if args.cmd == "launch":
        run_id = args.run_id or _new_run_id()
        return launch(
            Path(args.spec),
            run_id=run_id,
            allow_no_prereg=args.allow_no_prereg,
            trial_plan=args.trial_plan,
            design_id=args.design_id,
        )
    if args.cmd == "status":
        return status(args.run_id)
    if args.cmd == "stop":
        return stop(args.run_id)
    return 2


if __name__ == "__main__":
    sys.exit(main())
