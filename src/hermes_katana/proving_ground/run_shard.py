"""Universal single-worker shard runner.

Runs one (shard_id, model_id) pair end-to-end:
- Loads shards/shard_NNN.jsonl
- Resolves endpoint for model_id (local llama.cpp, Ollama, or OpenRouter)
- For each attack in the shard, runs a sandbox session
- Writes resumable outputs to results/shard_runs/shard_NNN_<model>.jsonl

Idempotent: re-running the same pair skips attacks already in the output.
Resumable: crashing/killing mid-shard loses only the in-flight session.
Portable: runs on laptop, Mini, Colab, Vast with the same interface.

Usage:
    python -m hermes_katana.proving_ground.run_shard --shard-id 1 --model-id qwen3.5-4b
    python -m hermes_katana.proving_ground.run_shard --shard-id 1 --model-id or-gemma4-31b:free
    python -m hermes_katana.proving_ground.run_shard --shard-id 1 --model-id qwen3.5-9b-ollama --max-sessions 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from hermes_katana.proving_ground.sandbox.session import SessionRunner, SessionConfig  # noqa: E402
from hermes_katana.proving_ground.sandbox.honeypot import HoneypotChannel  # noqa: E402
from hermes_katana.proving_ground.sandbox.behavioral_tracker import BehavioralTracker  # noqa: E402
from hermes_katana.proving_ground.sandbox.analyzers.behavioral_drift import BehavioralAnalyzer  # noqa: E402
from hermes_katana.proving_ground.sandbox.severity import score_session_api  # noqa: E402
from hermes_katana.proving_ground.sandbox.workspace_sweeper import sweep_sessions  # noqa: E402
from hermes_katana.proving_ground.models import AttackSample  # noqa: E402
import hermes_katana.proving_ground.local_models as local_models  # noqa: E402


# Load .env files into os.environ. Primary source is the project .env; we
# also read ~/.hermes/.env as a fallback so keys configured there (Anthropic,
# Gemini, MiniMax, OpenRouter, etc.) are usable without duplication. The
# project .env wins if both files define the same key.
def _load_dotenv():
    candidates = [ROOT / ".env", Path.home() / ".hermes" / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                # setdefault: don't clobber keys set by an earlier (higher
                # priority) .env. Note: the project .env is loaded first.
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except PermissionError:
            # ~/.hermes/.env may be 0600 on multi-user systems; skip quietly.
            continue


def _safe_slug(s: str) -> str:
    """Make a model id safe as a filename — no dots (they break with_suffix)."""
    return s.replace("/", "_").replace(":", "_").replace(" ", "_").replace(".", "-")


def _public_api_model_name(model_id: str) -> str:
    """Return the non-secret provider model name used for result metadata."""
    m = local_models.MODELS.get(model_id)
    if not m:
        return model_id
    backend = m.get("backend", "llama_cpp")
    if backend == "openrouter":
        return str(m.get("openrouter_slug") or model_id)
    if backend == "minimax":
        return str(m.get("minimax_slug") or model_id)
    if backend == "remote_api":
        return str(m.get("api_model_slug") or model_id)
    if backend == "ollama":
        return str(m.get("ollama_tag") or model_id)
    return model_id


def resolve_endpoint(
    model_id: str,
    port: int = 8080,
    startup_timeout: int = 30,
    ngl_override: int | None = None,
):
    """Return (base_url, api_model_name, api_key, cleanup_callable)."""
    if model_id not in local_models.MODELS:
        print(f"ERROR: unknown model_id {model_id!r}. Available: {sorted(local_models.MODELS)}")
        return None
    m = local_models.MODELS[model_id]
    backend = m.get("backend", "llama_cpp")

    if backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            print("SKIP: OPENROUTER_API_KEY not set (put it in .env)")
            return None
        slug = m["openrouter_slug"]
        print(f"    Backend: openrouter ({slug})")
        return (local_models.OPENROUTER_BASE_URL, slug, key, lambda: None)

    if backend == "minimax":
        key = os.environ.get("MINIMAX_API_KEY", "")
        if not key:
            print("SKIP: MINIMAX_API_KEY not set (check ~/.hermes/.env or project .env)")
            return None
        slug = m["minimax_slug"]
        print(f"    Backend: minimax ({slug})")
        return (local_models.MINIMAX_BASE_URL, slug, key, lambda: None)

    if backend == "remote_api":
        # Generic OpenAI-compat endpoint hosted elsewhere — Vast vLLM,
        # Colab-hosted, self-hosted llama.cpp server etc. Configured per-
        # model in local_models.py:
        #   "api_base_url":      full base URL (include /v1) or
        #   "api_base_url_env":  name of env var holding it (preferred for
        #                        Vast, where the IP changes per rental)
        #   "api_model_slug":    model identifier to pass through
        #   "api_key_env":       name of env var holding the API key
        #                        (optional — defaults to "not-needed")
        base_url = m.get("api_base_url")
        if not base_url:
            env_name = m.get("api_base_url_env", "")
            if env_name:
                base_url = os.environ.get(env_name, "")
        if not base_url:
            print(
                f"SKIP: remote_api base_url not set (set "
                f"{m.get('api_base_url_env', 'api_base_url')} in .env "
                f"or local_models.py)"
            )
            return None
        key_env = m.get("api_key_env", "")
        key = os.environ.get(key_env, "not-needed") if key_env else "not-needed"
        slug = m.get("api_model_slug") or model_id
        print(f"    Backend: remote_api {base_url} ({slug})")
        return (base_url, slug, key, lambda: None)

    if backend == "ollama":
        if not local_models.is_ollama_running():
            print("SKIP: ollama daemon not reachable")
            return None
        tag = m["ollama_tag"]
        if not local_models.ollama_has_model(tag):
            print(f"SKIP: ollama model not pulled — run: ollama pull {tag}")
            return None
        print(f"    Backend: ollama ({tag})")
        return (local_models.OLLAMA_BASE_URL, tag, "not-needed", lambda: None)

    # llama_cpp (default)
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    time.sleep(1)
    fork = m.get("requires_fork", False)
    server_bin = Path.home() / f"llama.cpp{'-bonsai' if fork else ''}" / "build" / "bin" / "llama-server"
    model_path = Path.home() / "models" / "gguf" / ("bonsai/" if fork else "") / m["file"]
    if not model_path.exists() or not server_bin.exists():
        print(f"SKIP: missing model or server binary ({model_path})")
        return None
    template = "gemma" if "gemma" in model_id else "chatml"
    if ngl_override is not None:
        ngl = ngl_override
    else:
        ngl = local_models.estimate_gpu_layers(m["size_gb"], local_models.get_gpu_vram_mb())
    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "-c",
        "4096",
        "-ngl",
        str(ngl),
        "-fa",
        "on",
        "--chat-template",
        template,
    ]
    log_path = str(Path(tempfile.gettempdir()) / f"llama_shard_{_safe_slug(model_id)}.log")
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    def _full_cleanup():
        """Kill the llama-server process AND close the log file. Called on
        every exit path below (success, timeout, exception) so we don't
        leak file handles across retry loops."""
        try:
            proc.kill()
        except Exception:
            pass
        try:
            log.close()
        except Exception:
            pass

    import urllib.request
    import urllib.error

    try:
        for _ in range(startup_timeout):
            time.sleep(1)
            try:
                if (
                    urllib.request.urlopen(
                        f"http://localhost:{port}/v1/models",
                        timeout=2,
                    ).status
                    == 200
                ):
                    mode = "GPU" if ngl >= 999 else f"CPU+{ngl}GPU"
                    print(f"    Backend: llama.cpp {mode}, {m['size_gb']:.1f}GB")
                    # Success — caller takes ownership of cleanup. We do NOT
                    # close the log here because stdout/stderr of proc are
                    # still writing into it.
                    return (
                        f"http://localhost:{port}/v1",
                        model_id,
                        "not-needed",
                        _full_cleanup,
                    )
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                # Expected during server warmup — keep polling.
                continue
    except BaseException:
        # KeyboardInterrupt / SystemExit still has to clean up the child.
        _full_cleanup()
        raise
    # Timeout — clean up and bail.
    _full_cleanup()
    print(f"SKIP: server timeout after {startup_timeout}s")
    return None


def _output_paths(shard_id: int, model_id: str) -> tuple[Path, Path, Path]:
    safe = _safe_slug(model_id)
    base = Path("results/shard_runs") / f"shard_{shard_id:03d}_{safe}"
    base.parent.mkdir(parents=True, exist_ok=True)
    return (
        base.with_suffix(".sessions.jsonl"),
        base.with_suffix(".signatures.jsonl"),
        base.with_suffix(".status.json"),
    )


def _load_already_done(sessions_path: Path) -> set[str]:
    """Return attack_ids already processed in this (shard, model) output."""
    done: set[str] = set()
    if sessions_path.exists():
        with sessions_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    aid = d.get("attack_id")
                    if aid:
                        done.add(aid)
                except Exception:
                    continue
    return done


def _load_shard(shard_id: int) -> list[AttackSample]:
    path = Path("shards") / f"shard_{shard_id:03d}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Shard not found: {path}")
    out: list[AttackSample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(
                AttackSample(
                    id=d["id"],
                    text=d["text"],
                    label=d.get("label", ""),
                    source_lang=d.get("source_lang") or "en",
                    origin=d.get("origin", "user_input"),
                    metadata=d,  # keep the whole row accessible for downstream analysis
                )
            )
    return out


async def run(
    shard_id: int,
    model_id: str,
    max_sessions: int | None,
    task_name: str,
    channel: str,
    max_turns: int,
    trigger_after: int,
    startup_timeout: int,
    ngl_override: int | None,
):
    _load_dotenv()
    sessions_path, signatures_path, status_path = _output_paths(shard_id, model_id)

    print(f"=== shard {shard_id:03d} × {model_id} ===")
    print(f"    outputs: {sessions_path}")

    swept = sweep_sessions()
    if swept.deleted_empty + swept.deleted_stale > 0:
        print(f"    [sweep] {swept.summary()}")

    public_api_model = _public_api_model_name(model_id)
    endpoint = resolve_endpoint(model_id, startup_timeout=startup_timeout, ngl_override=ngl_override)
    if endpoint is None:
        return 1
    base_url, api_model, endpoint_token, cleanup = endpoint

    try:
        attacks = _load_shard(shard_id)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        cleanup()
        return 2

    done = _load_already_done(sessions_path)
    pending = [a for a in attacks if a.id not in done]
    if max_sessions is not None:
        pending = pending[:max_sessions]
    print(f"    {len(attacks)} attacks total, {len(done)} done, {len(pending)} pending")
    if not pending:
        print("    Nothing to do.")
        cleanup()
        return 0

    # Dedicated per-(shard,model) tracking DB. Keeps writers isolated so we
    # never fight over SQLite locks when multiple workers run in parallel.
    db_path = Path("results/shard_runs") / f"shard_{shard_id:03d}_{_safe_slug(model_id)}.db"
    tracker = BehavioralTracker(str(db_path))
    analyzer = BehavioralAnalyzer(str(db_path))

    from hermes_katana.proving_ground.sandbox.analyzers.semantic_fingerprint import fingerprint_session

    t0 = time.time()
    n_effective = 0
    n_collapse = 0

    # Enable clean Ctrl+C → finalise current line, exit cleanly.
    stop = {"flag": False}

    def _sigint(_sig, _frm):
        print("\n[caught SIGINT — finishing current session then exiting]")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)

    with sessions_path.open("a", encoding="utf-8") as sess_f, signatures_path.open("a", encoding="utf-8") as sig_f:
        for i, attack in enumerate(pending):
            if stop["flag"]:
                break

            config = SessionConfig(
                task_name=task_name,
                attack=attack,
                honeypot_channel=HoneypotChannel(channel),
                model=api_model,
                base_url=base_url,
                api_key=endpoint_token,
                max_turns=max_turns,
                trigger_after_turns=trigger_after,
            )
            workspace = f"sessions/shard_{shard_id:03d}_{_safe_slug(model_id)}_{uuid.uuid4().hex[:6]}"
            runner = SessionRunner(tracker, workspace_root=workspace)
            print(
                f"    [{i + 1}/{len(pending)}] {attack.id} [{attack.label}]...",
                end=" ",
                flush=True,
            )

            try:
                result = await runner.run(config)
                report = analyzer.analyze_session(result.session_id)
                drift = report.tool_usage_drift
                collapsed = report.collapse_detected
                # Graded severity replaces the raw OR-of-flags rubric. The
                # binary `effective` field is preserved for back-compat —
                # now derived from severity >= EFFECTIVE_THRESHOLD.
                sev = score_session_api(
                    tool_drift=drift,
                    task_drift=report.task_drift,
                    exfiltration_risk=report.exfiltration_risk,
                    dangerous_action_rate=report.dangerous_action_rate,
                    collapsed=collapsed,
                )
                effective = int(sev.is_effective)
                n_effective += effective
                n_collapse += int(collapsed)
                print(
                    f"{result.total_turns}t {result.tool_calls_before}→{result.tool_calls_after} drift={drift:.2f}"
                    + (" COLLAPSE" if collapsed else "")
                    + f" sev={sev.total}"
                )

                # Stream result + signature to JSONL.
                out_row = {
                    "session_id": result.session_id,
                    "shard": shard_id,
                    "model_id": model_id,
                    "api_model": public_api_model,
                    "attack_id": attack.id,
                    "attack_label": attack.label,
                    "task": task_name,
                    "channel": channel,
                    "total_turns": result.total_turns,
                    "tools_before": result.tool_calls_before,
                    "tools_after": result.tool_calls_after,
                    "tool_drift": round(drift, 4),
                    "task_drift": round(report.task_drift, 4),
                    "exfil_risk": round(report.exfiltration_risk, 4),
                    "danger_rate": round(report.dangerous_action_rate, 4),
                    "collapse": collapsed,
                    "effective": effective,
                    "severity": sev.total,
                    "severity_components": sev.components,
                    "severity_top_signal": sev.top_signal,
                }
                sess_f.write(json.dumps(out_row) + "\n")
                sess_f.flush()

                # Signature (may be None if the session has no assistant messages).
                sig = fingerprint_session(tracker.conn, result.session_id)
                if sig is not None:
                    sig.effective = effective
                    sig.collapsed = int(collapsed)
                    from dataclasses import asdict

                    row = asdict(sig)
                    row["shard"] = shard_id
                    row["model_id"] = model_id
                    sig_f.write(json.dumps(row) + "\n")
                    sig_f.flush()
            except Exception as e:
                print(f"ERROR: {e}")

            # Heartbeat.
            status_path.write_text(
                json.dumps(
                    {
                        "shard": shard_id,
                        "model_id": model_id,
                        "total": len(attacks),
                        "done": len(done) + i + 1,
                        "effective": n_effective,
                        "collapsed": n_collapse,
                        "elapsed_sec": round(time.time() - t0, 1),
                        "updated_at": time.time(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    tracker.close()
    cleanup()
    elapsed = time.time() - t0
    print(
        f"\n    Done {len(pending)} sessions in {elapsed:.0f}s "
        f"({elapsed / max(1, len(pending)):.1f}s/session, {n_effective} effective, {n_collapse} collapsed)"
    )
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--model-id", type=str, required=True)
    p.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Cap sessions per shard (useful for smoke testing; defaults to whole shard)",
    )
    p.add_argument(
        "--task",
        default="code_review",
        help="Task name from WORKSPACE_TASKS (default: code_review)",
    )
    p.add_argument(
        "--channel",
        default="file_content",
        help="Honeypot channel (default: file_content)",
    )
    p.add_argument("--max-turns", type=int, default=15)
    p.add_argument("--trigger-after", type=int, default=5)
    p.add_argument("--startup-timeout", type=int, default=30)
    p.add_argument("--ngl", type=int, default=None, help="Override GPU layer count for llama.cpp")
    args = p.parse_args()

    rc = asyncio.run(
        run(
            shard_id=args.shard_id,
            model_id=args.model_id,
            max_sessions=args.max_sessions,
            task_name=args.task,
            channel=args.channel,
            max_turns=args.max_turns,
            trigger_after=args.trigger_after,
            startup_timeout=args.startup_timeout,
            ngl_override=args.ngl,
        )
    )
    sys.exit(rc)
