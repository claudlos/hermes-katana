"""Watch submitted Anthropic batches and auto-fetch+score as they complete.

Reads every batch/jobs/*.json record, polls the provider, and when a batch
reports status=ended: downloads responses and scores them, exactly like
`batch_run.py fetch + score` would. Idempotent — skips batches whose raw
response file already exists.

Usage:
  python scripts/batch_watcher.py                  # poll once, exit
  python scripts/batch_watcher.py --loop           # poll every 60s until all done
  python scripts/batch_watcher.py --loop --sleep 30

Handles: anthropic (real batch), openai (real batch), gemini_direct (already
synchronous — just re-scores if raw present). Gemini native batch is not
watched because we rejected that path (paid-tier only).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BATCH_JOBS = ROOT / "batch" / "jobs"
BATCH_OUT = ROOT / "batch" / "out"
RESULTS_DIR = ROOT / "results" / "batch_runs"


def _load_dotenv():
    import os

    for p in (ROOT / ".env", Path.home() / ".hermes" / ".env"):
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except PermissionError:
            continue


def batch_done(job: dict) -> tuple[str, dict | None]:
    import os

    prov = job["provider"]
    if prov == "anthropic":
        from anthropic import Anthropic

        b = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=60.0).messages.batches.retrieve(job["batch_id"])
        return b.processing_status, b.request_counts.model_dump() if hasattr(b.request_counts, "model_dump") else None
    if prov == "openai":
        from openai import OpenAI

        b = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0).batches.retrieve(job["batch_id"])
        return b.status, None
    if prov in ("gemini_direct", "minimax_direct"):
        # Always "completed" — direct-async is synchronous.
        return "completed", None
    return "unknown", None


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, check=False).returncode


def process_one(job_path: Path) -> str:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    raw_path = BATCH_OUT / f"{job['batch_id'].replace('/', '_')}.raw.jsonl"
    scored_path = RESULTS_DIR / f"{job['batch_id'].replace('/', '_')}.jsonl"
    # Already fully processed?
    if scored_path.exists() and scored_path.stat().st_size > 0:
        return "already_scored"

    status, _ = batch_done(job)
    is_done = status in ("ended", "completed")
    if not is_done:
        return status

    # Fetch if raw missing or empty.
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        rc = _run(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/batch_run.py",
                "fetch",
                "--job",
                str(job_path),
            ]
        )
        if rc != 0:
            return "fetch_failed"
    # Score if not already.
    if not scored_path.exists() or scored_path.stat().st_size == 0:
        rc = _run(
            [
                str(ROOT / ".venv/bin/python"),
                "scripts/batch_run.py",
                "score",
                "--job",
                str(job_path),
            ]
        )
        if rc != 0:
            return "score_failed"
    return "scored"


def poll_once() -> dict:
    counts: dict[str, int] = {}
    for j in sorted(BATCH_JOBS.glob("*.json")):
        try:
            r = process_one(j)
        except Exception as e:
            r = f"error:{type(e).__name__}"
        counts[r] = counts.get(r, 0) + 1
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--sleep", type=int, default=60)
    p.add_argument("--max-iters", type=int, default=1000)
    args = p.parse_args()

    _load_dotenv()

    iters = 0
    while True:
        iters += 1
        stamp = time.strftime("%H:%M:%S")
        counts = poll_once()
        still_open = counts.get("in_progress", 0) + counts.get("validating", 0) + counts.get("finalizing", 0)
        print(f"[{stamp}] iter={iters}  {counts}")
        sys.stdout.flush()
        if not args.loop:
            break
        if still_open == 0:
            print(f"[{stamp}] all batches drained — exiting")
            break
        if iters >= args.max_iters:
            print(f"[{stamp}] hit max_iters={args.max_iters} — exiting")
            break
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
