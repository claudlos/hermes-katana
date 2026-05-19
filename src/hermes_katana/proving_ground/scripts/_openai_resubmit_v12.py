"""Fleet v12 Track A: resubmit failed OpenAI batches with rolling window.

Keeps ~6 in-flight (within the 2M enqueued-tokens org cap).
Submits next batch whenever the number of non-terminal (validating/in_progress/
finalizing) OpenAI batches drops below the target.

Checks every 60 seconds. Exits when all remaining inputs have been submitted
at least once this session.

Log: /tmp/openai_resubmit_v12.log
Pending state: /tmp/resubmit_pending.json (consumed left-to-right)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from batch_run import (  # noqa: E402
    _load_dotenv,
    submit_openai,
    _job_record_path,
    _json_default,
)

LOG = Path("/tmp/openai_resubmit_v12.log")
PENDING = Path("/tmp/resubmit_pending.json")
TARGET_IN_FLIGHT = 6
POLL_SEC = 60


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def _count_active(client, submitted_ids: list[str]) -> int:
    active = 0
    for bid in submitted_ids:
        try:
            b = client.batches.retrieve(bid)
            if b.status in ("validating", "in_progress", "finalizing"):
                active += 1
        except Exception:
            pass
    return active


def main() -> int:
    _load_dotenv()
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=60.0)

    pending = json.loads(PENDING.read_text()) if PENDING.exists() else []
    log(f"=== v12 Track A start ({len(pending)} pending) ===")

    session_submissions: list[str] = []
    while pending:
        active = _count_active(client, session_submissions)
        room = TARGET_IN_FLIGHT - active
        if room <= 0:
            log(f"active={active}, waiting {POLL_SEC}s")
            time.sleep(POLL_SEC)
            continue
        to_submit = min(room, len(pending))
        log(f"active={active} → submitting {to_submit}")
        for _ in range(to_submit):
            r = pending.pop(0)
            try:
                rec = submit_openai(Path(r["input"]), r["model"])
                job_path = _job_record_path(rec["batch_id"])
                job_path.write_text(json.dumps(rec, indent=2, default=_json_default))
                session_submissions.append(rec["batch_id"])
                log(f"  ✓ {Path(r['input']).name} → {rec['batch_id']}")
            except Exception as e:
                msg = str(e)[:300]
                # If token_limit — put back and wait longer
                if "token_limit_exceeded" in msg or "enqueue" in msg.lower():
                    pending.insert(0, r)
                    log(f"  ✗ {Path(r['input']).name}: token cap — requeued")
                    time.sleep(POLL_SEC)
                    break
                log(f"  ✗ {Path(r['input']).name}: {msg}")
        # persist pending so restarts don't re-submit
        PENDING.write_text(json.dumps(pending))
        if pending:
            time.sleep(5)

    log(f"=== v12 Track A done: submitted {len(session_submissions)} in session ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
