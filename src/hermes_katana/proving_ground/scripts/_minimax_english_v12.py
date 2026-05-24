"""Fleet v12 Track C: MiniMax M2.7 × English shards 001-010 × 4 channels.

One-shot driver: builds the batch input if missing, then calls
submit_minimax_direct which runs the 200 requests synchronously at 5 rps.

Writes:
  - batch/in/shard_NNN_MiniMax-M2.7_<channel>.jsonl
  - batch/out/direct-minimax-<ts>-shard_NNN_MiniMax-M2.7_<channel>.raw.jsonl
  - batch/jobs/direct-minimax-<ts>-...json (job record)
  - results/batch_runs/direct-minimax-<ts>-...jsonl (scored, via cmd_score)

Log:
  - /tmp/minimax_english_v12.log
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from batch_run import (  # noqa: E402
    _load_dotenv,
    submit_minimax_direct,
    _job_record_path,
    _json_default,
    build_batch,
    BATCH_IN,
)

LOG = Path("/tmp/minimax_english_v12.log")


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> int:
    _load_dotenv()
    log("=== v12 Track C start ===")
    CHANNELS = ["file_content", "code_comment", "tool_output", "data_row"]
    SHARDS = list(range(1, 11))
    MODEL = "MiniMax-M2.7"
    total = len(CHANNELS) * len(SHARDS)
    done = err = 0
    for shard in SHARDS:
        for ch in CHANNELS:
            safe_model = MODEL.replace("/", "_").replace(":", "_")
            inp = BATCH_IN / f"shard_{shard:03d}_{safe_model}_{ch}.jsonl"
            if not inp.exists():
                log(f"build shard_{shard:03d} {ch}")
                try:
                    build_batch(shard, "code_review", ch, MODEL, "minimax_direct", BATCH_IN)
                except Exception as e:
                    log(f"build FAIL shard_{shard:03d} {ch}: {str(e)[:200]}")
                    err += 1
                    continue

            log(f"submit shard_{shard:03d} {ch} (n={done + err + 1}/{total})")
            try:
                rec = submit_minimax_direct(inp, MODEL)
                job_path = _job_record_path(rec["batch_id"])
                job_path.write_text(json.dumps(rec, indent=2, default=_json_default), encoding="utf-8")
                done += 1
                # score immediately since minimax returns completed data
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/batch_run.py",
                        "score",
                        "--job",
                        str(job_path),
                    ],
                    check=False,
                    capture_output=True,
                )
                log(f"  ok: batch_id={rec['batch_id']}  n_ok={rec.get('n_ok')}  n_err={rec.get('n_err')}")
            except Exception as e:
                err += 1
                log(f"  submit FAIL: {str(e)[:300]}")
                # Back off briefly on error
                time.sleep(5)

    log(f"=== v12 Track C done: done={done} err={err} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
