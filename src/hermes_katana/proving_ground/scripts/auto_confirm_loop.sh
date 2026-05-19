#!/bin/bash
# Auto-confirmation loop — run cross-reference + semantic fingerprinting
# on a schedule so confirmed_attacks.jsonl grows continuously while the
# fleet runs. Without this, confirmation is manual and the corpus only
# updates when someone remembers.
#
# Run from a tmux pane on the main box (the only box with the canonical
# results/ folder). Other boxes rsync into the same namespace.
#
# Usage:
#     bash scripts/auto_confirm_loop.sh                  # 1h interval
#     INTERVAL_SEC=1800 bash scripts/auto_confirm_loop.sh  # 30 min
#     ONCE=1 bash scripts/auto_confirm_loop.sh           # single pass + exit
#
# Stop with Ctrl+C — handles SIGINT cleanly.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer project venv — batch_fingerprint needs sentence_transformers which
# usually isn't installed system-wide. Fall back to system python3 if no venv.
if [ -x "$ROOT/.venv/bin/python" ]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="python3"
fi

INTERVAL_SEC="${INTERVAL_SEC:-3600}"
ONCE="${ONCE:-0}"
LOG_DIR="$ROOT/results/auto_confirm_logs"
mkdir -p "$LOG_DIR"

trap 'echo "[$(date)] auto_confirm_loop interrupted"; exit 0' INT TERM

run_pass() {
    local ts=$(date +%Y%m%d_%H%M%S)
    local logf="$LOG_DIR/pass_${ts}.log"
    echo "[$(date '+%F %T')] === pass start, log=$logf ==="

    {
        echo "=== pass ${ts} ==="
        echo

        echo "--- cross_reference_confirm ---"
        "$PY" scripts/cross_reference_confirm.py 2>&1
        echo

        echo "--- batch_fingerprint --mode agent ---"
        # Cap the fingerprint pass so it doesn't take forever — only the
        # most recent shard runs need re-scoring.
        "$PY" scripts/batch_fingerprint.py --mode agent 2>&1
        echo

        echo "--- confirmed_attacks count ---"
        if [ -f results/confirmed_attacks.jsonl ]; then
            wc -l results/confirmed_attacks.jsonl
        else
            echo "  (no confirmed_attacks.jsonl yet)"
        fi
        echo

        echo "--- agent_shard_runs row count ---"
        # How many trial rows have been written across all shards. Use this
        # to track fleet progress trend over time.
        find results/agent_shard_runs -name "*.jsonl" \
            ! -name "*.fp.jsonl" -print0 \
        | xargs -0 wc -l 2>/dev/null \
        | tail -1
    } > "$logf" 2>&1

    # Print just the headline numbers to stdout (so a tmux watcher can see).
    local conf=$(grep -E "Confirmed.*[0-9]" "$logf" | head -1 | tr -s ' ')
    local rows=$(grep -E "total$" "$logf" | tail -1 | tr -s ' ')
    echo "  $conf"
    echo "  trial rows: $rows"
    echo "[$(date '+%F %T')] === pass done ==="
}

while true; do
    run_pass
    if [ "$ONCE" = "1" ]; then
        break
    fi
    echo "[$(date '+%F %T')] sleeping ${INTERVAL_SEC}s ..."
    sleep "$INTERVAL_SEC"
done
