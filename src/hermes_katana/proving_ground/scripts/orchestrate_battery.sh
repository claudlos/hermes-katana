#!/usr/bin/env bash
# Continuous orchestrator: walk (shard, model) pairs, running run_shard.py
# sequentially. Idempotent — skipping done attacks — so re-running is safe.
#
# Usage:
#   scripts/orchestrate_battery.sh <worker_tag> <model1,model2,...>  [max_sessions] [start_shard] [end_shard]
#
# Example:
#   scripts/orchestrate_battery.sh local-nvidia qwen3.5-4b,nemotron-4b,tinyllama-1b 50
#   scripts/orchestrate_battery.sh local-qwen qwen3-4b,qwen3.5-4b 40 25 60
#
# Stops on SIGINT (Ctrl+C) or SIGTERM. Safe to kill.

set -u

TAG="${1:?worker tag, e.g. local-nvidia}"
MODELS="${2:?comma-separated model ids}"
MAX_SESS="${3:-}"
START_SHARD="${4:-1}"
END_SHARD="${5:-100}"

cd "$(dirname "$0")/.."

MAX_ARG=""
if [[ -n "$MAX_SESS" ]]; then
    MAX_ARG="--max-sessions $MAX_SESS"
fi

IFS=',' read -ra MODEL_ARR <<< "$MODELS"

echo "[$(date '+%F %T')] orchestrator starting: tag=$TAG, models=${MODEL_ARR[*]}"
echo "[$(date '+%F %T')] shards ${START_SHARD}..${END_SHARD} cycled; per-shard max_sessions=${MAX_SESS:-all}"

for shard in $(seq "$START_SHARD" "$END_SHARD"); do
    for model in "${MODEL_ARR[@]}"; do
        echo "=============================================="
        echo "[$(date '+%F %T')] $TAG: shard $shard × $model"
        echo "=============================================="
        .venv/bin/python run_shard.py \
            --shard-id "$shard" \
            --model-id "$model" \
            $MAX_ARG \
            2>&1 | tee -a "/tmp/orchestrator_${TAG}.log"
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[$(date '+%F %T')] worker returned $rc; continuing to next pair"
        fi
    done
done

echo "[$(date '+%F %T')] orchestrator finished all 100 shards"
