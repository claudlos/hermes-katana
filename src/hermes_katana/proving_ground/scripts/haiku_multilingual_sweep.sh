#!/usr/bin/env bash
# Submit all 22 multilingual shards (IDs 101-122) as Anthropic batch jobs
# against Claude Haiku 4.5, code_comment channel. ~200 requests per batch.
#
# Cost estimate (Haiku 4.5 batch, 50% off):
#   input:  ~310 tok × $0.40/M = $0.00012/req
#   output: ~400 tok × $2.00/M = $0.00080/req
#   per req ~$0.001  →  22 × 200 = 4,400 reqs  ≈ $4.40 total
#
# Why 22 batches not 1 giant: per-shard telemetry (each (lang, shard) cell
# visible separately), smaller blast radius if one fails, parallel progress.
# Anthropic allows many concurrent batches on one API key.
#
# Usage: scripts/haiku_multilingual_sweep.sh [first_shard] [last_shard] [channel]

set -u
FIRST="${1:-101}"
LAST="${2:-122}"
CHANNEL="${3:-code_comment}"
MODEL="claude-haiku-4-5"
PROVIDER="anthropic"
TASK="code_review"

cd "$(dirname "$0")/.."

echo "Haiku multilingual sweep — shards $FIRST..$LAST × channel=$CHANNEL"
echo "Model: $MODEL   Provider: $PROVIDER"
echo

submitted=0
failed=0
for shard in $(seq "$FIRST" "$LAST"); do
    echo "=== shard $shard ==="
    .venv/bin/python scripts/batch_run.py build \
        --shard-id "$shard" --task "$TASK" --channel "$CHANNEL" \
        --model "$MODEL" --provider "$PROVIDER" 2>&1 | tail -1

    input_path="batch/in/shard_$(printf %03d "$shard")_${MODEL}_${CHANNEL}.jsonl"
    if [ ! -s "$input_path" ]; then
        echo "  SKIP: no input file at $input_path"
        failed=$((failed + 1))
        continue
    fi

    .venv/bin/python scripts/batch_run.py submit \
        --provider "$PROVIDER" --model "$MODEL" \
        --input "$input_path" 2>&1 | tail -2
    rc=$?
    if [ $rc -eq 0 ]; then
        submitted=$((submitted + 1))
    else
        failed=$((failed + 1))
    fi
done

echo
echo "=== summary ==="
echo "submitted: $submitted"
echo "failed:    $failed"
echo
echo "Track via:  .venv/bin/python scripts/batch_run.py list"
echo "Poll one:   .venv/bin/python scripts/batch_run.py poll --job batch/jobs/<msgbatch_id>.json"
