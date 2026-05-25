#!/usr/bin/env bash
# Vast.ai on-instance boot script. Drop into the Vast "On-start script" field
# when creating an instance, OR paste into an SSH session after first login.
#
# What it does:
#   1. Installs vLLM (+ CUDA runtime) via pip.
#   2. Downloads the target HF model (huggingface-cli; falls back to hf_hub).
#   3. Starts `vllm serve ...` on 0.0.0.0:8000 with OpenAI-compat.
#   4. Prints the instance's public URL for our local run_shard.py to hit.
#
# Variables (set BEFORE running via `-e` or export):
#   HF_MODEL          — e.g. "Qwen/Qwen3.6-35B-A3B"
#   HF_REVISION       — (opt) commit SHA or tag to pin
#   VLLM_PORT         — default 8000
#   MAX_MODEL_LEN     — context, default 16384. Keep tight on 24 GB cards.
#   VLLM_EXTRA_ARGS   — (opt) extra flags, e.g. for NVFP4: "--quantization fp8"
#   PUBLIC_IPV4       — (opt) your IP for firewall; Vast usually exposes this
#   HF_TOKEN          — (opt) for gated models
#
# Notes:
#   - Vast's default Docker images have CUDA + Python 3.10/3.11 preinstalled.
#   - vllm serve binds to 0.0.0.0; Vast maps the port to a public one.
#     After boot, get the public port with `vastai show instance <ID>` — the
#     `8000/tcp` entry maps to something like 55555.
#   - This script is idempotent-ish: re-running after model download is fast.
#   - It logs to /var/log/katana-vast.log.

set -u

LOG=/var/log/katana-vast.log
mkdir -p "$(dirname "$LOG")" 2>/dev/null || LOG=/tmp/katana-vast.log

log() { echo "[$(date +%FT%T)] $*" | tee -a "$LOG"; }

HF_MODEL="${HF_MODEL:?set HF_MODEL to the HuggingFace repo id, e.g. Qwen/Qwen3.6-35B-A3B}"
VLLM_PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

log "katana vast bootstrap starting — model=$HF_MODEL port=$VLLM_PORT"

# -------- 1. Install vLLM --------
if ! python3 -c "import vllm" 2>/dev/null; then
    log "installing vllm..."
    python3 -m pip install --no-cache-dir --upgrade pip==26.1.1 wheel==0.47.0 2>&1 | tail -5 | tee -a "$LOG"
    # vllm 0.6+ needed for NVFP4; pin to known-good range.
    python3 -m pip install --no-cache-dir "vllm==0.21.0" "huggingface_hub[cli]==1.16.1" 2>&1 | tail -20 | tee -a "$LOG"
else
    log "vllm already installed"
fi

# -------- 2. Download the model --------
CACHE_DIR="${HF_HOME:-/root/.cache/huggingface}/hub"
mkdir -p "$CACHE_DIR"
log "downloading $HF_MODEL to $CACHE_DIR ..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>&1 | tail -3 | tee -a "$LOG" || true
fi
# --local-dir-use-symlinks False to avoid leaving behind a symlink mess if the
# instance is rebooted; --resume-download handles interrupted pulls.
huggingface-cli download "$HF_MODEL" \
    ${HF_REVISION:+--revision "$HF_REVISION"} \
    --local-dir-use-symlinks False 2>&1 | tail -15 | tee -a "$LOG"
log "download complete"

# -------- 3. Start vllm serve --------
# Tool-calling on most Qwen 3.x models works with the `hermes` tool parser;
# NVFP4 models are served with --quantization fp8 (vLLM's NVFP4 path).
TOOL_ARGS="--enable-auto-tool-choice --tool-call-parser hermes"
if [[ "$HF_MODEL" == *NVFP4* || "$HF_MODEL" == *nvfp4* ]]; then
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS --quantization fp8"
fi
if [[ "$HF_MODEL" == *gemma* ]]; then
    TOOL_ARGS=""   # Gemma doesn't have a native vLLM tool parser yet
fi

log "launching vllm serve — this occupies the shell; Ctrl-C to stop"
log "run_shard.py base_url will be: http://$(curl -s ifconfig.me):$VLLM_PORT/v1"

exec vllm serve "$HF_MODEL" \
    --host 0.0.0.0 --port "$VLLM_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype auto \
    --trust-remote-code \
    $TOOL_ARGS \
    $VLLM_EXTRA_ARGS 2>&1 | tee -a "$LOG"
