#!/usr/bin/env python3
"""Local model manager — downloads and serves GGUF models via llama.cpp server.

For Carlos's RTX 3050Ti (4GB VRAM):
- Bonsai 8B 1-bit: 1.15 GB, ~67 tok/s (best quality that fits)
- Bonsai 4B 1-bit: ~0.6 GB, ~96 tok/s (fastest)
- Qwen3 4B Q4_K_M: ~2.5 GB, ~49 tok/s (good quality, standard quant)
- Nemotron 4B: similar size range

Usage:
    python -m hermes_katana.proving_ground.local_models list
    python -m hermes_katana.proving_ground.local_models download <id>
    python -m hermes_katana.proving_ground.local_models serve <id>
    python -m hermes_katana.proving_ground.local_models stop
    python -m hermes_katana.proving_ground.local_models status
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Model registry — small models that fit in 4GB VRAM
MODELS = {
    # 1-bit models (PrismML Bonsai) — smallest, fastest, need custom fork
    "bonsai-8b": {
        "name": "Bonsai 8B (1-bit)",
        "repo": "prism-ml/Bonsai-8B-gguf",
        "file": "Bonsai-8B.gguf",
        "size_gb": 1.1,
        "vram_gb": 1.5,
        "tok_per_sec": 67,
        "context": 4096,
        "description": "8B 1-bit model. Fits easily in 4GB. Good quality for size.",
        "requires_fork": True,
    },
    "bonsai-4b": {
        "name": "Bonsai 4B (1-bit)",
        "repo": "prism-ml/Bonsai-4B-gguf",
        "file": "Bonsai-4B.gguf",
        "size_gb": 0.55,
        "vram_gb": 1.0,
        "tok_per_sec": 96,
        "context": 4096,
        "description": "4B 1-bit model. Very fast, minimal VRAM.",
        "requires_fork": True,
    },
    "bonsai-1.7b": {
        "name": "Bonsai 1.7B (1-bit)",
        "repo": "prism-ml/Bonsai-1.7B-gguf",
        "file": "Bonsai-1.7B.gguf",
        "size_gb": 0.24,
        "vram_gb": 0.5,
        "tok_per_sec": 175,
        "context": 4096,
        "description": "1.7B 1-bit model. Tiniest Bonsai. Ultra fast.",
        "requires_fork": True,
    },
    # Uncensored models — zero refusals, ideal for attack testing
    "gemma4-e4b-obliterated": {
        "name": "Gemma-4 E4B OBLITERATED v2 (Q4_K_M)",
        "repo": "OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
        "file": "gemma-4-E4B-it-OBLITERATED-Q4_K_M.gguf",
        "size_gb": 5.0,
        "vram_gb": 5.5,
        "tok_per_sec": 35,
        "context": 4096,
        "description": "0% refusal rate. Gemma-4 E4B (8B params, 4.5B effective). Needs CPU offloading on 4GB VRAM.",
        "requires_fork": False,
    },
    "gemma4-e4b-obliterated-q5": {
        "name": "Gemma-4 E4B OBLITERATED v2 (Q5_K_M)",
        "repo": "OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
        "file": "gemma-4-E4B-it-OBLITERATED-Q5_K_M.gguf",
        "size_gb": 5.4,
        "vram_gb": 6.0,
        "tok_per_sec": 30,
        "context": 4096,
        "description": "Higher quality quant of OBLITERATED. ~15 layers on GPU, rest on CPU.",
        "requires_fork": False,
    },
    "qwen3.5-4b-abliterated": {
        "name": "Qwen3.5-4B abliterated (Q4_K_M)",
        "repo": "mradermacher/Qwen3.5-4B-abliterated-GGUF",
        "file": "Qwen3.5-4B-abliterated.Q4_K_M.gguf",
        "size_gb": 2.8,
        "vram_gb": 3.2,
        "tok_per_sec": 50,
        "context": 4096,
        "description": "Abliterated Qwen3.5-4B. Light, fast, uncensored. Good tool use.",
        "requires_fork": False,
    },
    "qwen3.5-4b-abliterated-q5": {
        "name": "Qwen3.5-4B abliterated (Q5_K_M)",
        "repo": "mradermacher/Qwen3.5-4B-abliterated-GGUF",
        "file": "Qwen3.5-4B-abliterated.Q5_K_M.gguf",
        "size_gb": 3.2,
        "vram_gb": 3.6,
        "tok_per_sec": 45,
        "context": 4096,
        "description": "Higher quality quant. Still fits in 4GB.",
        "requires_fork": False,
    },
    # Standard GGUF models — work with upstream llama.cpp (may refuse some attacks)
    "qwen3.5-4b": {
        "name": "Qwen3.5-4B (Q4_K_M)",
        "repo": "unsloth/Qwen3.5-4B-GGUF",
        "file": "Qwen3.5-4B-Q4_K_M.gguf",
        "size_gb": 2.8,
        "vram_gb": 3.2,
        "tok_per_sec": 50,
        "context": 4096,
        "description": "Qwen3.5 4B, 4-bit quantized. Has safety — will refuse some attacks. Useful as baseline comparison.",
        "requires_fork": False,
    },
    "gemma4-e4b": {
        "name": "Gemma-4 E4B instruct (Q4_K_M)",
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "file": "gemma-4-E4B-it-Q4_K_M.gguf",
        "size_gb": 4.9,
        "vram_gb": 5.5,
        "tok_per_sec": 35,
        "context": 2048,
        "description": "Original Google Gemma-4 E4B with safety. 98.8% refusal rate. Baseline for comparing against OBLITERATED.",
        "requires_fork": False,
    },
    "gemma4-e2b": {
        "name": "Gemma-4 E2B instruct (Q4_K_M)",
        "repo": "unsloth/gemma-4-E2B-it-GGUF",
        "file": "gemma-4-E2B-it-Q4_K_M.gguf",
        "size_gb": 2.9,
        "vram_gb": 3.5,
        "tok_per_sec": 60,
        "context": 4096,
        "description": "Smaller Gemma-4 (2.3B effective params). Multimodal (text+image+audio). Fits in VRAM.",
        "requires_fork": False,
    },
    "qwen3.5-9b": {
        "name": "Qwen3.5-9B (Q4_K_M)",
        "repo": "unsloth/Qwen3.5-9B-GGUF",
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "size_gb": 5.3,
        "vram_gb": 6.0,
        "tok_per_sec": 25,
        "context": 4096,
        "description": "Qwen3.5 9B. Best quality available in the small range. Needs CPU offloading. ~15 layers on GPU.",
        "requires_fork": False,
    },
    "qwen3-4b": {
        "name": "Qwen3 4B (Q4_K_M)",
        "repo": "Qwen/Qwen3-4B-GGUF",
        "file": "Qwen3-4B-Q4_K_M.gguf",
        "size_gb": 2.5,
        "vram_gb": 3.0,
        "tok_per_sec": 49,
        "context": 4096,
        "description": "Qwen3 4B, 4-bit quantized. Good general quality. Older gen but proven.",
        "requires_fork": False,
    },
    "nemotron-4b": {
        "name": "NVIDIA Nemotron 4B (Q4_K_M)",
        "repo": "bartowski/NVIDIA-Nemotron-Mini-4B-Instruct-GGUF",
        "file": "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf",
        "size_gb": 2.6,
        "vram_gb": 3.0,
        "tok_per_sec": 50,
        "context": 4096,
        "description": "NVIDIA's small instruct model. Good at tool use.",
        "requires_fork": False,
    },
    "tinyllama-1b": {
        "name": "TinyLlama 1.1B (Q4_K_M)",
        "repo": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_gb": 0.67,
        "vram_gb": 1.0,
        "tok_per_sec": 150,
        "context": 4096,
        "description": "Tiny baseline model. Useful for speed testing and as a control.",
        "requires_fork": False,
    },
    # Ollama-backed models — used when a model can't load in our llama.cpp build.
    # Ollama runs as a separate daemon and exposes an OpenAI-compatible endpoint.
    "gemma4-e2b-ollama": {
        "name": "Gemma-4 E2B (via Ollama)",
        "backend": "ollama",
        "ollama_tag": "gemma4:e2b",
        "size_gb": 7.2,  # As stored by Ollama (BF16 weights for some shards)
        "vram_gb": 4.0,
        "tok_per_sec": 30,
        "context": 8192,
        "description": "Gemma-4 E2B served by the Ollama daemon. Use when llama.cpp can't load Gemma 4.",
        "requires_fork": False,
    },
    "gemma4-e4b-ollama": {
        "name": "Gemma-4 E4B (via Ollama)",
        "backend": "ollama",
        "ollama_tag": "gemma4:e4b",
        "size_gb": 11.0,
        "vram_gb": 4.0,
        "tok_per_sec": 20,
        "context": 8192,
        "description": "Gemma-4 E4B served by the Ollama daemon.",
        "requires_fork": False,
    },
    "qwen3.5-9b-ollama": {
        "name": "Qwen 3.5 9B (via Ollama)",
        "backend": "ollama",
        "ollama_tag": "qwen3.5:9b",
        "size_gb": 6.6,
        "vram_gb": 4.0,
        "tok_per_sec": 15,
        "context": 8192,
        "description": "Qwen 3.5 9B via Ollama. Uses Gated DeltaNet; llama.cpp "
        "hits CUDA OOM + partial-DeltaNet warnings on 4GB cards.",
        "requires_fork": False,
    },
    # OpenRouter free-tier tool-capable models. Keys loaded from $OPENROUTER_API_KEY.
    # Rate limits: 20 req/min per key across all :free models, ~200 req/day per model.
    # tool calling confirmed available on all entries below (Apr 2026).
    "or-gemma4-26b-a4b:free": {
        "name": "Gemma 4 26B-A4B (OpenRouter free)",
        "backend": "openrouter",
        "openrouter_slug": "google/gemma-4-26b-a4b-it:free",
        "context": 32768,
        "tok_per_sec": 40,
        "description": "Gemma 4 mid-range via OpenRouter free tier.",
    },
    "or-gemma4-31b:free": {
        "name": "Gemma 4 31B (OpenRouter free)",
        "backend": "openrouter",
        "openrouter_slug": "google/gemma-4-31b-it:free",
        "context": 32768,
        "tok_per_sec": 30,
        "description": "Gemma 4 31B via OpenRouter free tier.",
    },
    "or-nemotron3-super-120b:free": {
        "name": "Nemotron 3 Super 120B-A12B (OpenRouter free)",
        "backend": "openrouter",
        "openrouter_slug": "nvidia/nemotron-3-super-120b-a12b:free",
        "context": 32768,
        "tok_per_sec": 25,
        "description": "NVIDIA Nemotron 3 Super MoE via OpenRouter free tier.",
    },
    "or-minimax-m2.5:free": {
        "name": "MiniMax M2.5 (OpenRouter free)",
        "backend": "openrouter",
        "openrouter_slug": "minimax/minimax-m2.5:free",
        "context": 200000,
        "tok_per_sec": 30,
        "description": "MiniMax M2.5 via OpenRouter free tier; long context.",
    },
    "or-arcee-trinity-large:free": {
        "name": "Arcee Trinity Large Preview (OpenRouter free)",
        "backend": "openrouter",
        "openrouter_slug": "arcee-ai/trinity-large-preview:free",
        "context": 32768,
        "tok_per_sec": 30,
        "description": "Arcee Trinity large preview via OpenRouter free tier.",
    },
    "or-liquid-lfm-2.5-1.2b:free": {
        "name": "Liquid LFM 2.5 1.2B thinking (OpenRouter free)",
        "backend": "openrouter",
        "openrouter_slug": "liquid/lfm-2.5-1.2b-thinking:free",
        "context": 32768,
        "tok_per_sec": 60,
        "description": "Tiny Liquid thinking model; fast baseline.",
    },
    # Paid OpenRouter models — cheap per-session, gives us Claude + Gemini
    # coverage without needing to set up direct provider SDKs. Budget-capped
    # because they use the $1 OpenRouter credit.
    "or-claude-haiku-4-5": {
        "name": "Claude Haiku 4.5 (OpenRouter, paid)",
        "backend": "openrouter",
        "openrouter_slug": "anthropic/claude-haiku-4-5",
        "context": 200000,
        "tok_per_sec": 70,
        "description": "Anthropic's cheapest frontier model, via OpenRouter.",
    },
    "or-claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6 (OpenRouter, paid)",
        "backend": "openrouter",
        "openrouter_slug": "anthropic/claude-sonnet-4-6",
        "context": 200000,
        "tok_per_sec": 40,
        "description": "Anthropic mid-tier; stronger tool-use than Haiku.",
    },
    "or-gemini-2-flash": {
        "name": "Google Gemini 2 Flash (OpenRouter, paid)",
        "backend": "openrouter",
        "openrouter_slug": "google/gemini-2-flash",
        "context": 1000000,
        "tok_per_sec": 90,
        "description": "Google Gemini fast tier with huge context window.",
    },
    "or-gpt-4o-mini": {
        "name": "OpenAI GPT-4o Mini (OpenRouter, paid)",
        "backend": "openrouter",
        "openrouter_slug": "openai/gpt-4o-mini",
        "context": 128000,
        "tok_per_sec": 60,
        "description": "OpenAI cheapest frontier for comparison.",
    },
    # MiniMax direct via the international endpoint (minimaxi.chat).
    # The MINIMAX_API_KEY with sk-cp- prefix in .env has M2 + M2.5 access;
    # Text-01, M1, abab series all return 2061 "plan not supported".
    "minimax-m2": {
        "name": "MiniMax M2",
        "backend": "minimax",
        "minimax_slug": "MiniMax-M2",
        "context": 245000,
        "tok_per_sec": 40,
        "description": "MiniMax M2 MoE. Accessible via sk-cp- key.",
    },
    "minimax-m2.5": {
        "name": "MiniMax M2.5",
        "backend": "minimax",
        "minimax_slug": "MiniMax-M2.5",
        "context": 245000,
        "tok_per_sec": 35,
        "description": "MiniMax M2.5 reasoning. Accessible via sk-cp- key.",
    },
    "minimax-m2.7": {
        "name": "MiniMax M2.7",
        "backend": "minimax",
        "minimax_slug": "MiniMax-M2.7",
        "context": 245000,
        "tok_per_sec": 40,
        "description": "MiniMax M2.7 latest. Via MINIMAX_API_KEY.",
    },
    "mimo-v2.5-pro": {
        "name": "MiMo v2.5 Pro (Xiaomi direct)",
        "backend": "remote_api",
        "api_base_url_env": "XIAOMI_BASE_URL",
        "api_model_slug": "xiaomi/mimo-v2.5-pro",
        "api_key_env": "XIAOMI_API_KEY",
        "context": 131072,
        "tok_per_sec": 50,
        "description": "Xiaomi MiMo v2.5 Pro via direct API. Key + URL in .env.",
    },
    "nous-step-3.5-flash": {
        "name": "StepFun Step 3.5 Flash (Nous free)",
        "backend": "remote_api",
        "api_base_url": "https://inference.nousresearch.com/v1",
        "api_model_slug": "stepfun/step-3.5-flash",
        "api_key_env": "NOUS_PORTAL_API_KEY",
        "context": 131072,
        "tok_per_sec": 80,
        "description": "StepFun Step 3.5 Flash via Nous Portal (free).",
    },
    "nous-gemma-4-31b-free": {
        "name": "Gemma 4 31B (Nous free)",
        "backend": "remote_api",
        "api_base_url": "https://inference.nousresearch.com/v1",
        "api_model_slug": "google/gemma-4-31b-it:free",
        "api_key_env": "NOUS_PORTAL_API_KEY",
        "context": 32768,
        "tok_per_sec": 30,
        "description": "Google Gemma 4 31B via Nous Portal (free).",
    },
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# International endpoint — api.minimax.chat (CN) rejects the sk-cp- key scheme.
MINIMAX_BASE_URL = "https://api.minimaxi.chat/v1"


# --- Remote vLLM / Vast entries ---
# Register one per Vast rental. The api_base_url_env convention lets you
# leave the model entry static and set the IP at rental time:
#   export VAST_QWEN36_35B_URL=http://193.123.45.67:8000/v1
# then `run_shard.py --model-id vast-qwen3-6-35b-a3b`.
#
# The 3 models the user flagged for Vast (April 2026):
#   - Qwen/Qwen3.6-35B-A3B                            (MoE, 3B active, ~22GB Q4)
#   - HauhauCS/Qwen3.6-35B-A3B-Uncensored-Aggressive  (same base, uncensored)
#   - LilaRest/gemma-4-31B-it-NVFP4-turbo             (needs NVFP4 vLLM)
MODELS["vast-qwen3-6-35b-a3b"] = {
    "name": "Qwen 3.6 35B-A3B (Vast vLLM)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_QWEN36_35B_URL",
    "api_model_slug": "Qwen/Qwen3.6-35B-A3B",
    "api_key_env": "VAST_API_TOKEN",  # optional — vLLM accepts any token
    "context": 32768,
    "tok_per_sec": 30,
    "description": "Qwen 3.6 35B-A3B base via vLLM on Vast. Set VAST_QWEN36_35B_URL in .env.",
}
# --- Fleet slots (same model, different Vast hosts) ---
# One entry per rented GPU. Each has its own env var + tunnel port.
# Run parallel shards: shard 3 on 8765, shard 4 on 8766, etc.
MODELS["vast-qwen3-6-35b-a3b-4090-tx"] = {
    "name": "Qwen 3.6 35B-A3B (Vast 4090-TX)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_QWEN36_35B_4090_TX_URL",
    "api_model_slug": "Qwen/Qwen3.6-35B-A3B",
    "api_key_env": "VAST_API_TOKEN",
    "context": 32768,
    "tok_per_sec": 120,
    "description": "Qwen 3.6 35B on Vast 4090-TX (30224). Tunnel 8766.",
}
MODELS["vast-qwen3-6-35b-a3b-3090-fi"] = {
    "name": "Qwen 3.6 35B-A3B (Vast 3090-FI)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_QWEN36_35B_3090_FI_URL",
    "api_model_slug": "Qwen/Qwen3.6-35B-A3B",
    "api_key_env": "VAST_API_TOKEN",
    "context": 32768,
    "tok_per_sec": 120,
    "description": "Qwen 3.6 35B on Vast 3090-FI (13194). Tunnel 8767.",
}
MODELS["vast-qwen3-6-35b-a3b-3090-ky"] = {
    "name": "Qwen 3.6 35B-A3B (Vast 3090-KY)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_QWEN36_35B_3090_KY_URL",
    "api_model_slug": "Qwen/Qwen3.6-35B-A3B",
    "api_key_env": "VAST_API_TOKEN",
    "context": 32768,
    "tok_per_sec": 120,
    "description": "Qwen 3.6 35B on Vast 3090-KY (15556). Tunnel 8768.",
}
MODELS["vast-qwen3-6-35b-a3b-4090-mi2"] = {
    "name": "Qwen 3.6 35B-A3B (Vast 4090-MI2)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_QWEN36_35B_4090_MI2_URL",
    "api_model_slug": "Qwen/Qwen3.6-35B-A3B",
    "api_key_env": "VAST_API_TOKEN",
    "context": 32768,
    "tok_per_sec": 120,
    "description": "Qwen 3.6 35B on Vast 4090-MI2 (33930). Tunnel 8769.",
}

MODELS["vast-qwen3-6-35b-a3b-uncensored"] = {
    "name": "Qwen 3.6 35B-A3B Uncensored Aggressive (Vast vLLM)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_QWEN36_35B_UNC_URL",
    "api_model_slug": "HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
    "api_key_env": "VAST_API_TOKEN",
    "context": 32768,
    "tok_per_sec": 30,
    "description": "Uncensored Qwen 3.6 35B variant via vLLM on Vast.",
}
MODELS["vast-gemma4-31b-nvfp4-turbo"] = {
    "name": "Gemma 4 31B NVFP4 Turbo (Vast vLLM)",
    "backend": "remote_api",
    "api_base_url_env": "VAST_GEMMA4_31B_URL",
    "api_model_slug": "LilaRest/gemma-4-31B-it-NVFP4-turbo",
    "api_key_env": "VAST_API_TOKEN",
    "context": 32768,
    "tok_per_sec": 55,
    "description": "Gemma 4 31B in NVFP4 via vLLM — needs vLLM 0.6+ with NVFP4 kernels.",
}

# --- xAI (Grok) — OpenAI-compatible API at api.x.ai/v1 ---
# Auth via XAI_API_KEY env. Grok-4 is the flagship; grok-4-fast is cheaper
# + faster; grok-3-mini is the tiny / cheapest for breadth testing.
XAI_BASE_URL = "https://api.x.ai/v1"

MODELS["xai-grok-4"] = {
    "name": "Grok 4 (xAI)",
    "backend": "remote_api",
    "api_base_url": XAI_BASE_URL,
    "api_model_slug": "grok-4",
    "api_key_env": "XAI_API_KEY",
    "context": 131072,
    "tok_per_sec": 50,
    "description": "xAI Grok 4 via api.x.ai (flagship). Paid pay-as-you-go.",
}
MODELS["xai-grok-4-fast"] = {
    "name": "Grok 4 Fast (xAI)",
    "backend": "remote_api",
    "api_base_url": XAI_BASE_URL,
    "api_model_slug": "grok-4-fast",
    "api_key_env": "XAI_API_KEY",
    "context": 131072,
    "tok_per_sec": 120,
    "description": "xAI Grok 4 Fast — cheaper + faster variant.",
}
MODELS["xai-grok-3-mini"] = {
    "name": "Grok 3 Mini (xAI)",
    "backend": "remote_api",
    "api_base_url": XAI_BASE_URL,
    "api_model_slug": "grok-3-mini",
    "api_key_env": "XAI_API_KEY",
    "context": 131072,
    "tok_per_sec": 200,
    "description": "xAI Grok 3 Mini — smallest/cheapest; breadth testing.",
}

# Ollama daemon defaults
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_BASE = "http://localhost:11434/api"


def is_ollama_running() -> bool:
    """Check if the Ollama daemon is reachable."""
    import urllib.request

    try:
        urllib.request.urlopen(f"{OLLAMA_API_BASE}/version", timeout=2)
        return True
    except Exception:
        return False


def ollama_has_model(tag: str) -> bool:
    """Check whether an Ollama-backed model has been pulled."""
    import urllib.request

    try:
        resp = urllib.request.urlopen(f"{OLLAMA_API_BASE}/tags", timeout=5)
        data = json.loads(resp.read())
        return any(
            m.get("name", "").startswith(tag.split(":")[0]) and m.get("name") == tag for m in data.get("models", [])
        )
    except Exception:
        return False


MODELS_DIR = Path.home() / "models" / "gguf"
LLAMA_CPP_DIR = Path.home() / "llama.cpp"
PID_FILE = Path("/tmp/llama_server.pid")
LOG_FILE = Path("/tmp/llama_server.log")
DEFAULT_PORT = 8080


def list_models():
    """List available models with VRAM requirements."""
    print(f"{'ID':<16} {'Name':<32} {'Size':>6} {'VRAM':>6} {'Speed':>8} {'Fits 3050Ti?':<12}")
    print("-" * 90)
    for mid, m in MODELS.items():
        vram_gb = m.get("vram_gb")
        size_gb = m.get("size_gb")
        if vram_gb is None:
            fits_str = "remote"
            vram_str = "n/a"
        else:
            fits = "YES" if vram_gb <= 4.0 else "TIGHT" if vram_gb <= 4.5 else "NO"
            fits_str = (
                f"\033[32m{fits}\033[0m"
                if fits == "YES"
                else f"\033[33m{fits}\033[0m"
                if fits == "TIGHT"
                else f"\033[31m{fits}\033[0m"
            )
            vram_str = f"{vram_gb:>5.1f}G"
        size_str = f"{size_gb:>5.1f}G" if size_gb is not None else "  n/a "
        fork = " [fork]" if m.get("requires_fork") else ""
        print(
            f"{mid:<16} {m['name']:<32} {size_str:>6} {vram_str:>6} {m.get('tok_per_sec', 'n/a'):>5}t/s {fits_str}{fork}"
        )
    print(f"\nModels dir: {MODELS_DIR}")
    print(f"llama.cpp:  {LLAMA_CPP_DIR}")


def download_model(model_id: str):
    """Download a GGUF model from HuggingFace."""
    if model_id not in MODELS:
        print(f"[!] Unknown model: {model_id}")
        print(f"    Available: {', '.join(MODELS.keys())}")
        return

    m = MODELS[model_id]
    if "file" not in m or "repo" not in m:
        print(f"[!] {model_id} is a remote/API model, not a downloadable GGUF.")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / m["file"]

    if target.exists():
        print(f"[+] Already downloaded: {target}")
        return

    print(f"[*] Downloading {m['name']} ({m['size_gb']:.1f} GB)...")
    print(f"    Repo: {m['repo']}")
    print(f"    File: {m['file']}")

    cmd = [
        "hf",
        "download",
        m["repo"],
        m["file"],
        "--local-dir",
        str(MODELS_DIR),
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[+] Downloaded to {target}")
    except FileNotFoundError:
        print("[!] huggingface-cli not found. Install with: pip install huggingface-hub")
        print(f"    Or manually download from: https://huggingface.co/{m['repo']}/resolve/main/{m['file']}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Download failed: {e}")


def get_gpu_vram_mb() -> int:
    """Get available GPU VRAM in MB."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return int(out.stdout.strip().split()[0])
    except Exception:
        return 0


def estimate_gpu_layers(model_size_gb: float, vram_mb: int) -> int:
    """Estimate how many layers can fit on GPU.

    Rough heuristic: each layer is ~model_size/num_layers GB.
    Leave ~500MB overhead for KV cache at short context.
    Usable VRAM = total - overhead.
    """
    if vram_mb <= 0:
        return 0
    usable_gb = (vram_mb - 500) / 1024
    if model_size_gb <= usable_gb:
        return 999  # Everything fits
    # Partial offload: use fraction of VRAM
    fraction = usable_gb / model_size_gb
    # Typical models have 32-42 layers
    return max(1, int(fraction * 32))


def serve_model(model_id: str, port: int = DEFAULT_PORT, context: int = 0, ngl: int = 0):
    """Start llama.cpp server with the specified model."""
    # Stop existing server first
    stop_server(silent=True)
    time.sleep(1)

    if model_id not in MODELS:
        print(f"[!] Unknown model: {model_id}")
        return

    m = MODELS[model_id]
    if "file" not in m or "size_gb" not in m:
        print(f"[!] {model_id} is a remote/API model; serve it through its provider, not llama.cpp.")
        return

    # Bonsai models use custom fork
    if m.get("requires_fork"):
        server_bin = Path.home() / "llama.cpp-bonsai" / "build" / "bin" / "llama-server"
        model_path = Path.home() / "models" / "gguf" / "bonsai" / m["file"]
    else:
        server_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-server"
        model_path = MODELS_DIR / m["file"]

    if not model_path.exists():
        print(f"[!] Model not downloaded: {model_path}")
        print(f"    Run: python -m hermes_katana.proving_ground.local_models download {model_id}")
        return

    if not server_bin.exists():
        print(f"[!] llama-server not found at {server_bin}")
        if m.get("requires_fork"):
            print(
                "    Build Bonsai fork: cd ~/llama.cpp-bonsai && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)"
            )
        else:
            print(
                "    Build llama.cpp: cd ~/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)"
            )
        return

    ctx = context or m.get("context", 4096)

    # Calculate GPU layers
    vram_mb = get_gpu_vram_mb()
    if ngl > 0:
        gpu_layers = ngl
    else:
        gpu_layers = estimate_gpu_layers(m["size_gb"], vram_mb)

    # Pick chat template based on model family. Gemma GGUFs carry an embedded
    # template; overriding it with llama.cpp's legacy "gemma" template causes
    # garbage output on Gemma-4.
    if "gemma" in model_id.lower():
        template = None
    elif "qwen" in model_id.lower():
        template = "chatml"
    elif "nemotron" in model_id.lower():
        template = "chatml"
    elif "tinyllama" in model_id.lower():
        template = "chatml"
    elif "bonsai" in model_id.lower():
        template = "chatml"
    else:
        template = "chatml"

    cmd = [
        str(server_bin),
        "-m",
        str(model_path),
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "-c",
        str(ctx),
        "-ngl",
        str(gpu_layers),
        "-fa",
        "on",
    ]
    if template:
        cmd.extend(["--chat-template", template])

    mode = "full GPU" if gpu_layers >= 999 else f"{gpu_layers} layers on GPU, rest on CPU"

    print("[*] Starting llama.cpp server")
    print(f"    Model: {m['name']} ({m['size_gb']:.1f} GB)")
    print(f"    Port: {port}")
    print(f"    Context: {ctx}")
    print(f"    VRAM: {vram_mb} MB | Offload: {mode}")
    print(f"    Binary: {server_bin}")
    print(f"    Command: {' '.join(cmd)}")

    log = open(LOG_FILE, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    # Save PID
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")

    # Wait a moment and check if it started
    time.sleep(3)
    if proc.poll() is not None:
        print(f"[!] Server crashed. Check log: {LOG_FILE}")
        log_out = LOG_FILE.read_text(encoding="utf-8")[-2000:] if LOG_FILE.exists() else ""
        print(log_out)
        return

    print(f"[+] Server started (PID: {proc.pid})")
    print(f"    Endpoint: http://localhost:{port}/v1")
    print(f"    Log: {LOG_FILE}")
    print("\n    Test with:")
    print(f"    curl http://localhost:{port}/v1/chat/completions \\")
    print("      -H 'Content-Type: application/json' \\")
    print(
        f'      -d \'{{"model": "{model_id}", "messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 50}}\''
    )


def stop_server(silent: bool = False):
    """Stop the running llama.cpp server."""
    if not PID_FILE.exists():
        if not silent:
            print("[*] No server running (no PID file)")
        return

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 15)  # SIGTERM
        if not silent:
            print(f"[+] Stopped server (PID: {pid})")
    except (ProcessLookupError, ValueError):
        if not silent:
            print("[*] Process already gone")
    except Exception as e:
        if not silent:
            print(f"[!] Error stopping: {e}")

    PID_FILE.unlink(missing_ok=True)


def server_status():
    """Check if server is running."""
    if not PID_FILE.exists():
        print("Server: NOT RUNNING")
        return False

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)  # Check if process exists
        print(f"Server: RUNNING (PID: {pid})")

        # Try to hit the endpoint
        import urllib.request

        try:
            resp = urllib.request.urlopen(f"http://localhost:{DEFAULT_PORT}/v1/models", timeout=2)
            data = json.loads(resp.read())
            print(f"Endpoint: http://localhost:{DEFAULT_PORT}/v1")
            print(f"Models: {json.dumps(data, indent=2)}")
        except Exception:
            print("Endpoint: not responding yet")

        return True
    except (ProcessLookupError, ValueError):
        print("Server: DEAD (stale PID)")
        PID_FILE.unlink(missing_ok=True)
        return False


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "list":
        list_models()
    elif cmd == "download":
        if len(sys.argv) < 3:
            print("Usage: python -m hermes_katana.proving_ground.local_models download <model_id>")
            list_models()
        else:
            download_model(sys.argv[2])
    elif cmd == "serve":
        if len(sys.argv) < 3:
            print("Usage: python -m hermes_katana.proving_ground.local_models serve <model_id> [port] [context]")
        else:
            model_id = sys.argv[2]
            port = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_PORT
            ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
            serve_model(model_id, port, ctx)
    elif cmd == "stop":
        stop_server()
    elif cmd == "status":
        server_status()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
