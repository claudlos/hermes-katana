#!/usr/bin/env python3
"""Download the torch-free ONNX sentence-embedder used by the Scabbard
cosine-similarity false-positive softener.

The softener (``hermes_katana.scabbard.similarity_allowlist``) needs a small
general-purpose sentence encoder that runs under ``onnxruntime`` -- NOT the
Scabbard zvec embedder (which clusters attacks and is unsafe as an allowlist
signal), and NOT a torch model (the production gateway venv is torch-free).

This installs ``Xenova/all-MiniLM-L6-v2`` (ONNX export, ~88 MB) into the Scabbard
artifact cache. If the artifact is absent the softener simply fails closed
(no softening), so this step is optional but recommended on any deployment that
wants false-positive relief on benign security-domain content.

Usage::

    python scripts/setup_similarity_embedder.py
    # or pin a location:
    KATANA_SIM_EMBEDDER_DIR=/opt/models/minilm-onnx python scripts/setup_similarity_embedder.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = "Xenova/all-MiniLM-L6-v2"
_FILES = [
    "onnx/model.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def _target_dir() -> Path:
    override = os.environ.get("KATANA_SIM_EMBEDDER_DIR")
    if override:
        return Path(override).expanduser()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hermes_katana.artifacts import default_artifact_cache_dir

    return default_artifact_cache_dir() / "onnx_embedder_allMiniLM"


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        return 1

    dst = _target_dir()
    dst.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {_REPO} (ONNX) -> {dst}")
    for fname in _FILES:
        try:
            path = hf_hub_download(repo_id=_REPO, filename=fname, local_dir=str(dst))
            print(f"  ok  {fname} ({os.path.getsize(path) // 1024} KB)")
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {fname}: {type(exc).__name__}: {exc}", file=sys.stderr)

    model = dst / "onnx" / "model.onnx"
    if not model.is_file():
        print(f"ERROR: {model} not present after download", file=sys.stderr)
        return 1
    print(f"Done. Similarity softener will use: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
