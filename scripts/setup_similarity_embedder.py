#!/usr/bin/env python3
"""Download the ONNX sentence embedder used by the Scabbard similarity softener.

Usage::

    python scripts/setup_similarity_embedder.py
    # or pin a location:
    KATANA_SIM_EMBEDDER_DIR=/opt/models/minilm-onnx python scripts/setup_similarity_embedder.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _skip_setup() -> bool:
    return str(os.environ.get("HERMES_KATANA_SKIP_SETUP") or "").strip().lower() in {"1", "true", "yes", "on"}


def _main() -> int:
    if _skip_setup():
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hermes_katana.scabbard.similarity_embedder_setup import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_main())
