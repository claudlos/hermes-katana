"""Known false-positive allowlist for the Scabbard classifier.

Each entry is a (text_hash_prefix, top_category) pair for benign content
that the v17 origin-aware MiniLM mis-classifies as an attack. Full text
is NOT stored, only the SHA-256 prefix, so this file does not become a
target list for crafting bypasses.

To add a new entry: re-run tests/smoke/false_positive_gate.py, parse
the FP lines, append a (text_hash, top_category) pair.

Lookup is O(1) over a frozenset of (hash, category) tuples. The set is
loaded once at import time from the path resolved by
_default_known_fps_path() (override with KATANA_KNOWN_FPS_PATH).
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Tuple


_ENV_PATH = "KATANA_KNOWN_FPS_PATH"
_DEFAULT_RELATIVE = Path("policies") / "scabbard_known_fps.yaml"


def _default_known_fps_path() -> Path:
    override = os.environ.get(_ENV_PATH)
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    # search ancestors: src/hermes_katana/scabbard -> src/hermes_katana -> src -> repo root
    for ancestor in (here.parent.parent.parent.parent, here.parent.parent.parent):
        candidate = ancestor / _DEFAULT_RELATIVE
        if candidate.is_file():
            return candidate
    return here.parent.parent.parent / _DEFAULT_RELATIVE


@lru_cache(maxsize=1)
def _load_known_fps() -> FrozenSet[Tuple[str, str]]:
    path = _default_known_fps_path()
    if not path.is_file():
        return frozenset()
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return frozenset()
    entries = data.get("entries", []) or []
    out = set()
    for e in entries:
        h = e.get("text_hash")
        c = e.get("top_category")
        if h and c:
            out.add((h, c))
    return frozenset(out)


def text_hash(text: str) -> str:
    """Return the 16-char SHA-256 prefix used as the allowlist key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def is_known_fp(text: str, top_category: str) -> bool:
    """Return True if (text, top_category) is in the known-FP allowlist.

    The lookup is hash-based, so it does not match rephrasings. Callers
    should pass the exact post-normalization text that Scabbard classified.
    """
    if not text or not top_category:
        return False
    return (text_hash(text), top_category) in _load_known_fps()


def reload_known_fps() -> int:
    """Force re-read of the allowlist file. Returns the new entry count.

    Useful for tests; not for hot-reload in production (use a SIGHUP handler).
    """
    _load_known_fps.cache_clear()
    return len(_load_known_fps())


__all__ = ["is_known_fp", "reload_known_fps", "text_hash"]
