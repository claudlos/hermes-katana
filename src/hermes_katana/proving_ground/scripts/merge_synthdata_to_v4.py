"""Merge synthdata/checkpoints/<run>/synthdata_final.jsonl into
hermes-katana/training/data_v4/.

data_v4/ = data_v3/combined.jsonl + synthdata rows, with the v3 schema
applied (source, source_family, is_attack, text_sha256, split, …).

Split assignment uses a deterministic hash of text_sha256 so the same
attack always lands in the same split across runs. No data leakage
between train/val/test.

Usage:
    python scripts/merge_synthdata_to_v4.py \
        --synth synthdata/checkpoints/katana_synth_v1/synthdata_final.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path


KATANA_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(Path(__file__).resolve().parents[2] / "hermes-katana"),
    )
)
DEFAULT_V3 = KATANA_ROOT / "training" / "data_v3" / "combined.jsonl"
DEFAULT_V4_DIR = KATANA_ROOT / "training" / "data_v4"


ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿‪-‮]")
WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = ZERO_WIDTH_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip().lower()
    return s


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _split_for(text_sha: str, *, train_pct: float, val_pct: float) -> str:
    """Deterministic 3-way split from the sha."""
    bucket = int(text_sha[:8], 16) % 1000 / 1000.0
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def _row_from_synth(synth: dict, *, train_pct: float, val_pct: float) -> dict:
    text = synth["text"]
    label = synth["label"]
    norm = _normalize(text)
    raw_sha = _sha(text)
    norm_sha = _sha(norm)
    # family = taxonomy leaf_id if present, else norm_sha (self-family)
    family_sha = _sha(synth.get("leaf_id") or norm_sha)
    split = _split_for(raw_sha, train_pct=train_pct, val_pct=val_pct)
    return {
        "text": text,
        "label": label,
        "source": "synthdata_v1_simula",
        "source_family": f"synth_leaf_{synth.get('leaf_id', '')}",
        "is_attack": label != "clean",
        "binary_label": "attack" if label != "clean" else "clean",
        "text_sha256_normalized": norm_sha,
        "text_sha256": raw_sha,
        "family_sha256": family_sha,
        "text_length": len(text),
        "origin": synth.get("origin", "user_input")
        if synth.get("origin", "").startswith(
            (
                "user_input",
                "retrieved_web",
                "mcp_tool_description",
                "mcp_tool_result",
                "prior_session_memory",
                "delegated_agent_output",
            )
        )
        else "user_input",
        "split": split,
        "id": f"synth_{raw_sha[:16]}",
        "quality_tier": "simula_dual_critic",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=Path, required=True, help="path to synthdata_final.jsonl")
    ap.add_argument("--v3", type=Path, default=DEFAULT_V3)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_V4_DIR)
    ap.add_argument("--train-pct", type=float, default=0.80)
    ap.add_argument("--val-pct", type=float, default=0.10)
    ap.add_argument(
        "--dedup",
        action="store_true",
        default=True,
        help="drop synth rows whose normalized text already exists in v3",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load v3 and index by normalized sha for dedup
    v3_rows: list[dict] = []
    seen_norm: set[str] = set()
    with args.v3.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            v3_rows.append(r)
            seen_norm.add(r.get("text_sha256_normalized", ""))

    # Load synth, convert, dedup
    synth_converted: list[dict] = []
    dropped_dup = 0
    with args.synth.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            row = _row_from_synth(s, train_pct=args.train_pct, val_pct=args.val_pct)
            if args.dedup and row["text_sha256_normalized"] in seen_norm:
                dropped_dup += 1
                continue
            seen_norm.add(row["text_sha256_normalized"])
            synth_converted.append(row)

    # Write combined
    out = args.out_dir / "combined.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in v3_rows:
            f.write(json.dumps(r) + "\n")
        for r in synth_converted:
            f.write(json.dumps(r) + "\n")

    # Report
    from collections import Counter

    n_v3 = len(v3_rows)
    n_add = len(synth_converted)
    by_split = Counter(r["split"] for r in synth_converted)
    by_label = Counter(r["label"] for r in synth_converted)
    print(f"v3 rows         : {n_v3:,}")
    print(f"synth added     : {n_add:,}  ({dropped_dup} dedup-dropped)")
    print(f"total v4 rows   : {n_v3 + n_add:,}")
    print(f"synth by split  : {dict(by_split)}")
    print("synth by label  :")
    for lbl, n in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<28} {n:>6}")
    print(f"→ {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
