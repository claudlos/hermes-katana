"""Build a focused, lean training corpus for katana_v10 from scratch.

Inputs:
  - results/confirmed_attacks.jsonl
        Empirically-validated attacks (n_models_effective >= 3 each).
  - synthdata/checkpoints/katana_synth_v{1,2,3...}/synthdata_final.jsonl
        LLM-generated, dual-critic-passed synthetic attacks.
  - hermes-katana/training/data_v3/combined.jsonl
        Reused only for `clean` rows: awesome_prompts + deepset (high
        quality real prompts/benchmark) + a sampled top-up from
        synthetic_benign for class-magnitude balance.

This replaces the prior approach of stacking synth on top of v3's
63,924-row legacy mix. The legacy v3 corpus has a lot of historical
noise (multilingual translation augmentations, homoglyph fuzz from old
runs, many sources of dubious provenance) that drag training quality.
Focused v4 keeps only known-good rows.

Output:
  hermes-katana/training/data_v4/combined.jsonl  (overwritten)

Schema matches v3 row shape so katana_v10.yaml works unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


PG_ROOT = Path(
    os.environ.get(
        "KATANA_PROVING_GROUND_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
)
KATANA_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(PG_ROOT.parent / "hermes-katana"),
    )
)
DEFAULT_V3 = KATANA_ROOT / "training" / "data_v3" / "combined.jsonl"
DEFAULT_OUT = KATANA_ROOT / "training" / "data_v4" / "combined.jsonl"
DEFAULT_SYNTH_FINALS = [
    PG_ROOT / "synthdata" / "checkpoints" / "katana_synth_v1" / "synthdata_final.jsonl",
    PG_ROOT / "synthdata" / "checkpoints" / "katana_synth_v2_opus_elite" / "synthdata_final.jsonl",
    PG_ROOT / "synthdata" / "checkpoints" / "katana_synth_v3_gap6" / "synthdata_final.jsonl",
]
DEFAULT_CONFIRMED = PG_ROOT / "results" / "confirmed_attacks.jsonl"

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
    bucket = int(text_sha[:8], 16) % 1000 / 1000.0
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def _row(
    *,
    text: str,
    label: str,
    source: str,
    train_pct: float,
    val_pct: float,
    origin: str = "user_input",
    quality_tier: str = "default",
) -> dict:
    norm = _normalize(text)
    raw_sha = _sha(text)
    norm_sha = _sha(norm)
    split = _split_for(raw_sha, train_pct=train_pct, val_pct=val_pct)
    is_attack = label != "clean"
    return {
        "text": text,
        "label": label,
        "source": source,
        "source_family": source,
        "is_attack": is_attack,
        "binary_label": "attack" if is_attack else "clean",
        "text_sha256_normalized": norm_sha,
        "text_sha256": raw_sha,
        "family_sha256": norm_sha,
        "text_length": len(text),
        "origin": origin,
        "split": split,
        "id": f"{label[:6]}_{raw_sha[:16]}",
        "quality_tier": quality_tier,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--confirmed",
        type=Path,
        default=DEFAULT_CONFIRMED,
        help="confirmed_attacks.jsonl path",
    )
    ap.add_argument(
        "--synth",
        type=Path,
        nargs="+",
        default=DEFAULT_SYNTH_FINALS,
        help="one or more synthdata_final.jsonl paths",
    )
    ap.add_argument(
        "--v3-clean-source",
        type=Path,
        default=DEFAULT_V3,
        help="v3 combined.jsonl — used only as a clean-row source",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--clean-target",
        type=int,
        default=4500,
        help="target number of clean rows to include",
    )
    ap.add_argument("--train-pct", type=float, default=0.80)
    ap.add_argument("--val-pct", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_rows: list[dict] = []
    seen_norm: set[str] = set()
    by_source: Counter = Counter()
    by_label: Counter = Counter()

    def _add(row: dict) -> bool:
        if row["text_sha256_normalized"] in seen_norm:
            return False
        seen_norm.add(row["text_sha256_normalized"])
        out_rows.append(row)
        by_source[row["source"]] += 1
        by_label[row["label"]] += 1
        return True

    # 1) Confirmed attacks (every row, no sampling — these are the spine)
    if args.confirmed.exists():
        with args.confirmed.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = (r.get("text") or "").strip()
                label = r.get("label") or ""
                if not text or label == "clean" or len(text) < 8:
                    continue
                _add(
                    _row(
                        text=text,
                        label=label,
                        source="confirmed_attacks",
                        train_pct=args.train_pct,
                        val_pct=args.val_pct,
                        quality_tier=f"confirmed_n{r.get('n_models_effective', 0)}",
                    )
                )

    # 2) Synthetic attacks (every kept row, no sampling)
    for synth_path in args.synth:
        if not synth_path.exists():
            print(f"[skip] synth not found: {synth_path}")
            continue
        with synth_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = (r.get("text") or "").strip()
                label = r.get("label") or ""
                if not text or label == "clean" or len(text) < 8:
                    continue
                _add(
                    _row(
                        text=text,
                        label=label,
                        source=f"synthdata_{r.get('origin', 'v?')}",
                        train_pct=args.train_pct,
                        val_pct=args.val_pct,
                        origin=r.get("origin", "user_input"),
                        quality_tier="simula_dual_critic",
                    )
                )

    # 3) Clean rows from v3 — quality-tiered:
    #    awesome_prompts and deepset are real human/benchmark prompts (priority);
    #    synthetic_benign / synthetic_benign_extra / benign are formulaic but useful filler.
    QUALITY_ORDER = (
        "awesome_prompts",
        "deepset",
        "synthetic_benign_extra",
        "synthetic_benign",
        "benign",
    )
    pools: dict[str, list[dict]] = {s: [] for s in QUALITY_ORDER}
    if args.v3_clean_source.exists():
        with args.v3_clean_source.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("label") != "clean":
                    continue
                src = r.get("source", "")
                if src in pools:
                    pools[src].append(r)
        for s in QUALITY_ORDER:
            rng.shuffle(pools[s])

    clean_added = 0
    for s in QUALITY_ORDER:
        if clean_added >= args.clean_target:
            break
        for r in pools[s]:
            if clean_added >= args.clean_target:
                break
            text = (r.get("text") or "").strip()
            if not text or len(text) < 8:
                continue
            ok = _add(
                _row(
                    text=text,
                    label="clean",
                    source=f"v3_clean:{s}",
                    train_pct=args.train_pct,
                    val_pct=args.val_pct,
                    origin="user_input",
                    quality_tier=f"clean_{s}",
                )
            )
            if ok:
                clean_added += 1

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    # Report
    print(f"\nwrote {len(out_rows)} rows → {args.out}")
    print("\nby label:")
    for lbl in [
        "clean",
        "content_injection",
        "semantic_manipulation",
        "behavioral_control",
        "exfiltration_attempt",
        "jailbreak",
        "cognitive_state_attack",
        "encoding_evasion",
        "persona_jailbreak",
    ]:
        print(f"  {lbl:<25} {by_label[lbl]:>5}")
    print("\nby source:")
    for s, n in sorted(by_source.items(), key=lambda x: -x[1])[:15]:
        print(f"  {s:<35} {n:>5}")
    splits = Counter(r["split"] for r in out_rows)
    print(f"\nby split: train={splits['train']}  val={splits['val']}  test={splits['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
