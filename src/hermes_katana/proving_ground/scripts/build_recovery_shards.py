#!/usr/bin/env python3
"""Build proving-ground shards from the recovered backup corpus.

The HermesKatana 2026-04-17 handoff describes 64,608 rows of recovered
English attack text in `training/data_v3/backup_recovery_normalized.jsonl`.
After dropping label=clean rows and deduping (within-recovered + against
existing shards 1-122 / 200-222), ~21,800 net-new attacks remain. This
script writes them as new shards in the 400+ range so the fleet can
test them without colliding with anything already on disk.

Output:
    shards/shard_400.jsonl   ... shards/shard_NNN.jsonl
    shards/recovery_manifest.json   (provenance + per-label counts + dedup stats)

Run:
    python scripts/build_recovery_shards.py
    python scripts/build_recovery_shards.py --first-shard 400 --per-shard 177
    python scripts/build_recovery_shards.py --dry-run        # don't write anything

After it completes, point a fleet at the new shard range:

    python scripts/fleet_preflight.py --spec scripts/fleet_recovery_qwen.json
    python scripts/fleet.py launch    --spec scripts/fleet_recovery_qwen.json

Idempotent: re-running on an unchanged source produces identical shard
files (deterministic shuffle, seed=42) and refuses to overwrite existing
shard files unless `--overwrite` is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KATANA_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(ROOT.parent / "hermes-katana"),
    )
)

DEFAULT_SOURCE = KATANA_ROOT / "training" / "data_v3" / "backup_recovery_normalized.jsonl"
DEFAULT_OUT_DIR = ROOT / "shards"
DEFAULT_FIRST_SHARD = 400  # leaves 1-122 (empirical), 200-222 (synth),
# 300-321 (multilingual) untouched
DEFAULT_PER_SHARD = 177  # match existing granularity
DEFAULT_SEED = 42  # match existing manifest seed

ZW_RE = re.compile(r"[​-‏‪-‮⁠﻿⁦-⁩]")
WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = ZW_RE.sub("", s)
    return WS_RE.sub(" ", s).strip().lower()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return None


def rows_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Existing-shard fingerprint loader
# ---------------------------------------------------------------------------


def load_existing_fingerprints(shards_dir: Path) -> set[str]:
    """Collect normalized-text shas from every existing shard so we can
    dedupe new rows against them. Cheap (~22K rows in ~50 files)."""
    seen = set()
    for sp in sorted(shards_dir.glob("shard_*.jsonl")):
        try:
            with sp.open(encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    ns = d.get("text_sha256_normalized") or sha(normalize(d.get("text", "")))
                    seen.add(ns)
        except Exception as e:
            print(f"  warn: could not read {sp.name}: {e}", file=sys.stderr)
    return seen


# ---------------------------------------------------------------------------
# Row schema — match existing shards
# ---------------------------------------------------------------------------


def _quality_tier(rec_quality: str) -> str:
    if rec_quality == "high":
        return "high_confidence_attack"
    if rec_quality == "medium":
        return "medium_high_confidence_attack"
    return "recovered_unconfirmed"


def make_row(
    *,
    text: str,
    label: str,
    source: str,
    quality: str,
    backup_source: str,
    shard_id: int,
) -> dict:
    raw_sha = sha(text)
    norm_sha = sha(normalize(text))
    return {
        "binary_label": "attack",
        "family_sha256": norm_sha,
        "id": f"atk_{raw_sha[:16]}",
        "is_attack": True,
        "label": label,
        "origin": "user_input",
        "quality_tier": _quality_tier(quality),
        "shard": shard_id,
        "source": source or "wild_corpus_recovered",
        "source_family": source or "wild_corpus_recovered",
        "split": "train",
        "text": text,
        "text_length": len(text),
        "text_sha256": raw_sha,
        "text_sha256_normalized": norm_sha,
        # New provenance fields specific to recovery shards
        "recovery_origin": backup_source,  # canonical_en / wild_clean / synthetic_clean
        "shard_origin": "backup_recovery_2026-04-17",
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    source: Path,
    out_dir: Path,
    first_shard: int,
    per_shard: int,
    seed: int,
    dry_run: bool,
    overwrite: bool,
) -> int:
    if not source.exists():
        print(f"ERROR: source not found: {source}", file=sys.stderr)
        return 1

    print(f"Reading: {source}")
    print(f"Writing: {out_dir}/shard_{first_shard:03d}.jsonl ...")
    print()

    print("[1] Indexing existing shard fingerprints ...")
    existing = load_existing_fingerprints(out_dir)
    print(f"     {len(existing):,} normalized hashes loaded from existing shards")

    print("[2] Reading + deduping recovered rows ...")
    n_total = 0
    n_clean = 0
    n_short = 0
    n_dup_existing = 0
    n_dup_within = 0
    seen_within = set()
    kept: list[dict] = []
    label_in = Counter()

    with source.open(encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except Exception:
                continue
            text = d.get("text", "")
            label = d.get("label", "")
            label_in[label] += 1
            if not text or len(text) < 8:
                n_short += 1
                continue
            if label == "clean" or not d.get("is_attack", True):
                n_clean += 1
                continue
            ns = sha(normalize(text))
            if ns in existing:
                n_dup_existing += 1
                continue
            if ns in seen_within:
                n_dup_within += 1
                continue
            seen_within.add(ns)
            kept.append(
                {
                    "text": text,
                    "label": label,
                    "source": d.get("source", "wild_corpus_recovered"),
                    "quality": d.get("quality", "unknown"),
                    "backup_source": d.get("_backup_source", "unknown"),
                }
            )

    print(f"     read:                 {n_total:,}")
    print(f"     skipped short:        {n_short:,}")
    print(f"     skipped clean:        {n_clean:,}")
    print(f"     dup vs existing:      {n_dup_existing:,}")
    print(f"     dup within recovered: {n_dup_within:,}")
    print(f"     KEPT:                 {len(kept):,}")

    print("\n[3] Deterministic shuffle (seed=%d) ..." % seed)
    rng = random.Random(seed)
    rng.shuffle(kept)

    # Pre-flight collision check on the target shard range.
    n_shards = -(-len(kept) // per_shard)
    target_paths = [out_dir / f"shard_{first_shard + i:03d}.jsonl" for i in range(n_shards)]
    existing_targets = [p for p in target_paths if p.exists()]
    if existing_targets and not overwrite:
        print(
            f"\nERROR: {len(existing_targets)} target shard files already exist "
            f"(first: {existing_targets[0].name}). "
            f"Pass --overwrite to replace, or --first-shard to pick a different "
            f"id range.",
            file=sys.stderr,
        )
        return 2

    print(f"[4] Writing {n_shards} shard files (~{per_shard} attacks each) ...")

    label_out = Counter()
    by_quality = Counter()
    written_paths = []
    written_entries = []
    all_output_rows: list[dict] = []
    for i in range(n_shards):
        shard_id = first_shard + i
        chunk = kept[i * per_shard : (i + 1) * per_shard]
        if not chunk:
            break
        rows = [
            make_row(
                text=r["text"],
                label=r["label"],
                source=r["source"],
                quality=r["quality"],
                backup_source=r["backup_source"],
                shard_id=shard_id,
            )
            for r in chunk
        ]
        for row in rows:
            label_out[row["label"]] += 1
            by_quality[row["quality_tier"]] += 1
        all_output_rows.extend(rows)

        if not dry_run:
            sp = out_dir / f"shard_{shard_id:03d}.jsonl"
            with sp.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written_paths.append(sp.name)
            written_entries.append(
                {
                    "file": sp.name,
                    "size_bytes": sp.stat().st_size,
                    "sha256": file_sha256(sp),
                    "n_rows": len(rows),
                }
            )

    print(f"     wrote: {len(written_paths)} shard files")
    if dry_run:
        print("     (dry-run — nothing actually written)")

    # Manifest
    manifest = {
        "version": 1,
        "kind": "recovery_unconfirmed",
        "source": str(source),
        "first_shard": first_shard,
        "n_shards": n_shards,
        "per_shard_target": per_shard,
        "total_attacks": sum(label_out.values()),
        "by_label": dict(label_out.most_common()),
        "by_quality_tier": dict(by_quality.most_common()),
        "input_label_distribution": dict(label_in.most_common()),
        "dedup": {
            "read_total": n_total,
            "skipped_short": n_short,
            "skipped_clean": n_clean,
            "dup_vs_existing_shards": n_dup_existing,
            "dup_within_recovered": n_dup_within,
            "kept": len(kept),
        },
        "seed": seed,
        "files": written_paths,
        "provenance": {
            "builder": "scripts/build_recovery_shards.py",
            "builder_git_head": git_head(),
            "source_size_bytes": source.stat().st_size,
            "source_sha256": file_sha256(source),
            "selected_row_count": len(all_output_rows),
            "selected_rows_sha256": rows_digest(all_output_rows),
            "shards": written_entries,
        },
    }
    if not dry_run:
        manifest_path = out_dir / "recovery_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nWrote: {manifest_path}")
    else:
        print("\nDry-run manifest:")
        print(json.dumps(manifest, indent=2))

    print(f"\nDone. {sum(label_out.values()):,} attacks across {n_shards} new shards.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="recovered-attacks JSONL")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--first-shard", type=int, default=DEFAULT_FIRST_SHARD)
    ap.add_argument("--per-shard", type=int, default=DEFAULT_PER_SHARD)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing shard files in the target range",
    )
    args = ap.parse_args()

    return build(
        source=args.source,
        out_dir=args.out_dir,
        first_shard=args.first_shard,
        per_shard=args.per_shard,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
