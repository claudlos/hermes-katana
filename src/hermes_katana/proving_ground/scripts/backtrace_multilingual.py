#!/usr/bin/env python3
"""Multilingual back-trace: build focused shards from translations of
already-confirmed English attacks.

The brute-force alternative — running every multilingual factory row
through every agent — would mean ~215K rows × N agents × M channels.
Instead we observe: an attack that doesn't work in English almost
certainly doesn't work in French either. So we only test the 11
language counterparts of attacks that ALREADY confirmed effective in
English. That's ~3,500 confirmed × 11 ≈ ~38K trials per agent —
two orders of magnitude smaller than brute force, and produces a
*useful* per-language transferability matrix for the release.

Pipeline:

    confirmed_attacks.jsonl   ┐
                              ├─→  scripts/backtrace_multilingual.py
    factory/manifest.db       ┘
                              │
                              ▼
              shards/shard_600.jsonl ... shard_NNN.jsonl
              shards/multilingual_backtrace_manifest.json

Output rows match the standard shard schema with three extra fields:
    - language               (ar / de / es / fr / hi / it / ja / ko / pt / ru / zh)
    - original_atk_id        (the English confirmed source_id)
    - translation_source     (latest_source from the factory manifest, e.g. mimo_v2_pro)

Usage:
    python scripts/backtrace_multilingual.py
    python scripts/backtrace_multilingual.py --languages ar de es     # subset
    python scripts/backtrace_multilingual.py --first-shard 600 --per-shard 177
    python scripts/backtrace_multilingual.py --dry-run

Idempotent: deterministic shuffle (seed=42), refuses to overwrite
existing shard files unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
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

DEFAULT_CONFIRMED = ROOT / "results" / "confirmed_attacks.jsonl"
DEFAULT_DB = KATANA_ROOT / "training" / "data_v3" / "factory" / "manifest.db"
DEFAULT_OUT_DIR = ROOT / "shards"
DEFAULT_FIRST_SHARD = 600  # 1-122 empirical, 200-222 synth, 300-321 multiling
# factory, 400-523 recovery — leaves 600+ free
DEFAULT_PER_SHARD = 177
DEFAULT_SEED = 42
DEFAULT_LANGUAGES = ("ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "zh")

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
# Read confirmed-attack ids + their labels (so we can stamp each multilingual
# row with the same label its English ancestor carried)
# ---------------------------------------------------------------------------


def load_confirmed(path: Path) -> dict[str, dict]:
    """source_id → {label, n_models_effective, n_platforms_effective}"""
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("id")
            if not sid:
                continue
            out[sid] = {
                "label": d.get("label", ""),
                "n_models_effective": d.get("n_models_effective", 0),
                "n_platforms_effective": d.get("n_platforms_effective", 0),
                "english_text_sha256": d.get("text_sha256") or sha(d.get("text", "")),
            }
    return out


# ---------------------------------------------------------------------------
# Query factory manifest for translations
# ---------------------------------------------------------------------------


def query_translations(
    db_path: Path,
    source_ids: list[str],
    languages: tuple[str, ...],
    chunk: int = 900,
) -> list[dict]:
    """For each (source_id, language) accepted in the factory, return the
    translated text + metadata. We chunk the IN clause to stay below
    SQLite's parameter limit (1000 in older builds, 32K modern but
    chunked anyway for clarity)."""
    conn = sqlite3.connect(str(db_path))
    rows: list[dict] = []
    sids = list(source_ids)
    for i in range(0, len(sids), chunk):
        slice_ = sids[i : i + chunk]
        sid_placeholders = ",".join("?" * len(slice_))
        lang_placeholders = ",".join("?" * len(languages))
        q = (
            "SELECT source_id, language, latest_text, latest_source, latest_bucket "
            "FROM translation_jobs "
            f"WHERE status='accepted' AND source_id IN ({sid_placeholders}) "
            f"AND language IN ({lang_placeholders})"
        )
        for r in conn.execute(q, list(slice_) + list(languages)):
            sid, lang, text, src, bucket = r
            if not text or len(text) < 8:
                continue
            rows.append(
                {
                    "source_id": sid,
                    "language": lang,
                    "text": text,
                    "latest_source": src or "factory_unknown",
                    "latest_bucket": bucket or "",
                }
            )
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Build a row for a translation
# ---------------------------------------------------------------------------


def make_row(
    *,
    text: str,
    label: str,
    source_id: str,
    language: str,
    translation_source: str,
    translation_bucket: str,
    n_models_effective: int,
    n_platforms_effective: int,
    english_text_sha256: str,
    shard_id: int,
) -> dict:
    raw_sha = sha(text)
    norm_sha = sha(normalize(text))
    return {
        "binary_label": "attack",
        "family_sha256": norm_sha,
        # The id is keyed off the translated text — same English source can
        # have 11 distinct multilingual atks, each with its own id.
        "id": f"atk_{raw_sha[:16]}",
        "is_attack": True,
        "label": label,
        "language": language,
        "origin": "user_input",
        "quality_tier": "multilingual_backtrace",
        "shard": shard_id,
        "source": f"multilingual_backtrace/{translation_source}",
        "source_family": "multilingual_backtrace",
        "split": "train",
        "text": text,
        "text_length": len(text),
        "text_sha256": raw_sha,
        "text_sha256_normalized": norm_sha,
        # Provenance back to the English confirmed attack
        "original_atk_id": source_id,
        "english_text_sha256": english_text_sha256,
        "translation_source": translation_source,
        "translation_bucket": translation_bucket,
        # Useful priors for downstream analysis
        "english_n_models_effective": n_models_effective,
        "english_n_platforms_effective": n_platforms_effective,
        "shard_origin": "multilingual_backtrace_2026-04-28",
    }


# ---------------------------------------------------------------------------
# Dedup against existing shards
# ---------------------------------------------------------------------------


def load_existing_fingerprints(shards_dir: Path) -> set[str]:
    seen = set()
    for sp in sorted(shards_dir.glob("shard_*.jsonl")):
        try:
            with sp.open(encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    ns = d.get("text_sha256_normalized") or sha(normalize(d.get("text", "")))
                    seen.add(ns)
        except Exception:
            continue
    return seen


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(
    confirmed_path: Path,
    db_path: Path,
    out_dir: Path,
    first_shard: int,
    per_shard: int,
    seed: int,
    languages: tuple[str, ...],
    dry_run: bool,
    overwrite: bool,
) -> int:
    if not confirmed_path.exists():
        print(f"ERROR: confirmed_attacks not found: {confirmed_path}", file=sys.stderr)
        return 1
    if not db_path.exists():
        print(f"ERROR: factory manifest not found: {db_path}", file=sys.stderr)
        return 1

    print(f"[1] Reading confirmed attacks: {confirmed_path}")
    confirmed = load_confirmed(confirmed_path)
    print(f"     {len(confirmed):,} confirmed source_ids loaded")

    print("[2] Querying factory manifest for translations ...")
    print(f"     languages: {','.join(languages)}")
    translations = query_translations(db_path, list(confirmed), languages)
    n_unique_sids = len({r["source_id"] for r in translations})
    by_lang = Counter(r["language"] for r in translations)
    print(f"     {len(translations):,} translation rows fetched")
    print(f"     covering {n_unique_sids:,} confirmed source_ids ({100 * n_unique_sids / max(len(confirmed), 1):.1f}%)")
    print("     per-language counts:")
    for lang in languages:
        print(f"       {lang:<4} {by_lang.get(lang, 0):,}")

    print("[3] Indexing existing shard fingerprints ...")
    existing = load_existing_fingerprints(out_dir)
    print(f"     {len(existing):,} normalized hashes already on disk")

    print("[4] Building rows + dedup ...")
    rows: list[dict] = []
    seen_within = set()
    n_dup_existing = 0
    n_dup_within = 0
    label_count = Counter()
    src_count = Counter()
    for tr in translations:
        sid = tr["source_id"]
        meta = confirmed.get(sid)
        if not meta:
            continue
        text = tr["text"]
        ns = sha(normalize(text))
        if ns in existing:
            n_dup_existing += 1
            continue
        if ns in seen_within:
            n_dup_within += 1
            continue
        seen_within.add(ns)
        # shard_id is filled in below after shuffle
        rows.append(
            {
                "_pre": True,
                "text": text,
                "label": meta["label"],
                "source_id": sid,
                "language": tr["language"],
                "translation_source": tr["latest_source"],
                "translation_bucket": tr["latest_bucket"],
                "n_models_effective": meta["n_models_effective"],
                "n_platforms_effective": meta["n_platforms_effective"],
                "english_text_sha256": meta["english_text_sha256"],
            }
        )
        label_count[meta["label"]] += 1
        src_count[tr["latest_source"]] += 1
    print(f"     dup vs existing shards:   {n_dup_existing:,}")
    print(f"     dup within back-trace:    {n_dup_within:,}")
    print(f"     KEPT:                     {len(rows):,}")

    if not rows:
        print("     nothing to write — exiting")
        return 0

    print(f"\n[5] Deterministic shuffle (seed={seed}) ...")
    rng = random.Random(seed)
    rng.shuffle(rows)

    n_shards = -(-len(rows) // per_shard)
    target_paths = [out_dir / f"shard_{first_shard + i:03d}.jsonl" for i in range(n_shards)]
    existing_targets = [p for p in target_paths if p.exists()]
    if existing_targets and not overwrite:
        print(
            f"\nERROR: {len(existing_targets)} target shard files already "
            f"exist (first: {existing_targets[0].name}). Pass --overwrite or "
            f"--first-shard.",
            file=sys.stderr,
        )
        return 2

    print(f"[6] Writing {n_shards} shard files (~{per_shard} rows each) ...")
    written: list[str] = []
    written_entries: list[dict] = []
    all_shard_rows: list[dict] = []
    out_lang = Counter()
    for i in range(n_shards):
        shard_id = first_shard + i
        chunk = rows[i * per_shard : (i + 1) * per_shard]
        if not chunk:
            break
        shard_rows = [
            make_row(
                text=r["text"],
                label=r["label"],
                source_id=r["source_id"],
                language=r["language"],
                translation_source=r["translation_source"],
                translation_bucket=r["translation_bucket"],
                n_models_effective=r["n_models_effective"],
                n_platforms_effective=r["n_platforms_effective"],
                english_text_sha256=r["english_text_sha256"],
                shard_id=shard_id,
            )
            for r in chunk
        ]
        for sr in shard_rows:
            out_lang[sr["language"]] += 1
        all_shard_rows.extend(shard_rows)
        if not dry_run:
            sp = out_dir / f"shard_{shard_id:03d}.jsonl"
            with sp.open("w", encoding="utf-8") as f:
                for sr in shard_rows:
                    f.write(json.dumps(sr, ensure_ascii=False) + "\n")
            written.append(sp.name)
            written_entries.append(
                {
                    "file": sp.name,
                    "size_bytes": sp.stat().st_size,
                    "sha256": file_sha256(sp),
                    "n_rows": len(shard_rows),
                }
            )
    print(f"     wrote: {len(written)} shard files{' (dry-run)' if dry_run else ''}")

    manifest = {
        "version": 1,
        "kind": "multilingual_backtrace",
        "confirmed_source": str(confirmed_path),
        "factory_manifest": str(db_path),
        "first_shard": first_shard,
        "n_shards": n_shards,
        "per_shard_target": per_shard,
        "total_rows": len(rows),
        "by_language": dict(out_lang.most_common()),
        "by_label": dict(label_count.most_common()),
        "by_translation_source": dict(src_count.most_common()),
        "seed": seed,
        "languages": list(languages),
        "files": written,
        "provenance": {
            "builder": "scripts/backtrace_multilingual.py",
            "builder_git_head": git_head(),
            "confirmed_source_sha256": file_sha256(confirmed_path) if confirmed_path.exists() else None,
            "factory_manifest_size_bytes": db_path.stat().st_size if db_path.exists() else None,
            "factory_manifest_sha256": file_sha256(db_path) if db_path.exists() else None,
            "selected_row_count": len(all_shard_rows),
            "selected_rows_sha256": rows_digest(all_shard_rows),
            "shards": written_entries,
        },
        "dedup": {
            "translations_fetched": len(translations),
            "dup_vs_existing_shards": n_dup_existing,
            "dup_within_backtrace": n_dup_within,
            "kept": len(rows),
        },
    }
    if not dry_run:
        mpath = out_dir / "multilingual_backtrace_manifest.json"
        mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote manifest: {mpath}")
    else:
        print("\nDry-run manifest:")
        print(json.dumps(manifest, indent=2, ensure_ascii=False)[:1500])

    print(
        f"\nDone. {len(rows):,} multilingual rows across "
        f"{n_shards} shards ({first_shard:03d}-{first_shard + n_shards - 1:03d})."
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confirmed", type=Path, default=DEFAULT_CONFIRMED)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--first-shard", type=int, default=DEFAULT_FIRST_SHARD)
    ap.add_argument("--per-shard", type=int, default=DEFAULT_PER_SHARD)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    return build(
        confirmed_path=args.confirmed,
        db_path=args.db,
        out_dir=args.out_dir,
        first_shard=args.first_shard,
        per_shard=args.per_shard,
        seed=args.seed,
        languages=tuple(args.languages),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
