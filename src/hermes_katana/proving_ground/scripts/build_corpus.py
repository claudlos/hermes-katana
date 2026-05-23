"""Build sharded corpora for proving-grounds fleet runs.

Unified replacement for build_shards.py, build_benign_shards.py, and
build_multilingual_shards.py. Pick a mode:

    python scripts/build_corpus.py --mode attack
    python scripts/build_corpus.py --mode benign
    python scripts/build_corpus.py --mode multilingual

Shared invariants across modes:
- Read JSONL, apply mode-specific filter, dedup by a stable hash field,
  stratified-shuffle, write N near-equal shards, emit a manifest.
- Shards are gitignored; manifest is safe to share (hashes + counts only).

Mode specifics:

  attack        is_attack=True, split∈{train,val}, tier∈{high,medium_high},
                dedup by family_sha256, stratify by label,
                writes shards/shard_NNN.jsonl (IDs 1..N).

  benign        is_attack=False, split∈{train,val}, source∈ENGLISH_BENIGN,
                dedup by text_sha256_normalized, stratify by source,
                writes shards/control/shard_ctrl_NNN.jsonl, extra field
                {"is_control": True, "control_shard": N}.

  multilingual  per-language loop, tier filter only (test split OK — it IS
                the eval set), dedup by family_sha256, stratify by label,
                writes shards/shard_1NN.jsonl (IDs start at 101).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable


KATANA_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(Path(__file__).resolve().parents[2] / "hermes-katana"),
    )
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _rows_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _source_checksum_manifest(source: Path, rows: list[dict], entries: list[dict], *, builder: str) -> dict:
    """Checksum/provenance block shared by corpus builders."""
    shard_hash = hashlib.sha256()
    for entry in entries:
        shard_hash.update(str(entry.get("file", "")).encode("utf-8"))
        shard_hash.update(str(entry.get("sha256", "")).encode("utf-8"))
    return {
        "builder": builder,
        "builder_git_head": _git_head(),
        "source": str(source),
        "source_exists": source.exists(),
        "source_size_bytes": source.stat().st_size if source.exists() else None,
        "source_sha256": _file_sha256(source) if source.exists() and source.is_file() else None,
        "selected_row_count": len(rows),
        "selected_rows_sha256": _rows_digest(rows),
        "shards_sha256": shard_hash.hexdigest(),
    }


def _stratified_interleave(rows: list[dict], key: str, seed: int) -> list[dict]:
    """Round-robin interleave rows by `key` so every shard sees a proportional slice."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get(key, "?")].append(r)
    for lst in buckets.values():
        rng.shuffle(lst)
    keys = list(buckets)
    out: list[dict] = []
    while True:
        exhausted = 0
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop())
            else:
                exhausted += 1
        if exhausted == len(keys):
            break
    return out


def _filter_rows(
    path: Path,
    keep_fn: Callable[[dict], str | None],
    dedup_key: str,
) -> tuple[list[dict], Counter]:
    """Stream JSONL, drop rows where keep_fn returns a skip-reason string, dedup by dedup_key."""
    kept: list[dict] = []
    seen: set[str] = set()
    skipped: Counter = Counter()
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                skipped["bad_json"] += 1
                continue
            reason = keep_fn(d)
            if reason is not None:
                skipped[reason] += 1
                continue
            dk = d.get(dedup_key, "")
            if dk and dk in seen:
                skipped["dup"] += 1
                continue
            if dk:
                seen.add(dk)
            kept.append(d)
    return kept, skipped


def _write_shard(
    rows: list[dict],
    path: Path,
    shard_id: int,
    extra_fields: dict,
    tag_key: str,  # "label" or "source"
) -> dict:
    h = hashlib.sha256()
    tag_counts: Counter = Counter()
    with path.open("w") as f:
        for r in rows:
            r = {**r, "shard": shard_id, **extra_fields}
            line = json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
            tag_counts[r.get(tag_key, "?")] += 1
    return {
        "shard": shard_id,
        "file": path.name,
        "n_rows": len(rows),
        "sha256": h.hexdigest(),
        tag_key + "s": dict(tag_counts),
    }


def _split_into_shards(rows: list[dict], num_shards: int) -> list[list[dict]]:
    per = len(rows) // num_shards
    rem = len(rows) - per * num_shards
    out: list[list[dict]] = []
    cursor = 0
    for i in range(num_shards):
        size = per + (1 if i < rem else 0)
        out.append(rows[cursor : cursor + size])
        cursor += size
    return out


# ---------------------------------------------------------------------------
# Mode: attack
# ---------------------------------------------------------------------------

ATTACK_CORPUS = str(KATANA_ROOT / "training" / "data_v3" / "combined.jsonl")
ATTACK_TIERS = {"high_confidence_attack", "medium_high_confidence_attack"}
ATTACK_SPLITS = {"train", "val"}


def _attack_filter(d: dict) -> str | None:
    if not d.get("is_attack"):
        return "not_attack"
    if d.get("split") not in ATTACK_SPLITS:
        return "bad_split"
    if d.get("quality_tier") not in ATTACK_TIERS:
        return "bad_tier"
    if not d.get("text"):
        return "no_text"
    return None


def build_attack(args: argparse.Namespace) -> None:
    rows, skipped = _filter_rows(Path(args.corpus), _attack_filter, dedup_key="family_sha256")
    print(f"Read + filtered: kept {len(rows):,} attacks. skipped={dict(skipped)}")

    rows = _stratified_interleave(rows, key="label", seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = _split_into_shards(rows, args.num_shards)
    entries = []
    for i, chunk in enumerate(chunks):
        sid = i + 1
        shard_path = out_dir / f"shard_{sid:03d}.jsonl"
        entries.append(_write_shard(chunk, shard_path, sid, extra_fields={}, tag_key="label"))

    total_labels: Counter = Counter()
    for e in entries:
        for k, v in e.get("labels", {}).items():
            total_labels[k] += v

    manifest = {
        "version": 1,
        "mode": "attack",
        "source": args.corpus,
        "num_shards": args.num_shards,
        "total_rows": sum(e["n_rows"] for e in entries),
        "seed": args.seed,
        "filter": {
            "is_attack": True,
            "splits": sorted(ATTACK_SPLITS),
            "quality_tiers": sorted(ATTACK_TIERS),
            "dedup_by": "family_sha256",
        },
        "label_totals": dict(total_labels),
        "shards": entries,
        "provenance": _source_checksum_manifest(
            Path(args.corpus), rows, entries, builder="scripts/build_corpus.py attack"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\nWrote {args.num_shards} attack shards to {out_dir}/ "
        f"({manifest['total_rows']:,} attacks, ~{manifest['total_rows'] / args.num_shards:.0f}/shard)"
    )
    for lbl, n in total_labels.most_common():
        print(f"  {lbl:<28} {n:,}")


# ---------------------------------------------------------------------------
# Mode: benign
# ---------------------------------------------------------------------------

BENIGN_CORPUS = str(KATANA_ROOT / "training" / "data_v3" / "benign.jsonl")
ENGLISH_BENIGN_SOURCES = {
    "hackaprompt",
    "synthetic_benign",
    "synthetic_benign_extra",
    "awesome_prompts",
    "tensortrust",
    "benign",
}
BENIGN_SPLITS = {"train", "val"}


def _benign_filter(d: dict) -> str | None:
    if d.get("is_attack"):
        return "not_benign"
    if d.get("split") not in BENIGN_SPLITS:
        return "bad_split"
    if d.get("source") not in ENGLISH_BENIGN_SOURCES:
        return "bad_source"
    if not d.get("text"):
        return "no_text"
    return None


def build_benign(args: argparse.Namespace) -> None:
    rows, skipped = _filter_rows(Path(args.corpus), _benign_filter, dedup_key="text_sha256_normalized")
    print(f"Read + filtered: kept {len(rows):,} benign rows. skipped={dict(skipped)}")

    rows = _stratified_interleave(rows, key="source", seed=args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = _split_into_shards(rows, args.num_shards)
    entries = []
    for i, chunk in enumerate(chunks):
        sid = i + 1
        shard_path = out_dir / f"shard_ctrl_{sid:03d}.jsonl"
        entries.append(
            _write_shard(
                chunk,
                shard_path,
                sid,
                extra_fields={"is_control": True, "control_shard": sid},
                tag_key="source",
            )
        )

    manifest = {
        "version": 1,
        "mode": "benign",
        "source": args.corpus,
        "num_shards": args.num_shards,
        "total_rows": sum(e["n_rows"] for e in entries),
        "seed": args.seed,
        "filter": {
            "is_attack": False,
            "splits": sorted(BENIGN_SPLITS),
            "sources": sorted(ENGLISH_BENIGN_SOURCES),
            "dedup_by": "text_sha256_normalized",
        },
        "shards": entries,
        "provenance": _source_checksum_manifest(
            Path(args.corpus), rows, entries, builder="scripts/build_corpus.py benign"
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\nWrote {args.num_shards} benign control shards to {out_dir}/ "
        f"({manifest['total_rows']:,} rows, ~{manifest['total_rows'] / args.num_shards:.0f}/shard)"
    )


# ---------------------------------------------------------------------------
# Mode: multilingual
# ---------------------------------------------------------------------------

MULTILINGUAL_CORPUS_DIR = str(KATANA_ROOT / "training" / "data_v3" / "factory" / "accepted")
MULTILINGUAL_LANGS = ["ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pt", "ru", "zh"]
MULTILINGUAL_SHARD_BASE = 101
MULTILINGUAL_TIERS = {"high_confidence_attack", "medium_high_confidence_attack"}


def _multilingual_filter(d: dict) -> str | None:
    if d.get("quality_tier") not in MULTILINGUAL_TIERS:
        return "bad_tier"
    if not d.get("text"):
        return "no_text"
    return None


def build_multilingual(args: argparse.Namespace) -> None:
    corpus_dir = Path(args.corpus_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_id = MULTILINGUAL_SHARD_BASE
    all_entries: list[dict] = []
    all_sampled_rows: list[dict] = []
    source_files: list[dict] = []
    total = 0

    for lang in args.languages:
        lang_path = corpus_dir / f"{lang}.jsonl"
        if not lang_path.exists():
            print(f"  [{lang}] MISSING: {lang_path}")
            continue
        source_files.append(
            {
                "language": lang,
                "path": str(lang_path),
                "size_bytes": lang_path.stat().st_size,
                "sha256": _file_sha256(lang_path),
            }
        )
        rows, _ = _filter_rows(lang_path, _multilingual_filter, dedup_key="family_sha256")
        rows = _stratified_interleave(rows, key="label", seed=args.seed)

        want = args.shards_per_lang * args.attacks_per_shard
        sample = rows[:want]
        all_sampled_rows.extend(sample)
        lang_entries: list[dict] = []
        for i in range(args.shards_per_lang):
            start = i * args.attacks_per_shard
            chunk = sample[start : start + args.attacks_per_shard]
            shard_path = out_dir / f"shard_{shard_id:03d}.jsonl"
            entry = _write_shard(chunk, shard_path, shard_id, extra_fields={}, tag_key="label")
            entry["language"] = lang
            lang_entries.append(entry)
            all_entries.append(entry)
            shard_id += 1
            total += len(chunk)
        first, last = lang_entries[0]["shard"], lang_entries[-1]["shard"]
        print(
            f"  [{lang}] wrote {len(lang_entries)} shards ({first}-{last}), "
            f"{sum(e['n_rows'] for e in lang_entries)} attacks"
        )

    manifest = {
        "version": 1,
        "mode": "multilingual",
        "source_dir": str(corpus_dir),
        "languages": args.languages,
        "shards_per_language": args.shards_per_lang,
        "attacks_per_shard": args.attacks_per_shard,
        "seed": args.seed,
        "total_rows": total,
        "shard_id_range": [MULTILINGUAL_SHARD_BASE, shard_id - 1],
        "shards": all_entries,
        "provenance": {
            **_source_checksum_manifest(
                corpus_dir, all_sampled_rows, all_entries, builder="scripts/build_corpus.py multilingual"
            ),
            "source_files": source_files,
        },
    }
    (out_dir / "manifest_multilingual.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(
        f"\nWrote {len(all_entries)} multilingual shards "
        f"(IDs {MULTILINGUAL_SHARD_BASE}-{shard_id - 1}), {total:,} attacks total."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)

    pa = sub.add_parser("attack", help="Attack shards from hermes-katana combined.jsonl")
    pa.add_argument("--corpus", default=ATTACK_CORPUS)
    pa.add_argument("--out-dir", default="shards")
    pa.add_argument("--num-shards", type=int, default=100)
    pa.add_argument("--seed", type=int, default=42)

    pb = sub.add_parser("benign", help="Benign control shards from hermes-katana benign.jsonl")
    pb.add_argument("--corpus", default=BENIGN_CORPUS)
    pb.add_argument("--out-dir", default="shards/control")
    pb.add_argument("--num-shards", type=int, default=20)
    pb.add_argument("--seed", type=int, default=42)

    pm = sub.add_parser("multilingual", help="Per-language shards from factory/accepted/")
    pm.add_argument("--corpus-dir", default=MULTILINGUAL_CORPUS_DIR)
    pm.add_argument("--out-dir", default="shards")
    pm.add_argument("--shards-per-lang", type=int, default=2)
    pm.add_argument("--attacks-per-shard", type=int, default=200)
    pm.add_argument("--seed", type=int, default=42)
    pm.add_argument("--languages", nargs="+", default=MULTILINGUAL_LANGS)

    # Accept `--mode` form too for convenience (maps to subparser)
    args = p.parse_args()
    if args.mode == "attack":
        build_attack(args)
    elif args.mode == "benign":
        build_benign(args)
    elif args.mode == "multilingual":
        build_multilingual(args)


if __name__ == "__main__":
    main()
