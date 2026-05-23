"""Convert all synth `examples_raw.jsonl` outputs into proving-ground shards.

Reads every synthdata/checkpoints/*/examples_raw.jsonl, normalizes each
row into the same schema as shards/shard_NNN.jsonl, dedupes by
normalized text sha, and writes shard files in the next-available
shard-id range (starts at 200 by default to avoid colliding with the
existing 1-125 shards).

Output:
  shards/shard_200.jsonl, shard_201.jsonl, ...
  shards/synth_manifest.json    (provenance + counts)

Run agents against these via the fleet:
    python scripts/fleet.py launch \\
        --spec scripts/fleet_synth_confirm.json \\
        --run-id synth_v4_confirm

After completion, scripts/cross_reference_confirm.py will pick up rows
from results/agent_shard_runs/ matching shard_2*.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "shards"
DEFAULT_FIRST_SHARD = 200
DEFAULT_PER_SHARD = 177  # match existing shard granularity

ZW_RE = re.compile(r"[​-‏⁠﻿‪-‮]")
WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    s = ZW_RE.sub("", s)
    s = WS_RE.sub(" ", s).strip().lower()
    return s


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return None


def _rows_digest(rows: list[dict]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _build_synth_manifest(
    *,
    input_paths: list[Path],
    written_paths: list[Path],
    rows: list[dict],
    first_shard: int,
    per_shard: int,
    by_run: dict,
    by_label: dict,
) -> dict:
    shard_entries = []
    for p in written_paths:
        shard_entries.append(
            {
                "file": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "sha256": _file_sha256(p) if p.exists() else None,
            }
        )
    return {
        "version": 2,
        "kind": "synth_unconfirmed",
        "builder": "scripts/synth_to_shards.py",
        "builder_git_head": _git_head(),
        "first_shard": first_shard,
        "n_shards": len(written_paths),
        "per_shard_target": per_shard,
        "total_attacks": len(rows),
        "by_run": dict(by_run),
        "by_label": dict(by_label),
        "input_files": [
            {
                "path": str(p),
                "size_bytes": p.stat().st_size if p.exists() else None,
                "sha256": _file_sha256(p) if p.exists() else None,
            }
            for p in input_paths
        ],
        "selected_rows_sha256": _rows_digest(rows),
        "shards": shard_entries,
        "files": [p.name for p in written_paths],
    }


def _row(*, text: str, label: str, source: str, shard_id: int) -> dict:
    raw_sha = _sha(text)
    norm_sha = _sha(_normalize(text))
    return {
        "binary_label": "attack",
        "family_sha256": norm_sha,
        "id": f"atk_{raw_sha[:16]}",
        "is_attack": True,
        "label": label,
        "origin": "user_input",
        "quality_tier": "synthdata_unconfirmed",
        "shard": shard_id,
        "source": source,
        "source_family": source,
        "split": "train",
        "text": text,
        "text_length": len(text),
        "text_sha256": raw_sha,
        "text_sha256_normalized": norm_sha,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-glob", type=str, default="synthdata/checkpoints/*/examples_raw.jsonl")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--first-shard", type=int, default=DEFAULT_FIRST_SHARD)
    ap.add_argument("--per-shard", type=int, default=DEFAULT_PER_SHARD)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[str] = set()
    by_run: Counter = Counter()
    by_label: Counter = Counter()

    paths = sorted(ROOT.glob(args.ckpt_glob))
    print(f"reading {len(paths)} checkpoint files:")
    for p in paths:
        run = p.parent.name
        added = 0
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                text = (r.get("text") or "").strip()
                label = r.get("label") or ""
                if not text or label == "clean":
                    continue
                if len(text) < 16 or len(text) > 1500:
                    continue
                norm_sha = _sha(_normalize(text))
                if norm_sha in seen:
                    continue
                seen.add(norm_sha)
                rows.append({"text": text, "label": label, "_source_run": run})
                by_run[run] += 1
                by_label[label] += 1
                added += 1
        print(f"  {run}: +{added}")

    n_total = len(rows)
    print(f"\ndedupe-pass total: {n_total}")
    print("by run:")
    for r, n in by_run.most_common():
        print(f"  {r:<30} {n}")
    print("by label:")
    for label, n in by_label.most_common():
        print(f"  {label:<25} {n}")

    # Shard out
    n_shards = (n_total + args.per_shard - 1) // args.per_shard
    print(f"\nwriting {n_shards} shards starting at shard_{args.first_shard:03d}")
    written_paths: list[Path] = []
    for s_idx in range(n_shards):
        sid = args.first_shard + s_idx
        chunk = rows[s_idx * args.per_shard : (s_idx + 1) * args.per_shard]
        out_path = args.out_dir / f"shard_{sid:03d}.jsonl"
        with out_path.open("w") as f:
            for r in chunk:
                source = f"synthdata/{r['_source_run']}"
                f.write(
                    json.dumps(
                        _row(
                            text=r["text"],
                            label=r["label"],
                            source=source,
                            shard_id=sid,
                        )
                    )
                    + "\n"
                )
        written_paths.append(out_path)
        print(f"  {out_path.name}: {len(chunk)} rows")

    # Manifest
    manifest = _build_synth_manifest(
        input_paths=paths,
        written_paths=written_paths,
        rows=rows,
        first_shard=args.first_shard,
        per_shard=args.per_shard,
        by_run=by_run,
        by_label=by_label,
    )
    manifest_path = args.out_dir / "synth_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nmanifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
