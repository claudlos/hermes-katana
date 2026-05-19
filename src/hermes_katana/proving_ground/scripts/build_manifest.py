"""Build results/MANIFEST.json — lineage tracker for canonical artifacts.

Records, for each canonical output file, the script that produced it and
when. Used by downstream consumers (hermes-katana scanner, paper pipeline)
to know what's fresh and where it came from.

Run after any pipeline stage completes:

    python scripts/build_manifest.py

Inspect a manifest entry:

    jq '.outputs["results/confirmed_attacks.jsonl"]' results/MANIFEST.json
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "results" / "MANIFEST.json"

# (path, producer_script) — canonical outputs only. Stale siblings (v1_legacy,
# .legacy, batch_tracking.db etc.) are intentionally NOT tracked here.
OUTPUTS = [
    # Cross-reference corpus
    ("results/confirmed_attacks.jsonl", "scripts/cross_reference_confirm.py"),
    ("results/rejected_attacks.jsonl", "scripts/cross_reference_confirm.py"),
    ("results/provisional_attacks.jsonl", "scripts/cross_reference_confirm.py"),
    # Scanner feeds — trigger n-grams
    (
        "results/scanner_feeds/trigger_ngrams.json",
        "scripts/features/extract_trigger_ngrams.py",
    ),
    (
        "results/scanner_feeds/fast_patterns_extension.json",
        "scripts/features/extract_trigger_ngrams.py",
    ),
    (
        "results/scanner_feeds/attack_seed_phrases_extension.json",
        "scripts/features/extract_trigger_ngrams.py",
    ),
    # Scanner feeds — semantic centroids
    (
        "results/scanner_feeds/semantic_centroids.json",
        "scripts/features/build_semantic_centroids.py",
    ),
    (
        "results/scanner_feeds/semantic_centroids.npz",
        "scripts/features/build_semantic_centroids.py",
    ),
    (
        "results/scanner_feeds/attack_vectors.jsonl",
        "scripts/features/build_semantic_centroids.py",
    ),
    # Scanner feeds — cross-model effect clusters
    (
        "results/scanner_feeds/cross_model_effect_clusters.json",
        "scripts/features/cluster_cross_model_effects.py",
    ),
    (
        "results/scanner_feeds/cross_model_effect_clusters.jsonl",
        "scripts/features/cluster_cross_model_effects.py",
    ),
    # Scanner feeds — channel weights + behavioral signature model
    ("results/scanner_feeds/channel_weights.json", "scripts/export_channel_weights.py"),
    (
        "results/scanner_feeds/behavioral_signature_scanner.json",
        "scripts/features/train_behavioral_scanner.py",
    ),
    (
        "results/scanner_feeds/behavioral_signature_scanner.joblib",
        "scripts/features/train_behavioral_scanner.py",
    ),
]


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            text=True,
        ).strip()
    except Exception:
        return None


def _count_rows_jsonl(path: Path) -> int:
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


def build_entry(rel_path: str, producer: str) -> dict:
    p = ROOT / rel_path
    entry: dict = {
        "producer_script": producer,
        "present": p.exists(),
    }
    if not p.exists():
        entry["size_bytes"] = 0
        return entry
    st = p.stat()
    entry["size_bytes"] = st.st_size
    entry["mtime"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    if rel_path.endswith(".jsonl"):
        entry["row_count"] = _count_rows_jsonl(p)
    return entry


def main() -> None:
    manifest = {
        "schema_version": 1,
        "built_at": datetime.now(tz=timezone.utc).isoformat(),
        "git_head": _git_head(),
        "outputs": {path: build_entry(path, prod) for path, prod in OUTPUTS},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    present = sum(1 for v in manifest["outputs"].values() if v.get("present"))
    missing = sum(1 for v in manifest["outputs"].values() if not v.get("present"))
    print(f"Wrote {MANIFEST_PATH}  (present={present} missing={missing}, git={manifest['git_head']})")
    if missing:
        print("\nMissing (not built yet or pipeline stage not run):")
        for path, meta in manifest["outputs"].items():
            if not meta.get("present"):
                print(f"  - {path}  (producer: {meta['producer_script']})")


if __name__ == "__main__":
    main()
