"""Expand the 6 katana labels that v2's taxonomy never recursed into.

Usage:
    python -m synthdata.expand_gap_labels \
        --run synthdata/checkpoints/katana_synth_v2_opus_elite \
        --config synthdata/configs/v2_opus_elite.json

For each gap label, deletes the stale root-leaf node from taxonomy.jsonl
and runs the propose/critic/refine recursion in-place. Updates
run_meta.json counts on success. Idempotent: a label whose subtree
already has >1 node is skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm import LLMClient, LLMConfig
from .schema import GenerationRun
from .taxonomy import (
    TAXONOMY_MAX_DEPTH,
    TAXONOMY_N_CANDIDATES_PER_SPLIT,
    _expand,
    _node_id_for,
    _TOP_LEVEL_DESCRIPTIONS,
    load_taxonomy,
    save_taxonomy,
    seed_sampler,
    _format_seed_hint,
)


GAP_LABELS = (
    "behavioral_control",
    "exfiltration_attempt",
    "jailbreak",
    "cognitive_state_attack",
    "encoding_evasion",
    "persona_jailbreak",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--labels",
        type=str,
        default=None,
        help="comma-sep labels to expand (default: 6 v3 gap labels)",
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    teacher = LLMClient(LLMConfig(**cfg["teacher"]))

    target_labels = tuple(args.labels.split(",")) if args.labels else GAP_LABELS

    tax_path = args.run / "taxonomy.jsonl"
    nodes = load_taxonomy(tax_path)

    # Seed corpus for domain hints (optional but improves tree quality)
    seed_corpus = Path(cfg.get("seed_corpus", "results/elite_attacks_n5.jsonl"))
    seeds = seed_sampler(seed_corpus, n_per_label=cfg.get("seeds_per_label", 3)) if seed_corpus.exists() else {}

    expanded_count = 0
    for label in target_labels:
        # Find the existing root node for this label. It will be a
        # node_id derived from `_node_id_for(None, label)`.
        root_id = _node_id_for(None, label)
        if root_id not in nodes:
            print(f"[skip] {label}: no root node in taxonomy")
            continue
        root = nodes[root_id]
        # Count children and descendants for this root.
        descendants = sum(1 for n in nodes.values() if n.parent_id == root_id)
        if descendants > 0:
            print(f"[skip] {label}: already has {descendants} children")
            continue
        # Reset the root to non-leaf and run the recursion.
        print(f"[expand] {label} (was leaf, will recurse)...")
        root.is_leaf = False
        root.children = []
        # Override the description to the canonical one (in case the
        # earlier silent-failure run left a stale "[propose_failed: …]"
        # tag on it).
        if "[propose_failed" in root.description:
            root.description = _TOP_LEVEL_DESCRIPTIONS.get(label, root.description)
        seed_hint = _format_seed_hint(label, seeds.get(label, []))
        before = len(nodes)
        _expand(
            teacher,
            root,
            nodes,
            seed_hint=seed_hint,
            n_candidates=cfg.get("n_candidates_per_split", TAXONOMY_N_CANDIDATES_PER_SPLIT),
            max_depth=cfg.get("max_depth", TAXONOMY_MAX_DEPTH),
        )
        added = len(nodes) - before
        print(f"   added {added} nodes under {label}")
        expanded_count += 1

    # Save updated taxonomy.
    save_taxonomy(nodes, tax_path)

    # Update run_meta counts.
    meta_path = args.run / "run_meta.json"
    if meta_path.exists():
        meta = GenerationRun(**json.loads(meta_path.read_text(encoding="utf-8")))
        meta.n_taxonomy_nodes = len(nodes)
        meta.n_taxonomy_leaves = sum(1 for n in nodes.values() if n.is_leaf)
        meta.taxonomy_done = True
        # Force step-2a to rerun so the new gap leaves get scenarios.
        meta.meta_prompts_done = False
        meta.generation_done = False
        meta.critics_done = False
        meta_path.write_text(json.dumps(meta.to_json(), indent=2), encoding="utf-8")

    print(f"\n[done] expanded {expanded_count} gap labels")
    print(f"[done] taxonomy now: {meta.n_taxonomy_nodes} nodes, {meta.n_taxonomy_leaves} leaves")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
