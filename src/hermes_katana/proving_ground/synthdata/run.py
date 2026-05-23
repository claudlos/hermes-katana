"""Orchestrator CLI for the Simula-for-katana pipeline.

Usage:
    python -m synthdata.run --smoke          # tiny test: 1 leaf, 1 scenario, 1 text
    python -m synthdata.run --config <path>  # real run from YAML

Each step writes to <checkpoint_dir> and is resume-safe — rerunning
skips any step whose output already exists.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .complexify import complexify_batch
from .critics import judge, summarize
from .llm import LLMClient, LLMConfig
from .meta_prompt import (
    generate_texts_for_meta,
    load_meta_prompts,
    meta_prompts_from_leaves,
    save_examples,
    save_meta_prompts,
)
from .schema import GenerationRun, SynthExample
from .taxonomy import (
    build_taxonomy,
    iter_leaves,
    load_taxonomy,
    save_taxonomy,
    seed_sampler,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "synthdata" / "checkpoints"
DEFAULT_SEED_CORPUS = ROOT / "results" / "confirmed_attacks.jsonl"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _make_llms(cfg: dict) -> tuple[LLMClient, LLMClient, LLMClient]:
    """Return (teacher, critic_a, critic_b)."""

    def mk(role: str) -> LLMClient:
        role_cfg = cfg[role]
        return LLMClient(LLMConfig(**role_cfg))

    return mk("teacher"), mk("critic_a"), mk("critic_b")


def run(cfg_path: Path | None, *, smoke: bool = False) -> int:
    if smoke:
        cfg = _smoke_config()
    else:
        cfg = json.loads(Path(cfg_path).read_text())

    run_id = cfg.get("run_id") or f"synth-{int(time.time())}"
    ckpt = Path(cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT)) / run_id
    ckpt.mkdir(parents=True, exist_ok=True)

    teacher, critic_a, critic_b = _make_llms(cfg)

    # ---- init run manifest --------------------------------------------------
    manifest_path = ckpt / "run_meta.json"
    if manifest_path.exists():
        manifest = GenerationRun(**json.loads(manifest_path.read_text()))
        print(f"[resume] {manifest_path}")
    else:
        manifest = GenerationRun(
            run_id=run_id,
            started_at_iso=_now_iso(),
            config_path=str(cfg_path) if cfg_path else "<smoke>",
            teacher_model=teacher.cfg.model,
            critic_a_model=critic_a.cfg.model,
            critic_b_model=critic_b.cfg.model,
        )
        manifest_path.write_text(json.dumps(manifest.to_json(), indent=2))

    # ---- Step 1: taxonomy ---------------------------------------------------
    tax_path = ckpt / "taxonomy.jsonl"
    if tax_path.exists() and manifest.taxonomy_done:
        nodes = load_taxonomy(tax_path)
        print(f"[skip] taxonomy — loaded {len(nodes)} nodes")
    else:
        print("[step 1] building taxonomy...")
        seed_path = Path(cfg.get("seed_corpus", DEFAULT_SEED_CORPUS))
        seeds = seed_sampler(seed_path, n_per_label=cfg.get("seeds_per_label", 3)) if seed_path.exists() else {}
        nodes = build_taxonomy(
            teacher,
            seed_samples_by_label=seeds,
            max_depth=cfg.get("max_depth", 3),
            n_candidates_per_split=cfg.get("n_candidates_per_split", 7),
        )
        save_taxonomy(nodes, tax_path)
        manifest.n_taxonomy_nodes = len(nodes)
        manifest.n_taxonomy_leaves = sum(1 for n in nodes.values() if n.is_leaf)
        manifest.taxonomy_done = True
        manifest_path.write_text(json.dumps(manifest.to_json(), indent=2))
        print(f"[step 1] done: {manifest.n_taxonomy_nodes} nodes, {manifest.n_taxonomy_leaves} leaves")

    # ---- Step 2a: meta-prompts ---------------------------------------------
    meta_path = ckpt / "meta_prompts.jsonl"
    if meta_path.exists() and manifest.meta_prompts_done:
        metas = load_meta_prompts(meta_path)
        print(f"[skip] meta_prompts — loaded {len(metas)}")
    else:
        print("[step 2a] generating meta-prompts...")
        leaves = list(iter_leaves(nodes))
        if cfg.get("max_leaves"):
            leaves = leaves[: cfg["max_leaves"]]
        metas = meta_prompts_from_leaves(
            teacher,
            leaves,
            n_scenarios=cfg.get("scenarios_per_leaf", 4),
        )
        # ---- Step 3: complexification (ADDS to the list) ------------------
        if cfg.get("complexify_fraction", 0.3) > 0:
            print(f"[step 3] complexifying {int(cfg.get('complexify_fraction', 0.3) * 100)}% ...")
            metas = complexify_batch(
                teacher,
                metas,
                fraction=cfg.get("complexify_fraction", 0.3),
                ops_budget=tuple(cfg.get("ops_budget", (1, 2))),
            )
        save_meta_prompts(metas, meta_path)
        manifest.n_meta_prompts = len(metas)
        manifest.meta_prompts_done = True
        manifest_path.write_text(json.dumps(manifest.to_json(), indent=2))
        print(f"[step 2a] done: {manifest.n_meta_prompts} scenarios (incl. complexified)")

    # ---- Step 2b: generate concrete texts ----------------------------------
    raw_path = ckpt / "examples_raw.jsonl"
    from .meta_prompt import load_examples

    if raw_path.exists() and manifest.generation_done:
        examples = load_examples(raw_path)
        print(f"[skip] generation — loaded {len(examples)}")
    else:
        # Resume-safe: load any prior partial examples_raw.jsonl and skip
        # metas whose meta_id is already covered. Allows pause/resume across
        # SIGTERM, quota outages, machine reboots.
        examples: list[SynthExample] = []
        already_covered_meta_ids: set[str] = set()
        if raw_path.exists():
            examples = load_examples(raw_path)
            already_covered_meta_ids = {ex.meta_id for ex in examples}
            print(
                f"[step 2b] resuming with {len(examples)} prior examples "
                f"({len(already_covered_meta_ids)} metas already covered)"
            )
        else:
            print("[step 2b] generating attack texts...")
        n_texts = cfg.get("texts_per_scenario", 3)
        for i, meta in enumerate(metas):
            if meta.meta_id in already_covered_meta_ids:
                continue
            batch = generate_texts_for_meta(
                teacher,
                meta,
                n_texts=n_texts,
                teacher_model=teacher.cfg.model,
            )
            examples.extend(batch)
            if (i + 1) % 25 == 0:
                print(f"   ... {i + 1}/{len(metas)} scenarios -> {len(examples)} examples")
                save_examples(examples, raw_path)  # incremental
        save_examples(examples, raw_path)
        manifest.n_examples_generated = len(examples)
        manifest.generation_done = True
        manifest_path.write_text(json.dumps(manifest.to_json(), indent=2))
        print(f"[step 2b] done: {manifest.n_examples_generated} examples")

    # ---- Step 4: dual-critic gate ------------------------------------------
    judged_path = ckpt / "examples_judged.jsonl"
    if judged_path.exists() and manifest.critics_done:
        from .meta_prompt import load_examples

        judged = load_examples(judged_path)
        print(f"[skip] critics — loaded {len(judged)}")
    else:
        critic_workers = int(cfg.get("critic_workers", 1))
        print(f"[step 4] running dual-critic gate (workers={critic_workers})...")
        from .meta_prompt import save_examples as _save_examples_helper

        def _save_partial(judged_so_far: list) -> None:
            _save_examples_helper(judged_so_far, judged_path)

        judged = judge(
            examples,
            critic_a_llm=critic_a,
            critic_b_llm=critic_b,
            max_workers=critic_workers,
            save_callback=_save_partial,
            save_every=50,
        )
        save_examples(judged, judged_path)
        summary = summarize(judged)
        (ckpt / "critic_summary.json").write_text(json.dumps(summary, indent=2))
        manifest.n_examples_kept = summary["n_kept"]
        manifest.critics_done = True
        manifest_path.write_text(json.dumps(manifest.to_json(), indent=2))
        print(f"[step 4] done: {summary['n_kept']}/{summary['n_total']} kept (keep_rate={summary['keep_rate']:.1%})")

    # ---- final export -------------------------------------------------------
    kept_only = [ex for ex in judged if ex.keep]
    final_path = ckpt / "synthdata_final.jsonl"
    with final_path.open("w") as f:
        for ex in kept_only:
            f.write(
                json.dumps(
                    {
                        "text": ex.text,
                        "label": ex.label,
                        "channel": ex.channel,
                        "origin": "synthdata_v1",
                        "meta_id": ex.meta_id,
                        "leaf_id": ex.leaf_id,
                        "complexity_level": 0,  # populated if we extend meta→complexity link
                        "teacher_model": ex.teacher_model,
                    }
                )
                + "\n"
            )
    print(f"[final] {len(kept_only)} examples → {final_path}")
    return 0


def _smoke_config() -> dict:
    """Tiny config for a <$0.10 end-to-end run."""
    return {
        "run_id": f"smoke-{int(time.time())}",
        "checkpoint_dir": str(DEFAULT_CHECKPOINT),
        "teacher": {
            "model": "claude-haiku-4-5",
            "provider": "anthropic",
            "temperature": 0.8,
            "max_tokens": 1200,
        },
        "critic_a": {
            "model": "claude-haiku-4-5",
            "provider": "anthropic",
            "temperature": 0.2,
            "max_tokens": 500,
        },
        "critic_b": {
            "model": "gpt-4o-mini",
            "provider": "openai",
            "temperature": 0.2,
            "max_tokens": 500,
        },
        "max_depth": 1,
        "n_candidates_per_split": 3,
        "max_leaves": 2,
        "scenarios_per_leaf": 2,
        "texts_per_scenario": 2,
        "complexify_fraction": 0.5,
        "seed_corpus": str(DEFAULT_SEED_CORPUS),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not args.smoke and args.config is None:
        ap.error("need --smoke or --config")
    return run(args.config, smoke=args.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
