"""Re-apply critics to an existing examples_raw.jsonl.

Use case: the generation pass succeeded but one or both critics errored
(auth, rate-limit, bug). Reload the raw examples, re-judge, overwrite
examples_judged.jsonl + critic_summary.json + synthdata_final.jsonl.

No API-call waste: raw texts are reused in place.

Usage:
    python -m synthdata.rerun_critics \
        --run synthdata/checkpoints/katana_synth_v1 \
        --config synthdata/configs/v1_claude.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .critics import judge, summarize
from .llm import LLMClient, LLMConfig
from .meta_prompt import load_examples, save_examples
from .schema import GenerationRun


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="path to the run checkpoint directory")
    ap.add_argument(
        "--config",
        type=Path,
        required=True,
        help="JSON config (for critic_a/critic_b model+provider)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="ThreadPoolExecutor size for parallel judging (safe values: 1-8 for CLI providers).",
    )
    ap.add_argument(
        "--only-errors",
        action="store_true",
        help="If examples_judged.jsonl exists, only re-judge "
        "rows where critic_a or critic_b had failure_mode="
        "'critic_error'. Other rows are kept verbatim.",
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    critic_a = LLMClient(LLMConfig(**cfg["critic_a"]))
    critic_b = LLMClient(LLMConfig(**cfg["critic_b"]))

    raw_path = args.run / "examples_raw.jsonl"
    if not raw_path.exists():
        print(f"[err] {raw_path} missing; nothing to re-judge")
        return 2

    examples = load_examples(raw_path)
    print(f"[rerun] loaded {len(examples)} raw examples")
    print(f"[rerun] critic A = {critic_a.cfg.provider}/{critic_a.cfg.model}")
    print(f"[rerun] critic B = {critic_b.cfg.provider}/{critic_b.cfg.model}")
    print(f"[rerun] workers = {args.workers}")

    judged_path = args.run / "examples_judged.jsonl"
    keep_clean: list = []
    to_redo = examples
    if args.only_errors and judged_path.exists():
        prior = load_examples(judged_path)
        prior_by_id = {ex.example_id: ex for ex in prior}
        to_redo = []
        for ex in examples:
            p = prior_by_id.get(ex.example_id)
            if p is None:
                to_redo.append(ex)
                continue
            a_err = p.critic_a is not None and p.critic_a.failure_mode == "critic_error"
            b_err = p.critic_b is not None and p.critic_b.failure_mode == "critic_error"
            if a_err or b_err:
                to_redo.append(ex)
            else:
                keep_clean.append(p)
        print(f"[rerun] only-errors: {len(keep_clean)} prior verdicts kept, {len(to_redo)} to re-judge")

    # Incremental save: every 50 critics, write the full judged list
    # (kept_clean prior verdicts + completed redone) so a kill mid-run
    # doesn't lose everything.
    judged_path_partial = args.run / "examples_judged.jsonl"

    def _save_partial(redone_so_far: list) -> None:
        full = keep_clean + redone_so_far
        from .meta_prompt import save_examples

        save_examples(full, judged_path_partial)

    redone = judge(
        to_redo,
        critic_a_llm=critic_a,
        critic_b_llm=critic_b,
        max_workers=args.workers,
        save_callback=_save_partial,
        save_every=50,
    )

    judged = keep_clean + redone
    summary = summarize(judged)

    save_examples(judged, args.run / "examples_judged.jsonl")
    (args.run / "critic_summary.json").write_text(json.dumps(summary, indent=2))

    kept = [ex for ex in judged if ex.keep]
    final = args.run / "synthdata_final.jsonl"
    with final.open("w") as f:
        for ex in kept:
            f.write(
                json.dumps(
                    {
                        "text": ex.text,
                        "label": ex.label,
                        "channel": ex.channel,
                        "origin": "synthdata_v1",
                        "meta_id": ex.meta_id,
                        "leaf_id": ex.leaf_id,
                        "complexity_level": 0,
                        "teacher_model": ex.teacher_model,
                    }
                )
                + "\n"
            )

    # Refresh run_meta counts
    meta_path = args.run / "run_meta.json"
    if meta_path.exists():
        meta = GenerationRun(**json.loads(meta_path.read_text()))
        meta.n_examples_kept = summary["n_kept"]
        meta.critics_done = True
        meta_path.write_text(json.dumps(meta.to_json(), indent=2))

    print(f"[rerun] done: {summary['n_kept']}/{summary['n_total']} kept (keep_rate={summary['keep_rate']:.1%})")
    print(f"[final] {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
