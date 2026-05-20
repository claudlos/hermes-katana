# Reproducibility

Everything in this document is the contract for being able to regenerate the v5.1 corpus, the v1.0 trained checkpoint, and the leaderboard numbers byte-for-byte (or close enough for confidence intervals to overlap).

## Seeds

| seed | purpose | location |
|---|---|---|
| `42` | corpus build, dataset shuffle, dry-run sampling, fleet sampling | every script accepts `--seed`, default 42 |
| `42 + 17` | hard-negative selection (offset to keep distinct from clean shuffle) | `scripts/build_focused_v5_1_20260506.py` |

Family-level split assignment is **deterministic and seed-free**: `int(family_sha256[:8], 16) % 100`. So splits are stable across seed changes.

## Hardware

| stage | requirement | wall time |
|---|---|---:|
| Corpus build (12.8K rows) | any laptop | <30 s |
| Trainer dry-run (100 rows, CPU) | any 8 GB machine | ~1 min |
| Trainer full run (5 epochs DeBERTa-v3-large) | A100 40GB / L4 24GB | 60 / 150 min |
| Eval (1,271 rows on RTX 3050 4GB at batch=8) | 4 GB GPU | ~90 s |
| Bootstrap CIs (1000 resamples) | adds ~5 s on top of eval | |

T4 (16 GB) **does not fit** v3-large at the default `batch_size=4 grad_accum=4 max_length=256`; drop to `batch_size=2 grad_accum=8` or fall back to `microsoft/deberta-v3-base`.

## Dependency lock

The historical training-time environment used these key versions:

```
torch==2.11.0
transformers==4.57.6
datasets==3.6.0
scikit-learn==1.8.0
numpy==2.4.4
sentencepiece==0.2.1
tokenizers==0.22.2
safetensors==0.7.0
accelerate==1.13.0
```

Both repos also have a full `requirements.lock` from `pip freeze`. To regenerate cleanly with proper resolution:

```bash
uv pip compile pyproject.toml -o requirements.lock --extra=ml --extra=dev
```

## Reproducing the v5.1 corpus from scratch

Inputs (all in `hermes-katana`):

- `training/data_v3/combined.jsonl` (legacy v3 spine for the v4 baseline)
- `training/data_v3/benign.jsonl` (clean controls)
- `training/data_v3/hard_negatives.jsonl` (HackaPrompt + Awesome Prompts curated benigns)
- `synthdata/incoming/v5_balanced_synthdata_final.jsonl` in this repo (the proving-ground v5 attack pool)

Build:

```bash
cd katana-proving-ground
python scripts/build_focused_v5_1_20260506.py
# Produces:
#   hermes-katana/training/data_v5_1/{combined,attacks,controls,hard_negatives}.jsonl
#   hermes-katana/training/data_v5_1/splits/{train,val,test}.jsonl
#   hermes-katana/training/data_v5_1/metadata.json
#   katana-proving-ground/results/reports/focused_v5_1_*/report.md
```

The build is deterministic. Re-running on unchanged inputs produces byte-identical outputs.

## Reproducing the public corpus (sanitized for HF release)

```bash
cd katana-proving-ground
python scripts/sanitize_v5_1_for_publish.py
# Produces hermes-katana/training/data_v5_1_public/...
```

This applies email/SSN/CC/anthropic.com redactions. Family hashes / split IDs / labels are preserved.

## Reproducing the v1.0 model

```bash
cd hermes-katana
python training/train_katana.py --config training/configs/katana_v11.yaml --strict-determinism
```

Add `--strict-determinism` for paper-quality reproducibility (~5–10% slower; cudnn deterministic; no nondeterministic algorithms allowed).

For hosted notebook runs, use the same commands with mounted artifact and dataset directories. Keep generated checkpoints outside the Git repository and publish release artifacts through the artifact registry.

Expected outputs:

- `training/checkpoints/katana_v11/best/` — best epoch's weights + tokenizer
- `training/checkpoints/katana_v11/final/` — last epoch's weights
- `training/checkpoints/katana_v11/metrics.json` — best_val_macro_f1, epochs_run, vocab_size, train/val row counts

The v1.0 reference run produced `best_val_macro_f1 = 0.8643` at epoch 4 (Colab A100 40GB, 2026-05-06). With the same seed and dependency lock you should land within 0.005 macro F1 on val.

## Reproducing the leaderboard number

```bash
cd hermes-katana
python training/eval_katana_v11.py \
  --checkpoint training/checkpoints/katana_v11/best \
  --data evals/benchmarks/confirmed_only_v1/test.jsonl \
  --config training/configs/katana_v11.yaml \
  --bootstrap 1000 \
  --out-dir results/eval_v1_repro
```

The bootstrap resampling uses seed=42 internally; the 95% CI bounds should match the published numbers within rounding. The point estimate (macro F1) is fully deterministic.

## Source provenance & licenses

| source | what we use | license |
|---|---|---|
| awesome_prompts | clean rows | CC0 |
| deepset/prompt-injections | clean rows | CC-BY-4.0 |
| HackaPrompt | hard-negative benigns only | CC-BY-SA-4.0 |
| confirmed attacks (proving-ground) | this corpus's empirical core | CC-BY-4.0 (this release) |
| synthetic critic-passed attacks | enrichment for thin labels | CC-BY-4.0 (this release) |

The HackaPrompt CC-BY-SA-4.0 license is share-alike. If your downstream use is incompatible, filter `release_tier != "hard_negative_control"` to drop those 1,000 rows.

## Versioning

| artifact | current version | next planned |
|---|---|---|
| corpus | `v5_1` | `v6` (when proving-ground v6 confirms more empirical attacks; v5 silver_v5_enrichment shrinks correspondingly) |
| trainer | `katana_v11` | `katana_v12` (LR-decay tweak, AMP, distributed support) |
| benchmark | `confirmed_only_v1` | versions are append-only; older benchmarks stay frozen for historical comparison |

## Known nondeterminism without `--strict-determinism`

- CUDA convolution algorithm selection (cudnn benchmark mode)
- Some matmul implementations select algorithms based on heuristics
- DataLoader ordering when `num_workers > 0` (default trainer uses 0; safe)

With `--strict-determinism`:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.use_deterministic_algorithms(True)`

Trade-off: ~5–10% slower; on Colab A100 the v1.0 reference run takes ~50 minutes instead of ~46.

## Citation

```bibtex
@misc{katana_v1_2026,
  title  = {Katana: An origin-aware prompt-injection scanner with empirical confirmation},
  author = {Anonymous},
  year   = {2026},
  note   = {Code: https://github.com/.../hermes-katana,
           Corpus: https://huggingface.co/datasets/.../katana-v5_1,
           Model: https://huggingface.co/.../katana_v11}
}
```
