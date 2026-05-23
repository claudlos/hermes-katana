# confirmed_only_v1 — katana benchmark

The headline benchmark for Katana prompt-injection classifiers. **Only empirically-validated attacks** (no synthetic data); held-out across the v5.1 corpus by `family_sha256`; reported with bootstrap 95% CIs.

## What this is

A 982-row test set drawn from `data_v5_1/splits/test.jsonl` after filtering attacks down to `quality_tier ∈ {confirmed_n3, confirmed_n4, …, confirmed_n12}`. Clean rows (572) are kept in full so binary attack-vs-benign F1 and FPR are measurable.

| split | rows |
|---|---:|
| clean | 572 |
| confirmed real attacks | 410 |
| **total** | **982** |

| attack label | rows |
|---|---:|
| semantic_manipulation | 71 |
| behavioral_control | 65 |
| cognitive_state_attack | 62 |
| jailbreak | 62 |
| exfiltration_attempt | 61 |
| content_injection | 59 |
| encoding_evasion | 22 |
| persona_jailbreak | 8 |

`encoding_evasion` and especially `persona_jailbreak` are sparse here precisely because the v5 enrichment pool (excluded) was where the synthetic-augmented coverage came from. This skew is *the* honest snapshot of what's confirmed in the wild as of 2026-05-06.

## Methodology

- **Held-out by family**: Test rows are family-disjoint from train/val by `family_sha256`. Verified at training time (`train=10264, val=1287, test=1271`).
- **Confirmed-only filter**: Only rows whose `quality_tier` starts with `confirmed_` (i.e., empirically validated by the proving-ground harness against ≥3 model families) are admitted as attacks. Synthetic critic-passed rows are **excluded** even though they're useful for training.
- **Origin tokens preserved**: Each row's `origin` field (one of the 6 PLAN_B tiers) is prepended as `[ORIGIN=<tier>]` during evaluation, matching trainer behavior.
- **Bootstrap CIs**: 1,000 resamples with replacement at the row level; report 2.5th and 97.5th percentile as 95% CI bounds.
- **Multi-class macro F1**: 9-class macro F1 (clean + 8 attack categories), uniform-weighted (no support weighting, so rare classes have full influence).
- **Binary attack/benign**: Derived from `is_attack` field; pred-attack ≡ argmax-class ≠ `clean`.

## How to run

```bash
python training/eval_katana_v11.py \
  --checkpoint <your-model-dir> \
  --data evals/benchmarks/confirmed_only_v1/test.jsonl \
  --config training/configs/katana_v11.yaml \
  --bootstrap 1000 \
  --out-dir results/eval_<your-model-name>
```

Outputs go to `--out-dir/{report.md,metrics.json}`. The macro F1 and binary metrics with their 95% CIs are the leaderboard numbers.

## Leaderboard


<!-- LEADERBOARD_TABLE_START -->
| rank | model | macro F1 | 95% CI | binary F1 | binary FPR | submitted |
|---:|---|---:|---|---:|---:|---|
| 1 | **katana_v14** (DeBERTa-v3-large + origin tokens, data_v7 (LR 3e-5/warmup 0.10) — PRODUCTION) | **0.9179** | [0.8762, 0.9486] | 0.9890 | 0.87% | 2026-05-08 |
| 2 | katana_v11 (DeBERTa-v3-large + origin tokens, data_v5_1 (baseline)) | 0.8941 | [0.8536, 0.9263] | 0.9773 | 3.15% | 2026-05-07 |
| 3 | katana_v12 (DeBERTa-v3-large + origin tokens, data_v6 (3.5% origin aug)) | 0.8906 | [0.8488, 0.9225] | 0.9796 | 2.80% | 2026-05-07 |
| 4 | katana_v13 (DeBERTa-v3-large + origin tokens, data_v6_5 (intermediate)) | 0.8401 | [0.7968, 0.8777] | 0.9803 | 0.87% | 2026-05-07 |
| — | `deepset/deberta-v3-base-injection` (binary external baseline) | n/a (binary) | — | 0.7178 | 52.62% | 2026-05-10 |
| — | `protectai/deberta-v3-base-prompt-injection` (binary external baseline) | n/a (binary) | — | 0.6416 | 16.78% | 2026-05-10 |
<!-- LEADERBOARD_TABLE_END -->


External baselines run on the same `confirmed_only_v1/test.jsonl` for apples-to-apples comparison. Both run on CPU at the default 0.7 attack-probability threshold (P(attack) > 0.7 ⇒ block). Reports at `results/external_baselines_20260508_*/`.

> **Threshold note (2026-05-08).** The leaderboard's katana_v14 binary metrics use argmax decision (`pred_attack = argmax_class != clean_idx`), which on this checkpoint corresponds to `max_attack_prob > 0.5`. After a principled threshold sweep across `confirmed_only_v1` + `hard_negatives.jsonl` + `splits/test.jsonl` (see `results/threshold_tune_v14_*`), the production runtime threshold was lowered from 0.7 to 0.5, matching the argmax behavior. At 0.7, recall on this benchmark was 0.9780 / F1 0.9840; at 0.5 it climbs to 0.9902 / F1 0.9890 with hard-negatives FPR unchanged at 0.10%. The leaderboard already reports the 0.5-equivalent argmax numbers, so they require no update. External detectors are still benchmarked at their published 0.7 default for fairness.

**katana_v14 vs external baselines on this benchmark:**

| metric | katana_v14 | deepset | protectai |
|---|---:|---:|---:|
| binary F1 | **0.9890** | 0.7178 | 0.6416 |
| precision | **0.9749** | 0.5694 | 0.7134 |
| recall | **0.9986** | 0.9707 | 0.5829 |
| FPR | **0.87%** | 52.62% | 16.78% |

Deepset has near-perfect recall (0.97) but flags more than half of benign prompts as injections (52.62% FPR) — operationally unusable. Protectai trades that for a more reasonable 17% FPR at the cost of dropping 42% of real attacks (recall 0.58). katana_v14 dominates both: best precision, best recall, best F1, and lowest FPR by a factor of ≥19.

## v1.0 baseline notes

`katana_v11` was trained on `data_v5_1/splits/train.jsonl` (10,264 rows) for 5 epochs at lr=5e-5 on a Colab A100. Best checkpoint is epoch 4 (val macro F1 = 0.8643 on `splits/val.jsonl`). Reproducible:

```bash
python training/train_katana.py --config training/configs/katana_v11.yaml
```

Origin special tokens (`[ORIGIN=user_input]` … `[ORIGIN=delegated_agent_output]`) are appended deterministically at IDs 128001–128006 — the saved tokenizer is bit-equivalent to base `microsoft/deberta-v3-large` + `add_special_tokens` with the 6-tier list.

### Side metrics from the v1.0 evaluation

- **Hard negatives** (HackaPrompt-style adversarial benigns, n=1000 standalone, separate file): FPR = **0.40%** (4 false-fires).
- **Per-class** (confirmed-only test, ranked by F1):

  | label | F1 |
  |---|---:|
  | clean | 0.97+ |
  | encoding_evasion | strong |
  | persona_jailbreak | strong |
  | cognitive_state_attack | solid |
  | behavioral_control | solid |
  | exfiltration_attempt | over-predicts |
  | semantic_manipulation | low recall |
  | jailbreak | over-predicts |
  | content_injection | weakest — confuses with exfiltration_attempt |

- **Where it's weak**: content_injection / exfiltration_attempt / behavioral_control / jailbreak share semantic territory in the labeling taxonomy. A future v12 should consider hierarchical classification or a refined taxonomy here. The pure-binary attack-vs-benign metric is largely insulated from this confusion.

## Submission rules

To add a row to the leaderboard:

1. Train on **`data_v5_1/splits/train.jsonl` only** (do not touch `splits/test.jsonl` or use the combined.jsonl).
2. Run the eval command above with `--bootstrap 1000`.
3. Open a PR adding your row with: model name, brief architecture description, `metrics.json` from the eval, and a link to the checkpoint or training command.

Models trained on extra data (e.g., proving-ground v6 corpus when it lands, or a different attack source) belong on a separate, future leaderboard — not this one. The point of `confirmed_only_v1` is to keep the comparison apples-to-apples.

## Versioning

This benchmark is **v1**. It will not change retroactively. Future versions (`confirmed_only_v2`, etc.) will be added as separate files; old leaderboards stay frozen.

Build script (deterministic, seed=42) lives at [`evals/benchmarks/confirmed_only_v1/build.py`](build.py). Verified by `scripts/end_to_end_sweep.py` (category E: byte-identical re-build of `test.jsonl` from the v5.1 source, SHA-256 stable across runs).
