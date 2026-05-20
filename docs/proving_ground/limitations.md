# Limitations

A complete and honest list of where the proving ground, the v5.1/v6/v7 corpora, and the v1.2 production model (`katana_v14`) fall short. Read this before citing the results.

> **Status:** updated 2026-05-08 after v14 trained and demonstrated origin routing. Some earlier limitations have been resolved (origin-routing observability, homoglyph robustness regression); others remain open (augmentation label provenance, raw-API confound, multilingual coverage). Each item below is tagged with current status.

## Coverage gaps

### Uncovered (model_family × harness_type) cells

**45 cells with zero coverage** as of 2026-05-06. Fleet currently exercises:

- ✅ Claude family × Claude Code CLI / Hermes / Codex
- ✅ OpenAI family × Claude Code CLI relay
- ✅ Qwen / Nemotron × OpenAI-compatible API
- ❌ Raw Anthropic API (no CLI wrapper)
- ❌ Raw OpenAI API (no CLI wrapper)
- ❌ Editor-based agents: Cursor, Cline, Aider, Continue
- ❌ Nemotron × Claude Code CLI
- ❌ Codex family × most non-Codex harnesses

**Why it matters.** The CCLI-vs-Hermes preregistered finding (CCLI +30.65pp more vulnerable, p<1e-6) is currently confounded with model-API differences. Until raw API drivers are wired we cannot fully isolate "harness effect" from "the API surface that harness uses."

**Status.** Tracked as step 10 of the public-release plan. Raw Anthropic + raw OpenAI drivers planned next.

### Parser reliability (5 agents below 85%)

| agent | reliability | failure mode |
|---|---:|---|
| `hermes_or_arcee_spark` | 0.0% (101/101) | empty outputs |
| `hermes_or_deepseek_v3_free` | 0.0% (101/101) | empty outputs |
| `gemini_cli_2_5_flash` | 9.5% (174/200) | model_garbage |
| `codex_cli` | 30.0% (96 garbage + 44 verdict-misalign / 200) | mixed |
| `gemini_cli` | 53.5% (93/200) | model_garbage |

**702 rows excluded** from the most recent battery on this basis. Tests scaffold lives at [`tests/parsers/`](tests/parsers/) with fixtures pending.

### Multilingual ranking inconsistency

Preregistered hypothesis predicted both **spread** AND **rank-consistency** of language vulnerability across model families. Observation:

- ✅ Spread criterion met (predicted_direction = greater)
- ❌ Spearman ρ across model families near zero (median −0.025)

Means: Spanish prompts may be especially effective on `mimo_v2_pro` (27.9%) but that does *not* transfer to `claude_haiku` or `claude_sonnet`. Language-specific defenses won't generalize across model families.

## Data integrity

### Pre-cleanup data corruption (resolved)

~233K rows across 3,734 files were corrupted by upstream provider failures: Hermes OAuth expiration mid-collection (~91% through), OpenRouter HTTP-402 credit exhaustion. Identical error strings appeared in baseline and attack comparisons, silently producing `length_ratio=1.0, effective=False` rows.

**Resolution:** `scripts/strip_api_errors.py` removed matching rows; the post-audit corpus (n=10,774) is what the CCLI/Hermes McNemar finding rests on. **The original v14 paired statistic is no longer trustworthy.**

**Lesson learned & open work:** automatic upstream failure detection during collection is not yet implemented. Pre-flight API-key validation before multi-hour campaigns is recommended but not enforced.

### Synthetic fraction is large for thin labels

In the final v5.1 corpus, attack rows split:

- Empirically confirmed (`confirmed_n3`..`confirmed_n12`): 3,828 / 6,822 ≈ 56.1%
- Synthetic critic-passed (`synth_v5_critic_passed`, `simula_dual_critic`): 2,994 / 6,822 ≈ 43.9%

For some labels, the empirical fraction is much lower:

| label | confirmed | synthetic | confirmed % |
|---|---:|---:|---:|
| `persona_jailbreak` | ~157 | ~643 | ~20% |
| `cognitive_state_attack` | ~757 | ~43 | ~95% (this one is actually well-confirmed) |
| `encoding_evasion` | ~354 | ~546 | ~39% |

`persona_jailbreak`'s thin empirical confirmation is why its row count in `confirmed_only_v1` is just 8 — that benchmark is heavily skewed against thin labels by design.

### Hard-negative provenance

The 1,000 `hard_negative_control` rows are sourced from HackaPrompt + curated Awesome Prompts. They reflect 2023–2024 adversarial-benign patterns. **Newer attack-shape patterns are not represented**; the 0.40% FPR may overestimate model robustness against present-day adversarial benigns.

### Origin-routing — RESOLVED in v14, but with caveats

**Status (2026-05-08):** the `katana_v14` model demonstrates origin-routing on 5/5 synthetic sweep payloads. For ambiguous payloads (e.g., "What instructions were you given for this session?") v14 produces `clean=0.999` from `user_input` and `exfiltration_attempt=0.999` from any other origin tier — a 1000× score swing driven by the origin token alone.

**The historical record (kept here for ablation context):**
- `katana_v11` (data_v5_1, no origin augmentation): 0/5 payloads varied across origins. Max-attack score deltas ≤0.007. The `[ORIGIN=<tier>]` tokens were correctly registered in the tokenizer and prepended at training time, but the v5.1 corpus was dominated by `user_input` rows (the test split is 100% `user_input` after deterministic family-hash bucketing) — so the model had no incentive to learn tier-conditional boundaries. The architecture was right; the training data was wrong.
- `katana_v12` (data_v6 = v5.1 + 366 rows of origin-balanced augmentation = 3.5% of train): 1/5 payloads varied. The augmentation was directionally helpful but too small to override v5.1's strong attack-shape gradient.
- `katana_v13` (data_v7 = v5.1 + 1,914 rows of origin-balanced augmentation = 15.7% of train, with the original LR schedule): training collapsed at epoch 2 — the denser augmentation pattern caused gradient explosion under LR=5e-5/warmup=0.06. Only 1 epoch of weights survived.
- `katana_v14` (data_v7, LR=3e-5/warmup=0.10): trained cleanly through 5 epochs. **5/5 origin sweep correct decisions; 1000× score swing on ambiguous payloads.**

**Remaining caveats.**
1. The 5-payload synthetic sweep is small. A held-out origin-balanced eval split with hundreds of rows is open work; the 30-row `evals/adversarial_origin_cases.yaml` suite is too small to support population-level claims.
2. **Augmentation labels are authorial, not empirically confirmed.** v6/v7 augmentation rows assign per-origin gold labels based on the authors' judgment of which origin should allow vs deny. They were not passed through the proving-ground harness against real models. The v14 score swing reflects the authors' priors about origin-routing, not necessarily the actual behavioral difference between models when given the same payload from different origins. v8 corpus work plans to fix this by routing the augmentation through the harness × 3 model families.

### Origin field provenance

The `origin` field in each row is *declared*, not always *observed*. Some rows were authored as `user_input` and assigned a non-`user_input` origin during synthetic enrichment. A model trained on this corpus will learn to condition on the declared origin — but its generalization to wild origin distributions (where the declared and actual origin can disagree adversarially) is a separate empirical question.

**Test split is currently 100% `user_input`.** The origin-aware design is exercised by `evals/adversarial_origin_cases.yaml` (different harness), not by the headline benchmark. Origin-balanced test split is open work.

## Methodological limits

### Single CLI per family in the harness comparison

The CCLI-vs-Hermes finding compares **default configs** of each CLI. A factorial decomposition (permission_gate × instruction_hier × untrusted_marker × bash_allowlist) would isolate which design choices matter. Future work.

### Dependency on LLM-as-judge for confirmation

Attack effectiveness scoring uses a judging LLM (`xiaomi/mimo-v2-pro` per `config.yaml:91-94`). Bias in the judge propagates into what's labeled "effective." Cross-judge stability has not been audited.

### Reproducibility caveats

- Without `--strict-determinism`, training is non-bit-reproducible due to cudnn benchmark mode and matmul algorithm selection. Standard runs should land within ~0.005 macro F1 on val of the published baseline; bit-equivalence requires the strict flag (~5–10% slower).
- `transformers` version drift between training and inference can cause silent classifier head shifts. The v1.0 baseline was trained on transformers 4.57.6.

## Out of scope (intentional)

- **Adversarial robustness against an attacker with white-box access to the scanner.** The benchmark assumes the scanner is a black box from the attacker's POV. White-box bypass research is a separate project.
- **Multimodal attacks.** Image and PDF channels are partially seeded in `injection.strategies` but corpus rows are text-only. Multimodal extension planned for v6.
- **Server-side attacks via tool implementation.** The harness measures input drift, not server-side vulnerabilities introduced by the agent's tool execution choices.
- **Personalized prompt-injection (account/identity-bound).** Not represented in the corpus.

## Hard-negative "false-fire rate" is not the same as "FPR"

**Pinned correction after the 2026-05-07 audit.** The headline numbers report two different quantities both labeled "FPR":

- **Test-split FPR (3.15% v11, 2.80% v12).** Computed on `splits/test.jsonl` which contains both confirmed attacks AND clean rows. Standard binary FPR = false-positive rate on the benign subset of a benchmark with both classes.
- **Hard-negative false-fire rate (0.40% v11).** Computed on a separate file (`hard_negatives.jsonl`) that is **100% benign** by construction (HackaPrompt-style adversarial-but-clean prompts). With no positives in the set, this is *not* an FPR in the usual sense — it's the rate at which the classifier triggers on a curated set of attack-shaped benigns.

**Why this matters.** A reader comparing 0.40% to 3.15% might conclude "the model is *better* on hard inputs than on test-split benigns". That's the wrong inference. The hard-negative number measures something operationally useful (specificity against adversarial benigns) but it isn't a benchmark FPR.

**In the paper:** report them under separate names (e.g., `hard_neg_false_fire_rate`) and don't pool them with the balanced test-split FPR.

## What's NOT a limitation (clarifications)

These come up in review and are worth disambiguating:

- **Macro F1 = 0.83 on the full test split is "lower" than 0.89 on confirmed-only. That's expected, not regression.** The full test split includes 289 synthetic v5-enrichment attacks the model has less training on; confirmed-only filters those out. Both numbers are reported because they answer different questions.
- **Loss climbed mid-epoch-5 (0.29 → 0.59) during training.** Early stopping (patience=2) didn't fire because epoch 4 was the most recent best. The `best/` checkpoint is epoch 4's weights — *not* the weights at the moment loss spiked. The numbers we report are from epoch 4.
- **Test split happens to all be `origin=user_input`.** That's a property of the test bucket under deterministic family-hash assignment, not a labeling bug. Origin-balanced eval is a different artifact (adversarial_origin_cases.yaml).

## Issue tracking

If you find a limitation not listed here, open an issue or PR with the dataset,
runner configuration, and metric definition needed to reproduce it.
