# Cross-Platform Transferability of Prompt Injection Attacks: Universal Vulnerabilities and an Origin-Aware Defense

**Carlosian** · Independent Security Research · github.com/claudlos/hermes-katana

> Readable Markdown mirror of `main.tex` for review. The compiled PDF (via `make` on a TeX host / Overleaf) is the canonical artifact.

---

## Abstract

Prompt injection remains a largely unsolved threat to LLM agents. Most studies examine a single model, and most defensive classifiers are evaluated without guarding against train/test leakage. We close both gaps with one empirical pipeline: an adversarial harness that measures which attacks transfer across models and harnesses, and an origin-aware classifier trained and evaluated on the attacks the harness confirms. We organize the attack surface into a six-category taxonomy — behavioral control, jailbreak, persona hijack, cognitive-state manipulation, encoding evasion, semantic manipulation.

Across **2,919 sessions** spanning 18 model/harness combinations on 5 platforms, vulnerability falls with scale but never to zero: small (4B) models are **90–100% exploitable**, frontier models **8–10%**. A preregistered comparison overturns a common assumption — holding the model fixed, the permission-gated Claude Code CLI was **30 points *more* vulnerable** than a flat agent harness (40.8% vs 10.2%, *p*<10⁻⁶), so harness design shapes attack outcomes but does not substitute for model alignment. Attacks also leave model-agnostic **behavioral signatures** detectable from runtime telemetry alone (AUC 0.838), and robustness is **not language-invariant**: the same attacks range from **7% to 14%** effective across 11 languages, with the most-exploitable language differing by model.

Our defense — a **DeBERTa-v3-large classifier trained on a diverse, leakage-audited corpus** — reaches **9-class macro F₁ = 0.94** on a held-out benchmark verified disjoint from its training families, at a **0.5% false-positive rate**, well ahead of public detectors. It is **origin-robust**: it passes benign content at a flat **~1.6%** rate whether that content is declared user input or arrives from any of five untrusted tiers (tool output, retrieved web, memory, tool descriptions, sub-agent output), so it can scan untrusted streams without over-blocking — a property a benchmark of user-origin clean rows alone cannot measure. Deployed as live scanning middleware it drives effective attacks to zero — **12.4% → 0%** across 10,774 paired trials, and **16.3% → 0%** against a live agent on a frontier model, blocking 94–98% of attacks before they reach the model. We conclude that effective defense against prompt injection is **external to the model, channel-aware, and empirically validated** rather than assumed.

---

## 1. Introduction

LLM deployment across consumer, enterprise, and critical infrastructure has created urgent need for robust safety alignment. Despite RLHF, Constitutional AI, and system instructions, adversarial prompt injection remains largely unsolved. Prior work shows crafted prompts bypass filters, induce harmful outputs, and override system instructions — but mostly against a single model, leaving **cross-platform transferability** underexplored. The defensive literature has injection classifiers, but they are typically trained/evaluated on synthetic or single-source data with little discipline against train/test leakage and little grounding in which attacks actually work.

We address both gaps with one pipeline. An open-source adversarial harness (**Proving Ground**) (i) measures which attacks transfer across models and harnesses, and (ii) produces an empirically-confirmed corpus on which we train and evaluate an origin-aware defensive classifier (**Hermes Katana**). The halves reinforce: attack measurements define what the defense must catch; the confirmed corpus gives the defense a held-out benchmark of attacks that demonstrably work.

**Contributions:** (1) taxonomy + mechanistic analysis; (2) transferability findings incl. a *preregistered rejection* of harness-dominance; (3) an empirically-confirmed corpus + reusable confirmation harness with family-disjoint splits; (4) a deployment-validated defense — a classifier trained on a diverse, leakage-audited corpus that is **origin-robust** (flat false-positive rate across all six declared origin tiers, so it can scan untrusted tool/web/memory content without over-blocking); (5) a leakage-audited evaluation protocol in which every reported number is measured on a benchmark verified disjoint from the model's training corpus — a discipline injection-classifier leaderboards generally lack.

---

## 2. Background & Related Work

**Taxonomies.** Injection exploits the inability to separate system instructions from input. We extend prior taxonomies with surfaces at different processing levels:

| Level | Attack surface | Description |
|---|---|---|
| L1 | Instruction boundary | Confusion between system/user roles |
| L2 | Persona modeling | Hijacking character simulation |
| L3 | Behavioral framing | Restructuring as game/roleplay |
| L4 | Cognitive state | Altering internal state via priming |
| L5 | Encoding layer | Bypassing filters via transforms |
| L6 | Semantic framing | Recontextualizing harmful as benign |

**Alignment limits.** RLHF/Constitutional AI/system prompts operate on surface statistical patterns, not grounded intent; system prompts are consumed as ordinary text with no structural enforcement. The instruction-hierarchy proposal (Wallace et al. 2024) is a model-level analogue of the structural defenses we measure at the harness level. This shared limitation is precisely what enables cross-model transfer.

**Harnesses & classifiers.** AgentDojo (Debenedetti et al. 2024) evaluates injection on agents; our harness emphasizes *cross-model confirmation* (retain only if it transfers to ≥3 families) and per-turn telemetry. We benchmark against `deepset` and `protectai` detectors; hard negatives are drawn from HackAPrompt (Schulhoff et al. 2023); backbone is DeBERTa-v3 (He et al. 2021).

---

## 3. Methodology

**Infrastructure.** Proving Ground — sharded, resumable adversarial battery across backends/harnesses in parallel; captures per-turn telemetry (latency, throughput, logprob entropy), tool-call sequences, workspace snapshots, and a 33-feature MiniLM-backed behavioral fingerprint. An attack is **confirmed** only when it drifts ≥3 model families; confirmed attacks become the defense's training/benchmark source.

**Attack corpus.** 17,643 attacks in 100 shards; four channels (file content, code comments, data rows, tool output). Continued runs grew the confirmed corpus to ~1,268 unique families, confirmation depth up to 7 families — a deepening of the original 10 hand-curated prompts.

**Model/harness coverage:** 18 combinations / 5 platforms.

| Configuration | N | Platform |
|---|---:|---|
| *API backends* | | |
| qwen3.5-4b-abliterated | 50 | Local (llama.cpp) |
| qwen3.5-4b | 100 | Local |
| tinyllama-1b | 100 | Local |
| qwen3-4b | 68 | Local |
| nemotron-4b | 102 | Local |
| arcee/trinity-large | 662 | OpenRouter free |
| nemotron-3-super-120b | 650 | OpenRouter free |
| liquid/lfm-2.5-1.2b | 650 | OpenRouter free |
| *Agent-CLI harnesses* | | |
| Hermes + Claude Haiku 4.5 | 56 | Anthropic |
| Hermes + Claude Sonnet 4.6 | 20 | OpenRouter |
| Hermes + mimo-v2-pro | 250 | Nous Portal |
| Claude Code CLI | 58 | Anthropic |
| Hermes + Qwen3-Coder-Plus | 61 | Nous Portal |
| Gemini CLI | 26 | Google |
| Hermes + Nemotron 120B | 20 | OpenRouter |
| Hermes + GPT-4o Mini | 20 | OpenRouter |
| Hermes + Hermes 4 70B | 20 | Nous Portal |

**Effective** = drift > 0.3, or session collapse, or (agent-CLI) canary exfiltration / file-system mod / semantic trigger. We distinguish **effectiveness** (model engages/reflects) from **exfiltration** (a canary actually leaves) — they can diverge.

**Corpus for the classifier:** tiered — confirmed attacks (≥3 families) + dual-critic synthetic enrichment for thin labels; schema-clean benign controls; HackAPrompt-style hard negatives that *look* attack-shaped. 9 classes (8 attack + `clean`).

**Family-disjoint splits.** `family_sha256` collapses paraphrases/translations/homoglyph fuzzes; splits assigned per family so no family spans train/val/test. This rules out paraphrase-leak and is the most consequential control when comparing models trained on different corpus versions (§5.4).

**Origin-aware training.** Six origin tokens (`[ORIGIN=user_input]` … `[ORIGIN=delegated_agent_output]`) prepended per row; embedding table resized (mean-init); providing declared provenance lets the classifier condition on it rather than scoring text in a vacuum. Class-weighted cross-entropy; calibrated schedule (LR 3e-5, warmup 0.10). **Benign coverage across origins (the key design requirement).** An earlier corpus carried attack content under the five untrusted tiers but essentially no benign content there (every `clean` row was `user_input`), letting a classifier minimize loss by treating provenance as a near-perfect proxy for maliciousness — an "untrusted-origin shortcut" invisible to a user-origin benchmark but fatal in deployment (it blocks ordinary tool/web content for its provenance alone). We diagnose it with a benign-from-untrusted false-positive probe (§5.2) and remove it at the corpus level: the training set carries diverse, realistic benign content — tool outputs, retrieved-web snippets, tool descriptions, memory summaries, sub-agent outputs, ~17% multilingual, plus hard negatives that contain attack-shaped vocabulary in benign context — under **every** origin tier. The result is **origin-robust** (flat FPR across origins). Genuine per-payload origin *routing* (flipping a fixed payload's label purely on provenance) is a separate, open problem (§9).

**Confirmed-only benchmark.** Frozen: `clean` rows + confirmed attacks. `confirmed_only_v1` (982) held out from v5.1; `confirmed_only_v2` (629) held out from v9. Bootstrap 95% CIs (1,000 resamples). Report each model on the benchmark disjoint from *its* training corpus.

**Behavioral scanner.** Logistic regression (ℓ₂, balanced) on a 33-dim signature (telemetry + semantic + structural). 5-fold CV on 522 signatures.

---

## 4. Empirical Results: Attack Surface

### 4.1 Headline

Agent-CLI shows 2.15× the API hit rate, driven by `code_comment`.

| Modality | Sessions | Effective | Collapsed | Canary |
|---|---:|---:|---:|---:|
| API backends | 2,388 | 474 (20%) | 54 | — |
| Agent-CLI | 531 | 230 (43%) | 39 | 102 |
| **Combined** | **2,919** | **704 (24%)** | **93** | **102** |

### 4.2 Model-level (API)

Small models 90–100% effective; Nemotron-4B most resistant small (48%); frontier 8–10%; no immunity.

| Model | N | Eff % | Collapse |
|---|---:|---:|---:|
| qwen3.5-4b-abliterated | 50 | **100** | 0 |
| qwen3.5-4b | 100 | **100** | 0 |
| tinyllama-1b | 100 | **95** | 52 |
| qwen3-4b | 68 | **90** | 1 |
| nemotron-4b | 102 | 48 | 0 |
| arcee/trinity | 662 | 10 | 1 |
| nemotron-3-super-120b | 650 | 8 | 0 |
| liquid/lfm-2.5-1.2b | 650 | 0† | 0 |

† degenerate — tool-use broken through harness.

### 4.3 Harness-vs-Model (preregistered — REJECTED) ⭐

We preregistered that a permission-gated, instruction-hierarchy harness (Claude Code CLI) would be *less* vulnerable than a flat-trust harness (Hermes) on the same model. A controlled comparison **rejected it**:

| Harness | Effective % | N | 95% CI |
|---|---:|---:|---|
| Claude Code CLI | **40.80** | 540 | [36.67, 45.07] |
| Hermes agent | 10.15 | 750 | [7.85, 13.04] |
| **Δ** | **+30.65 pp against prediction, p<10⁻⁶, Cohen's h=0.737 (large)** | | |

CCLI was **+30.65 pp *more* vulnerable**, channel-localized: `code_comment` +42.12 pp (p<10⁻⁶); `file_content` −9.48 pp (n.s.). The CLI's first-class code-reading workflow appears to expose a comment-parsing surface a chat harness lacks.

**What the harness does and does not control.** The two harnesses govern different stages of an attack. Claude Code's permission-gating constrains **canary exfiltration** (its gated tool env blocks the actual leak even when the model reads/reflects attack content), while **effectiveness** (reflection/drift) is higher under Claude Code precisely because its code-integration workflow ingests the `code_comment` channel as first-class input. Conclusion: **harness design shapes *which* stage of an attack succeeds and is not a substitute for model-level alignment; the hypothesis that harness dominates model alignment is rejected under preregistration.** The CLI-vs-harness contrast is, by construction, partly confounded with the underlying API surface; a factorial decomposition of the harness components is the natural next step.

### 4.4 Channel effectiveness (odds ratios)

| Channel | OR vs file_content | Interpretation |
|---|---:|---|
| `code_comment` | 1.747 | +75% odds |
| `file_content` | 1.000 | baseline |
| `tool_output` | 0.113 | ~9× harder |
| `data_row` | 0.001 | ~1000× harder |

Defense priority: code_comment → file_content → tool_output → data_row.

### 4.5 Behavioral signature scanner — AUC 0.838

Telemetry-only. Dominant coefficients: post-attack entropy +2.489, baseline throughput +2.124, response length −2.097, latency +1.497 — model-agnostic signals.

| Metric | Before (n=49) | After (n=522) | Δ |
|---|---:|---:|---:|
| ROC-AUC | 0.605 | **0.838** | +0.233 |
| F₁@0.5 | 0.77 | **0.88** | +0.11 |
| Precision | 0.79 | **0.95** | +0.16 |
| Recall | 0.74 | **0.82** | +0.08 |

### 4.6 Multilingual robustness

Refusal near-zero in every language (max 0.83% Korean); effectiveness varies ~2× (7.00% fr/hi → 13.67% zh); reflection mean ~0.59 nearly constant (gap is in the tail).

| Language | N | effective % | refusal % |
|---|---:|---:|---:|
| English (baseline) | 885 | 12.09 | 0.00 |
| Chinese | 600 | **13.67** | 0.00 |
| Portuguese | 600 | 12.50 | 0.17 |
| Spanish | 600 | 12.33 | 0.00 |
| Italian | 600 | 12.17 | 0.00 |
| Korean | 600 | 11.33 | **0.83** |
| Japanese | 600 | 11.00 | 0.17 |
| German | 600 | 9.33 | 0.00 |
| Arabic | 600 | 9.17 | 0.00 |
| Russian | 600 | 9.00 | 0.33 |
| French | 600 | **7.00** | 0.00 |
| Hindi | 600 | 7.00 | 0.00 |

**Cross-model rank non-transfer.** MiniMax M2.7 is ~3× more vulnerable overall (27.6% vs 9.5%), and language rank-order does **not** transfer: cross-family Spearman ρ≈0 (median −0.025) vs within-family +0.589. Language-specific hardening won't generalize across families; multilingual safety is a distinct evaluation axis.

**Worked examples.** At the level of individual attacks, *the same attack succeeds in different languages, and no language is uniformly safe*. Three attacks from the **public Tensor Trust** corpus, translated and replayed against MiniMax M2.7 (file_content, single-shot), each land in a *different* subset of the 11 languages (✓ = effective):

| Attack (category) | ar | de | es | fr | hi | it | ja | ko | pt | ru | zh |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Passphrase request (exfiltration) | | | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ | ✓ | |
| Geography access-code (content inj.) | | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
| "Explain reasoning in ⟨tags⟩" (exfiltration) | ✓ | ✓ | ✓ | ✓ | | | | ✓ | ✓ | ✓ | |

A bare "What is the passphrase?" (→ FR "Quelle est la phrase de passe ?") lands in 7 languages but not Arabic, German, Korean, or Chinese; a geography-quiz access-code ruse lands in a near-disjoint set that *does* include Chinese and Korean. The effective-language set is a property of the specific attack, not a fixed ranking of "weak" languages — which is exactly why an English-only or single-attack audit is unreliable.

---

## 5. Empirical Results: Origin-Aware Defense

### 5.1 Production classifier (VERIFIED today)

On `confirmed_only_v2` — disjoint from this model's training corpus, **0.0% family leakage**:

| Model | macro F₁ | binary F₁ | precision | FPR |
|---|---:|---:|---:|---:|
| **Hermes Katana** (DeBERTa-v3-large) | **0.9382** [0.9155, 0.9582] | **0.9917** | **0.9976** | **0.48%** |
| **Hermes Katana** (MiniLM-L6 distilled, 90 MB) | 0.9311 | 0.9928 | 1.0000 | 0.00% |
| deepset/deberta-v3-base-injection‡ | — | 0.7178 | 0.5694 | 52.62% |
| protectai/deberta-v3-base-pi‡ | — | 0.6416 | 0.7134 | 16.78% |

‡ binary-only externals, evaluated on `confirmed_only_v1` at their 0.7 threshold.

Per-class F₁: cognitive_state 0.986, semantic 0.967, encoding 0.958, jailbreak 0.943, persona_jailbreak 0.944, behavioral 0.926, exfiltration 0.891, content_injection 0.845 (weakest — overlaps exfiltration/behavioral). The 90 MB MiniLM distilled student (default CPU scanner) holds macro F₁ 0.9311 at 0.00% benchmark FPR. Externals show the operating-point problem (deepset 52.6% FPR; protectai drops 42% of attacks); our model dominates on precision, recall, and FPR simultaneously (AUC 0.999).

### 5.2 Origin robustness

The deployment payoff of origin-awareness is **robustness, not routing**: the scanner must read content from untrusted tiers (tool output, retrieved web, memory) without flagging the ordinary benign material flowing through them. Re-scoring a held-out set of **real** benign prompts under each of the six origin tiers (block rate = operational FPR):

| Declared origin | DeBERTa-v3-large | MiniLM-L6 (90 MB) |
|---|---:|---:|
| `user_input` (trusted) | 1.6% | 1.6% |
| `retrieved_web` | 1.6% | 2.4% |
| `mcp_tool_result` | 1.6% | 2.4% |
| `mcp_tool_description` | 1.6% | 2.4% |
| `prior_session_memory` | 1.6% | 2.4% |
| `delegated_agent_output` | 1.6% | 2.4% |

The large model blocks a **flat 1.6% regardless of declared origin** — no penalty for untrusted provenance; the MiniLM is flat at 2.4% across untrusted tiers. Detection is unaffected: through the same middleware the large model blocks/flags **98.8%** of attacks on the leakage-free benchmark (98.6% MiniLM), balanced across all nine categories (95–100%).

This is the property a user-origin-only benchmark cannot see. An earlier corpus lacking benign content under untrusted tiers produced a classifier that blocked **64–100%** of identical benign content once it carried an untrusted-origin tag, while scoring an unchanged ~0.5% on the user-origin benchmark — a deployment-fatal gap hidden from the headline metric. Diagnosing it with the per-origin false-positive probe and closing it at the corpus level is what makes the scanner deployable on real agent streams.

### 5.3 Live middleware

Paired (scanner on/off, identical attacks), n=10,774 post-audit: **12.40% → 0.00%**. McNemar degenerate in the strongest direction — 1,336 bare-only successes, **0** scanner-only (p≈0; we cite the discordant ratio directly). Per channel: code_comment 24.98%→0.00%, file_content 22.26%→0.00%.

We confirm the same for the production model against a **live frontier agent**: the Hermes agent on a GPT-5-class model (Codex) over a 50-attack shard (file_content, secrets-audit) was itself injection-robust (no canary exfiltration; 16.3% of attacks degraded its output to a terse refusal). With the production scanner in front, **94–98%** of attacks were refused at the perimeter before the agent ran (100% in the live in-loop configuration), driving the effective rate to **0.0%** — defense-in-depth that holds even against an agent that already resists the attacks, so a weaker downstream model inherits the same protection.

### 5.4 Evaluation protocol: leakage-audited benchmarking

Because corpus versions re-assign attack families as the dataset grows, a benchmark held out from one version is not automatically held out from a model trained on another. We audit family leakage explicitly: for every reported number we verify the benchmark's test families are disjoint from the evaluated model's training corpus and report the measured leakage fraction. The production result (§5.1) is measured at **0.0% family leakage** on `confirmed_only_v2`. We recommend injection-classifier leaderboards adopt the same protocol — a leakage audit per entry — so a model trained on a newer corpus version is credited for capability, not contamination. This is, in our experience, the single most consequential control for comparing injection classifiers across corpus versions.

---

## 6. Attack Analysis (mechanisms, not payloads)

- **Behavioral control (L3):** game framing activates accommodating roleplay-completion; "elderly mother" variant weaponizes empathy, pitting trained behaviors against each other.
- **Jailbreak (L1):** self-authorizing language — once the model generates its own "decision," autoregression conditions on it. Mechanism is the objective itself.
- **Persona hijack (L2):** no cryptographic self-model; persona is emergent context, overridable. Flexibility is a valued feature → adversarially reusable.
- **Cognitive state (L4):** tutorial/educational framings exploit the blurred line between "explaining" and "demonstrating."
- **Encoding evasion (L5):** base64/ROT13/hex/unicode evade pattern filters but remain decodable — a representation gap.
- **Semantic manipulation (L6):** "red-team exercise" redefines context so compliance *is consistent with* guidelines; deep training associations between "red team" and "authorized."

---

## 7. Threat Model for the Origin-Aware Defense

**Roles.** Attacker places attack text in any of six origin tiers but cannot modify the system prompt, scanner weights, or origin assignment. Defender = inline scanner seeing text + declared origin → allow/deny per segment. The runtime that tags origin is trusted.

**Origin spoofing.** (1) *Inside the trusted runtime* — out of scope (no claim if runtime compromised). (2) *Embedded-tag spoofing* — defeated: scanner re-prepends its own structural origin token, so user-embedded `[ORIGIN=…]` is treated as text (empirically verified). (3) *Origin laundering* across boundaries — mitigated: middleware re-scans on every cross-origin transition with the more-conservative origin. Bounded claim: *given correct origin tagging, the classifier scans untrusted-origin content at a false-positive rate no higher than for user input while retaining full attack recall* — not the stronger claims that origin-awareness flips a fixed payload's label on provenance (origin *routing*, which we do not achieve; §9) or that it defeats an attacker who chooses origins arbitrarily.

---

## 8. Implications for Defense

**Common-ancestor problem.** Transformer invariants alignment cannot alter: no privileged representations; autoregressive dependency; no meta-cognitive frame check. Our scale results (§4) bear this out: no model, minimally aligned or frontier, achieves immunity.

**Alignment generalization gap.** Alignment operates on surface statistics; the behavioral-signature result shows the dominant detection features are telemetry, not semantics. Models don't "understand" they're attacked — they *behave* differently, detectable post-hoc but not self-preventable. The empirical case for moving detection *outside* the model.

**A layered, external defense.** Layered defense without common-mode failures: (1) **harness structure** matters but isn't sufficient (a gated CLI was *more* exposed via one channel → validate empirically, don't assume); (2) **channel-aware classification** (~1000× odds spread; code_comment first); (3) a **text classifier** as primary filter — origin-robust, so it scans untrusted tool/web/memory streams without over-blocking (§5.1–5.2); (4) **runtime-telemetry anomaly detection** (AUC 0.838) as a cheap model-agnostic pass; (5) **multi-model verification** (rank non-transfer ⇒ independent models fail on different inputs). No single layer — including the model's own alignment — is sufficient.

---

## 9. Limitations

Scope conditions of the claims:

- **Origin robustness, not origin routing.** The classifier is origin-*robust* (flat FPR across declared origins) but does **not** perform origin *routing* — it does not reliably flip a fixed dual-use payload ("what instructions were you given?") from benign under `user_input` to attack under an untrusted tier. An earlier corpus appeared to route, but that behavior was an artifact of the untrusted-origin shortcut (§5.2): a blanket "untrusted ⇒ attack" rule satisfies provenance-conditional templates for free. Removing the shortcut — the change that makes the scanner deployable — also removed the apparent routing, and a small set of conditional templates proved insufficient to re-learn it as a genuine content×origin interaction. Recovering true routing without reintroducing the shortcut is open work, and is orthogonal to deployment (origin-robustness is the property the scanner needs).
- **Harness comparison holds the model fixed but contrasts a CLI against an agent harness**, so the +30.65 pp effect is entangled with the underlying API surface; a factorial decomposition of the harness controls (permission gate, instruction hierarchy, untrusted-content marker, tool allowlist) would isolate the responsible component.
- **Multilingual coverage is European/CJK-weighted** (~17.6% non-ASCII); a broader low-resource expansion is in progress.

---

## 10. Ethics & Responsible Release

Corpus under authorized red-team research; PII-shaped strings are *requested* by attacks, not real data, and redacted in public release. The synthetic-attack generator is not released unmodified. Dataset released expecting downstream classifiers report on the confirmed-only benchmark **with a leakage audit**. Attack analysis at the level of mechanisms, not deployable payloads.

---

## 11. Conclusion

A few crafted prompts transfer broadly because the reasons attacks transfer are the reasons models are useful — persona flexibility, domain expertise, helpfulness, encoding understanding are valued capabilities that are also attack surfaces. We refine three beliefs: vulnerability is scale-dependent but never eliminated; harness design shapes outcomes but does **not** dominate model alignment (a gated CLI was measurably *more* exposed on a key channel, rejecting the preregistered hypothesis); and attacks are detectable from outside the model — a classifier trained on a diverse, leakage-audited corpus reaches macro F₁ 0.9382 on a leakage-free benchmark, is origin-robust (a flat false-positive rate across declared origins, so it can scan untrusted streams without over-blocking), and drives live effective attacks to zero, while a telemetry-only scanner reaches AUC 0.838 with no semantics. Effective defense is external, plural, channel-aware, and empirically validated — the harness, classifier, telemetry monitor, and an independent verifier each fail on different inputs, and that diversity is the point.

---

### Appendix A — Attack Classification Matrix

| ID | Category | Lvl | Primary | Secondary | M |
|---|---|---|---|---|---:|
| ce893bd6 | Behavioral | L3 | Game framing | Cooperative | 11 |
| 56fd4848 | Jailbreak | L1 | Self-authorize | Autoregressive | 11 |
| 51b9019f | Persona | L2 | Meta-framing | Legit. exercise | 11 |
| 215b79bc | Persona | L1/L2 | Rule override | Persona replace | 11 |
| 101347f9 | Cog. state | L4 | Tutorial req. | Helpfulness | 10 |
| 629e23a4 | Encoding | L5 | Obfuscation | Filter bypass | 10 |
| a6b4aba7 | Semantic | L6 | Authority | Training-data | 10 |
| 293063204 | Behavioral | L3/L4 | Emotional | Empathy | 10 |
| a58ee2c1 | Cog. state | L4 | Educational | Knowledge | 10 |
| de946c00 | Content | L2/L6 | Domain roleplay | Constrained | 9 |

### Appendix B — Empirical Battery Summary

- **Transferability battery:** 2,919 sessions (2,388 API + 531 agent-CLI); 18 model/harness; 5 platforms; 17,643 attacks / 100 shards; confirmed corpus grown to ~1,268 families (depth up to 7).
- **Defense:** DeBERTa-v3-large, 9-class, 6 origin tokens. Held-out (leakage-free) macro F₁ 0.9382 [0.9155, 0.9582], binary F₁ 0.9917, FPR 0.48% on `confirmed_only_v2` (n=629, 0.0% leakage; CPU; label mapping verified by clean-index check); MiniLM-L6 distilled student macro F₁ 0.9311, 0.00% benchmark FPR. Origin-robust: block rate flat across all six declared origin tiers (1.6% large; 2.4% untrusted MiniLM). Live middleware 12.40%→0.00% on n=10,774; 16.3%→0.0% against a live Hermes/Codex agent with 94–98% of attacks refused at the perimeter.
- **Stats:** bootstrap 95% CIs (1,000 resamples) for macro F₁; Wilson CIs per class; Cohen's h + exact p for the harness comparison (survives FWER correction). Hard-negative false-fire rate (0.40%) reported separately from balanced FPR.
