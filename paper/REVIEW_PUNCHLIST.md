# Consolidated Editorial Punch-List

**Paper:** *Cross-Platform Transferability of Prompt Injection Attacks: Universal Vulnerabilities and an Origin-Aware Defense*
**Reviews merged:** consistency, typography, tables/figures, prose, structure, citations (6 independent passes, de-duplicated)
**Generated:** 2026-06-02 by a 6-reviewer + synthesis pass. *Numbers were verified against source, never altered.*

> **Status (applied 2026-06-02):** Tiers (i) + (ii) are APPLIED to `main.tex`, `extra_pkgs.tex`, `bibliography.bib`, and mirrored into `PAPER_READABLE.md` — formatting/typography (B1–B5), cross-reference + figure/citation fixes (A5, A8, C2, C3, C4), the ~10 prose splits (D1–D10), and the restructure (E2 §8→§9 merge, E3 trust-box, E8 signposts). A 3-way adversarial check confirmed: compiles cleanly, no empirical number changed, tex/md consistent. **Still OPEN (need author data/decisions):** the Tier (iii) conflicts — **A1, A2, A3, A4, A6, A7, A9, A10, C1** — plus rigor additions (E1, E4, E5, E6, E7). Pristine pre-edit backup: `backup-pre-edit-20260602-023847/`.

---

## 1. Executive Summary

This is genuine, well-hedged empirical research with sound internal arithmetic: the Fig 3 confusion matrix (n=629), every named per-class F1, Cohen's *h*=0.737, the +30.65 pp harness delta, and the live-middleware 1,336/10,774=12.40% all reproduce exactly. The paper is publication-shaped but not submission-ready. The highest-leverage work is **resolving ~6 cross-source data conflicts** (multilingual N, API session sum, harness-N unit, the v1-vs-v2 baseline comparison, a duplicated/garbled pair of bib entries, and a `.tex`↔`.md` contribution mismatch) — none of which have been altered, all flagged with both locations for the author to adjudicate. After that, the back third (§8/§9/§12) restates the same three findings three to four times and should be compressed to free space for the **missing reproducibility/compute/dataset-statistics material** a NeurIPS/S&P/USENIX PC will demand. Prose is strong but carries several 60–95-word run-ons (the abstract, §3.3, §7, §9) that a tired reviewer cannot parse in one pass. A clean mechanical pass (hyperref boxes, dead packages, thousands separators) can ship immediately; the data and citation conflicts gate everything else.

---

## 2. A. Must-Fix Correctness & Consistency

These are the most important. Numbers are flagged, never changed. "AUTHOR" = requires a data/claim decision; "MECHANICAL" = safe to apply once intent confirmed.

### A1. Multilingual sample size — prose vs. Table (flagged by 4 reviewers) — **AUTHOR**
- **Locations:** `main.tex:322` (prose) vs. `tab:multilingual` rows `main.tex:334–344`; mirrored in `PAPER_READABLE.md §4.6`.
- **Conflict:** Prose says "**4,400** attacks … **400 per language**, evenly across categories." Table lists **N=600 for every one of the 11 languages** (+ English baseline N=885). 11×600 = 6,600 (verified), not 4,400; and 600 ≠ 400/language. The 4,400 total matches neither 6,600 nor 6,600+885.
- **Decision:** State the true per-language N and total; clarify whether English (885) is one of the "11 languages" or a 12th baseline. Fix both `main.tex` and the `.md` mirror together.

### A2. API session accounting — Table 2 rows sum to 2,382, not 2,388 — **AUTHOR**
- **Locations:** `tab:models` rows `main.tex:122–129` vs. Table 3 `main.tex:185` and Appendix B `main.tex:599`.
- **Conflict:** API per-model N's (50+100+100+68+102+662+650+650) sum to **2,382** (verified). Table 3 and Appendix B both state API = **2,388** and Combined = **2,919**. Off by 6 sessions. (Agent-CLI rows sum to exactly 531, correct.)
- **Decision:** Either a Table 2 row N is mis-transcribed or 6 API sessions are un-itemized. Reconcile; do not change either number.

### A3. Harness-comparison N unit collision (Table 5 vs. Tables 2/3) — **AUTHOR**
- **Locations:** `tab:harness` `main.tex:232–233` (Claude Code CLI N=540, Hermes agent N=750, total 1,290) vs. Table 2 Claude Code CLI N=58 `main.tex:135` and Agent-CLI total 531 `main.tex:186/599`.
- **Conflict:** 1,290 cannot be drawn from 531 agent-CLI sessions, nor reconciled with the 58 in Table 2. The unit of "N" in Table 5 (per-attack trials?) is never distinguished from "N=sessions" elsewhere. This is **load-bearing for the headline preregistered claim**.
- **Decision:** Add one clause defining the Table 5 unit (e.g., "N = paired attack trials") and stating it is a separate dedicated run outside the 2,919-session battery — or reconcile the counts.

### A4. External-baseline comparison spans two benchmarks (v1 vs v2) — **AUTHOR**
- **Locations:** `tab:defense` caption `main.tex:406` says baselines "evaluated on the **same rows**" but its footnote `main.tex:419` says deepset/protectai used **`confirmed_only_v1`** (982 rows) while Katana uses **`confirmed_only_v2`** (629 rows). The §5.1 "dominates both" claim and Fig 4 bars inherit this.
- **Conflict:** Caption ("same rows") directly contradicts its own footnote. The head-to-head therefore compares scores on different row sets of different size.
- **Decision:** Either (a) re-run the two training-free externals on v2 so all numbers share one row set, or (b) state in caption + §5.1 that the comparison spans two benchmarks and add Katana's v1 numbers for a like-for-like row. "Dominates"/"same rows" cannot stand as written.

### A5. Duplicate / garbled citations — `perez2022ignore` & `zou2023universal`
- **`perez2022ignore` (`bibliography.bib:39–45`) — AUTHOR.** Its title is **byte-identical** to `schulhoff2023hackaprompt` (`:69–70`): *"Ignore This Title and HackAPrompt…"*, same EMNLP-2023 venue. Authors listed "Perez, Felix and Ribeiro, Ian"; cited at `main.tex:72` as the origin of "ignore previous instructions." That is almost certainly **Perez & Ribeiro 2022, "Ignore Previous Prompt: Attack Techniques for Language Models" (arXiv:2211.09527)** — a different paper, and the first name is likely **Fabio**, not Felix. Internal inconsistency: `year={2023}` but key says 2022.
  - **Decision:** Confirm intent, then replace the corrupted entry's title/venue/year with the real Perez & Ribeiro 2022 metadata. Verify the arXiv ID and given name before editing.
- **`zou2023universal` (`bibliography.bib:17–19`) — MECHANICAL (verify).** Author field garbled: "Wang, Zico" should be **Zifan Wang**; "Zico" is duplicated (also Kolter's middle name). Likely-correct list (verify vs arXiv:2307.15043): `Zou, Andy and Wang, Zifan and Carlini, Nicholas and Nasr, Milad and Kolter, J. Zico and Fredrikson, Matt`.
- **`schulhoff2023hackaprompt`** itself is correct; needs no change once `perez2022ignore` is fixed.
- **Positive:** all 13 keys resolve, no orphans, no dangling keys, brace-protection and plainnat author-year style are clean.

### A6. Contribution #5 differs between `.tex` and `.md` — **AUTHOR**
- **Locations:** `main.tex:64` vs. `PAPER_READABLE.md:25`.
- **Conflict:** `.tex` makes a *protocol* claim ("a family-disjoint evaluation protocol … a discipline leaderboards generally lack"). `.md` makes a *quantitative* claim: "family leakage **inflates apparent scores by several F₁ points**." That before/after leakage delta appears **nowhere in the body**.
- **Decision:** Pick the canonical wording and sync both files. If the `.md` quantitative wording wins, it needs a supporting measurement.

### A7. "Confirmation depth up to seven families" vs. Appendix A M=11 — **AUTHOR**
- **Locations:** `main.tex:107` and `:599` ("up to seven / up to 7 families") vs. `tab:atkmatrix` column M (`main.tex:580–589`), captioned "M = model families exploited," listing **M=11** for five attacks.
- **Conflict:** As written both read as the same quantity; 11 > 7. Likely two different definitions (Appendix A = transferability battery over 18 combos; "depth up to 7" = depth in the grown ~1,268-family confirmed corpus).
- **Decision:** Disambiguate wording, or reconcile if they are the same.

### A8. `\ref{sec:defense-results}` points to §5.1 but cites §5.2 content — **MECHANICAL (verify intent)**
- **Locations:** `\label{sec:defense-results}` sits on §5.1 (Production Classifier, `main.tex:399`). §5.2 (Origin Robustness) has **no label**. Three references to origin-robust material (`main.tex:160, 456, 543`) point to `sec:defense-results` (=§5.1) but that material lives in §5.2. The `.md` mirror correctly targets §5.2.
- **Fix:** Add `\label{sec:origin-robust}` after `main.tex:452` and repoint the three refs.

### A9. Six-category taxonomy vs. eight attack classes — **AUTHOR**
- **Locations:** Abstract/intro "six-category taxonomy" (`main.tex:41,45`) vs. classifier "nine classes (eight attack categories plus clean)" (`main.tex:150`) and confusion matrix's 8 labels — which add `content_injection` and `exfiltration_attempt`, absent from the six.
- **Decision:** Add a bridging sentence relating the conceptual six-category taxonomy (L1–L6) to the eight operational classifier classes.

### A10. Lower-severity numeric flags to confirm (do not change) — **AUTHOR**
- **98.8% vs 98.58%/98.6%:** §5.2 `main.tex:454` "blocks or flags 98.8%" vs. Table 10 binary recall 0.9858 and ROC caption "98.6%" (`fig_roc.tex:21`). Confirm 98.8% (block-or-flag) is intentionally distinct from binary recall.
- **16.3% of a "50-attack shard":** `main.tex:481`. 16.3%×50 = 8.15 (non-integer; 8/50=16.0%, 9/50=18.0%). State the real evaluable denominator.
- **0.845 label:** prose attributes F1=0.845 to `content_injection` (`main.tex:401`); in Fig 3 the 0.845 cell is `exfiltration_attempt`'s **recall**. Clarify the matrix shows recall, and confirm the min-F1 class.
- **Appendix A attack ID `293063204`** (`main.tex:583`): 9 digits, all-numeric, vs. the other 8-char hex prefixes. Verify it is not a truncated/typo'd hash prefix.

### A11. Acceptable rounding — NO ACTION
- Abstract "30 points" vs body +30.65 pp; "7%–14%" vs 7.00–13.67; "2.15×"; 12.4% vs 12.40%; "local NVIDIA" vs "Local (llama.cpp)". All within normal abstract rounding / cosmetic. **Fig 3 confusion matrix verified fully self-consistent** (n=629; all 9 per-class F1 match the `.md`).

---

## 3. B. Formatting & Typography (safe mechanical fixes)

- **B1. hyperref colored boxes (HIGH).** `extra_pkgs.tex:5` loads hyperref with no setup → green citation / red reference boxes in the PDF. **Fix:** add `\hypersetup{colorlinks=true, linkcolor=black, citecolor=<dark blue>, urlcolor=<dark blue>}` or `\hypersetup{hidelinks}`.
- **B2. Dead math code.** `main.tex:7–11` loads `physics` and declares `\p/\n/\B` paired delimiters never used; `\metric` macro (`main.tex:16`) unused. **Fix:** delete lines 7–11 and 16.
- **B3. Unused packages.** `extra_pkgs.tex` lines 1 (adjustbox), 2 (algorithm2e), 9 (xurl), 11 (graphicx), 13 (tabularray), 22 (nicefrac), 24 (`\UseTblrLibrary{booktabs}`) appear unused. **Fix:** remove (keep pifont/patterns/pgfplots/groupplots; keep graphicx/algorithm2e only if a raster figure/algorithm is planned). *Verify each before deleting.*
- **B4. Thousands separators.** Convert plain-comma to braced form (digits unchanged): `2,919`, `2,388`, `17,643`, optionally `1,000`.
- **B5. Spelling/markup nits.** "HackaPrompt" → "HackAPrompt" (`main.tex:97,150`). Wrap chart legend strings in `\texttt`/smallcaps. Standardize AUC styling and `\geq3` spacing. Make author GitHub URL a real `\url{}`.
- **B6. AUTHOR-decision typography:** abstract `F1≈0.94` vs body `0.9382` — keep loose or align; "33-feature" vs "33-dimensional" — pick one.

---

## 4. C. Tables & Figures

- **C1. Fig 2 right panel plots only 10 of 11 languages (HIGH) — AUTHOR.** The "across families" addplot has 10 coordinate pairs. **Fix:** supply the missing (Sonnet-rank, MiMo-rank) pair, or caption the drop. Do not fabricate.
- **C2. Fig 2 "MiMo" undefined + Sonnet axis untabulated — AUTHOR.** Axes say "rank on Sonnet"/"rank on MiMo" but no Sonnet multilingual table exists and "MiMo" is never defined. **Fix (mechanical):** relabel "rank on MiniMax M2.7" or add "(MiMo = MiniMax M2.7)". **(AUTHOR):** add the underlying Sonnet numbers or confirm the axis.
- **C3. Fig 4 renders the 0.48% FPR as "0" (MEDIUM).** Erases a headline result. **Fix:** annotate "0.48%" explicitly. **Design:** consider cutting Fig 4 — Table 10 + Fig 5 (ROC) already cover it.
- **C4. Bolding semantics.** Table 4 bold marks the **worst** (highest) effectiveness (inverts "bold=best"). `tab:multilingual` bold marks max-effective, min-effective (but not the tied 7.00 hi), and max-refusal — three meanings + broken tie. **Fix:** bold one quantity consistently, or caption the convention.
- **C5. Figure/table duplication.** Fig 1-left ≈ Table 4; Fig 1-right ≈ Table 6 (full duplication). **Fix:** align Fig 1-left model strings to Table 4 exactly (or cut); drop Table 6's "Interpretation" column.
- **C6.** = A4 (resolve once).
- **C7. Low-priority polish.** Table 10 mixes 4-decimal fractions with 2-decimal percents; Table 3 Canary "---" should be footnoted; add a recall column for the externals; add `[tbp]` hints to pull Figs 3–5 nearer §5.

---

## 5. D. Prose & Clarity Rewrites (BEFORE→AFTER; preserve every number/claim)

1. **Abstract final sentence (`:45`, HIGH)** — ~70-word run-on. Split into: middleware result → "It blocks 94–98%…" → conclusion.
2. **Abstract "It is origin-robust…" (`:45`, HIGH)** — ~75 words; split after the five-tier parenthetical.
3. **Abstract second sentence (`:43`, MED)** — split the preregistered result from its conclusion.
4. **§3.3 benign-coverage sentence (`:160`, HIGH)** — ~95 words, hardest in the methodology; break at the colon into three sentences.
5. **§7 Threat Model closing (`:520`, HIGH)** — ~75-word double-negative; split into "We do not make two stronger claims. …"
6. **§4.6 worked-examples (`:375`, HIGH)** — ~90 words; split the Tensor-Trust aside, then one sentence per attack trace.
7. **§9 Limitations harness sentence (`:543`, HIGH)** — ~70 words; trim and cross-reference §4.3.
8. **§5.2 first measurement sentence (`:454`, MED)** — separate production-model flat-rate from the MiniLM rate.
9. **§5.1 first result sentence (`:401`, MED)** — lift the long `confirmed_only_v2` appositive into its own sentence.
10. **§5.4 McNemar sentence (`:479`, MED)** — lead with 1,336 vs 0 before the "degenerate / undefined at zero discordance" jargon.

**Repetition map (AUTHOR to approve trims):**
- **Origin-robust / flat-FPR / scan-untrusted** stated ~6×: abstract (`:45`), contribution 4 (`:63`), §3.3 (`:160`), §5.2 (`:454`), §8 (`:536`), §12 (`:555`). → State fully **once in §5.2** + abstract headline; cross-reference the rest.
- **"external / plural / channel-aware / empirically validated"** closes abstract, §8, §12 near-verbatim → keep two.
- **"invisible to a user-origin benchmark"** in abstract, contribution 4, §3.3 → let §5.2 own it.

---

## 6. E. Structure, Rigor & Related Work

- **E1. Missing Reproducibility/Availability + compute/hyperparameters + dataset-statistics table (HIGH).** No availability statement; scattered LR/warmup but no epochs/batch/max_len/seed/GPU-hours/checkpoint/versions; no train/val/test rows-per-split/class/origin table; v5.1/v9 vs v1/v2 relationship unstated. **Fix:** add a Reproducibility paragraph + compute table + dataset-stats table.
- **E2. Repetitive back third (MED).** Merge §8 into the head of §9; tighten §12 to ~5 sentences. Recovers ~½ page for E1.
- **E3. Threat model misplaced (MED).** §7 follows §5, which already assumes it. Move a short roles/trust-boundary box before §5; keep the spoofing decomposition in place.
- **E4. Under-specified multiple-comparison correction + unverifiable preregistration (MED) — AUTHOR.** Name the correction (Bonferroni/Holm?, m, corrected α) and give a checkable preregistration artifact (committed hypotheses file + hash/date, or registry URL).
- **E5. Thin related work — 13 refs (HIGH).** Add a "Defenses" paragraph (spotlighting / data-prompt separation; StruQ/SecAlign) + a "Benchmarks" sentence (InjecAgent; Tensor Trust as benchmark) + embedding / LLM-as-judge detectors as baselines. Strengthens the novelty argument.
- **E6. Threats-to-validity gaps (MED) — AUTHOR.** Add: (a) sensitivity of "effective" to drift>0.3 (report at {0.2,0.3,0.4}); (b) circularity — the "≥3 families" criterion defines both transfer and the classifier's data; (c) single-annotator labeling / no IRR; (d) checkpoint dating.
- **E7. "Drives attacks to zero" / "dominates" over-claim (MED) — AUTHOR.** Carry the §7/§10 scope ("on confirmed families, under correct origin tagging") into the headline; lean on the ROC (AUC 0.999) rather than single-threshold bars.
- **E8. Signposting (LOW).** Add "(Sec X)" to each contribution bullet; consider moving §6 (Attack Analysis) ahead of the defense half.

---

## 7. Top 10 Quick Wins (value-to-effort)

1. hyperref boxes → one `\hypersetup` line (B1).
2. Fix `zou2023universal` author field (A5).
3. Annotate Fig 4 "0" → "0.48%" (C3).
4. Delete dead packages + math code (B2, B3).
5. Thousands-separator pass (B4).
6. Add `\label{sec:origin-robust}` + repoint 3 refs (A8).
7. Split the abstract's two run-on sentences (D1, D2).
8. HackaPrompt→HackAPrompt + define "MiMo" (B5, C2).
9. Trim the 6× origin-robust restatement to 1 + cross-refs (D repetition map).
10. Drop Table 6 "Interpretation" column / align Fig 1-left labels (C5).

---

## 8. Recommended Work Plan

### Tier (i) — Safe mechanical pass (no data/claim decisions)
B1 hyperref; B2 delete dead math; B3 remove unused packages; B4 thousands separators; B5 spelling/markup/URL; A5 `zou2023universal` author field (verify vs arXiv); A8 `\label`+repoint; C2 "MiMo"→"MiniMax M2.7"; C3 Fig 4 FPR annotation; C4 bolding; C5 label alignment / Table 6 trim; C7 polish; D1–D10 prose splits.

### Tier (ii) — Clarity rewrite pass (light author sign-off)
D repetition trims; E2 compress §8/§9/§12; E3 threat-model box before §5; E8 signposts / §6 reorder; B6 abstract `0.94` vs `0.938`.

### Tier (iii) — Substantive / structural (need author data or claim decisions)
**Blocking data/claim conflicts:** A1 multilingual N; A2 API session sum; A3 Table 5 N unit; A4/C6 external baseline v1-vs-v2; A5 `perez2022ignore` identity; A6 Contribution #5 sync; A7 "seven families" disambiguation; A9 six-vs-eight bridge; A10 numeric confirms; C1 Fig 2 11th point.
**Rigor additions a PC will demand:** E1 reproducibility/compute/dataset tables; E4 correction + preregistration artifact; E5 expand related work; E6 threats-to-validity; E7 scope the headline claims.
