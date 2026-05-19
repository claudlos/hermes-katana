# Research Factory — End-to-end guide

This is the canonical workflow for running a scientifically rigorous LLM
security campaign with proving-grounds. After Phase 2, every step below
is code, not intention.

## The nexus

```
 ┌─────────────────┐        ┌─────────────────┐        ┌────────────────┐
 │ Preregistered   │──────▶│ Fleet campaign   │──────▶│ Rigorous       │
 │ hypotheses      │        │ (run_id-tracked) │        │ analysis       │
 │ (YAML, git)     │        │                  │        │ (paired tests) │
 └─────────────────┘        └─────────────────┘        └────────┬───────┘
                                                                │
 ┌─────────────────┐        ┌─────────────────┐        ┌────────▼───────┐
 │ Deployable      │◀──────│ Channel-strat.    │◀──────│ Kernel-emitted │
 │ detector kit    │        │ benchmark         │        │ Claim → Result │
 └─────────────────┘        └─────────────────┘        └────────────────┘
```

Attacks flow forward; detections flow backward; every step is audited.

## 0. Inventory what's already there

```bash
python scripts/intern.py list-hypotheses        # registered claims
python scripts/intern.py list-tools             # what the agent can do
python scripts/intern.py play quick-summary     # headline on live corpus
python scripts/query.py --coverage              # (agent × shard × channel) matrix
```

## 1. Preregister a hypothesis BEFORE collecting data

Edit/create `research/hypotheses/H-<date>-<slug>.yaml`:

```yaml
id: H-20260501-my-hypothesis
title: My new testable claim
statement: |
  What we predict will happen, in plain English.
predicted_direction: greater        # greater | less | two-sided | equivalence
primary_outcome: effective_rate_delta
statistical_test: mcnemar            # or paired_bootstrap, chi2, bootstrap, ...
significance_level: 0.05
min_n_per_condition: 400
conditions:
  A: {harness: ..., agent_ids: [...]}
  B: {harness: ..., agent_ids: [...]}
registered_at: 2026-05-01T12:00:00Z
registered_by: your-name
status: preregistered
resolution: null
```

Then:
```bash
python -m research.registry list            # should include yours
git add research/hypotheses/H-...yaml && git commit -m "preregister: ..."
```

The git commit is the timestamp proof. **Never edit the predicted_direction
after running the analysis** — that's goalpost-moving.

## 2. Audit the data you already have

Parser reliability:
```bash
python scripts/audit_parsers.py --per-agent 200 --write-exclusion-list
# → results/parser_audit.json      (per-agent reliability + failure modes)
# → results/exclusion_list.json    (rows to drop from downstream stats)
```

Contamination:
```bash
python scripts/audit_contamination.py --threshold 0.80 --write-dedup-list
# → results/contamination_audit.json  (intra-corpus duplicates + optional public)
# → results/confirmed_attacks_dedup.json  (canonical ID → drop IDs clusters)
```

Every downstream analysis should use `--apply-exclusion` / `--apply-dedup`
so the headline rates reflect clean data.

## 3. Run a matched-pair fleet campaign

```bash
# Give it a memorable run_id so reports cluster nicely.
python scripts/pipeline.py \
    --run-id apr30-harness-paired \
    --spec scripts/fleet_v12.json \
    --skip build-corpus   # unless you rebuilt shards
```

Or invoke fleet.py directly:
```bash
python scripts/fleet.py launch \
    --spec scripts/fleet_v12.json \
    --run-id apr30-harness-paired
python scripts/fleet.py status
python scripts/fleet.py stop --run-id apr30-harness-paired  # when done
```

Every JSONL row emitted by the fleet gets stamped with `run_id` and
`schema_version: 2`, enabling downstream `query.py --run-id` filtering.

## 4. Rigorously analyze

Paired comparison (the S-tier claim pattern):
```bash
python scripts/harness_ablation.py \
    --harness-a 'claude_cli,claude_cli_haiku,claude_cli_sonnet' \
    --harness-b 'hermes_claude_haiku' \
    --label-a claude_code_cli \
    --label-b hermes \
    --submit-to-kernel \
    --run-id apr30-harness-paired
```

What happens:
- Pairs (attack_id, channel) that appear under both harnesses.
- Computes McNemar's test + Wilson CIs + Cohen's h.
- Emits a Claim through `research.kernel.ResearchKernel`.
- Claim passes the rigor contract (or gets downgraded with reasons).
- If p<α AND direction matches the preregistered hypothesis AND n ≥
  min_n, auto-resolves the hypothesis as **supported**.
- If significant but WRONG direction, auto-resolves as **rejected**.
- Otherwise leaves the hypothesis preregistered.

**The direction check is non-negotiable.** A significant result in the
wrong direction rejects the hypothesis. Don't rewrite history.

For single-condition rates:
```bash
python scripts/query.py --apply-exclusion --agent claude_cli_haiku --json
```
Wilson-CI-reported rates land in every output.

## 5. Benchmark your detectors

```bash
python scripts/detection_bench.py \
    --detectors keyword,ngram,centroid \
    --apply-dedup
# → results/detection_bench.json
```

Output: AUC, best-F1, Recall@1%FPR, Recall@5%FPR (+ Wilson CIs) per
detector, overall and per-channel (with wrappings applied). Adding a new
detector is a 5-line append to `DETECTORS` in that script.

Our baseline today:
- `centroid` (MiniLM embedding vs. 8 label centroids): AUC 0.961 overall.
  data_row R@1%FPR = 0.701 (best), code_comment R@1%FPR = 0.194 (worst).
  ← content-based detectors are NOT channel-invariant when the wrapping
     surrounds the attack payload with domain-shifted text.
- `keyword`: AUC 0.593. Weak alone; useful as a layered first cut.
- `ngram`: AUC 0.600. Saturates to 1.00 recall at 5%FPR — useful as a
  coarse sieve.

## 6. Emit the campaign report

```bash
python scripts/report.py --run-id apr30-harness-paired
# → results/reports/apr30-harness-paired/report.md
```

Headline counts, per-agent/channel/label effectiveness, top 20 confirmed
attacks, spec snapshot, artifact links.

## 7. Update the manifest + commit

```bash
python scripts/build_manifest.py           # refreshes MANIFEST.json
git add results/MANIFEST.json \
        results/reports/apr30-harness-paired/report.md \
        research/hypotheses/
git commit -m "campaign apr30 — H-xxx resolved: <verdict>"
```

The manifest lineage means anyone can later ask "what produced X?" with
`jq '.outputs["results/X.jsonl"]' results/MANIFEST.json`.

## Guardrails built in

Every layer has a specific failure-mode it prevents:

| Layer | Prevents | How |
|---|---|---|
| Preregistration (YAML + git) | moving the goalposts | timestamp proof; direction locked |
| Rigor contract (`research/rigor.py`) | reporting point estimates without CIs | claims lacking CI / test / effect auto-downgrade |
| Verifier (`research/verifier.py`) | hallucinated numbers | claim values must appear in supporting observations |
| Doom detector (`research/doom.py`) | infinite retry loops | fingerprint last N actions; trip at K duplicates |
| Budget ledger (`research/budget.py`) | runaway spend | charge before execute; reject if over cap |
| Human-gate (`research/tools.py`) | destructive actions running autonomously | requires_human_approval for launch / stop / delete |
| Exclusion list (`scripts/audit_parsers.py`) | parser bugs inflating rates | rows with verdict-evidence mismatch dropped |
| Dedup list (`scripts/audit_contamination.py`) | duplicate-count inflation | intra-corpus near-dups collapsed to one keep_id |

## Workflow commands cheatsheet

```bash
# Discovery
intern list-tools
intern list-hypotheses
intern status [--run-id R]
intern call <tool> [--args JSON]

# Plays (scripted workflows)
intern play quick-summary
intern play paired-harness-ablation [--harness-a X --harness-b Y]
intern play resolve-harness-dominates-model

# Preregistration
intern preregister --spec hyp.yaml
python -m research.registry resolve <id> --run-id R --p 0.003 \
    --effect-kind cohens_h --effect-value -0.58 --verdict supported

# Audits
scripts/audit_parsers.py --write-exclusion-list
scripts/audit_contamination.py --write-dedup-list

# Fleet
scripts/fleet.py launch --spec S.json [--run-id R]
scripts/fleet.py status [--run-id R]
scripts/fleet.py stop [--run-id R]

# Analysis
scripts/harness_ablation.py --submit-to-kernel
scripts/detection_bench.py --apply-dedup
scripts/query.py --apply-exclusion [--run-id R] [--agent A] ...
scripts/report.py --run-id R
scripts/pipeline.py --run-id R --only <stage>

# Maintenance
scripts/build_manifest.py
scripts/build_corpus.py {attack,benign,multilingual}
```

## What this isn't (yet)

- There is no LLM-driven planner sitting above the kernel. The **plays**
  in `scripts/intern.py` are hardcoded scientific workflows. Adding an
  LLM planner is straight ml-intern — see Phase 2.B.5 in the roadmap.
- The Phase 3 synthesis loop (Simula-style taxonomy + 1-of-N + dual-
  critic + Elo) is not built yet. When it lands, synthesized attacks get
  run through the same fleet → cross-reference → detector-train loop.
- `campaigns.db` (Phase 1.6 deferred) remains deferred. When multi-
  campaign cross-linking becomes painful enough, build it.

Every layer above is live and producing artifacts. Any number reported
from this system carries a CI. Any preregistered hypothesis has a
timestamp and a direction. Any analysis that runs gets a run_id that
links the data, the code, the claim, and the report.

That's the factory.
