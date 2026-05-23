# synthdata — Simula-style synthetic data generation for katana

Implements the four-step recipe from Davidson & Harkous (2026),
*Reasoning-Driven Synthetic Data Generation and Evaluation*, adapted to
expand our 3,828-row confirmed-attack corpus into a high-quality
labeled training set for `hermes-katana`'s DeBERTa-v3 classifier and
`zvec` centroid detector.

## The method

1. **Global Diversification** (`taxonomy.py`) — A reasoning model
   recursively builds a hierarchical taxonomy of injection mechanisms
   under each of the 8 katana attack labels. Depth ≤ 3. At each level a
   *propose* pass generates candidate subcategories, then a separate
   *critic* pass merges duplicates, drops invalid ones, and tags
   leaves. The seed corpus is consulted *only* for domain grounding —
   Simula is seedless.
2. **Local Diversification** (`meta_prompt.py`) — Each leaf generates M
   distinct scenario briefs (who, what channel, what pretext). From
   each brief, K concrete attack texts are generated (1-of-N
   meta-prompting: separate scenario generation prevents the teacher
   from copy-pasting N rewordings of the same thing).
3. **Complexification** (`complexify.py`) — Orthogonal difficulty
   axis. A configurable fraction of scenarios are rewritten through
   one or more operators: `obfuscate_payload`, `multi_turn_setup`,
   `authority_framing`, `cross_language`. Originals and complexified
   twins are both kept so the training set spans difficulty strata.
4. **Dual-critic quality gate** (`critics.py`) — Two independent
   critics score each text. Critic A judges *plausibility* (is this a
   genuine instance of the declared mechanism?). Critic B judges
   *compliance-plausibility* (would a naive target model move on it?).
   Both must accept. Independence comes from different prompts AND
   different models where possible (anti-sycophancy).

## Run

```bash
# source your keys (ANTHROPIC_API_KEY + OPENAI_API_KEY) first
source ~/.hermes/.env

# smoke test — ~20 examples, a few minutes, costs cents
python -m synthdata.run --smoke

# real run — uses synthdata/configs/v1_claude.json
python -m synthdata.run --config synthdata/configs/v1_claude.json
```

Each step checkpoints to `synthdata/checkpoints/<run_id>/`:
- `run_meta.json` — manifest, resume-tracker
- `taxonomy.jsonl` — step 1 output
- `meta_prompts.jsonl` — step 2a + step 3 output
- `examples_raw.jsonl` — step 2b output (before critics)
- `examples_judged.jsonl` — step 4 output (with verdicts)
- `critic_summary.json` — per-critic reject-reason breakdown
- `synthdata_final.jsonl` — KEPT examples, ready to fold into
  `hermes-katana/training/data_v4/`

Reruns skip any step whose output already exists.

## Output format

`synthdata_final.jsonl` rows:
```json
{
  "text": "...attack text...",
  "label": "jailbreak",
  "channel": "file_content",
  "origin": "synthdata_v1",
  "meta_id": "abc123",
  "leaf_id": "def456",
  "complexity_level": 0,
  "teacher_model": "claude-haiku-4-5"
}
```

Maps directly onto `hermes-katana/training/configs/katana_v9.yaml`'s
text/label schema; `origin` field fits the v9 origin taxonomy.

## Provider / model configuration

`configs/v1_claude.json` defines per-role LLM configs. Any role can be
swapped to any supported provider (`anthropic`, `openai`,
`openrouter`, `nous`). Teacher + Critic A run through the same
provider by default for throughput; Critic B uses a *different*
provider family to minimize shared-bias sycophancy.

## Cost estimate (v1 config)

- Taxonomy: ~8 labels × 3 depths × 7 candidates × 2 passes (propose+critic)
  ≈ 340 calls × 800 tokens avg ≈ $0.30
- Scenarios: ~200 leaves × 5 scenarios/leaf ≈ 1,000 calls × 400 tokens
  ≈ $0.40
- Complexify: ~300 rewrites × 600 tokens ≈ $0.35
- Text generation: ~1,300 metas × 4 texts = 5,200 calls × 300 tokens
  ≈ $1.50
- Critics: 5,200 × 2 critics × 400 tokens ≈ $4.00

Total ≈ $6.50 per full run at Haiku + GPT-4o-mini prices. Scale ~10×
for Sonnet/GPT-4o teacher runs if quality at top-end is needed.

## Downstream

Expected output: 10–20k high-quality labeled synthetic examples. Fold
into `hermes-katana/training/data_v4/combined.jsonl` alongside the
existing `data_v3/` corpus, then train v10:

```bash
cd ../hermes-katana
python training/train_katana.py --config training/configs/katana_v10.yaml
# ... produces training/checkpoints/katana_v10/
```
