# Mini-PC Agent Guide — Parallel Research Handoff

**Written:** 2026-04-23
**For:** the agent running on the Mac/Linux mini off this T7 SSD copy
**From:** the agent on the Linux desktop (15 GiB RAM box, currently running fleet v14)
**Purpose:** let you do useful katana-proving-ground research *in parallel* with the desktop without duplicating work or corrupting shared output files.

Read this top-to-bottom before running anything.

---

## 0. Orientation in 60 seconds

This repo is a sandbox harness that injects prompt-injection attacks into LLM coding agents, tracks tool usage, and measures which harnesses / models / channels resist attacks. The goal is **rigorous empirical data** on prompt-injection defenses (specifically, the Hermes Katana scanner-in-harness layer).

Key docs — read these before you touch code:

- `HANDOFF.md` — full project arc (Phase 1 + data pipeline)
- `HANDOFF_PHASE3.md` — the four current open threads (**this is where your work comes from**)
- `FINDINGS.md` — nexus of resolved and open hypotheses; headline numbers live here
- `FACTORY.md` — ResearchKernel / factory substrate reference
- `research/hypotheses/` — preregistered hypotheses (git-committed before data → reviewer-defensible)
- `scripts/fleet.py` — the supervisor that launches and tracks agent-shard jobs
- `scripts/fleet_v14_katana_paired.json` — the spec the desktop is currently running

---

## 1. What the desktop is doing RIGHT NOW (do not duplicate)

**Run ID reserved:** `H-20260423-live-v2`
**Spec:** `scripts/fleet_v14_katana_paired.json`
**Purpose:** LIVE resolution of `H-20260423-scanner-in-harness-protects` — same attack, paired bare-harness vs `_katana` twin.
**Concurrency:** 12 (the box is RAM-limited; higher caused OOM reboot earlier today)
**Total jobs:** 3,280. Expect ~8–24 hours wall-time.

**Shards the desktop is consuming:**

| Agent family | Shards in use | Channels | max_attacks |
|---|---|---|---|
| CORE (claude_cli_haiku/_sonnet, hermes_claude_haiku, hermes_nous_qwen3_coder_plus, plus all `_katana` twins) | `1..40 ∪ 101..110` | all 4 | 200 |
| CROSS (hermes_minimax_m2_7, hermes_or_gpt4o_mini, hermes_nous_hermes4_70b, hermes_or_deepseek_v3_free, hermes_nous_step_flash, hermes_nous_kimi_k2_6, plus all `_katana` twins) | `1..35` | all 4 | 25–100 depending on agent |

Those cells are **claimed**. If you launch the same (agent × shard × channel) on the mini, both boxes will append to the same `results/agent_shard_runs/shard_NNN_agent.jsonl` file and corrupt each other. Don't.

---

## 2. Hard isolation rules (read before doing anything)

The output filename format is `results/agent_shard_runs/shard_{NNN}_{agent_id}[_{channel}].jsonl`. **No run_id in the filename.** So two boxes writing the same (agent × shard × channel) will interleave lines into one file.

You must satisfy ONE of these:

1. **Disjoint agent IDs.** If you register *new* agent drivers (e.g. `anthropic_api_haiku_raw`, `cursor_claude_sonnet`), your output filenames will not collide with the desktop's. **Safest path.**
2. **Disjoint shard IDs.** Use shards the desktop isn't touching (see §3 table). Same agent names are fine as long as shards don't overlap.
3. **Disjoint run_id AND disjoint anything-above.** `run_id` is recorded INSIDE each row but does NOT partition the output file. So this rule alone is insufficient — it only helps for filtering in analysis. Always combine with rule 1 or 2.

Every row you generate should carry a **unique run_id** anyway — pick a tag that identifies this box, e.g.:
- `H-20260423-mini-shardfill` (if you run option A)
- `H-20260423-mini-factorial`  (if you run option B)
- `H-20260423-mini-covfill`     (if you run option C)

The desktop is on `H-20260423-live-v2`. Do not reuse that value.

---

## 3. Suggested work, ranked by leverage

### Option A (lowest risk, highest drop-in value) — extend shard coverage on the v14 agents

**What it buys us:** more statistical power on the scanner-in-harness hypothesis and the factorial cells. Same spec, disjoint shards, zero collision.

**Shards free for you:**

- CORE agents: shards `41..100` and `111..125` are untouched
- CROSS agents: shards `36..125` are untouched

Concrete recipe:

1. Copy `scripts/fleet_v14_katana_paired.json` → `scripts/fleet_v14_katana_paired_mini.json`.
2. Replace every CORE worker's `shards` array with e.g. `[41..80]` (40 extra shards).
3. Replace every CROSS worker's `shards` array with e.g. `[36..70]` (35 extra shards).
4. Drop `max_concurrency` to whatever your mini can actually handle; on a 16 GiB box, 8–10 is safe. On a 32 GiB box, 16.
5. Run:
   ```bash
   .venv/bin/python scripts/fleet.py launch \
     --spec scripts/fleet_v14_katana_paired_mini.json \
     --run-id H-20260423-mini-shardfill
   ```

When done, the rows live in NEW `shard_041_*.jsonl` ... `shard_080_*.jsonl` files that the desktop never touched. Merge is `rsync --ignore-existing`.

**Required env:**
- `ANTHROPIC_API_KEY` (for claude_cli_haiku/_sonnet, hermes_claude_haiku)
- `OPENROUTER_API_KEY` (for hermes_or_gpt4o_mini, hermes_or_deepseek_v3_free)
- `NOUS_API_KEY` or equivalent (for hermes_nous_* — Nous Portal credential)
- `MINIMAX_API_KEY` (for hermes_minimax_m2_7)
- Claude Code CLI installed and logged in for the `claude_cli_*` agents
- Hermes binary available in `$PATH` for `hermes_*` agents

If a provider isn't available on the mini, just delete that worker from the spec.

---

### Option B (higher scientific leverage) — Thread 2: factorial component-identifiability

**What it buys us:** the "exact math" for which harness COMPONENT matters (`instruction_hier`, `permission_gate`, `untrusted_marker`, `bash_allowlist`, `scanner_layer`). Read `HANDOFF_PHASE3.md §Thread 2` for full rationale.

**Why the mini is well-suited:** this work uses raw Anthropic API calls, not local CLI agents. No CCLI/Hermes install needed — just `ANTHROPIC_API_KEY` and Python.

**What you actually build (about 300 LoC total):**

1. `sandbox/raw_api_adapter.py` — minimal Anthropic SDK wrapper (messages.create loop with tool_use).
2. Four new entries in `sandbox/agent_cli_runner.py` `AGENT_DRIVERS`:
   - `anthropic_api_haiku_raw`          — no harness features
   - `anthropic_api_haiku_hier`         — instruction hierarchy system prompt only
   - `anthropic_api_haiku_permgate`     — synthetic permission gate only
   - `anthropic_api_haiku_untrusted`    — untrusted-content markers only
3. Four new rows in `research/harness_profiles.yaml` with the matching feature flags (`instruction_hier`, `permission_gate`, `untrusted_marker`). Set `scanner_layer: false` on all four.
4. A new fleet spec `scripts/fleet_v15_factorial.json` running the 4 new agents across ~10 shards × 4 channels × 100 attacks = ~16k rows.
5. Launch under `run_id=H-20260423-mini-factorial`.
6. After it finishes, the desktop (or you) runs `python scripts/factorial_decompose.py --apply-exclusion` and the per-feature ORs appear in `results/factorial.json`.

**Expected cost:** ~180M tokens of Haiku (batch-able), ~$70 or $35 with batch. Confirm on-box API key has budget.

**Why this doesn't overlap the desktop:** the 4 `anthropic_api_*` agent IDs are brand new — no file-name collision possible.

---

### Option C — Thread 3: fill uncovered (model × harness) cells

**What it buys us:** cross-provider comparisons the reviewer will ask for (Cursor, Aider, Continue, Ollama, …).

**Priority cells + what to install on the mini:**

| Cell | CLI to install | Notes |
|---|---|---|
| claude × cursor_cli | `cursor-agent` CLI | `cursor-agent --print "task"` — verify non-interactive mode exists on your OS first |
| llama × ollama | `ollama` + pull `llama3` | entirely local execution, good baseline |
| gpt4 × aider | `pip install aider-chat` | needs `OPENAI_API_KEY` |
| claude × continue | `continue-cli` | third harness ecosystem |

Pattern per new driver (per `HANDOFF_PHASE3.md §Thread 3`):

1. Install the CLI, verify non-interactive invocation.
2. Add `AgentDriver` entry in `sandbox/agent_cli_runner.py` with the right `cmd_template`.
3. Add a parser in `sandbox/parsers.py` if the output schema differs from `claude_cli` format.
4. Smoke: `python run_agent_shard.py --shard-id 1 --agent-id <new> --max-attacks 3 --run-id smoke`.
5. Update `research/harness_profiles.yaml` with the new agent's feature vector.
6. Extend or create a fleet spec `scripts/fleet_v15_covfill.json` and launch under `run_id=H-20260423-mini-covfill`.

These are all **new agent IDs** → output files will never collide with desktop.

---

### Option D (no compute, just code) — Thread 4: LLM planner above the kernel

Read `HANDOFF_PHASE3.md §Thread 4`. This is a ~300-LOC Python task:

- Create `research/planner.py` (Anthropic SDK wrapper that drives `ResearchKernel.call(tool, args)`)
- Write the planner system prompt (`research/system_prompt.py`)
- Wire to `scripts/intern.py` as a `planner` subcommand
- Add doom-loop and budget gates

**Zero risk of collision** with anything the desktop is doing — it's pure code. Commit on a branch and we'll merge.

---

## 4. Execution recipe common to A/B/C

```bash
cd /path/to/katana-proving-ground

# 1. Re-create the venv on THIS box (T7 copy does not ship one)
python3 -m venv .venv
.venv/bin/pip install -e .
# If pyproject lists the Hermes Katana package as a local path dep, you may need
# pip install -e ../hermes-katana   (if that repo is present)

# 2. Smoke test the framework
.venv/bin/python run_agent_shard.py \
  --shard-id 1 --agent-id claude_cli_haiku --channel file_content \
  --max-attacks 3 --run-id smoke-local

# 3. Launch the real fleet (use one of the run_ids above)
nohup .venv/bin/python scripts/fleet.py launch \
  --spec scripts/fleet_v14_katana_paired_mini.json \
  --run-id H-20260423-mini-shardfill \
  > /tmp/fleet-mini.out 2>&1 &
disown

# 4. Monitor
.venv/bin/python scripts/fleet.py status --run-id H-20260423-mini-shardfill
tail -f results/fleet_runs/H-20260423-mini-shardfill/supervisor.log
```

---

## 5. Merge-back protocol (how we combine data later)

When your run finishes, **do not write to the desktop's disk directly**. Sync via the T7 SSD.

On the mini:
```bash
rsync -av --ignore-existing \
  /path/to/katana-proving-ground/results/agent_shard_runs/ \
  /media/<mount>/T7/katana-proving-ground/results/agent_shard_runs/
rsync -av \
  /path/to/katana-proving-ground/results/fleet_runs/H-20260423-mini-*/ \
  /media/<mount>/T7/katana-proving-ground/results/fleet_runs/
```

On the desktop (when I'm done with v14):
```bash
rsync -av --ignore-existing \
  /media/ssd/hermes-katana/results/agent_shard_runs/ \
  /path/to/hermes-katana/results/agent_shard_runs/
rsync -av \
  /media/ssd/hermes-katana/results/fleet_runs/H-20260423-mini-*/ \
  /path/to/hermes-katana/results/fleet_runs/
```

`--ignore-existing` is the safety net: if by accident we *did* target the same (agent×shard×channel), the later rsync will refuse to clobber the first file and we'll notice during the merge. Do not replace with `--update` or `-I`.

For code changes you make (new drivers, planner.py, etc.): commit on a branch named `mini/<your-slug>` on your local clone, bundle with `git bundle create mini.bundle master..mini/<slug>`, drop the bundle on T7, and the desktop pulls with `git bundle unbundle`. The repo has no remote — bundles are the cleanest cross-box path.

---

## 6. Gotchas

- **Python version pinned to 3.12.** `.venv` is not shipped on T7 — recreate it.
- **`fleet.py` writes `results/fleet_runs/<run_id>/supervisor.pid`.** Different run_ids = separate pid files = no conflict with desktop.
- **Row schema is `OUTPUT_SCHEMA_VERSION`** — see `run_agent_shard.py` near line 357. Any new driver must emit the same schema or analysis breaks.
- **Rate limits** — Claude Max has a 5h rolling window per user. If the desktop and mini share the same Anthropic account, you'll race each other. Prefer a raw-API key (Option B) or separate account for the mini.
- **`results/agent_shard_runs/*.db`** and `*.status.json` / `*.baselines.json` are per-box state — do NOT copy these between boxes. `.gitignore` already excludes them.
- **Preregistration discipline:** if you test a NEW hypothesis (not one already in `research/hypotheses/`), git-commit the YAML *before* you analyze the data. The commit timestamp is the preregistration proof.

---

## 7. What to skip

- Don't run the same fleet v14 spec under the same run_id as the desktop.
- Don't touch `research/hypotheses/H-20260423-scanner-in-harness-protects.yaml` — that one's being resolved LIVE by the desktop right now.
- Don't rerun shards 1–40 or 101–110 on CORE agents. Same for 1–35 on CROSS agents.
- Don't edit `results/agent_shard_runs/` files in place; only append via `run_agent_shard.py`.

---

## 8. Quick sanity check before launch

```bash
# Verify no collision with desktop
grep -c '"run_id": "H-20260423-live-v2"' \
  results/agent_shard_runs/shard_041_claude_cli_haiku.jsonl 2>/dev/null
# expected: 0 (desktop didn't touch 041)

# Verify the spec you're about to launch doesn't duplicate desktop shards
.venv/bin/python -c "
import json; s = json.load(open('scripts/fleet_v14_katana_paired_mini.json'))
desktop_core = set(range(1,41)) | set(range(101,111))
desktop_cross = set(range(1,36))
for w in s['workers']:
  claimed = set(w['shards'])
  core = 'minimax' not in w['agent'] and 'or_' not in w['agent'] and 'nous_hermes4' not in w['agent'] and 'step_flash' not in w['agent'] and 'kimi' not in w['agent']
  overlap = claimed & (desktop_core if core else desktop_cross)
  assert not overlap, f'{w[\"agent\"]} overlaps desktop on shards {sorted(overlap)}'
print('OK — no shard overlap with desktop')
"
```

Run that before launch. If it prints `OK` you're safe.

---

## 9. Ping back

When you're done (or stuck), drop a one-page summary at `MINI_AGENT_REPORT.md` in this folder. Include:

- Which option you ran (A/B/C/D)
- run_id
- Rows produced
- Any new drivers / code added (file paths + one-line rationale each)
- Any surprises or broken assumptions above

That's the handoff doc the desktop reads when merging.

Good luck.
