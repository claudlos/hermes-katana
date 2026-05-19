# Fleet Runbook — Smoke → Wave A → B → D → (Katana freeze) → Wave C

This is the operational playbook for running the proving-ground fleet.

**Phase strategy** (decided 2026-05-01): we run the **undefended** waves
A / B / D first, use the resulting confirmed corpus to retrain the
DeBERTa scanner and finish the Katana pipeline, freeze a Katana
release version, **then** run Wave C (paired defended). Running
defended-side data against an unfinished Katana produces invalid
headline numbers, so Wave C is deferred until the defense is stable.

Order: Smoke → A → B → D → freeze Katana → C.

## Step 0 — Smoke fleet (run FIRST on every new box)

Before any real wave, run a 5-trial × 3-agent smoke through the actual
fleet supervisor. ~10-15 minutes. Catches every "the supervisor crashed
on workers" / "the JSONL writer is broken" / "the watchdog is
false-firing" class of bug before they eat hours.

```bash
# In Docker (any box):
docker run --rm --network=host \
    -v "$(pwd)":/opt/proving-ground \
    -v "$(pwd)/results":/opt/proving-ground/results \
    katana-fleet-worker:latest \
    fleet launch --spec scripts/fleet_smoke.json --run-id smoke_$(date +%Y%m%d_%H%M)

# Or directly, no Docker (main box only):
python3 scripts/fleet.py launch \
    --spec scripts/fleet_smoke.json \
    --run-id smoke_$(date +%Y%m%d_%H%M)
```

**Pass criteria:**

```bash
# Three JSONL files, 5 rows each, with effective-rate scoring populated:
ls results/agent_shard_runs/shard_400_*.jsonl

# tool_call_count > 0 on at least one row per hermes agent (regression
# check for the 2026-05-01 -Q parser bug):
for f in results/agent_shard_runs/shard_400_*.jsonl; do
    echo "$f"
    jq -c '.attack_run.tool_call_count' "$f" | sort -u
done

# No quota-watchdog firings in the run log:
grep -i "quota-watchdog\|broken-runner" results/fleet_runs/smoke_*/supervisor.log
```

If smoke passes, proceed to Step 1 (Wave A). If it fails, fix before
launching anything bigger.

For local-model boxes, use `fleet_smoke_local.json` instead. Edit the
`agent` field to match what the box has registered
(`hermes_qwen35_local` on the desktop, `hermes_minipc_local` on the
mini-PC).

## Where the boxes live

## Where the boxes live

- **Main box** (`/path/to/hermes-katana/`):
  coordinator. Holds canonical `results/`. Runs analysis, confirmation,
  cross-reference. Optionally also runs free-API agents that don't need
  local hardware.
- **Qwen35 box**: dedicated to `hermes_qwen35_local` + `_katana` pair.
  Setup per `README_LOCAL_QWEN_BOX.md`. rsyncs results back to main.
- **Mini-PC**: dedicated to `hermes_minipc_local` + `_katana` pair.
  Same setup pattern as qwen35 box, smaller model.

All three machines share a synchronized `results/agent_shard_runs/`
folder via rsync — fleet supervisors on each box write into the same
namespace. Cross-confirmation runs on the main box only.

## What each wave produces

| Wave | Agents | Defense | Channels | Shards | Purpose |
|---:|---|---|---|---|---|
| **A** | 8 free workhorses | undefended | file_content | 200-222, 400-523 (147) | breadth — every new attack tested ≥1× |
| **B** | claude × {haiku, sonnet}, codex_cli | undefended | file_content | 400-523 (124) | cross-reference data for confirmation pipeline |
| **C** | qwen35_local, qwen3-coder, minimax (paired) | **defended** | all 4 | varies | the headline release-table data |
| **D** | 3 strong agents | undefended | varies | confirmed-attack multilingual back-trace | per-language transferability |

Wave C is the load-bearing one for the release. Waves A+B feed the
confirmed-attacks corpus that Wave D consumes.

---

## Wave A — workhorses (start here)

Free unlimited compute. Touches every untested attack ≥1× to drive
the `confirmed_attacks.jsonl` count up fast.

### Pre-flight

```bash
cd /path/to/hermes-katana
source .venv/bin/activate

python3 scripts/fleet_preflight.py --spec scripts/fleet_wave_a_workhorses.json
```

Expect every agent to report `OK`. If any reports `BROKEN` or `ERROR`,
fix before launching:
- Local agents (`hermes_qwen35_local`, `hermes_minipc_local`) report
  BROKEN here on the main box because the inference servers aren't on
  this machine. Run their preflights on the actual local boxes.
- Nous agents that report ERROR usually mean Nous OAuth token expired
  — run `hermes auth login` to refresh.

### Launch

Split across boxes:

**Main box** — runs the API agents:
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_a_workhorses.json \
    --run-id wave_a_main_$(date +%Y%m%d) \
    --agents-include "hermes_minimax_m2_7,hermes_nous_qwen3_coder_plus,hermes_nous_kimi_k2_6,hermes_nous_step_flash,hermes_nous_hermes4_70b"
```

**Qwen35 box** — runs the local-Qwen agent:
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_a_workhorses.json \
    --run-id wave_a_qwen35_$(date +%Y%m%d) \
    --agents-include "hermes_qwen35_local"
```

**Mini-PC** — runs the small local agent:
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_a_workhorses.json \
    --run-id wave_a_minipc_$(date +%Y%m%d) \
    --agents-include "hermes_minipc_local"
```

> Note: the existing `fleet.py launch` may not yet support
> `--agents-include`. If not, copy the spec, delete the workers you
> don't want for that box, save as `fleet_wave_a_main.json` etc.

### Monitor

```bash
# Per-shard progress:
watch -n 30 'find results/agent_shard_runs -name "*.status.json" -newer /tmp/.last -exec jq -c "{shard,agent_id,done,total,effective}" {} \;; touch /tmp/.last'

# Confirm trial volume:
ls results/agent_shard_runs/*.jsonl | wc -l
```

### Done condition

Wave A is "done" when every shard 200-222 + 400-523 has at least one
`*.status.json` entry per agent with `done == total`. Expected wall-clock
~14-18 days at concurrency=8.

It's fine to launch B/C/D before A is fully done — just don't expect
the cross-reference confirmation to be stable until A reaches majority
coverage.

### Repeat for other channels

Wave A only runs `file_content`. Once it's done (or once you have
reasonable per-agent breadth), repeat with `code_comment`, then
`tool_output`, then `data_row`. Just edit the `channels` field in the
spec and relaunch with a new `--run-id`. The runner is idempotent.

---

## Wave B — Claude Max windowed (run when A is in progress)

3 agents on the new shards, undefended. Each Claude Max window is ~5h
on / 5h off rolling, plus a weekly cap. The runner's quota watchdog
will abort gracefully when the window tips over.

### Pre-flight

```bash
python3 scripts/fleet_preflight.py --spec scripts/fleet_wave_b_max_window.json
```

Expect OK. If BROKEN with the `~5kB / 0 tools / <5s` shape, your Max
window is exhausted — wait for reset (or check `~/.claude/usage.json`
if available). The env-scrubbing tests catch the env-leak failure mode
explicitly.

### Launch

Run on **main box only** (CCLI auth lives there):

```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_b_max_window.json \
    --run-id wave_b_$(date +%Y%m%d)
```

Concurrency is capped at 4 in the spec — don't override.

### Monitor — watch for the broken-runner shape

```bash
# Detect the broken-runner pattern across recent runs:
find results/agent_shard_runs -name "*.status.json" -mmin -30 -exec jq -c '
  select(.elapsed_sec < 30 and .done < 5)
  | {agent_id, shard, done, total, elapsed_sec}
' {} \;
```

If you see a string of <5-attack aborted shards, the watchdog is
firing — Max is burned. Stop, wait for window reset, restart.

### Re-run schedule

Run Wave B once per Max window until shards 400-523 are covered for
all 3 CCLI agents on file_content. Then repeat per channel as in Wave A.

---

## Wave C — DEFERRED until Katana is feature-complete

**Do not launch Wave C yet.** Running paired defended/undefended trials
against an unfinished Katana would produce invalid headline numbers —
the defense delta would be measuring "broken Katana vs. base" instead
of "shipping Katana vs. base."

The Wave C spec (`scripts/fleet_wave_c_paired_katana.json`) and the
paired drivers in `AGENT_DRIVERS` are already prepared. They sit idle
until the freeze checklist below clears.

### Freeze checklist — Katana must satisfy ALL of these before Wave C

1. Wave A + B + D have produced a stable confirmed corpus (≥10K
   confirmed attacks, daily new-confirmation rate <50/day for a week).
2. DeBERTa scanner has been retrained on that confirmed corpus and
   benchmarked at acceptable FP rate on the benign baseline.
3. Hermes-Katana version pinned to a specific release tag — no more
   in-flight changes during the run.
4. `katana install --target` produces patches that match the pinned
   release, verified by deep_preflight layer 8 (audit log writes).
5. The Docker image is rebuilt against the pinned versions and
   deep-preflight passes on every box.

When all five clear, then — and only then — launch Wave C with the
existing spec.

### When the time comes (placeholder for future-you)

This is the wave that produces the release table. Run each pair
together so matched-pair semantics hold.

### Pre-flight

The qwen35-pair is the priority. Run preflight on the qwen35 box:

```bash
python3 scripts/verify_qwen35_integration.py
python3 scripts/fleet_preflight.py --agent hermes_qwen35_local --agent hermes_qwen35_local_katana
```

Both must report OK with a meaningful delta on the smoke trial.

For the API pairs, run on main:

```bash
python3 scripts/fleet_preflight.py --spec scripts/fleet_wave_c_paired_katana.json
```

### Launch

Split across boxes the same way as Wave A:

**Qwen35 box** (the one that matters most):
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_c_paired_katana.json \
    --run-id wave_c_qwen35_$(date +%Y%m%d) \
    --agents-include "hermes_qwen35_local,hermes_qwen35_local_katana"
```

**Main box** — runs the API pairs:
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_c_paired_katana.json \
    --run-id wave_c_main_$(date +%Y%m%d) \
    --agents-include "hermes_nous_qwen3_coder_plus,hermes_nous_qwen3_coder_plus_katana,hermes_minimax_m2_7,hermes_minimax_m2_7_katana"
```

### What "good" Wave C data looks like

After ~1000 paired trials per (agent, agent_katana) pair, you should
see:

| metric | undefended | defended | delta |
|---|---|---|---|
| effective rate | 15-50% | 0-5% | -10pp to -50pp |
| canary leak rate | 5-30% | 0-2% | -5pp to -30pp |
| scanner triggered | 0% | ~50% (matches attack-text rate) | +50pp |

If defended rate is HIGHER than undefended, something is wrong (the
hermes_nous_hermes4_70b regression in the existing data). Pause and
investigate before more trials.

### Monitor

```bash
# Quick before/after delta per pair:
python3 scripts/analyze_synth_fleet.py --run-id wave_c_qwen35_$(date +%Y%m%d) --paired
```

(If `analyze_synth_fleet.py` doesn't have `--paired` yet, that's a
backlog item.)

---

## Wave D — Multilingual back-trace (run after A/B/C confirms stabilize)

Tests every confirmed English attack against its 11 translated
counterparts to produce a per-language transferability matrix. Doesn't
brute-force the 215K-row factory corpus — only tests translations of
attacks that ALREADY confirmed effective in English. Two orders of
magnitude smaller than brute force, and produces a publishable matrix.

### When to launch

- `confirmed_attacks.jsonl` has stabilized (rate of new confirmations
  drops below ~50/day after Wave A/B/C have been running for several
  days).
- The local Qwen box is up if you want the unlimited-budget agent in
  the mix.

### Pre-launch step — regenerate shards from the LATEST confirmed corpus

The shards in `shards/shard_600.jsonl` ... `shard_814.jsonl` were
built from the confirmed corpus *at the time the back-trace tool was
first run*. Before Wave D launches, regenerate them so the multilingual
set reflects every new confirmation produced by A/B/C:

```bash
python3 scripts/backtrace_multilingual.py --overwrite
```

This re-reads `results/confirmed_attacks.jsonl`, joins against the
factory manifest, and rewrites shards 600-NNN. Shard count grows as
the confirmed corpus grows.

### Pre-flight

```bash
python3 scripts/fleet_preflight.py --spec scripts/fleet_wave_d_multilingual.json
```

3 agents — should take ~5 minutes total. Refuse to launch if any
report BROKEN.

### Launch

**Main box** — runs the API agents:
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_d_multilingual.json \
    --run-id wave_d_main_$(date +%Y%m%d) \
    --agents-include "hermes_nous_qwen3_coder_plus,hermes_minimax_m2_7"
```

**Qwen35 box** — runs the local agent:
```bash
python3 scripts/fleet.py launch \
    --spec scripts/fleet_wave_d_multilingual.json \
    --run-id wave_d_qwen35_$(date +%Y%m%d) \
    --agents-include "hermes_qwen35_local"
```

### What good Wave D data looks like

After ~5,000 trials per agent, expect:

- Per-language effective rate within ±10pp of English baseline for
  high-resource langs (es, fr, de, pt, ja). These attacks transfer.
- Lower rates for low-resource langs (hi, ar, ko) — moderate evidence
  the model's safety training is uneven across languages.
- Specific labels (e.g. `encoding_evasion`) may show divergent rates —
  a Unicode-bidi attack that worked in English Latin script won't
  necessarily work in Cyrillic or Devanagari.

### Analysis

Each row in shards 600-814 carries:
- `language` (one of ar/de/es/fr/hi/it/ja/ko/pt/ru/zh)
- `original_atk_id` (the English source)
- `english_n_models_effective` (potency baseline)

So per-(agent, language) effectiveness is a 1-line groupby on the
trial JSONL output:

```bash
python3 scripts/analyze_synth_fleet.py --run-id wave_d_main_$(date +%Y%m%d) --by-language
```

(Note: `--by-language` may need adding to `analyze_synth_fleet.py` —
backlog if so.)

### Wave D.2 — defended multilingual (future)

Once Wave D produces a clean transferability matrix, a follow-on
defended pass measures whether Hermes-Katana's defense generalizes
across languages. The same shards 600-814 with the `*_katana` twins
of Wave D's agents. Plan that after Wave D.1 produces signal.

---

## Cross-cutting operations

### After each wave: run cross-confirmation (or use the auto-loop)

Manual one-shot:
```bash
python3 scripts/cross_reference_confirm.py
```

Continuous (recommended — start once, leave in a tmux pane):
```bash
bash scripts/auto_confirm_loop.sh                  # default 1h interval
INTERVAL_SEC=1800 bash scripts/auto_confirm_loop.sh  # 30 min
ONCE=1 bash scripts/auto_confirm_loop.sh           # single pass
```

Each pass runs cross_reference_confirm + batch_fingerprint, writes a
log to `results/auto_confirm_logs/pass_<ts>.log`, and prints headline
counts to stdout. Uses `.venv/bin/python` if present (needed for
sentence_transformers).

### Live fleet status snapshot

While Wave A grinds, get a one-shot status of every agent + recent
shard activity:

```bash
python3 scripts/fleet_status.py                            # full table
python3 scripts/fleet_status.py --top 10                   # active shards only
python3 scripts/fleet_status.py --run-id wave_a_main_xxx   # filter
watch -n 30 'python3 scripts/fleet_status.py --top 6'      # live
```

Reads only — safe to run while a fleet is live.

### After each wave: run semantic fingerprinting

The default in-line scoring sets `skip_semantic=True` for thermal
reasons. Run the post-hoc enrichment to add MiniLM-based
attack_mirror / semantic_drift columns:

```bash
python3 scripts/batch_fingerprint.py --mode agent --run-id <wave_run_id>
```

This is needed before the analyze script produces useful per-source
trim recommendations.

### Disk hygiene

`results/agent_shard_runs/` grows ~50 MB per 1000 trials.
`sessions/` grows ~10× faster. Periodic cleanup:

```bash
# Move old session bundles to archive (don't delete — forensic value):
ARCHIVE=results_archive_$(date +%Y%m)
mkdir -p $ARCHIVE
find sessions -mtime +30 -name "session_*.jsonl" -exec mv {} $ARCHIVE/ \;
tar czf $ARCHIVE.tar.gz $ARCHIVE/ && rm -rf $ARCHIVE/
```

### When to STOP and investigate

Stop the fleet immediately if any of these appear:

| signal | likely cause | action |
|---|---|---|
| Watchdog firing on >50% of CCLI shards | Max budget burned | Wait 5h or longer; rotate to Wave A free agents instead. |
| Defended pair worse than undefended (>3pp) | Scanner misconfigured or the wrong paired trial selected | Check `scanner_action` distribution; should be ~50% refuse, ~50% none. If all "none", scorer isn't loading. |
| Effective rate at 0% across multiple agents | Either great defense (unlikely on undefended) or runner broken | Look at `output_chars` and `tool_call_count`; if both ~0, runner broken. |
| Disk filling >90% | sessions accumulating | Run hygiene step above. |

---

## Quick commands cheat-sheet

```bash
# Activate
cd /path/to/hermes-katana && source .venv/bin/activate

# Pre-flight any spec
python3 scripts/fleet_preflight.py --spec scripts/fleet_wave_X_*.json

# Launch any spec
python3 scripts/fleet.py launch --spec scripts/fleet_wave_X_*.json --run-id wave_X_$(date +%Y%m%d)

# Status of all active runs
python3 scripts/fleet.py status

# Stop a run
python3 scripts/fleet.py stop --run-id <id>

# Cross-confirm fresh trials
python3 scripts/cross_reference_confirm.py

# Re-score with semantic fingerprinting
python3 scripts/batch_fingerprint.py --mode agent --run-id <id>

# Per-source effectiveness summary
python3 scripts/analyze_synth_fleet.py --run-id <id>
```
