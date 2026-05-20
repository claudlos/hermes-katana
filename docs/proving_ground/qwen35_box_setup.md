# Qwen3.6-35B Box Setup — Hermes-Agent + Katana

This is the runbook for the dedicated Qwen3.6-35B inference box. Run the
helper Claude CLI (or yourself) through these steps once. After that,
the box is part of the proving-ground fleet at unlimited budget.

## Prerequisites already on the box

- The local inference server (vLLM / llama.cpp / LM Studio / Ollama) is
  serving Qwen3.6-35B over an OpenAI-compatible HTTP endpoint.
- `hermes-agent` is installed and configured to reach that server. A
  trivial `hermes chat -q "say hi" -Q --model <model> --provider <provider>`
  should already work.
- Python 3.10+ available.

If any of those isn't true, fix it before continuing.

## Step 1 — clone the project code

```bash
cd /path/to/workspace
# rsync from the main box, or:
git clone <katana-proving-ground-remote> katana-proving-ground
git clone <hermes-katana-remote>          hermes-katana
cd katana-proving-ground
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../hermes-katana
pip install -r requirements.txt 2>/dev/null || pip install pyyaml sentence-transformers
```

## Step 2 — install katana into the hermes-agent checkout

This is the call that wires the full 7-layer middleware chain into hermes.
Without it, you only get the harness-level input scanner (one layer).

```bash
# Find the hermes-agent checkout (where hermes was installed from):
HERMES_CHECKOUT=$(python3 -c "import hermes; print(__import__('os').path.dirname(__import__('os').path.dirname(hermes.__file__)))")
echo "$HERMES_CHECKOUT"

# Apply katana patches:
katana install --target "$HERMES_CHECKOUT"

# Verify:
katana status --target "$HERMES_CHECKOUT"
```

The `install` is idempotent and reversible (`katana uninstall --target ...`).
Patches are sentinel-based and add code *around* the original — no code is
removed.

## Step 3 — point the proving-ground driver at your model

Set these env vars in your shell rc (or in a `.env` you source before runs).
The exact values depend on how your hermes-agent provider config is set.

```bash
# What hermes calls the model (whatever string `hermes chat --model <X>` accepts):
export KATANA_LOCAL_QWEN35_MODEL="qwen3.6-35b"

# What hermes calls the provider (the entry in hermes's config that points
# at your local OpenAI-compat server). Common: "local", "qwen", "openai-compat".
export KATANA_LOCAL_QWEN35_PROVIDER="local"

# Optional: longer per-attack timeout (default 240s). Bump if your box
# is slower than ~22 tok/s.
export KATANA_LOCAL_TIMEOUT_SEC="300"

# So the verifier knows where to check patch status:
export HERMES_CHECKOUT="$HERMES_CHECKOUT"
```

## Step 4 — run the integration verifier

```bash
python3 scripts/verify_qwen35_integration.py
```

Expected output, in order:

```
[1] hermes CLI installed?
  OK   hermes --version → ...
[2] Local Qwen3.6-35B server reachable through hermes?
       using model=qwen3.6-35b provider=local
  OK   round-trip ok (3.4s, 412 chars)
[3] hermes-katana module + patches?
  OK   hermes_katana imported (...)
  OK   katana patches applied to /path/to/hermes-agent
[4] Paired smoke trial (1 attack)
  OK   both drivers present in AGENT_DRIVERS

  variant                         exit     chars  tools   canary    sec
  hermes_qwen35_local                0   18,432      4    safe    32.4s
  hermes_qwen35_local_katana         0      147      0    safe     0.8s
  OK   Smoke trial completed both legs.
  OK   Defended leg produced significantly less output (likely scanner-refused).

All checks passed. Ready to launch fleet on this box.
```

If any step fails, read the message — exit codes are:

| code | meaning |
|---:|---|
| 0 | all good |
| 1 | hermes CLI / model not reachable |
| 2 | hermes-katana not installed |
| 3 | hermes-katana installed but not patched into hermes (still usable for harness-scanner pairs) |
| 4 | paired smoke trial failed |

## Step 5 — preflight a real attack

Once verify passes, run one shard of one attack to be sure the runner end-to-end
is healthy on this box. From the project root:

```bash
python3 run_agent_shard.py \
    --shard-id 1 --agent-id hermes_qwen35_local \
    --max-attacks 1 --concurrency 1
```

Then look at `results/agent_shard_runs/shard_001_hermes_qwen35_local.status.json`
— `done` should be 1, `effective` 0 or 1, `elapsed_sec` should be roughly the
per-attack budget (30-60s).

## Step 6 — fleet handoff

Once Steps 1-5 are clean, the box is ready to take fleet jobs. The fleet spec
on the main box can target this box's agent IDs:

```json
{
  "agents": ["hermes_qwen35_local", "hermes_qwen35_local_katana"],
  "shards": [1, 2, 3, ...],
  "max_concurrency": 2,
  "watchdog_threshold": 3
}
```

Two-concurrency is a safe starting point for a 35B model: each attack does
multi-turn agent work and CPU-bound scoring on the same box.

## Operational notes

- **rsync results back to the main box** every few hours (or set up a
  cron). Confirmation runs and analyses live on the main box; the qwen35
  box just generates trial rows.
- **No quota to worry about**, but **watch disk and RAM**: each trial
  writes a session JSONL of 10-300 kB; embedding/scoring loads MiniLM
  (~80 MB) once per worker.
- **If the local server crashes**, `run_agent_shard.py`'s broken-runner
  watchdog (3 consecutive failures) aborts the shard. Restart the server
  and re-launch — the shard is idempotent and will skip done attacks.
- **Don't `katana uninstall`** while the fleet is running — it modifies
  hermes source files and a request mid-flight could see partial state.
