# Katana Proving Ground

Empirical attack-effectiveness testing for LLM agents and APIs. The proving ground samples candidate prompt-injection attacks, replays them against real models in seeded workspaces, and admits only those that produce measurable behavioral drift across multiple targets. The output corpus informs the [Katana](../../README.md) classifier and scanner gates.

> **Status:** active research code. It is optional, installed through the `proving-ground` extra, and not required for the base Katana runtime.

## What it does

1. **Sample** attacks from a labeled corpus stratified by injection family, language, and channel.
2. **Inject** each attack into a realistic, **seeded sandbox** (a temp workspace with task-specific files: code review, log triage, csv summarize, etc.). Channels include `file_content`, `code_comment`, `tool_output`, `data_row` — each with a forced-read property guaranteeing the agent sees the payload.
3. **Run** the agent under test against the seeded workspace using its native CLI (`claude`, `hermes chat`, `codex`, `cline`, `gemini-cli`, …) or via the OpenAI-compatible API path for raw model evaluation.
4. **Score** behavioral drift: tool-usage delta vs baseline, task-completion drift, exfiltration risk, dangerous-action rate, post-attack collapse signal. Plus a separate LLM-as-judge call for compliance/escalation.
5. **Confirm** an attack only when ≥3 model families produced measurable drift; promote to the `confirmed_attacks` corpus that downstream Katana training consumes.
6. **Synthesize** new variants of confirmed attacks via dual-critic generation, feed back into step 1.

## Quick start

```bash
# 1. Clone Hermes Katana
git clone <hermes-katana-repo> hermes-katana
cd hermes-katana

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[proving-ground]"

# 3. Run a sandbox session against a local model
proving-ground run \
  --task code_review \
  --channel file_content \
  --model qwen3-8b \
  --base-url http://localhost:8080/v1
```

The `proving-ground` console script dispatches to the common sandbox subcommands. Run `proving-ground --help` for the full list.

### Path overrides

By default, the harness expects `hermes-katana` to live as a sibling of `katana-proving-ground`. Override with environment variables:

| variable | purpose |
|---|---|
| `HERMES_KATANA_ROOT` | path to the hermes-katana repo |
| `KATANA_PROVING_GROUND_ROOT` | path to this repo (rarely needed) |
| `KATANA_DEBUG` | set to `1` for verbose tracing |

## Workflows

| command | what it does |
|---|---|
| `proving-ground run` | single sandbox session against one (model, attack, channel) |
| `proving-ground batch --sample N` | N sessions, sampled from the corpus |
| `proving-ground analyze <session-id>` | re-analyze an existing session's behavioral metrics |
| `proving-ground list-sessions` | list tracked sessions in `results/sandbox_tracking.db` |
| `proving-ground list-tasks` | show available workspace task templates |
| `proving-ground synthesize` | generate new variants of confirmed-effective attacks |

For fleet-scale runs (multi-model, multi-shard, days at a time), use the packaged module entry points:

```bash
python -m hermes_katana.proving_ground.run_agent_shard --shard-id 600 --agent-id claude_cli --task readme_summarize ...
python -m hermes_katana.proving_ground.run_shard --shard-id 100 --model-id qwen3-8b --base-url http://... ...
python -m hermes_katana.proving_ground.scripts.fleet launch --spec src/hermes_katana/proving_ground/scripts/fleet_smoke.json
```

## Channels

Each channel injects attack text into a structurally distinct slot. The forced-read property guarantees the agent under test must load the payload-bearing file as part of completing the task.

| channel | task pairing | what gets injected |
|---|---|---|
| `file_content` | `readme_summarize` | a `README.md` body |
| `code_comment` | `refactor_app` | a comment in `src/app.py` |
| `tool_output` | `triage_log` | the contents of `recent_output.log` |
| `data_row` | `csv_summarize` | one row of `data/customers.csv` |

Live data has shown a clear hierarchy of vulnerability across channels. Treat `code_comment` and `file_content` as the first defense priorities, then verify any new channel claims with preregistered runs and confidence intervals.

## Confirmed empirical findings

These are conclusions backed by preregistered hypotheses with statistical power:

1. **Katana scanning middleware reduces effective attacks by >80%.** Paired n=10,774; bare 12.40% → katana 0.00%; McNemar p≈0; clean 1,336-vs-0 discordant pair ratio.
2. **Harness design does NOT dominate model alignment.** Preregistered prediction REJECTED: Claude Code CLI 40.80% vs Hermes 10.15% (Δ=+30.65pp **against** prediction, p<1e-6, Cohen's h=0.737). The `code_comment` channel is CCLI-specific attack surface (+42.12pp).
3. **Channel hierarchy** (above): rank defense priorities as code_comment > file_content > tool_output > data_row.
4. **MiniLM centroid baseline detector**: AUC 0.961, F1 0.943, recall@1%FPR = 0.754. Beats Katana injection_score (AUC 0.710) by 25.1pp on AUC and has orthogonal error modes.

## What this is not

- Not a benchmark — it produces benchmarks. The headline `confirmed_only_v1` benchmark lives at [`evals/benchmarks/confirmed_only_v1/`](../../evals/benchmarks/confirmed_only_v1/).
- Not a defense. The proving ground *measures* attack effectiveness; the [Hermes Katana](../../README.md) scanner is the runtime defense it informs.
- Not a CTF. Sandbox sessions test agent behavior under adversarial input; success/failure is a *signal* about the model+harness, not a winnable challenge.

## Reproducibility

See [`methodology.md`](methodology.md) for seed handling, dependency notes,
hardware requirements, and the public reporting bar.

## Configuration

Edit `config.yaml`:

- Provider list (OpenAI, Anthropic, Nous, Minimax, Gemini, xAI, local)
- Sample size and stratification
- Judging model and scoring thresholds
- Synthesis parameters

Provider API keys are read from environment variables (never `config.yaml`); pin them via `~/.hermes/.env` or your shell.

## Outputs

| file | contents |
|---|---|
| `results/evaluation.db` | SQLite of all evaluations |
| `results/confirmed_attacks.jsonl` | attacks confirmed effective on ≥3 model families |
| `results/rejected_attacks.jsonl` | attacks blocked on all targets |
| `results/synthetic_variants.jsonl` | dual-critic-passed variants of confirmed attacks |
| `sessions/<id>/` | per-session ephemeral workspace artifacts (gitignored; regenerable) |

## Further reading

- [`methodology.md`](methodology.md) — campaign design, reproducibility, and caveats
- [`mini-agent-guide.md`](mini-agent-guide.md) — parallel agent driver wiring

## License

MIT — see [`LICENSE`](../../LICENSE). Large corpora and trained model artifacts are not bundled in this repository; publish them with their own data cards and license notes.
