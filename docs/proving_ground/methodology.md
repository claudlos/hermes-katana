# Proving Ground Methodology

Katana Proving Ground is optional research code for measuring whether prompt
injection payloads change agent behavior in realistic workspaces. It is not
required for the base scanner, policy engine, vault, audit trail, or proxy.

## Campaign Shape

1. Preregister the claim and the primary metric before collecting data.
2. Select attacks by shard, family, language, and channel.
3. Seed a temporary workspace that forces the agent to read the payload.
4. Run the target agent or model through the packaged worker modules.
5. Score behavioral drift, canary leakage, refusal changes, and dangerous action rate.
6. Promote an attack only when it produces measurable drift across multiple model families.

Useful entry points:

```bash
pip install -e ".[proving-ground]"

proving-ground run --task code_review --channel file_content --model qwen3-8b

python -m hermes_katana.proving_ground.run_agent_shard \
  --shard-id 1 --agent-id claude_cli --max-attacks 5

python -m hermes_katana.proving_ground.scripts.fleet launch \
  --spec src/hermes_katana/proving_ground/scripts/fleet_smoke.json \
  --run-id smoke-local
```

## Reproducibility Rules

- Keep generated corpora, checkpoints, run outputs, and session workspaces out of Git.
- Keep provider credentials in environment variables or local secret stores.
- Use stable `run_id` values so reports, JSONL rows, and fleet metadata can be traced.
- Use disjoint shard, channel, or agent ranges when multiple machines write results.
- Treat old campaign results as historical unless the current code and current policy gate can reproduce them.

## Known Caveats

- Results depend on the exact agent CLI, model endpoint, permissions, and tool harness.
- LLM-as-judge scoring can bias which attacks are labeled effective.
- Some historical corpora included synthetic rows; empirical and synthetic sources should not be pooled without labels.
- Multilingual and editor-agent coverage is incomplete.
- Hardware-specific fleet plans are intentionally not shipped in the public docs.

## Public Reporting Bar

Report confidence intervals, sample counts, excluded-row rules, and the exact
policy/scanner revision with any headline attack-effectiveness number. If a
claim cannot be traced to a run manifest and a reproducible command, treat it as
exploratory rather than a release claim.
