# Parallel Agent Operations

This guide covers running Proving Ground work from more than one machine or
agent process without corrupting shared outputs.

## Output Isolation

Proving Ground shard output filenames are based on shard, agent, and channel.
They do not include `run_id`, so two workers writing the same
`agent x shard x channel` cell can interleave rows in the same JSONL file.

Use at least one of these isolation strategies:

1. Use disjoint agent IDs for each worker family.
2. Use disjoint shard IDs for each worker family.
3. Use disjoint channels when the runner writes channel-specific outputs.

Always set a unique `run_id` as well. It helps analysis and audit trails, but
it is not enough by itself to prevent filename collisions.

## Safe Launch Pattern

```bash
python -m venv .venv
.venv/bin/pip install -e ".[proving-ground]"

.venv/bin/python -m hermes_katana.proving_ground.run_agent_shard \
  --shard-id 1 \
  --agent-id <agent-id> \
  --channel file_content \
  --max-attacks 3 \
  --run-id smoke-<operator>
```

For fleet runs, create a dedicated spec for each machine or worker group and
make the shard ranges explicit in the spec name and `run_id`.

## Merge Protocol

When combining results from separate machines:

```bash
rsync -av --ignore-existing \
  /source/results/agent_shard_runs/ \
  /destination/results/agent_shard_runs/

rsync -av \
  /source/results/fleet_runs/<run-id>/ \
  /destination/results/fleet_runs/<run-id>/
```

Use `--ignore-existing` for row files. If a collision happened, the merge will
leave the earlier file in place so the conflict can be reviewed instead of
silently overwritten.

## Operational Rules

- Do not copy SQLite state files between machines.
- Do not edit `results/agent_shard_runs/` JSONL files in place.
- Commit new hypotheses before analyzing their results.
- Keep provider credentials in environment variables or local secret stores.
- Keep generated corpora, checkpoints, and fleet outputs out of Git.
