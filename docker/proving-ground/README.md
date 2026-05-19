# Katana Proving Ground Docker image

Public-safe worker image for running Proving Ground code in a repeatable Python environment.

This Docker setup intentionally does not bake credentials, corpora, sessions, results, or `.env` files into the image.

## Build

From the HermesKatana repository root:

```bash
docker build -t hermes-katana-proving-ground:latest \
  -f docker/proving-ground/Dockerfile \
  .
```

## Run

Mount runtime state and pass credentials as environment variables or mounted files:

```bash
docker run --rm \
  --network=host \
  -v "$PWD":/workspace \
  -v "$HOME/.hermes":/opt/hermes_data:ro \
  -e KATANA_ATTACK_CORPUS=/workspace/examples/proving_ground/sample_attacks.jsonl \
  hermes-katana-proving-ground:latest \
  katana proving-ground list-tasks
```

For API-backed agents, pass only the provider keys needed for that run:

```bash
docker run --rm --network=host \
  -v "$PWD":/workspace \
  -e OPENROUTER_API_KEY \
  -e NOUS_PORTAL_API_KEY \
  hermes-katana-proving-ground:latest \
  katana proving-ground run --corpus /workspace/examples/proving_ground/sample_attacks.jsonl
```

## Not included

- `auth.json`
- `.env` files
- private attack corpora
- generated shards
- `sessions/`
- `results/`
- batch input/output

Keep those as runtime mounts or environment variables.
