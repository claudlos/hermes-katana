# Model and dataset artifacts

HermesKatana keeps code in GitHub and large model/dataset artifacts outside the repository. The default public checkout is usable without downloading anything: rule-based scanners and the Proving Ground harness work with local or user-supplied corpora.

Optional ML artifacts can be downloaded from Hugging Face when you explicitly ask for them.

## Default model artifact

The default CPU deployment artifact is the distilled MiniLM ONNX Scabbard model:

- Default repo placeholder: `claudlos/hermes-katana-v15-distill-minilm-onnx`
- Type: Hugging Face model repo
- Required files:
  - `model.onnx`
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
  - `added_tokens.json`
  - `vocab.txt`

The repo ID can be changed before release if the artifact lands under a Nous Research namespace.

## Check status

No network access:

```bash
katana artifacts status
```

## Download

Explicit network access:

```bash
katana artifacts download
```

Override the repo or revision:

```bash
katana artifacts download \
  --repo-id claudlos/hermes-katana-v15-distill-minilm-onnx \
  --revision main
```

## Offline/local deployment

Use an already-downloaded artifact:

```bash
export KATANA_MINILM_ONNX_DIR=/models/hermes-katana/katana_v15_distill_minilm/onnx
katana artifacts path
```

Or choose a cache root:

```bash
export KATANA_ARTIFACT_DIR=/models/hermes-katana-cache
katana artifacts download
```

## Environment variables

- `KATANA_ARTIFACT_DIR`: cache root for downloaded artifacts.
- `KATANA_MINILM_ONNX_DIR`: direct path to a ready MiniLM ONNX artifact directory.
- `KATANA_HF_REPO_ID`: default Hugging Face repo override.
- `KATANA_HF_REVISION`: default revision override. Pin this to a tag or commit SHA for reproducible deployments.
- `KATANA_ARTIFACT_AUTO_DOWNLOAD`: set to `1` only when runtime auto-download is desired.
- `KATANA_HF_TOKEN` or `HF_TOKEN`: optional token for private/gated Hugging Face repos.

## Datasets

Training datasets and large red-team corpora are not bundled in GitHub. Public code accepts user-supplied JSONL corpora. For Proving Ground:

```bash
katana proving-ground run --corpus /path/to/attacks.jsonl
# or
export KATANA_ATTACK_CORPUS=/path/to/attacks.jsonl
```

The repository includes only a tiny synthetic sample corpus for smoke tests:

```text
examples/proving_ground/sample_attacks.jsonl
```
