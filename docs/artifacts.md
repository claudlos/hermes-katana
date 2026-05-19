# Model and dataset artifacts

HermesKatana keeps code in GitHub and large model/dataset artifacts outside the repository. The default public checkout is usable without downloading anything: rule-based scanners and the Proving Ground harness work with local or user-supplied corpora.

Optional ML artifacts can be downloaded from Hugging Face when you explicitly ask for them. Runtime code does not
download models unless you opt in with `KATANA_ARTIFACT_AUTO_DOWNLOAD=1`.

## Registered model artifacts

The default CPU deployment artifact is the distilled MiniLM ONNX Scabbard model. The optional large model is intended
for local high-accuracy experiments and is never selected by default.

| Selector | Artifact | Default repo | Default setup |
| --- | --- | --- | --- |
| `minilm`, `small` | `katana_v15_distill_minilm_onnx` | `claudlos/hermes-katana-v15-distill-minilm-onnx` | yes |
| `large`, `v15_large` | `katana_v15_large` | `claudlos/hermes-katana-v15-large` | no |

Each Hugging Face repo should include an `artifact_manifest.json` with sha256s, file sizes, source commit, training/eval
summary, model role, and license notes.

## Check status

No network access:

```bash
katana artifacts status
katana artifacts status --all
katana artifacts status large
```

## Guided setup

Prompted setup is the recommended first-run path. It offers the small CPU model by default and asks separately before
downloading the larger optional model:

```bash
katana artifacts setup
```

For CI, scripts, or non-interactive terminals, make the choice explicit:

```bash
katana artifacts setup --yes          # default choices: small model only
katana artifacts setup --small        # small model only
katana artifacts setup --large        # larger optional model only
katana artifacts setup --all          # both registered models
```

## Direct download

Explicit network access:

```bash
katana artifacts download minilm
katana artifacts download large
```

Override the repo or revision:

```bash
katana artifacts download \
  minilm \
  --repo-id claudlos/hermes-katana-v15-distill-minilm-onnx \
  --revision main
```

## Offline/local deployment

Use an already-downloaded artifact:

```bash
export KATANA_MINILM_ONNX_DIR=/models/hermes-katana/katana_v15_distill_minilm/onnx
export KATANA_V15_LARGE_DIR=/models/hermes-katana/katana_v15_large
katana artifacts path
katana artifacts path large
```

Or choose a cache root:

```bash
export KATANA_ARTIFACT_DIR=/models/hermes-katana-cache
katana artifacts setup --yes
```

## Environment variables

- `KATANA_ARTIFACT_DIR`: cache root for downloaded artifacts.
- `KATANA_MINILM_ONNX_DIR`: direct path to a ready MiniLM ONNX artifact directory.
- `KATANA_V15_LARGE_DIR`: direct path to a ready large v15 artifact directory.
- `KATANA_HF_REPO_ID`: default MiniLM Hugging Face repo override.
- `KATANA_HF_REVISION`: default MiniLM revision override. Pin this to a tag or commit SHA for reproducible deployments.
- `KATANA_V15_LARGE_HF_REPO_ID`: large model Hugging Face repo override.
- `KATANA_V15_LARGE_HF_REVISION`: large model revision override.
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
