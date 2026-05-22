# Model and dataset artifacts

HermesKatana keeps code in GitHub and large model/dataset artifacts outside the repository. The default public checkout is usable without downloading anything: rule-based scanners and the Proving Ground harness work with local or user-supplied corpora.

Optional ML artifacts can be downloaded from Hugging Face when you explicitly ask for them. Runtime code does not
download models unless you opt in with `KATANA_ARTIFACT_AUTO_DOWNLOAD=1`.

Install the lightweight ML runtime before downloading the default CPU artifact:

```bash
pip install "hermes-katana[fast-cpu]"
```

## Registered model artifacts

The default CPU deployment artifact is the distilled MiniLM ONNX Scabbard model. The optional large model is intended
for local high-accuracy experiments and is never selected by default.

| Selector | Artifact | Default repo | Default setup |
| --- | --- | --- | --- |
| `minilm`, `small` | `katana_v15_distill_minilm_onnx` | `Carlosian/hermes-katana-v15-distill-minilm-onnx` | yes |
| `large`, `v15_large` | `katana_v15_large` | `Carlosian/hermes-katana-v15-large` | no |

Each Hugging Face repo must include an `artifact_manifest.json`. Katana treats the manifest as part of the artifact and
refuses to mark the artifact ready when required files are missing, hashes do not match, sizes do not match, or the
manifest contains unsafe paths.

Minimal manifest shape:

```json
{
  "artifact": "katana_v15_distill_minilm_onnx",
  "version": "3.0.0",
  "source_commit": "<training-or-export-commit>",
  "license": "MIT",
  "files": {
    "model.onnx": {
      "sha256": "<64-hex-sha256>",
      "size": 123456
    }
  }
}
```

The `files` value may also be a list of objects with `path`, `sha256`, and optional `size` or `size_bytes`. Every
required model/tokenizer file must be listed.

## Check status

No network access:

```bash
katana artifacts status
katana artifacts status --all
katana artifacts status large
```

## Guided setup

The top-level first-run wizard is the recommended interactive path. It offers
the small CPU model by default, asks separately before downloading the larger
optional model, and asks whether to install the optional Proving Ground research
harness dependencies:

```bash
katana setup
```

For model artifacts only:

```bash
katana artifacts setup
```

For CI, scripts, or non-interactive terminals, make the choice explicit:

```bash
katana artifacts setup --yes          # default choices: small model only
katana artifacts setup --small        # small model only
katana artifacts setup --large        # larger optional model only
katana artifacts setup --all          # both registered models
katana setup --proving-ground         # install Proving Ground dependencies
katana setup --yes --proving-ground   # small model plus Proving Ground
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
  --repo-id Carlosian/hermes-katana-v15-distill-minilm-onnx \
  --revision v3.0.0
```

Use a release tag or commit SHA for reproducible deployments. `main` is acceptable only for development smoke tests.

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
