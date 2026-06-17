# Model and dataset artifacts

HermesKatana keeps code in GitHub and large model/dataset artifacts outside the repository. The default public checkout is usable without downloading anything: rule-based scanners and the Proving Ground harness work with local or user-supplied corpora.

Optional ML artifacts can be downloaded from Hugging Face when you explicitly ask for them. Runtime code does not
download models unless you opt in with `KATANA_ARTIFACT_AUTO_DOWNLOAD=1`.

Install the runtime that matches the artifact you want to run:

```bash
pip install "hermes-katana[fast-cpu]"   # ONNX Runtime, small default artifact
pip install "hermes-katana[torch-cpu]"  # PyTorch, checkpoint artifacts
```

## Registered model artifacts

The default CPU deployment artifact is the distilled MiniLM ONNX Scabbard model. MiniLM PyTorch is also registered for
systems where PyTorch performs better or for users who prefer checkpoint runtimes. The optional large model is intended
for local high-accuracy experiments and is never selected by default.

| Selector | Artifact | Default repo | Default setup |
| --- | --- | --- | --- |
| `minilm`, `small` | `katana_v15_distill_minilm_onnx` | `Carlosian/hermes-katana-v15-distill-minilm-onnx` | yes |
| `minilm_torch`, `small_torch` | `katana_v15_distill_minilm_torch` | `Carlosian/hermes-katana-v15-distill-minilm` | no |
| `large`, `v15_large` | `katana_v15_large` | `Carlosian/hermes-katana-v15-large` | no |
| `v17_large`, `deberta_v17` | `katana_v17_large` | `Carlosian/hermes-katana-17` | no (research) |
| `v17_minilm`, `small_v17` | `katana_v17_minilm` | `Carlosian/hermes-katana-90` | no (research) |

The `v17_*` rows are the v3.1 paper's origin-aware classifiers (`Carlosian/hermes-katana-17` and `-90`). They are pinned by commit revision, carry an `artifact_manifest.json`, and are treated as research models: downloadable explicitly (e.g. `katana artifacts download v17_minilm`) but excluded from managed `setup --all` / `full`.

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

For the simplest non-interactive full setup:

```bash
katana setup full
```

This downloads every managed setup model artifact, installs the ONNX Runtime
CPU, PyTorch CPU, and Proving Ground dependency groups, installs the separate
similarity-softener embedder, then verifies that all of them are present.
Research artifacts such as `v17_minilm` and `v17_large` remain explicit
downloads.

For model artifacts only:

```bash
katana artifacts setup
```

For CI, scripts, or non-interactive terminals, make the choice explicit:

```bash
katana artifacts setup --yes          # default choices: small ONNX model only
katana artifacts setup --small        # small ONNX model only
katana artifacts setup --small-torch  # small PyTorch checkpoint only
katana artifacts setup --large        # larger PyTorch model only
katana artifacts setup --all          # every managed setup model
katana setup --fast-cpu               # ONNX Runtime dependencies only
katana setup --torch-cpu              # PyTorch CPU dependencies only
katana setup --proving-ground         # install Proving Ground dependencies
katana setup --yes --proving-ground   # small ONNX model, similarity embedder, ONNX Runtime, plus Proving Ground
katana setup full                     # managed setup models, setup dependencies, and verification
```

## Direct download

Explicit network access:

```bash
katana artifacts download minilm
katana artifacts download minilm_torch
katana artifacts download large
katana artifacts download v17_minilm
katana artifacts download v17_large
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
export KATANA_MINILM_TORCH_DIR=/models/hermes-katana/katana_v15_distill_minilm/torch
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
- `KATANA_MINILM_TORCH_DIR`: direct path to a ready MiniLM PyTorch artifact directory.
- `KATANA_V15_LARGE_DIR`: direct path to a ready large v15 artifact directory.
- `KATANA_V17_MINILM_DIR`: direct path to a ready v17 MiniLM artifact directory.
- `KATANA_V17_LARGE_DIR`: direct path to a ready v17 large artifact directory.
- `KATANA_HF_REPO_ID`: default MiniLM Hugging Face repo override.
- `KATANA_HF_REVISION`: default MiniLM revision override. Pin this to a tag or commit SHA for reproducible deployments.
- `KATANA_MINILM_TORCH_HF_REPO_ID`: MiniLM PyTorch Hugging Face repo override.
- `KATANA_MINILM_TORCH_HF_REVISION`: MiniLM PyTorch revision override.
- `KATANA_V15_LARGE_HF_REPO_ID`: large model Hugging Face repo override.
- `KATANA_V15_LARGE_HF_REVISION`: large model revision override.
- `KATANA_V17_MINILM_HF_REPO_ID`: v17 MiniLM Hugging Face repo override.
- `KATANA_V17_MINILM_HF_REVISION`: v17 MiniLM revision override.
- `KATANA_V17_LARGE_HF_REPO_ID`: v17 large Hugging Face repo override.
- `KATANA_V17_LARGE_HF_REVISION`: v17 large revision override.
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
