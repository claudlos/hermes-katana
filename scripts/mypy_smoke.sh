#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m mypy \
  src/hermes_katana/_version.py \
  src/hermes_katana/artifacts.py \
  src/hermes_katana/ml_artifacts.py \
  src/hermes_katana/runtime_artifacts.py \
  src/hermes_katana/policy/defaults.py \
  src/hermes_katana/policy/yaml_loader.py \
  src/hermes_katana/taint/codecs.py \
  src/hermes_katana/cli/main.py \
  src/hermes_katana/proxy/addon.py \
  src/hermes_katana/proxy/injector.py \
  src/hermes_katana/proxy/runner.py \
  --ignore-missing-imports \
  --no-error-summary
