#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
SKIP_FULL_TESTS=0
ALLOW_MISSING_GITLEAKS=0
WITH_ARTIFACT_DOWNLOADS=0

usage() {
  cat <<'EOF'
Usage: scripts/release_gate.sh [--dry-run] [--skip-full-tests] [--with-artifact-downloads] [--allow-missing-gitleaks]

Runs the V3 release gate:
  1. Ruff lint/format checks
  2. Generated policy asset check
  3. Full pytest suite
  4. Scanner-change verification gate
  5. Wheel/sdist build
  6. Twine metadata check
  7. Artifact status smoke
  8. Gitleaks secret scan

Options:
  --dry-run                 Print commands without executing them.
  --skip-full-tests         Skip the full pytest suite.
  --with-artifact-downloads Download and verify the default small HF artifact.
  --allow-missing-gitleaks  Do not fail locally when gitleaks is not installed.
  -h, --help                Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-full-tests)
      SKIP_FULL_TESTS=1
      ;;
    --with-artifact-downloads)
      WITH_ARTIFACT_DOWNLOADS=1
      ;;
    --allow-missing-gitleaks)
      ALLOW_MISSING_GITLEAKS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DIST_DIR="${DIST_DIR:-${ROOT_DIR}/.pytest_tmp/release-gate-dist/${RUN_ID}}"

cd "${ROOT_DIR}"

run_cmd() {
  local display="$1"
  shift
  echo "+ ${display}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@"
}

run_shell() {
  local display="$1"
  local command="$2"
  echo "+ ${display}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  bash -lc "${command}"
}

run_cmd "ruff check src/ tests/" "${PYTHON_BIN}" -m ruff check src/ tests/
run_cmd "ruff format --check src/ tests/" "${PYTHON_BIN}" -m ruff format --check src/ tests/
run_cmd "python3 scripts/generate_policy_assets.py --check" "${PYTHON_BIN}" scripts/generate_policy_assets.py --check
run_cmd "scripts/mypy_smoke.sh" scripts/mypy_smoke.sh

if [[ "${SKIP_FULL_TESTS}" -eq 0 ]]; then
  run_shell "python3 -m pytest tests/ -q" "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src '${PYTHON_BIN}' -m pytest tests/ -q"
fi

run_cmd "scripts/verify_scanner_change.sh --skip-lint" scripts/verify_scanner_change.sh --skip-lint

run_cmd "mkdir -p ${DIST_DIR}" mkdir -p "${DIST_DIR}"
run_cmd "python3 -m build --outdir ${DIST_DIR}" "${PYTHON_BIN}" -m build --outdir "${DIST_DIR}"
run_shell "python3 -m twine check ${DIST_DIR}/*" "'${PYTHON_BIN}' -m twine check '${DIST_DIR}'/*"
run_shell "katana artifacts status --all" "PYTHONPATH=src '${PYTHON_BIN}' -m hermes_katana.cli.main artifacts status --all"

if [[ "${WITH_ARTIFACT_DOWNLOADS}" -eq 1 ]]; then
  ARTIFACT_SMOKE_DIR="${ARTIFACT_SMOKE_DIR:-${ROOT_DIR}/.pytest_tmp/artifact-smoke/${RUN_ID}}"
  run_cmd "mkdir -p ${ARTIFACT_SMOKE_DIR}" mkdir -p "${ARTIFACT_SMOKE_DIR}"
  run_shell "katana artifacts setup --yes --target-dir ${ARTIFACT_SMOKE_DIR} --force" \
    "PYTHONPATH=src '${PYTHON_BIN}' -m hermes_katana.cli.main artifacts setup --yes --target-dir '${ARTIFACT_SMOKE_DIR}' --force"
  run_shell "katana artifacts status minilm --target-dir ${ARTIFACT_SMOKE_DIR}" \
    "PYTHONPATH=src '${PYTHON_BIN}' -m hermes_katana.cli.main artifacts status minilm --target-dir '${ARTIFACT_SMOKE_DIR}'"
fi

if command -v gitleaks >/dev/null 2>&1; then
  run_cmd "gitleaks detect --source . --redact --no-banner --config .gitleaks.toml" \
    gitleaks detect --source . --redact --no-banner --config .gitleaks.toml
elif [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "+ gitleaks detect --source . --redact --no-banner --config .gitleaks.toml"
elif [[ "${ALLOW_MISSING_GITLEAKS}" -eq 1 ]]; then
  echo "gitleaks not installed; skipping because --allow-missing-gitleaks was set"
else
  echo "error: gitleaks is not installed. Install it or rerun with --allow-missing-gitleaks for local smoke." >&2
  exit 127
fi

echo "Release gate passed."
