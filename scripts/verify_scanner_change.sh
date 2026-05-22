#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
RUN_EVAL=0
SKIP_LINT=0

usage() {
  cat <<'EOF'
Usage: scripts/verify_scanner_change.sh [--dry-run] [--eval] [--skip-lint]

Runs the mandatory scanner-change gate:
  1. Ruff lint/format checks
  2. False-positive smoke gate
  3. Evasion gate
  4. Adversarial dispatch integration pack

Options:
  --dry-run    Print commands without executing them.
  --eval       Also run the slower eval-suite comparison gates.
  --skip-lint  Skip ruff checks when lint was already run separately.
  -h, --help   Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --eval)
      RUN_EVAL=1
      ;;
    --skip-lint)
      SKIP_LINT=1
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

# The local laptop GPU is small; scanner verification should be stable and
# reproducible, not opportunistically consume VRAM. Opt in with
# HERMES_KATANA_VERIFY_USE_CUDA=1 when explicitly benchmarking GPU paths.
if [[ "${HERMES_KATANA_VERIFY_USE_CUDA:-0}" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/.pytest_tmp/scanner-change}"
LOG_PATH="${LOG_DIR}/verify-scanner-change-${RUN_ID}.log"

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

run_cmd() {
  local display="$1"
  shift
  echo "+ ${display}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  "$@" 2>&1 | tee -a "${LOG_PATH}"
}

run_shell() {
  local display="$1"
  local command="$2"
  echo "+ ${display}"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    return 0
  fi
  bash -lc "${command}" 2>&1 | tee -a "${LOG_PATH}"
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
  {
    echo "HermesKatana scanner-change verification"
    echo "run_id=${RUN_ID}"
    echo "root=${ROOT_DIR}"
    echo "python=${PYTHON_BIN}"
    echo
  } | tee "${LOG_PATH}"
fi

if [[ "${SKIP_LINT}" -eq 0 ]]; then
  run_cmd "ruff check src/ tests/" ruff check src/ tests/
  run_cmd "ruff format --check src/ tests/" ruff format --check src/ tests/
fi

run_shell "python3 tests/smoke/false_positive_gate.py" "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src '${PYTHON_BIN}' tests/smoke/false_positive_gate.py"
run_shell "python3 tests/smoke/evasion_gate.py" "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src '${PYTHON_BIN}' tests/smoke/evasion_gate.py"
run_shell "python3 -m pytest tests/integration/test_adversarial_eval_pack.py -q" "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src '${PYTHON_BIN}' -m pytest tests/integration/test_adversarial_eval_pack.py -q"

if [[ "${RUN_EVAL}" -eq 1 ]]; then
  run_shell "HERMES_KATANA_RUN_EVALS=1 python3 -m pytest tests/eval/ -q" "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src HERMES_KATANA_RUN_EVALS=1 '${PYTHON_BIN}' -m pytest tests/eval/ -q"
  run_shell "python3 tests/eval/run_eval.py --compare" "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src '${PYTHON_BIN}' tests/eval/run_eval.py --compare"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
  echo "Scanner-change verification passed. Log: ${LOG_PATH}"
fi
