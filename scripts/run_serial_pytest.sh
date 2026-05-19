#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_PATH="${1:-tests}"
shift $(( $# > 0 ? 1 : 0 ))

if [[ "${TARGET_PATH}" = /* ]]; then
  SEARCH_PATH="${TARGET_PATH}"
else
  SEARCH_PATH="${ROOT_DIR}/${TARGET_PATH}"
fi

if [[ ! -e "${SEARCH_PATH}" ]]; then
  echo "error: target path does not exist: ${TARGET_PATH}" >&2
  exit 2
fi

if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/.pytest_tmp/serial}"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/serial-pytest-${RUN_ID}.log"
SUMMARY_PATH="${LOG_DIR}/serial-pytest-${RUN_ID}-summary.txt"

mapfile -t TEST_FILES < <(find "${SEARCH_PATH}" -type f -name 'test_*.py' | sort)
if [[ "${#TEST_FILES[@]}" -eq 0 ]]; then
  echo "error: no pytest files found under ${TARGET_PATH}" >&2
  exit 2
fi

TOTAL=0
PASSED=0
FAILED=0

for test_file in "${TEST_FILES[@]}"; do
  TOTAL=$((TOTAL + 1))
  rel_path="${test_file#${ROOT_DIR}/}"
  echo "=== ${rel_path} ===" | tee -a "${LOG_PATH}"
  if "${PYTHON_BIN}" -m pytest -q "${test_file}" "$@" >> "${LOG_PATH}" 2>&1; then
    status=0
  else
    status=$?
  fi
  echo "EXIT:${status}" | tee -a "${LOG_PATH}"
  echo >> "${LOG_PATH}"
  if [[ "${status}" -eq 0 ]]; then
    PASSED=$((PASSED + 1))
    echo "PASS ${rel_path}"
  else
    FAILED=$((FAILED + 1))
    echo "FAIL ${rel_path}"
  fi
done

{
  echo "TOTAL=${TOTAL}"
  echo "PASSED=${PASSED}"
  echo "FAILED=${FAILED}"
  echo "LOG_PATH=${LOG_PATH}"
} | tee "${SUMMARY_PATH}"

if [[ "${FAILED}" -ne 0 ]]; then
  exit 1
fi
