#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=0
SKIP_FULL_TESTS=0
ALLOW_MISSING_GITLEAKS=0
ALLOW_UNTAGGED=0

usage() {
  cat <<'EOF'
Usage: scripts/release_checklist.sh [--dry-run] [--skip-full-tests] [--allow-missing-gitleaks] [--allow-untagged]

Runs the maintainer release checklist:
  1. Confirm the working tree is clean.
  2. Confirm the release commit is tagged, unless --allow-untagged is set.
  3. Verify generated policy assets.
  4. Run the local release gate.
  5. Verify the GitHub release workflow still produces SBOMs and attestations.
  6. Verify the documented PyPI path uses trusted publishing/OIDC, not long-lived tokens.

Options:
  --dry-run                 Print checks and commands without executing them.
  --skip-full-tests         Pass through to scripts/release_gate.sh.
  --allow-missing-gitleaks  Pass through to scripts/release_gate.sh.
  --allow-untagged          Permit local release rehearsal without a tag at HEAD.
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
    --allow-missing-gitleaks)
      ALLOW_MISSING_GITLEAKS=1
      ;;
    --allow-untagged)
      ALLOW_UNTAGGED=1
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

require_pattern() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  run_cmd "${label}: grep -q '${pattern}' ${file}" grep -q "${pattern}" "${file}"
}

run_shell "git working tree clean" "test -z \"\$(git status --porcelain)\""

if [[ "${ALLOW_UNTAGGED}" -eq 1 ]]; then
  run_cmd "git tag --points-at HEAD" git tag --points-at HEAD
else
  run_shell "release tag points at HEAD" "git tag --points-at HEAD | grep -Eq '^v?[0-9]'"
fi

run_cmd "python3 scripts/generate_policy_assets.py --check" "${PYTHON_BIN}" scripts/generate_policy_assets.py --check

release_gate_args=()
if [[ "${DRY_RUN}" -eq 1 ]]; then
  release_gate_args+=(--dry-run)
fi
if [[ "${SKIP_FULL_TESTS}" -eq 1 ]]; then
  release_gate_args+=(--skip-full-tests)
fi
if [[ "${ALLOW_MISSING_GITLEAKS}" -eq 1 ]]; then
  release_gate_args+=(--allow-missing-gitleaks)
fi
run_cmd "scripts/release_gate.sh ${release_gate_args[*]}" scripts/release_gate.sh "${release_gate_args[@]}"

require_pattern "release workflow generates CycloneDX SBOM" ".github/workflows/release-gate.yml" "Generate CycloneDX SBOM"
require_pattern "release workflow attests provenance" ".github/workflows/release-gate.yml" "Attest release artifact provenance"
require_pattern "release workflow attests SBOM" ".github/workflows/release-gate.yml" "Attest release artifact SBOM"
require_pattern "release docs require PyPI trusted publishing" "docs/release-checklist.md" "trusted publishing"
require_pattern "release docs require OIDC" "docs/release-checklist.md" "OIDC"

echo "Release checklist passed."
