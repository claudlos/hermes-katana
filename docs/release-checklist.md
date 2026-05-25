# Release Checklist

Use this checklist before publishing a GitHub release or package artifact.

## Dry Run

Run the checklist in dry-run mode before tagging:

```bash
scripts/release_checklist.sh --dry-run --allow-untagged --allow-missing-gitleaks
```

Run the full local checklist from the tagged release commit:

```bash
scripts/release_checklist.sh --allow-missing-gitleaks
```

Use `--skip-full-tests` only for a local rehearsal after the same commit already has passing CI and release-gate checks.

## Pre-Release Checks

1. Confirm the working tree is clean.
2. Confirm the release commit has a version tag at `HEAD`.
3. Confirm generated policy assets are current with `python scripts/generate_policy_assets.py --check`.
4. Run `scripts/release_gate.sh`.
5. Confirm the GitHub Release Gate workflow uploads wheel, sdist, and CycloneDX SBOM artifacts.
6. Confirm non-PR Release Gate runs create provenance and SBOM attestations.
7. Confirm changelog or release notes describe security-impacting changes and any operator action.

## Publishing Model

PyPI publishing should use trusted publishing with GitHub Actions OIDC. Do not add long-lived PyPI API tokens to repository or organization secrets for the normal release path.

Expected PyPI trusted publishing setup:

- Publisher: GitHub Actions.
- Repository: `claudlos/hermes-katana`.
- Workflow: the package publishing workflow when it is added.
- Environment: `pypi`, protected by reviewer approval.
- Identity: OIDC token issued by GitHub for the release workflow.

Until trusted publishing is configured for the PyPI project, stop after GitHub release artifact generation and attestations. Manual package upload remains an exception path and must be documented in the release notes.

## Post-Release Verification

1. Confirm the default-branch Release Gate completed successfully.
2. Confirm artifact attestations exist for the wheel and sdist.
3. Download the release artifacts into a clean environment and run `twine check`.
4. Install the wheel in a fresh virtual environment and run `katana artifacts status --all`.
5. Confirm code scanning and security scans have no new open alerts on `master`.
