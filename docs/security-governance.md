# Security Governance

This repository treats GitHub repository settings, workflow permissions, release artifacts, and scanner outputs as part of the security boundary.

Runtime assurance references:

- Proxy and vault trust boundaries: `docs/proxy-vault-threat-model.md`.
- Alert triage and SLAs: `docs/security-alert-runbook.md`.
- Release dry-run and publishing checklist: `docs/release-checklist.md`.

## Branch protection

The expected `master` branch protection payload is tracked in `.github/branch-protection-master.json`.

Apply it with:

```bash
repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"
gh api \
  --method PUT \
  "repos/${repo}/branches/master/protection" \
  --input .github/branch-protection-master.json
```

The rule requires the CI matrix, release gate, CodeQL analyses, and security scan jobs before merge. The `Analyze (python)` and `Analyze (actions)` checks come from GitHub CodeQL default setup; enable default setup or add an equivalent CodeQL workflow before applying this payload to a fresh fork. The rule also requires one approving review, CODEOWNERS review, stale-review dismissal, and conversation resolution. Admin enforcement is intentionally disabled so the repository owner can recover from broken protection settings.

## GitHub security settings

Expected repository settings:

- Dependency graph: enabled.
- Dependabot alerts: enabled.
- Dependabot security updates: enabled.
- Secret scanning: enabled.
- Secret scanning push protection: enabled.
- Secret scanning non-provider patterns: enabled when available for the repository plan.
- Secret scanning validity checks: enabled when available for the repository plan.
- Code scanning: fed by CodeQL, Semgrep, Hadolint, Trivy, OSV, and zizmor workflows.

Posture-only tools such as OpenSSF Scorecard should publish artifacts rather than SARIF. This avoids filling code scanning with broad repository posture findings that are better handled as governance work.

## Required scanner split

Upload SARIF for actionable code findings:

- CodeQL.
- Semgrep OSS.
- Bandit.
- Hadolint.
- Trivy filesystem and image scans.
- OSV scanner.
- zizmor workflow hardening.

Publish artifact-only reports for posture and deep scheduled context:

- OpenSSF Scorecard JSON.
- Deep scheduled Semgrep JSON.
- Deep scheduled Bandit JSON.
- Deep scheduled pip-audit JSON.
- Deep scheduled Trivy JSON.
- Deep scheduled zizmor JSON.

## Release artifacts

The release gate builds the wheel and source distribution, generates a CycloneDX SBOM, uploads all release artifacts, and creates GitHub artifact attestations on non-PR events. Pull requests generate the SBOM and upload artifacts, but attestation is skipped because fork and PR token permissions should stay constrained.

Artifact attestation runs in a separate non-PR job with `attestations: write` and `id-token: write`. The build/test/package job keeps only `contents: read`, so ordinary pull requests do not receive release signing scopes.

Package publishing should use PyPI trusted publishing through GitHub Actions OIDC. Long-lived PyPI API tokens are not part of the intended release path.

## Dependabot policy

Dependabot is allowed to open GitHub Actions and Python dependency update PRs. Updates remain manual by default. Low-risk bot PRs can be merged after the required checks pass, but automatic merge is not enabled until branch protection and the required-check set have proven stable.

Recommended merge order for bot PRs:

1. GitHub Actions patch/minor updates.
2. Python patch updates.
3. Python minor updates.
4. Major version updates only after reading release notes.

Security updates may bypass the normal cooldown when GitHub opens them as Dependabot security updates.

## Alert triage

Handle open alerts in this order:

1. Confirm the finding is on the current default branch.
2. Fix real code or workflow findings in a PR.
3. Wait for the post-merge default-branch analysis to finish.
4. Delete stale analysis records only when every remaining alert points to obsolete analysis uploads.
5. Dismiss alerts only with a concrete reason and only when the finding is intentionally accepted.
