# Security Alert Runbook

This runbook defines how to triage repository security alerts and runtime security regressions.

## Ownership

Security-sensitive paths are covered by `CODEOWNERS`. Changes to workflows, vault, proxy, scanner, release scripts, Docker, and security policy files need an approving review before merge.

## Severity SLAs

| Severity | Examples | First response | Target fix |
|---|---|---:|---:|
| Critical | Active secret exposure, exploitable credential injection, publish compromise | 4 hours | 24 hours |
| High | CodeQL high alert, proxy scan bypass with secret impact, vulnerable release dependency | 1 business day | 3 business days |
| Medium | Hardening gap, noisy but real SAST finding, stale vulnerable dev dependency | 3 business days | 10 business days |
| Low | Documentation drift, posture finding, non-exploitable false positive candidate | 5 business days | Next planned maintenance |

## Triage Flow

1. Confirm the alert is on the current default branch and not from stale analysis.
2. Reproduce locally where practical with the smallest command or test case.
3. Classify the finding by impact, exploitability, affected release path, and whether secrets can cross a boundary.
4. Fix real code or workflow findings in a PR with a targeted regression test.
5. Wait for default-branch analysis after merge before considering the alert resolved.
6. Delete stale analysis records only when all remaining alerts point to obsolete uploads.
7. Dismiss only with a concrete reason: false positive, test-only fixture, accepted risk, or used in tests.

## Alert Types

### Code Scanning

- CodeQL, Semgrep, Bandit, OSV, Trivy, Hadolint, and zizmor findings should be fixed in code unless a clear false-positive reason exists.
- Avoid broad suppressions. Prefer narrow source changes or test fixture adjustments.
- If multiple tools report the same root cause, fix once and wait for every analyzer to refresh.

### Secret Scanning

- Treat verified or provider-recognized tokens as critical.
- Revoke first, then rotate any dependent credentials.
- Remove the value from source and history if it is real and reachable.
- Add or update tests using secret-shaped placeholders only when the scanner configuration intentionally allows them.

### Dependency Alerts

- Security updates may bypass normal Dependabot cooldown.
- Prefer patch updates, then minor updates after reading release notes.
- Major version updates require a compatibility note and full CI.

### Runtime Regressions

Runtime alerts include proxy scanning bypasses, vault memory/cache regressions, credential injection mistakes, and log redaction failures. Required evidence for a fix:

- A regression test under `tests/` that fails before the fix.
- A focused explanation of the trust boundary involved.
- Confirmation that strict/max mode fails closed where appropriate.

## Escalation

Escalate to a release-blocking issue when any of the following are true:

- A release artifact was published without required provenance or SBOM evidence.
- A live credential may have been logged, committed, or injected into the wrong provider.
- The proxy can forward unscanned secret-bearing traffic in strict or max mode.
- Branch protection or required checks were bypassed.

## Closeout

Every security closeout should record:

- Alert URL or run ID.
- Root cause.
- Fix PR and merge commit.
- Tests added or updated.
- Any residual risk or operator action, such as token revocation.
