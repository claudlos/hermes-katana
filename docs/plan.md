# Current Slice

Date: 2026-03-24

Goal: require provenance verification when refreshing the pinned Hermes
compatibility snapshots so the support matrix can only be updated from a
verified release archive or a verified source tree.

Acceptance criteria:

- Non-dry-run snapshot refreshes are rejected without provenance verification.
- Archive checksum verification and source-tree checksum verification are both supported.
- The refresh workflow and refusal path are covered by unit tests.
- Docs and handoff explain the verified refresh contract.

Checklist:

- [x] Add provenance verification helpers to the snapshot refresh module.
- [x] Require verified provenance for non-dry-run snapshot refreshes.
- [x] Extend snapshot-refresh tests for verified and rejected flows.
- [x] Update maintainer docs and handoff for the verified refresh workflow.
- [x] Run the full validation suite.
