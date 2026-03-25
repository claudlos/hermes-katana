# HermesKatana Handoff

Date: 2026-03-25  
Repo: `C:\Users\Carlos\HermesKatana\hermes-katana`

## Current status

The current hardening work is complete.

HermesKatana now has:

- checkout-driven runtime bootstrap
- audit-visible short-circuit and escalation denials
- installer `dry-run`, backup, and restore flows
- explicit Hermes compatibility snapshots with a support matrix
- automated snapshot refresh tooling
- required provenance verification for non-dry-run snapshot refreshes
- backfilled tree-checksum provenance on the existing `0.1.0` snapshot entries

## Main outcomes

### Runtime and enforcement

- `katana run --target <checkout>` now loads checkout-local Katana state through [bootstrap.py](/C:/Users/Carlos/HermesKatana/hermes-katana/src/hermes_katana/bootstrap.py).
- Hermes dispatch bootstrap is wired end to end through [patches.py](/C:/Users/Carlos/HermesKatana/hermes-katana/src/hermes_katana/installer/patches.py).
- Denied pre-dispatch calls and denied escalation outcomes now reach the tamper-evident audit trail through [chain.py](/C:/Users/Carlos/HermesKatana/hermes-katana/src/hermes_katana/middleware/chain.py) and [integration.py](/C:/Users/Carlos/HermesKatana/hermes-katana/src/hermes_katana/middleware/integration.py).

### Installer and recovery

- [installer.py](/C:/Users/Carlos/HermesKatana/hermes-katana/src/hermes_katana/installer/installer.py) supports:
  - install and uninstall previews
  - manifest-backed backups
  - `katana restore --manifest ...`
- Checkout-local backup state lives under `<checkout>/.katana-backups/`.

### Compatibility and provenance

- Supported Hermes fixtures are now explicit snapshots:
  - [hermes-v0.1.0-core-snapshot](/C:/Users/Carlos/HermesKatana/hermes-katana/tests/fixtures/hermes_compat/hermes-v0.1.0-core-snapshot)
  - [hermes-v0.1.0-extended-snapshot](/C:/Users/Carlos/HermesKatana/hermes-katana/tests/fixtures/hermes_compat/hermes-v0.1.0-extended-snapshot)
- Registry metadata lives in [fixtures.json](/C:/Users/Carlos/HermesKatana/hermes-katana/tests/fixtures/hermes_compat/fixtures.json).
- Snapshot refresh automation lives in [compat_snapshots.py](/C:/Users/Carlos/HermesKatana/hermes-katana/src/hermes_katana/installer/compat_snapshots.py) and [refresh_compat_snapshots.py](/C:/Users/Carlos/HermesKatana/hermes-katana/scripts/refresh_compat_snapshots.py).
- Non-dry-run refreshes now require one of:
  - `--source-archive` plus `--archive-sha256`
  - `--source-tree-sha256`
- The current `0.1.0` snapshot entries now include backfilled tree-checksum provenance from the pinned snapshot directories.

### CI, tests, and docs

- Operator-contract coverage runs in [ci.yml](/C:/Users/Carlos/HermesKatana/hermes-katana/.github/workflows/ci.yml).
- Compatibility refresh coverage lives in [test_compat_snapshots.py](/C:/Users/Carlos/HermesKatana/hermes-katana/tests/unit/test_compat_snapshots.py).
- Main operator docs are:
  - [README.md](/C:/Users/Carlos/HermesKatana/hermes-katana/README.md)
  - [quickstart.md](/C:/Users/Carlos/HermesKatana/hermes-katana/docs/quickstart.md)
  - [runbook.md](/C:/Users/Carlos/HermesKatana/hermes-katana/docs/runbook.md)
  - [compatibility.md](/C:/Users/Carlos/HermesKatana/hermes-katana/docs/compatibility.md)

## Validation

Commands run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -p pytest_asyncio.plugin -q
python -m py_compile src/hermes_katana/bootstrap.py src/hermes_katana/cli/main.py src/hermes_katana/installer/__init__.py src/hermes_katana/installer/installer.py src/hermes_katana/installer/patches.py src/hermes_katana/installer/compat_snapshots.py src/hermes_katana/middleware/chain.py src/hermes_katana/middleware/integration.py scripts/refresh_compat_snapshots.py tests/hermes_compat.py tests/unit/test_compat_snapshots.py tests/unit/test_bootstrap.py tests/unit/test_middleware.py tests/unit/test_installer.py tests/unit/test_cli.py tests/integration/test_cli_flow.py tests/integration/test_adversarial_eval_pack.py
```

Results:

- `259 passed`
- compile pass succeeded

## Residual risks

- Provenance verification still depends on a maintainer-supplied trusted checksum or verified archive path. The repo does not fetch authoritative upstream checksums on its own.
- The `0.1.0` fixture entries have tree-backed provenance, not archive-backed provenance.
- Restore is manifest-driven and assumes the referenced backup tree is still present and readable.

## Recommended next action

If you continue hardening this area, the next step is to automate retrieval or verification of authoritative Hermes release checksums so snapshot refresh does not depend on manually transcribed digests.
