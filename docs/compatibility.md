# Hermes Compatibility Matrix

HermesKatana supports the Hermes checkout layouts listed in the pinned snapshot
registry at `tests/fixtures/hermes_compat/fixtures.json`. Anything outside this
matrix is unsupported until a new snapshot is checked in and the installer
contract tests pass against it.

## Supported snapshots

| Snapshot ID | Hermes version | Profile | Checkout path | Expected patch behavior | Status |
|-------------|----------------|---------|---------------|-------------------------|--------|
| `hermes-v0.1.0-core-snapshot` | `0.1.0` | Core | `tests/fixtures/hermes_compat/hermes-v0.1.0-core-snapshot` | Required dispatcher patches apply; optional UI, Docker, and gateway patches skip cleanly | Supported |
| `hermes-v0.1.0-extended-snapshot` | `0.1.0` | Extended | `tests/fixtures/hermes_compat/hermes-v0.1.0-extended-snapshot` | All current patches apply | Supported |
| `hermes-v0.15.2-core-snapshot` | `0.15.2` | Core | `tests/fixtures/hermes_compat/hermes-v0.15.2-core-snapshot` | Required `model_tools.py`, registry, and terminal patches apply | Supported |
| `hermes-v0.15.2-extended-snapshot` | `0.15.2` | Extended | `tests/fixtures/hermes_compat/hermes-v0.15.2-extended-snapshot` | All current patches apply, including banner, Docker, and gateway integration | Supported |

## Support contract

- Installer detection must accept every snapshot in the support matrix.
- Install, verify, uninstall, backup, and restore must pass on every supported snapshot.
- Snapshot metadata must match the checkout's own `pyproject.toml` version.
- New or refreshed snapshots must include verified provenance metadata.
- New Hermes layouts are not supported by assumption. They become supported only after:
  - a pinned snapshot is added
  - `tests/unit/test_installer.py` and the operator-contract CI job pass
  - this matrix is updated

## What this protects

- Installer detection stays tied to real patch targets instead of generic repo markers.
- Patch application and reversion stay stable when multiple patches touch the same file.
- Release support becomes explicit instead of being inferred from abstract `minimal` and `full` labels.

## Refresh procedure

Refresh the pinned matrix from a real Hermes release checkout or extracted source tree with verified provenance:

```bash
python scripts/refresh_compat_snapshots.py --source /path/to/hermes-release --source-archive /path/to/hermes-vX.Y.Z.tar.gz --archive-sha256 <published_sha256> --source-ref vX.Y.Z --replace-existing
```

Notes:

- The refresh tool derives the snapshot file set from the current Katana patch definitions.
- `core` snapshots include only the required patch targets plus Hermes markers.
- `extended` snapshots include the full current patch surface.
- Use `--dry-run` first to preview the snapshot ids and current source tree hash.
- If you do not have a release archive, you can verify the extracted source tree directly with `--source-tree-sha256 <trusted_sha256>`.
- Non-dry-run refreshes are rejected unless archive or source-tree provenance has been verified.
- The current `0.1.0` bootstrap snapshots now include backfilled tree-checksum provenance from the pinned snapshot directories. Refresh them from a verified release archive if you want archive-backed provenance instead.

## Validation

Run the compatibility coverage with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -p pytest_asyncio.plugin -q tests/unit/test_compat_snapshots.py tests/unit/test_installer.py tests/unit/test_bootstrap.py tests/integration/test_cli_flow.py
```
