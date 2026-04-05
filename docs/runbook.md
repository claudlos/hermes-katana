# HermesKatana Runbook

## Primary operator commands

- Health: `katana doctor` and `katana doctor --target /path/to/hermes`
- Local state: `katana status`
- Hermes checkout state: `katana status --target /path/to/hermes`
- Install and remove patches: `katana install --target /path/to/hermes`, `katana uninstall --target /path/to/hermes`
- Runtime bootstrap: `katana run --target /path/to/hermes -- --task "hello"`
- Policy: `katana policy list`, `katana policy use paranoid`, `katana policy export policy.yaml`
- Vault: `katana vault list`, `katana vault set KEY VALUE`, `katana vault verify`, `katana vault rotate`
- Audit: `katana audit show --limit 20`, `katana audit verify`, `katana audit stats`
- Proxy: `katana proxy start`, `katana proxy status`, `katana proxy stop`

## State locations

- Global config file: `~/.hermes-katana/config.yaml`
- Vault file: `~/.config/hermes-katana/vault.json`
- Vault lock sentinel: `~/.config/hermes-katana/vault.lock`
- Audit trail: `~/.config/hermes-katana/audit/audit.jsonl`
- Proxy pid file: OS temp directory as `hermes_katana_proxy.pid`
- Checkout config: `<hermes>/.katana/katana.yaml`
- Checkout audit dir: `<hermes>/.katana/audit/`
- Checkout certs: `<hermes>/.katana/certs/`
- Checkout backups: `<hermes>/.katana-backups/`

## Routine workflows

### Inspect a machine

1. Run `katana doctor`.
2. If a Hermes checkout is involved, run `katana doctor --target /path/to/hermes`.
3. Run `katana audit stats` and `katana vault verify` if local state already exists.

### Change the active policy

1. Run `katana policy use balanced` or `katana policy use paranoid`.
2. Confirm with `katana policy list`.
3. If needed, export the current set with `katana policy export current-policy.yaml`.

`policy use` persists to `~/.hermes-katana/config.yaml`. It is not a process-only
environment toggle anymore.

### Install into Hermes

1. Verify the checkout with `katana doctor --target /path/to/hermes`.
2. Preview the patch set with `katana install --target /path/to/hermes --dry-run`.
3. Run `katana install --target /path/to/hermes --backup`.
4. Confirm with `katana status --target /path/to/hermes`.
5. Start Hermes through `katana run --target /path/to/hermes -- ...`.

The checkout-local config now drives runtime behavior. `katana run --target`
loads `.katana/katana.yaml`, exports the matching Katana environment
variables, and starts the configured proxy if needed.

### Recover from install or uninstall mistakes

1. Find the most recent manifest under `<hermes>/.katana-backups/`.
2. Preview the rollback with `katana restore --manifest <manifest> --dry-run`.
3. Apply the rollback with `katana restore --manifest <manifest>`.
4. Re-run `katana status --target /path/to/hermes` to confirm the checkout
   matches the expected state.

### Recover from proxy issues

1. Run `katana proxy status`.
2. Run `katana doctor` and confirm `mitmdump` is available.
3. If the proxy is stopped, run `katana proxy start`.
4. If `proxy status` shows stopped, the pidfile is cleared automatically; retry a clean start.

### Recover from vault issues

1. Run `katana vault verify`.
2. If the vault is locked, remove the circuit breaker with `katana vault unlock`.
3. If the key must be rotated, run `katana vault rotate`.

### Reset audit state

1. Verify the chain with `katana audit verify`.
2. Review recent entries with `katana audit show --limit 20`.
3. Clear the current file only when needed with `katana audit clear`.

## CI and validation

The repo CI should run on Linux and Windows with:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -p pytest_asyncio.plugin -q
```

Use the explicit `pytest_asyncio.plugin` entry point to avoid config warnings
when plugin autoload is disabled.

Compatibility and adversarial validation:

- Compatibility fixtures: `tests/fixtures/hermes_compat/`
- Adversarial eval pack: `evals/adversarial_dispatch.yaml`
- Dedicated operator-contract CI job: `.github/workflows/ci.yml`

## Maintainer workflow: refresh Hermes compatibility snapshots

1. Check out or extract the Hermes release source tree you want to support.
2. Preview the refresh with:
   `python scripts/refresh_compat_snapshots.py --source /path/to/hermes-release --source-ref vX.Y.Z --dry-run`
3. Refresh the pinned snapshots with:
   `python scripts/refresh_compat_snapshots.py --source /path/to/hermes-release --source-archive /path/to/hermes-vX.Y.Z.tar.gz --archive-sha256 <published_sha256> --source-ref vX.Y.Z --replace-existing`
4. If you do not have the release archive, use:
   `python scripts/refresh_compat_snapshots.py --source /path/to/hermes-release --source-tree-sha256 <trusted_sha256> --source-ref vX.Y.Z --replace-existing`
5. Run:
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest -p pytest_asyncio.plugin -q tests/unit/test_compat_snapshots.py tests/unit/test_installer.py tests/unit/test_bootstrap.py tests/integration/test_cli_flow.py`
6. Update [compatibility.md](/C:/Users/Carlos/HermesKatana/hermes-katana/docs/compatibility.md) if the supported snapshot list changed.
