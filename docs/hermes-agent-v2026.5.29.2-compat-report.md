# Hermes Agent v2026.5.29.2 Compatibility Report

Date: 2026-05-29

## Scope

- HermesKatana repo: `claudlos/hermes-katana`, starting from commit `a4a05c2`.
- Latest Hermes Agent release checked: `v2026.5.29.2`, Hermes Agent `0.15.2`, commit `77a1650c78a4cb1813d8a81fa1da40a15b6a3ec5`.
- Upstream `origin/main` also checked after fixes: commit `7379f175567bd0f1d833eec3a3599d0665b2a491`.

## Executive Summary

The newest Hermes Agent did break HermesKatana in multiple places. The critical source-patch anchors for dispatch, escalation audit, and proxy env injection no longer matched, and the native plugin had become fail-open under modern Hermes because hook exceptions are swallowed and result mutation now requires `transform_tool_result`.

All findings from this audit are fixed in this worktree. The current patch templates apply cleanly to both the `v2026.5.29.2` release and current upstream `origin/main`, install verification passes, and the latest Hermes plugin manager now receives proper block directives and transformed tool results.

## Findings And Fixes

### 1. Native pre-tool hook was fail-open

Modern Hermes catches plugin hook exceptions and only blocks tools when a `pre_tool_call` callback returns:

```python
{"action": "block", "message": "..."}
```

HermesKatana previously raised `KatanaSecurityError` or `EscalationRequired`, so DENY/ESCALATE decisions could be logged and then ignored by Hermes. Fixed in `src/hermes_katana/hermes_plugin.py` by returning Hermes block directives for initialization failure, chain failure, DENY, and ESCALATE.

### 2. Native output redaction was ignored

Hermes now treats `post_tool_call` as observational. Replacing tool output must happen through `transform_tool_result`. HermesKatana previously mutated `ctx.tool_output` inside `post_tool_call`, which did not affect the final model-visible result.

Fixed by registering `transform_tool_result`, caching the processed post-hook result, and returning the scanned/redacted string from the transform hook. Post-dispatch failures now return a JSON error string instead of passing the original tool output through.

### 3. Native plugin config loading changed

Latest Hermes `PluginContext` no longer provides `context.config`, and entry-point plugins are opt-in through `plugins.enabled`. HermesKatana was reading an empty config under the new context.

Fixed by loading Katana settings from current Hermes config shapes:

- `plugins.katana`
- `plugins.entries.katana`
- `plugins.entries.<manifest key/name>.config`

The API docs now state that users must enable the plugin with `plugins.enabled: ["katana"]` when using the native plugin path.

### 4. Current source patches targeted stale Hermes internals

The old `tool_dispatch_hook` and `dispatcher_escalation_audit` patches targeted `tools/registry.py` internals that changed. Latest Hermes dispatch enforcement now belongs around `model_tools.py::handle_function_call`.

Fixed current patch templates:

- `tool_dispatch_hook` now injects pre-dispatch Katana enforcement in `model_tools.py`.
- `dispatcher_escalation_audit` now runs post-dispatch scanning in `model_tools.py`.
- Source-patched Hermes now discovers `.katana` from the patched module path, so protection does not depend on the process cwd.
- `proxy_env_vars` now matches the latest local environment creation.
- Docker and gateway optional patches now match latest files.

### 5. Installer wrote success marker after critical failures

Before this fix, `KatanaInstaller.install()` could write `.katana-installed` even when critical patches failed. That made a broken install look valid until `verify()` was run.

Fixed by raising `RuntimeError` before marker creation when any critical patch returns `ERROR`.

### 6. Compatibility fixtures were stale

The current fixture still represented an April 2026 Hermes snapshot and the compatibility registry only listed Hermes `0.1.0`.

Fixed by:

- Refreshing `hermes-current-snapshot` to release commit `77a1650...`.
- Adding `model_tools.py` to current snapshot coverage.
- Adding generated `0.15.2` core and extended snapshots.
- Updating `tests/fixtures/hermes_compat/fixtures.json`.
- Updating `docs/compatibility.md`.

## Verification

Focused tests:

```text
150 passed, 1 warning
```

Command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest -q \
  tests/unit/test_hermes_plugin.py \
  tests/unit/test_patches.py \
  tests/unit/test_installer_patch_fail_closed.py \
  tests/unit/test_installer.py \
  tests/unit/test_compat_snapshots.py \
  tests/unit/test_batch3_production_profiles.py \
  tests/unit/test_bootstrap.py \
  tests/integration/test_cli_flow.py
```

Latest release patch preview:

```text
tool_dispatch_hook: planned
dispatcher_bootstrap: planned
dispatcher_escalation_audit: planned
proxy_env_vars: planned
banner_integration: planned
docker_proxy_forwarding: planned
gateway_command_scanning: planned
```

Latest release temp install:

```text
all 7 patches applied
verify_valid: True
issues: []
warnings: []
```

Current upstream `origin/main` temp install:

```text
all 7 patches applied
verify_valid: True
issues: []
warnings: []
```

Patched Hermes files compiled successfully after install for both release and `origin/main`.

Native latest-Hermes plugin-manager smoke:

- DENY returned `Tool call 'terminal' denied: synthetic deny`.
- `post_tool_call` and `transform_tool_result` returned `[redacted by katana]`.

Additional live-path smoke after the report fixes:

- A patched Hermes temp checkout allowed `read_file` on a benign file.
- The same checkout blocked `terminal` with `Katana blocked tool 'terminal': Scanner blocked...`.
- An isolated Hermes config with `plugins.enabled: ["katana"]` loaded the entry-point plugin and returned a native hook block message for the same dangerous terminal command.
- The live installed Hermes checkout at `/home/carlos/.hermes/hermes-agent` was patched, verified, and its venv now imports the editable `/home/carlos/hermes-katana` package.
- The live installed Hermes native plugin check now reports `katana_enabled: True` and blocks the dangerous terminal command.
- The live installed Hermes direct `model_tools.handle_function_call()` path allows `echo KATANA_INSTALLED_OK` and blocks the dangerous terminal command.
- An isolated source-patched Hermes checkout with `cwd=/tmp`, no `KATANA_CHECKOUT_ROOT`, and an empty `HERMES_HOME` still allowed `echo KATANA_CWD_OK` and blocked the dangerous terminal command.
- Second-pass live check after restarting the Hermes terminal still allowed `echo KATANA_SECOND_PASS_OK` and blocked the dangerous terminal command.
- The gateway process that was already running before patching still needs a restart before gateway sessions pick up these changes.

Note: in the checked Hermes CLI, `hermes plugins enable katana` did not recognize entry-point plugins even though the plugin manager can discover them. Direct config activation through `plugins.enabled` is the reliable native-plugin activation path for this release.

Hermes proving-ground CLI flags checked on `v2026.5.29.2` and still present:

- `--ignore-user-config`
- `--ignore-rules`
- `--source`
- `--max-turns`
- `--pass-session-id`

## Residual Risk

Source patches are still exact-anchor patches. They are now correct for `v2026.5.29.2` and `origin/main` as of 2026-05-29, but future Hermes refactors can still require another snapshot refresh and patch-template update. This risk is now monitored — see the `hermes-drift` CI job below.

The native plugin path depends on Hermes plugin activation. Users who rely on native hooks rather than source patching must enable `katana` in `plugins.enabled`; otherwise Hermes discovers the entry point but does not load it.

## Follow-up Hardening (2026-05-29)

A second pass addressed the audit's open items. All changes are covered by tests
(focused suite: see Verification below; full suite green apart from one
pre-existing, unrelated `test_semantic_recall` failure).

### A. ESCALATE is now a configurable policy, not a hard-coded block

The first-pass fix made ESCALATE always block. Investigation showed the *old*
escalation handler never actually prompted a human on this Hermes either: it
probed the dispatcher for approval callbacks (`request_approval`,
`confirm_tool_use`, …) that do not exist in Hermes, then fell back to
`KATANA_AUTO_APPROVE_ESCALATIONS` (default deny). So no working human-in-the-loop
was lost — but real HITL is now available.

A new shared resolver, `hermes_katana.escalation.resolve_escalation`, governs the
outcome via `escalate_action`:

- `block` (default) — fail-closed; correct for CLI/gateway/proving-ground.
- `acp_prompt` — prompt the human through Hermes' **generic** approval callback
  (`tools.terminal_tool` approval callback, the same `request_permission` bridge
  Zed/ACP binds). Falls back to `block` when no approver is bound, so headless
  runs never silently allow.
- `auto_approve` — allow with a loud warning; trusted automation only.

Both integration paths use the one resolver: the native plugin reads
`plugins.katana.escalate_action`; the source patch reads
`policy.escalate_action` from the checkout's `.katana/katana.yaml` (plumbed onto
`CheckoutRuntimeState`). Decision: rather than add a brittle card-creation patch,
Katana **invokes Hermes' existing generic approver**, giving a correct tool
approval card with zero new source anchors.

### B. Native output redaction depends on `transform_tool_result` — now asserted

The native plugin redacts output through `transform_tool_result` because Hermes
treats `post_tool_call` as observational. A new contract test
(`tests/unit/test_compat_snapshots.py::TestHermesResultHookContract`) fails the
build if a future snapshot drops the `transform_tool_result` invocation or runs
it before `post_tool_call`, turning a silent fail-open into a loud test failure.

### C. Source-patch + native-plugin double-enforcement guarded

`bootstrap_dispatcher_failsafe` now sets `KATANA_SOURCE_PATCHED=1`. The native
plugin's `pre`/`post`/`transform` hooks check this at hook time and defer when
source patches are active, so a checkout that is both pip-installed and
source-patched no longer scans and denies every tool call twice.

### D. `dispatcher_bootstrap` redundancy clarified (not removed)

The dispatch hook self-discovers its runtime and no longer needs the chain that
`dispatcher_bootstrap` attaches. The patch is retained because it still owns
process-level startup (fail-closed bootstrap state, runtime/proxy env priming,
and the `KATANA_SOURCE_PATCHED` marker). This is documented in `patches.py`; the
two patches have distinct responsibilities.

### E. Anchor-drift early warning (CI)

`scripts/check_hermes_drift.py` applies the current patch templates to a Hermes
checkout and verifies every critical patch lands and all patched files compile.
The `.github/workflows/hermes-drift.yml` job runs it weekly (and on demand)
against the latest Hermes Agent, so anchor drift surfaces as a red build with an
actionable message instead of a field incident.
