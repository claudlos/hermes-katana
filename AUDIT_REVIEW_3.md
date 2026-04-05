# AUDIT REVIEW 3

Project: /home/carlos/Documents/Code/hermes-katana
Reviewer stance: hostile code review

## Executive summary
This codebase is not clean. The worst problems are not cosmetic either: the vault rotation path writes master keys to disk, vault writes are still race-prone across processes, installer restore trusts a tamperable manifest and can overwrite or delete paths outside the checkout, bootstrap accepts checkout-local path escapes, and the proxy leaks scanner output back to clients/logs while also claiming success when no proxy process was actually started.

I also found multiple fake-comfort tests: tests that inspect source instead of behavior, a literally tautological assertion, and several critical paths with little or no adversarial coverage.

## Findings

### 1. Critical: vault key rotation writes both old and new master keys to disk
- File: `src/hermes_katana/vault/store.py:663-675`
- Problem:
  `rotate_key()` serializes both `old_key` and `new_key` into `.rotation_journal` as base64 strings on disk.
- Why this is bad:
  The vault advertises a security model where the master key does not touch disk. Rotation violates that model completely. Any local attacker, backup system, crash dump, or stale temp artifact can capture both the old and new vault master keys.
- Extra sloppiness:
  The journal is weaker than the vault file itself: no hardening to 0600 is evident here, and there is no fsync before proceeding, so crash consistency is also weaker.
- Impact:
  Total vault compromise during/after rotation.

### 2. High: interrupted rotation recovery validates only the first entry
- File: `src/hermes_katana/vault/store.py:741-763`
- Problem:
  `recover_rotation()` decides whether the vault is in old-key or new-key state by trying to decrypt only the first entry.
- Why this is bad:
  A partially re-encrypted or partially corrupted vault can still have a decryptable first entry. Recovery can then wrongly declare success, delete the journal, and strand the rest of the vault in an unreadable mixed state.
- Impact:
  Integrity failure and irreversible data loss during crash recovery.

### 3. High: vault `set()` / `remove()` are still vulnerable to lost-update races across processes
- Files: `src/hermes_katana/vault/store.py:454-456, 481-496, 551-557, 570-579`
- Problem:
  The code locks `_read_vault()` and `_write_vault()` individually, but does not hold one file lock over the entire read-modify-write transaction in `set()` and `remove()`.
- Why this is bad:
  Two processes can read the same old state, apply different updates, and whichever writes last silently clobbers the other. That means the advertised locking is incomplete.
- Impact:
  Silent secret loss/corruption under concurrent use.

### 4. Medium: `get()` does not uniformly enforce integrity if entries exist but HMAC is missing
- File: `src/hermes_katana/vault/store.py:523-538`
- Problem:
  `get()` verifies integrity only when both `entries` and `stored_hmac` are present. If entries exist and HMAC is absent, it continues into decryption instead of failing hard.
- Why this is bad:
  A stripped/downgraded vault can still be processed on reads even though `verify_integrity()` treats missing HMAC on non-empty vaults as invalid.
- Impact:
  Inconsistent tamper handling and possible acceptance of corrupted vault state.

### 5. Low/Medium: malformed master key material is not rejected at ingestion
- File: `src/hermes_katana/vault/store.py:297-312`
- Problem:
  `_get_master_key()` base64-decodes env/keyring values without clearly rejecting malformed input or enforcing 32-byte key length at the boundary.
- Why this is bad:
  Bad keys propagate deeper into cryptographic code and fail later, unpredictably.
- Impact:
  Reliability/DoS issue and poor key hygiene.

### 6. Medium: access-log integrity key is not secret by default
- File: `src/hermes_katana/vault/access_log.py:250-257`
- Problem:
  The default HMAC key is derived from `sha256(b"hermes-katana-access-log:" + path)`. That is deterministic and knowable to anyone who knows the log path.
- Why this is bad:
  That is not authentication. An attacker who can edit the file can recompute valid tags.
- Impact:
  The default access-log integrity protection is fake unless `HERMES_KATANA_LOG_KEY` is securely configured.

### 7. Medium: access-log read path ignores HMAC validation
- File: `src/hermes_katana/vault/access_log.py:197-223`
- Problem:
  `_query()` parses and returns entries without verifying their appended HMACs.
- Why this is bad:
  Consumers of `get_access_history()` / `get_all_access()` can ingest attacker-modified records as if they were trusted unless they separately remember to call `verify_integrity()`.
- Impact:
  Integrity is opt-in on the main read path, which defeats the whole point.

### 8. Medium: access-log claims file locking but only uses an in-process mutex
- Files: `src/hermes_katana/vault/access_log.py:7, 108, 144-153, 225-235`
- Problem:
  The module docstring claims file locking, but the implementation uses `threading.Lock()` only.
- Why this is bad:
  Multiple processes can still race on append and rotation. Rename-based rotation can race writers and drop or split records.
- Impact:
  Cross-process corruption/loss and misleading documentation.

### 9. Medium: expiry metadata writes can fail silently while callers assume success
- File: `src/hermes_katana/vault/expiry.py:195-217`
- Problem:
  `_write()` catches `OSError`, logs a warning, and returns without surfacing failure. Callers like `set_expiry()`, `extend_expiry()`, `remove_expiry()`, and `sync_with_vault()` do not check for failure.
- Why this is bad:
  The API can report success while TTL metadata was never persisted.
- Impact:
  Silent policy failure and inconsistent expiry behavior.

### 10. Critical: installer restore trusts `manifest.json` and can overwrite/delete arbitrary paths
- File: `src/hermes_katana/installer/installer.py:497-526`
- Problem:
  `restore()` trusts `manifest.target`, `manifest.backup_root`, `manifest.files`, and `manifest.missing_paths` from `manifest.json` without containment validation.
- Why this is bad:
  A tampered manifest can:
  - copy arbitrary readable files into arbitrary destinations,
  - escape the checkout via `..`,
  - use absolute destination paths,
  - delete arbitrary files/directories through `missing_paths`.
- Concrete abuse cases:
  - `manifest.files = ["/etc/passwd"]`
  - `manifest.files = ["../../.ssh/authorized_keys"]`
  - `manifest.missing_paths = ["../../important_dir"]`
- Impact:
  Arbitrary file overwrite and delete outside the repo. This is a real exploitation primitive, not a theoretical style nit.

### 11. Medium/High: restore path operations do not defend against symlink tricks
- File: `src/hermes_katana/installer/installer.py:510-526`
- Problem:
  `restore()` uses `copytree`, `copy2`, `rmtree`, and `unlink` on manifest-derived paths without symlink containment checks.
- Why this is bad:
  Even if manifest contents are nominally relative, symlinked destinations or roots can redirect writes/deletes outside the intended tree.
- Impact:
  More arbitrary filesystem damage through symlink abuse.

### 12. High: bootstrap trusts checkout-local `katana.yaml` paths that can escape the checkout
- Files: `src/hermes_katana/bootstrap.py:131-156, 259-267, 275-281, 337-338`
- Problem:
  `load_checkout_state()` reads `.katana/katana.yaml` and resolves `policy.custom_dir`, `audit.trail_dir`, and `proxy.ca_cert` using `_resolve_checkout_path()` without enforcing containment under the checkout.
- Why this is bad:
  A malicious checkout-local config can redirect:
  - policy loading to arbitrary filesystem locations,
  - audit logs to arbitrary paths,
  - CA cert path exports to arbitrary files.
- Impact:
  Path traversal / symlink escape from local project config, bypassing stronger validation in the main config loader.

### 13. Medium/High: plugin context recovery ignores `task_id` and can mix contexts between concurrent calls
- File: `src/hermes_katana/hermes_plugin.py:477-488`
- Problem:
  `_pop_context(tool_name, task_id)` ignores `task_id` entirely and returns the most recent context matching only `tool_name`.
- Why this is bad:
  Interleaved or concurrent calls to the same tool can pick up the wrong pre-call context.
- Impact:
  Broken audit attribution, taint/context corruption, and possible security policy mismatches under concurrency.

### 14. Medium: metrics API drops dimensions it pretends to collect
- File: `src/hermes_katana/metrics.py:160-175`
- Problem:
  `record_policy_eval(preset, tool_name, result, latency_ms)` only increments `self._policy_evals[(result,)]`.
- Why this is bad:
  `preset`, `tool_name`, and `latency_ms` are thrown away despite being part of the interface.
- Impact:
  Observability is materially weaker than the API suggests. Incident analysis by tool/preset is impossible.

### 15. Medium: injected Google credentials are not excluded from proxy scanning
- Files: `src/hermes_katana/proxy/addon.py:318-323`, `src/hermes_katana/proxy/injector.py:64-72, 224-230`
- Problem:
  The addon suppresses scanning for injected `authorization`, `api-key`, and `x-api-key`, but the injector also supports `x-goog-api-key` and that header is not exempted.
- Why this is bad:
  A proxy-injected Google credential can immediately be re-scanned and potentially blocked/logged/audited as outbound secret material.
- Impact:
  Self-inflicted leakage and broken credential injection behavior.

### 16. High: proxy block responses echo scanner summaries back to clients
- Files: `src/hermes_katana/proxy/addon.py:332, 346, 359, 375, 403-405, 459-460, 490-492, 573`
- Problem:
  The proxy returns `scan_result["summary"]` directly to requesters in block responses.
- Why this is bad:
  If scanner summaries include matched fragments, detector names, or user-controlled content, the proxy reflects sensitive detection output back to clients.
- Impact:
  Information leakage and easier evasion by attackers.

### 17. High: proxy logs and audit records also ingest scanner summaries and raw exception text
- Files: `src/hermes_katana/proxy/addon.py:398-411, 458-463, 484-499, 540-543, 572-580, 335-336, 349-350, 362-363, 378-379, 422-423, 465-466, 508-509, 545-546, 582-583`
- Problem:
  `scan_result["summary"]` is logged and stored in audit details; exception paths also log raw exception strings.
- Why this is bad:
  Logs and audit trails become secret sinks if summaries or exceptions contain payload excerpts or sensitive values.
- Impact:
  Internal leakage channel for the very secrets the proxy is supposed to protect.

### 18. High: `runner.start()` can claim the proxy started when mitmproxy is missing
- File: `src/hermes_katana/proxy/runner.py:384-406, 408-442`
- Problem:
  On `FileNotFoundError`, the code falls back to `pid = os.getpid()`, writes the PID file, starts watchdog/health machinery, and logs success.
- Why this is bad:
  The system enters a false-running state even though no proxy child exists.
- Impact:
  Operational lies, broken lifecycle management, misleading health checks, and automation failures.

### 19. Medium: response metrics count blocked responses as passed
- File: `src/hermes_katana/proxy/addon.py:482-520`
- Problem:
  After blocking a response body, the code increments `responses_blocked_scan`, rewrites the response, but does not return before later incrementing `responses_passed`.
- Why this is bad:
  Metrics are internally inconsistent.
- Impact:
  Monitoring understates blocking severity and overstates clean pass-through traffic.

### 20. Medium: provider matching in injector is exact-host only
- Files: `src/hermes_katana/proxy/injector.py:155-164, 196-198`
- Problem:
  Provider resolution is exact domain lookup only.
- Why this matters:
  Real-world provider traffic often uses alternate hostnames or subdomains. Injection silently fails there.
- Impact:
  Not a direct exploit, but a likely robustness hole that can lead to confusing misbehavior and accidental credential non-injection.

## Test failures in the test suite itself

### 21. Worthless test: `_get_master_key` test patches the function and asserts the mock behavior
- File: `tests/test_vault_safety.py:104-112`
- Problem:
  It patches `hermes_katana.vault.store._get_master_key` and then asserts the patched function returns `None`.
- Why this is bad:
  This tests the mocking framework, not the code.
- Impact:
  False confidence.

### 22. Worthless test: locking tests inspect source code instead of behavior
- File: `tests/test_vault_safety.py:122-134`
- Problem:
  The tests assert that `"_file_lock"` appears in source.
- Why this is bad:
  The actual race condition in vault writes still exists. The tests are ceremonial.
- Impact:
  They completely miss concurrent lost updates.

### 23. Proxy test contains a tautological assertion
- File: `tests/test_proxy_scanning.py:323-332`
- Problem:
  It asserts `flow.response is None or True`.
- Why this is bad:
  That is always true.
- Impact:
  The intended security property is not being tested at all.

### 24. Access-log tests do not verify integrity enforcement on reads
- File: `tests/unit/test_access_log.py` (notably around `112-122`)
- Problem:
  Tests cover shape and normalization but do not assert tampered entries are rejected by normal query methods.
- Impact:
  The broken read-path integrity behavior went unnoticed.

### 25. Expiry tests miss persistence failure and boundary cases
- File: `tests/unit/test_expiry.py:36-71` and overall file
- Problem:
  No coverage for `_write()` failure propagation, `ttl_seconds == 0`, corrupt metadata, or concurrent updates.
- Impact:
  Silent expiry persistence failures are untested.

### 26. No test covers rotation journal secret leakage or partial-recovery correctness
- Relevant code: `src/hermes_katana/vault/store.py:663-675, 741-763`
- Problem:
  The suite never asserts that rotation avoids writing keys to disk, and never simulates mixed-state recovery.
- Impact:
  Two dangerous rotation flaws survived.

### 27. Installer restore tests are happy-path only
- File: `tests/unit/test_installer.py:82-115`
- Problem:
  No tests for tampered manifests, absolute paths, traversal, symlinks, or containment checks.
- Impact:
  A critical filesystem overwrite/delete bug is uncovered by default tests.

### 28. Bootstrap tests do not cover malicious checkout-local path escapes
- File: `tests/unit/test_bootstrap.py`
- Problem:
  Only normal install-generated config seems covered.
- Impact:
  Local config path traversal/symlink escapes are untested.

### 29. Hermes plugin concurrency/context mixup is untested
- File: `tests/unit/test_hermes_plugin.py:264-287`
- Problem:
  The tests cover simple stash/pop behavior, not two same-tool calls with distinct `task_id`s.
- Impact:
  The context mix-up bug can survive indefinitely.

### 30. Metrics tests lock in a weak implementation
- File: `tests/unit/test_metrics.py:60-68`
- Problem:
  Tests assert the current reduced-by-result-only behavior.
- Impact:
  The tests enshrine underpowered metrics instead of validating the richer API signature.

## Coverage run
Command executed:
`python3 -m pytest tests/ --cov=hermes_katana --cov-report=term-missing -q 2>&1 | tail -40`

Result:
- `1214 passed in 10.74s`
- Total coverage: `78%`

Important coverage gaps that matter:
- `src/hermes_katana/vault/store.py` at 67%: terrible for the most security-sensitive file, including major uncovered rotation/recovery paths (`658-720`, `732-769`). That matters a lot.
- `src/hermes_katana/proxy/runner.py` at 58%: startup failure, lifecycle, watchdog, and health branches are under-tested. That matters because one of the real bugs is exactly in startup failure handling.
- `src/hermes_katana/installer/patches.py` at 64%: patch application/edge cases are still lightly tested. Moderate concern.
- `src/hermes_katana/installer/installer.py` at 79%: restore/uninstall edge cases still have meaningful blind spots. This absolutely matters because restore is exploitable.
- `src/hermes_katana/vault/access_log.py` at 79%: integrity/rotation/error branches are missing. That matters.
- `src/hermes_katana/policy/engine.py` at 59%: large gap, but outside the exact scope requested here. Worth future audit, not my top finding today.

Coverage gaps that matter less for this review:
- Some scanner/taint internals have moderate gaps, but the highest-risk issues I found are in vault/installer/proxy/bootstrap and already enough to fail this review.

## Bottom line
This is not review-clean.

If I were blocking release, I would require fixes before trust in this order:
1. Stop writing vault master keys to disk during rotation.
2. Fix installer restore containment and reject tampered/escaped manifest paths.
3. Enforce atomic vault read-modify-write locking across processes.
4. Reject checkout-local bootstrap paths that escape the checkout.
5. Stop reflecting/logging raw scanner summaries and exceptions in the proxy.
6. Make proxy startup fail loudly when mitmproxy is absent.
7. Replace fake tests with adversarial behavioral tests.
