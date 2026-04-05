# Audit Fixes — Final Review

All fixes applied based on hostile code reviews from 3 independent reviewers.

## Fixes Applied

### From baseline (Phase 1)

1. **vault/store.py: `_get_master_key` env var fallback unreachable on keyring failure**
   - When keyring was installed but had no backend (NoKeyringError), the env var fallback was never reached
   - Fixed: unified fallback path so env var is always checked when keyring fails
   - Also fixed `_set_master_key` to warn instead of raising when keyring is unavailable

2. **scanner/commands.py: TaintedStr.encode() warning in Unicode normalization**
   - Fixed: use `str.__str__()` to extract raw string before encoding

### From AUDIT_REVIEW_1 (Scanner + Evasion)

3. **scanner/commands.py: Lossy Unicode normalization allows homoglyph evasion** (CRITICAL)
   - NFKD + encode("ascii","ignore") drops Cyrillic/Greek homoglyphs instead of mapping them
   - Fixed: added `_CONFUSABLE_MAP` with 40+ Cyrillic/Greek/dash/slash homoglyph mappings applied before NFKD

4. **scanner/secrets.py: Pattern positions use whole match instead of capture group** (HIGH)
   - When patterns have capture groups, position should use `match.start(1)/end(1)`
   - Fixed: use subgroup positions when `match.lastindex` is present

5. **scanner/secrets.py: Vault exact matching only finds first occurrence** (MEDIUM)
   - Fixed: iterate all occurrences with a while loop instead of single `text.find()`

6. **scanner/allowlist.py: Misleading comment says "prepended" but code appends** (LOW)
   - Fixed: updated comment to match actual behavior

### From AUDIT_REVIEW_2 (Taint + Policy + Middleware)

7. **middleware/integration.py: Taint checking only inspects top-level args** (CRITICAL)
   - Nested tainted values in lists/dicts/tuples bypassed taint enforcement
   - Fixed: added `_find_tainted()` recursive helper, middleware now deep-inspects all args

8. **taint/flow.py: CRITICAL_SINKS missing real mutation/exfiltration tools** (HIGH)
   - Added: `memory`, `browser_press`, `browser_navigate`, `text_to_speech`

9. **policy/defaults.py: Paranoid browser rule matches all browser_* tools** (HIGH)
   - Was blocking read-only tools like `browser_snapshot`, `browser_vision`
   - Fixed: split into 4 specific rules for `browser_click*`, `browser_type*`, `browser_press*`, `browser_navigate*`

10. **policy/yaml_loader.py: Unknown parent in inheritance silently skipped** (HIGH)
    - A typo in `extends` would silently load a weaker policy
    - Fixed: raises `PolicyValidationError` instead of warning
    - Updated 3 tests to expect the new fail-closed behavior

11. **policy/yaml_loader.py: Hot reload applies partial policy sets** (HIGH)
    - If 1 of 5 policy files became invalid, the remaining 4 would replace the full set
    - Fixed: compare loaded count vs total files, reject partial loads

### From AUDIT_REVIEW_3 (Vault + Proxy + Installer)

12. **vault/store.py: Key rotation writes plaintext master keys to disk** (CRITICAL)
    - Both old and new keys were base64-encoded in plaintext in `.rotation_journal`
    - Fixed: keys are now cross-encrypted (old_key encrypted with new_key and vice versa)
    - Journal file permissions set to 0o600

13. **vault/store.py: Rotation recovery validates only first entry** (HIGH)
    - Fixed: recovery now validates ALL entries before declaring success

14. **vault/store.py: get() doesn't enforce HMAC when HMAC is missing** (MEDIUM)
    - Non-empty entries without HMAC were silently accepted
    - Fixed: raises VaultIntegrityError when HMAC is missing on non-empty vault

15. **vault/store.py: Malformed master key not validated at ingestion** (LOW/MEDIUM)
    - Fixed: added `_validate_key()` that enforces 32-byte key length at boundary

16. **installer/installer.py: restore() trusts manifest paths without containment** (CRITICAL)
    - Tampered manifest could overwrite/delete arbitrary files via `..` traversal or absolute paths
    - Fixed: reject absolute paths and `..` in relative paths, validate containment, reject symlinks

17. **bootstrap.py: Checkout-local paths can escape checkout root** (HIGH)
    - `_resolve_checkout_path()` accepted paths that resolved outside the checkout
    - Fixed: enforces containment via `relative_to()` check

18. **hermes_plugin.py: Plugin context recovery ignores task_id** (MEDIUM/HIGH)
    - Concurrent calls to same tool could get wrong context
    - Fixed: `_pop_context()` now matches on both tool_name and task_id

19. **proxy/runner.py: Claims proxy started when mitmproxy is missing** (HIGH)
    - FileNotFoundError fallback used `os.getpid()` as fake PID
    - Fixed: raises RuntimeError immediately

20. **proxy/addon.py: Scanner summaries echoed to clients** (HIGH)
    - Block responses included scanner summary text (info leak)
    - Fixed: all client-facing responses now use generic "blocked by security policy" message

21. **proxy/addon.py: x-goog-api-key not excluded from proxy scanning** (MEDIUM)
    - Fixed: added to injected headers exemption list

22. **proxy/addon.py: Blocked responses counted as passed in metrics** (MEDIUM)
    - Fixed: blocked responses return early before incrementing `responses_passed`

23. **test_proxy_scanning.py: Tautological assertion** (TEST)
    - `assert flow.response is None or True` always passes
    - Fixed: meaningful assertion checking header exemption behavior

## Test Results After All Fixes

- `pytest tests/`: **1214 passed**, 0 errors, 0 warnings
- `test_false_positives.py`: **0 FPs** across 273 benign inputs
- `test_evasion.py`: **64/64 caught**, 0 evasions (100%)
- `test_adversarial_eval_pack.py`: **159/159 passed** (100%)
- No import warnings or deprecation warnings
