# HermesKatana Test Coverage Gap Analysis

**Date:** 2026-04-04
**Overall Coverage:** 70% (2082 of 6970 statements missed)
**Tests:** 773 passed in 5.24s

---

## 1. Modules with 0% Coverage (CRITICAL)

| Module | Stmts | Description |
|--------|-------|-------------|
| `proxy/addon.py` | 211 | mitmproxy addon - request/response scanning, credential injection, rate limiting, thread-safe rate tracking |
| `proxy/addon_script.py` | 39 | mitmproxy addon script loader |
| `proxy/injector.py` | 50 | API key injection for 12+ LLM providers |
| `vault/migrate.py` | 157 | Secret discovery & migration from env/config/.env into vault, secure deletion |

**Risk:** These are security-critical modules handling credential injection, secret migration, and proxy interception with ZERO test coverage. The proxy addon includes thread-safe rate tracking and anomaly escalation — completely untested.

---

## 2. Modules Below 80% Coverage

| Module | Cover | Miss | Key Gaps |
|--------|-------|------|----------|
| `cli/main.py` | 55% | 358 | Majority of CLI commands untested |
| `proxy/runner.py` | 58% | 122 | Proxy lifecycle, start/stop, signal handling |
| `installer/patches.py` | 63% | 55 | Hermes patching logic |
| `vault/store.py` | 66% | 92 | Vault CRUD, encryption paths, error handling |
| `policy/engine.py` | 67% | 109 | Policy evaluation logic, rule matching, action resolution |
| `taint/tracker.py` | 67% | 65 | Taint propagation, tracking lifecycle |
| `audit/trail.py` | 71% | 86 | Audit logging, file I/O, rotation |
| `hermes_plugin.py` | 72% | 51 | Plugin lifecycle, registration, hooks |
| `scanner/ensemble.py` | 73% | 46 | Ensemble scanner coordination |
| `taint/value.py` | 76% | 83 | Tainted value operations, propagation |
| `config.py` | 77% | 28 | Configuration loading, validation |
| `secrets scanner` | 77% | 51 | Secret detection patterns |
| `vault/access_log.py` | 78% | 23 | Vault access audit logging |
| `policy/yaml_loader.py` | **16%** | 149 | YAML policy loading — nearly entirely untested |

**Worst offender:** `policy/yaml_loader.py` at 16% — this loads security policies from YAML and is almost completely untested. A bug here could silently disable security rules.

---

## 3. Missing Test Categories

### 3a. No Property-Based Tests (hypothesis)
- **Status:** hypothesis is available but NOT used anywhere in the test suite
- **Impact:** No fuzz-like testing of scanner patterns, policy evaluation, or taint propagation
- **Recommendation:** Add hypothesis tests for:
  - Scanner pattern matching (random strings, unicode edge cases)
  - Taint value arithmetic/propagation
  - Policy rule evaluation with random inputs
  - Config parsing with malformed inputs

### 3b. No Mutation Testing
- **Status:** No mutmut or similar mutation testing configured
- **Impact:** Cannot verify that tests actually catch regressions (tests may pass trivially)
- **Recommendation:** Add `mutmut` to CI pipeline targeting scanner and policy modules

### 3c. No Race Condition / Concurrency Tests
- **Status:** Files mentioning "thread"/"async" exist in source (proxy/addon.py uses threading) but NO concurrency tests exist
- **Impact:** `proxy/addon.py` has thread-safe rate tracking with locks — completely untested under contention
- **Recommendation:** Add threading stress tests for:
  - Rate limiter under concurrent requests
  - Vault access under concurrent reads/writes
  - Audit trail under concurrent log writes

### 3d. No Fuzzing Tests
- **Status:** No fuzzing infrastructure (no AFL, no pythonfuzz, no hypothesis strategies generating adversarial inputs)
- **Impact:** Scanner bypass discovery relies entirely on hand-crafted adversarial evals
- **Recommendation:** Add fuzzing for injection detection and secret scanning patterns

---

## 4. Adversarial Eval Coverage Gaps

The adversarial eval pack (`tests/integration/test_adversarial_eval_pack.py`) loads cases from `evals/adversarial_dispatch.yaml` and tests the full chain (taint → scan → policy → audit). However:

- **Missing scanner categories in adversarial evals:**
  - Unicode homoglyph attacks (scanner/unicode.py is 94% covered by unit tests but not via adversarial evals)
  - Ensemble scanner disagreement scenarios
  - Context-dependent scanner behavior
  - Secret patterns in non-obvious formats (base64-encoded, split across args)
- **Known gaps are marked** with `gap` decision type — good practice, but gaps should be tracked as issues
- **No adversarial evals for:**
  - Proxy addon request/response scanning path
  - Vault credential injection path
  - Config manipulation attacks

---

## 5. Integration Test Gaps

### What EXISTS:
- `test_adversarial_eval_pack.py` — dispatch chain (taint→scan→policy→audit) ✓
- `test_middleware_chain.py` — middleware chain integration ✓
- `test_flow.py` — taint flow integration ✓
- `test_cli_flow.py` — CLI integration ✓

### What's MISSING:
- **Full proxy pipeline test:** input → proxy addon → scan → policy → credential inject → audit → forward
  - The proxy modules (addon, injector, runner) are 0-58% covered
  - No test sends a mock HTTP request through the full mitmproxy pipeline
- **Vault lifecycle test:** migrate secrets → store → access → rotate → expire → audit
  - `vault/migrate.py` is 0%, `vault/store.py` is 66%
- **Installer end-to-end:** install → patch → verify → snapshot compatibility
  - `installer/patches.py` is 63%
- **Policy loading chain:** YAML file → loader → engine → decision
  - `policy/yaml_loader.py` is 16%, `policy/engine.py` is 67%
- **Error cascade test:** What happens when scanner raises, policy DB is corrupt, vault is locked, audit disk is full?

---

## 6. Untested Error Paths & Edge Cases

### CLI (`cli/main.py` — 55%)
- Most CLI subcommands are untested (lines 756-821, 837-869, 902-918, etc.)
- Error handling for invalid arguments, missing config, permission errors
- Interactive prompts and user confirmation flows

### Config (`config.py` — 77%)
- Missing config file handling (lines 189-197)
- Corrupt/malformed config parsing
- Environment variable override logic (lines 302-305, 307-310)
- Config validation edge cases

### Vault Store (`vault/store.py` — 66%)
- Encryption/decryption error paths (lines 172-188, 197-209)
- Concurrent access patterns (lines 518-554)
- Corrupt vault file recovery
- Key rotation during active reads

### Policy Engine (`policy/engine.py` — 67%)
- Complex rule matching (lines 203-241)
- Rule conflict resolution (lines 691-708)
- Policy hot-reload
- Malformed policy handling

### Taint Tracker (`taint/tracker.py` — 67%)
- Taint propagation through complex operations (lines 176-189, 215-239)
- Tracker cleanup/GC (lines 384-409)
- Deep taint chains

### Audit Trail (`audit/trail.py` — 71%)
- File rotation under load (lines 396-421)
- Disk full / permission denied scenarios
- Concurrent write handling
- Corrupt log recovery

---

## 7. Security-Critical Untested Paths

1. **Credential injection (proxy/injector.py — 0%):** Maps domains to vault keys for 12+ LLM providers. A bug could leak credentials to wrong domains or fail to inject, causing auth failures.

2. **Secret migration (vault/migrate.py — 0%):** Discovers and migrates secrets, with secure deletion (zero-overwrite). Untested secure deletion means secrets may persist on disk.

3. **Proxy addon rate limiting (proxy/addon.py — 0%):** Thread-safe rate tracking with anomaly escalation. Untested means rate limits may not work under load.

4. **YAML policy loader (policy/yaml_loader.py — 16%):** Parses security policy definitions. Bugs could silently disable security rules or create policy bypasses.

5. **Policy engine rule resolution (policy/engine.py — 67%):** Rule conflict resolution and action determination. Missing coverage on complex rule interactions.

6. **Vault encryption paths (vault/store.py — 66%):** Encryption key management, encrypt/decrypt operations only partially tested.

---

## 8. Recommendations (Priority Order)

### P0 — Critical (security impact)
1. Add tests for `proxy/injector.py` — credential injection domain mapping
2. Add tests for `vault/migrate.py` — secret discovery, migration, secure deletion
3. Add tests for `proxy/addon.py` — request/response scanning, rate limiting
4. Increase `policy/yaml_loader.py` from 16% to >90%
5. Add threading stress tests for proxy addon rate limiter

### P1 — High (reliability impact)
6. Increase `policy/engine.py` coverage to >90% (rule resolution, conflict handling)
7. Increase `vault/store.py` coverage to >90% (encryption paths, error handling)
8. Increase `cli/main.py` coverage to >80% (all subcommands)
9. Add full proxy pipeline integration test
10. Add vault lifecycle integration test

### P2 — Medium (quality impact)
11. Add hypothesis property-based tests for scanner patterns
12. Add hypothesis tests for taint value propagation
13. Add config edge case tests (missing, corrupt, env var overrides)
14. Increase `taint/tracker.py` and `taint/value.py` to >90%
15. Add error cascade integration tests

### P3 — Long-term (maturity)
16. Set up mutation testing with mutmut
17. Add fuzzing infrastructure for scanner bypass discovery
18. Add concurrency stress tests for vault and audit trail
19. Expand adversarial evals to cover unicode, ensemble, and context-dependent scenarios
20. Add performance regression tests for scanner hot paths

---

## Summary Table

| Category | Status |
|----------|--------|
| Overall coverage | 70% (target: >85%) |
| Modules at 0% | 4 (all security-critical) |
| Modules below 80% | 14 |
| Property-based tests | None |
| Mutation testing | None |
| Concurrency tests | None |
| Fuzzing | None |
| E2E proxy pipeline test | Missing |
| Vault lifecycle test | Missing |
| Policy load chain test | Missing |
| Hypothesis available | Yes, unused |
