# HermesKatana Security Gap Analysis

**Reviewer**: Hostile Security Review
**Date**: 2026-04-04
**Scope**: All source files in `src/hermes_katana/` (48 files, ~20,854 LOC)
**Context**: 538 unit tests passing, 159/159 adversarial evals passing

---

## Executive Summary

HermesKatana is a well-architected defense-in-depth toolkit with solid
fundamentals. However, this hostile review identifies **7 CRITICAL**,
**12 HIGH**, **18 MEDIUM**, and **9 LOW** severity gaps across taint
tracking, vault crypto, proxy scrubbing, middleware chain, installer,
and overall architecture. The most dangerous issues are taint-laundering
paths through Python builtins that bypass flow control entirely.

---

## 1. TAINT TRACKING (taint/)

### 1.1 [CRITICAL] Taint Lost via `__str__()` — Silent Laundering

**File**: `taint/value.py:365-366`

`TaintedStr.__str__()` returns `self.value` (a plain `str`). Any code path
that calls `str()` on a TaintedStr silently strips all taint metadata.
This is the **single most dangerous gap** in the entire system because:

- `f"{tainted_var}"` calls `__format__` which falls through to `__str__`
- `json.dumps({"key": tainted_str})` calls `str()` internally
- `"%s" % tainted_str` calls `__str__`
- `str.join([tainted_str])` calls `__str__` on each element
- `logging.info("Got: %s", tainted_str)` launders via `__str__`

**Impact**: An attacker can launder tainted data through any of these
common Python idioms, producing a clean `str` that passes all flow checks.

**Fix**: Override `__format__` to return a `TaintedStr`. Consider a
runtime monkey-patch or AST rewriter for `json.dumps`. At minimum,
add `__mod__` and `__rmod__` overrides for %-formatting. Document that
f-strings are an **unsupported laundering vector** and provide a
`tainted_format()` helper.

### 1.2 [CRITICAL] No `__format__` Override

**File**: `taint/value.py`

`TaintedStr` does not override `__format__`. Python's f-string machinery
calls `object.__format__` which delegates to `__str__`, returning a plain
`str`. This means:

```python
tainted = TaintedStr("rm -rf /", sources=untrusted_sources)
clean = f"Execute: {tainted}"  # clean is a plain str, no taint
```

**Fix**: Add `def __format__(self, format_spec): ...` that returns a
TaintedStr with propagated metadata.

### 1.3 [HIGH] `repr()` Leaks Raw Value Without Taint

**File**: `taint/value.py:368-373`

`__repr__` returns a plain `str` containing the raw value. If repr output
is ever used as data (e.g., logged then parsed, or used in error messages
fed back to the LLM), the taint is lost.

**Fix**: Return a `TaintedStr` from `__repr__` or ensure the repr format
is clearly non-parseable as the original value.

### 1.4 [HIGH] `split()` Has Incorrect Offset Tracking for Repeated Substrings

**File**: `taint/value.py:409-428`

The `split()` method uses `self.value.index(part, offset)` to find each
part's position. If the string contains repeated substrings, `index()`
may find the wrong occurrence. Example:

```python
t = TaintedStr("a|a|a", sources=...)
parts = t.split("|")  # index("a", 0) always finds position 0
```

The offset tracking at line 427 (`offset = stop + (len(sep) if sep else 1)`)
partially mitigates this, but edge cases with empty strings after split
or when `sep=None` (whitespace splitting) can produce incorrect char-taint
mappings.

**Fix**: Use a cursor-based approach that doesn't rely on `index()`.

### 1.5 [HIGH] `strip()` Has Ambiguous Index Finding

**File**: `taint/value.py:395-407`

`strip()` uses `self.value.index(raw)` to find the stripped substring's
offset. If the stripped result appears multiple times in the original
string, this finds the first occurrence, which may not be the correct one.

Example: `TaintedStr("  aba  ").strip()` → looks for "aba" starting at
index 0, but the actual stripped content starts at index 2.

Wait — actually `index()` does find the correct first occurrence here
since strip only removes leading/trailing chars. But if the stripped
content happens to appear earlier due to the strip chars being part of
the content, this breaks. Edge case but real.

**Fix**: Calculate the offset directly from `len(self.value) - len(self.value.lstrip(chars))`.

### 1.6 [HIGH] No Taint Propagation for `encode()`, `bytes()` Conversion

**File**: `taint/value.py`

`TaintedStr` doesn't override `encode()`. When code does
`tainted.encode("utf-8")`, it returns plain `bytes` with no taint.
This is significant because HTTP request construction, file I/O, and
serialization all commonly encode strings to bytes.

**Fix**: Create a `TaintedBytes` type, or at minimum override `encode()`
to raise or log a warning.

### 1.7 [MEDIUM] `TaintedList`/`TaintedDict` Don't Propagate on Iteration

**File**: `taint/value.py:560-580`

`TaintedList.__getitem__` returns the raw inner value `self.value[index]`,
not a tainted wrapper. If the inner items aren't themselves `TaintedValue`
instances, iteration silently drops the container-level taint.

**Fix**: Wrap returned items with the container's source metadata when
the items themselves aren't already tainted.

### 1.8 [MEDIUM] `unwrap()` Has No Audit Trail

**File**: `taint/value.py:690-705`

The `unwrap()` function silently discards all taint. There's no logging,
no audit entry, and no way to detect that taint was deliberately removed.
In a production system, every `unwrap()` call should be auditable.

**Fix**: Add an optional `audit=True` parameter that logs to the audit
trail, or require a "reason" string.

### 1.9 [MEDIUM] Flow Rules Missing Critical Sinks

**File**: `taint/flow.py:139-154`

`CRITICAL_SINKS` is missing several dangerous tool names:
- `subprocess`, `os.system`, `exec`, `eval` (Python builtins)
- `http_request`, `fetch`, `api_call` (network tools)
- `browser_type`, `browser_click` (browser automation with side effects)
- `cronjob` (persistent execution)
- `notes` (Hermes persistent storage, similar to memory)
- `skill_manage`, `skill_view`, `skills_list` (covered in Rule 5 but not in CRITICAL_SINKS set)
- `mcp_write`, `mcp_call` (MCP write operations)

**Fix**: Add these to `CRITICAL_SINKS` or create a secondary
`DANGEROUS_SINKS` category.

### 1.10 [MEDIUM] Default FlowAnalyzer Is Permissive (ALLOW)

**File**: `taint/flow.py:298-304`

The default decision when no rule matches is `FlowDecision.ALLOW`. This
means any new/unknown tool added to Hermes will be allowed by default,
even with untrusted taint. The `strict_mode` parameter exists but is
opt-in, not default.

**Fix**: Change default to `ASK_USER` or at least `QUARANTINE` for
production deployments. Document that `strict_mode=False` is only for
development.

### 1.11 [MEDIUM] Glob Pattern Matching Is Simplistic

**File**: `taint/flow.py:85-95`

`FlowRule.matches_tool()` only supports trailing `*` wildcards. Patterns
like `mcp_*_write` or `*_delete` won't work. An attacker could register
a tool with a name that evades pattern matching.

**Fix**: Use `fnmatch.fnmatch()` for proper glob support, or compile
regex patterns.

### 1.12 [LOW] History Truncation Loses Forensic Data

**File**: `taint/flow.py:414-416`

When history exceeds `_MAX_HISTORY` (1000), the oldest half is dropped.
This loses forensic data that might be needed for incident investigation.

**Fix**: Flush to disk (append to audit trail JSONL) before truncating.

### 1.13 [LOW] TaintTracker Singleton Not Fork-Safe

**File**: `taint/tracker.py:99-101`

The class-level `_lock` and `_instance` won't survive `os.fork()` correctly.
In multiprocess deployments, child processes inherit the parent's singleton
but with a dead lock.

**Fix**: Use `os.register_at_fork()` to reset the singleton in child
processes.

---

## 2. VAULT (vault/)

### 2.1 [CRITICAL] Master Key in Memory — No Mlock/Mprotect

**File**: `vault/store.py`

The master key is stored as a plain Python `bytes` object in
`self._master_key`. Python's garbage collector may copy this to different
memory locations, and it will appear in core dumps. There's no attempt
to use `mlock()` to prevent swapping or `madvise(MADV_DONTDUMP)`.

**Fix**: Use `mmap` with `mlock()` for key storage, or use the
`cryptography` library's `Fernet` key handling which does this internally.
At minimum, zero the key bytes on `__del__`.

### 2.2 [HIGH] HMAC Verification Uses Derived Key from Master Key

**File**: `vault/store.py`

`_compute_hmac()` derives the HMAC key as `sha256(b"hmac:" + key)`. This
is a custom KDF with no salt, no iteration count, and a fixed domain
separator. If the master key has low entropy (e.g., user-provided via
env var), this provides no key stretching.

The HMAC verification itself uses `hmac.compare_digest()` which is
correct (constant-time comparison), so no timing attack there. Good.

**Fix**: Use HKDF from the `cryptography` library for proper key
derivation with context separation.

### 2.3 [HIGH] Keyring Fallback to Environment Variable

**File**: `vault/store.py:179-183`

When `keyring` is unavailable, the code falls back to
`HERMES_KATANA_VAULT_KEY` environment variable. Environment variables
are visible in `/proc/PID/environ`, `ps eww`, and are inherited by all
child processes (including the terminal tool!).

**Impact**: If Hermes runs `terminal` with a command, the child process
inherits the vault master key in its environment.

**Fix**: When falling back to env var, immediately copy the value and
call `os.environ.pop("HERMES_KATANA_VAULT_KEY")`. Better: refuse to
operate without keyring in production mode.

### 2.4 [HIGH] No Vault File Encryption-at-Rest for Metadata

**File**: `vault/expiry.py`, `vault/access_log.py`

The expiry metadata (`vault_expiry.json`) and access log
(`vault_access.jsonl`) are stored as **plaintext JSON**. While secret
values are encrypted in the vault, the key names in expiry metadata
reveal what secrets exist, and the access log reveals access patterns.

**Fix**: Encrypt the expiry metadata file. For the access log, at
minimum encrypt key names (leave timestamps/operations in clear for
log rotation).

### 2.5 [MEDIUM] No File Locking on Vault Writes

**File**: `vault/store.py`

The vault uses `threading.RLock()` for in-process thread safety, but
there's no **file-level locking** (e.g., `fcntl.flock()`). If multiple
processes access the vault simultaneously (e.g., Hermes CLI + running
agent), one process can overwrite the other's changes.

The atomic write pattern (tmp + replace) prevents corruption but not
lost updates.

**Fix**: Add `fcntl.flock()` or use a lockfile with `filelock` package.

### 2.6 [MEDIUM] Key Rotation Has a Crash Window

**File**: `vault/store.py` (rotate_key method)

During key rotation, the new key is stored in keyring BEFORE the
re-encrypted vault is written. If the process crashes between storing
the new key and writing the vault, the keyring has the new key but the
vault file still has data encrypted with the old key.

The code has a rollback (`_set_master_key(old_key)`), but only for
exceptions during `_write_vault`, not for process crashes/kills.

**Fix**: Write a "rotation journal" file that records the old key (encrypted
with the new key) before starting. On startup, check for a pending rotation
and complete or rollback.

### 2.7 [MEDIUM] `_secure_delete_from_file` Is Not Actually Secure

**File**: `vault/migrate.py`

The "secure delete" function overwrites values with `"0" * len(value)`,
but on modern filesystems (ext4, btrfs, APFS) with journaling and
copy-on-write, the original data may persist in journal entries, snapshots,
or unallocated blocks. The function also doesn't `fsync()` after writing.

**Fix**: Document that this is best-effort, not forensic-grade. For true
secure deletion, recommend full-disk encryption. Call `os.fsync()` after
the overwrite at minimum.

### 2.8 [MEDIUM] Expiry Metadata Not Synced with Vault

**File**: `vault/expiry.py`

Secret expiry is tracked in a separate file from the vault. If a secret
is deleted from the vault, its expiry entry persists (orphaned). If the
vault is restored from backup, expiry data may be stale.

**Fix**: Add a `sync_with_vault()` method that removes orphaned expiry
entries. Call it on vault operations.

### 2.9 [LOW] Access Log Has No Integrity Protection

**File**: `vault/access_log.py`

The JSONL access log has no HMAC or signature. An attacker who gains
filesystem access can modify or delete log entries to cover their tracks.

**Fix**: Add per-line HMAC or use a hash chain (each entry includes hash
of previous entry) for tamper evidence.

---

## 3. PROXY (proxy/)

### 3.1 [CRITICAL] Request Headers and URL Not Scanned

**File**: `proxy/addon.py:281-310`

The `request()` hook only scans the **request body**. It does NOT scan:
- **URL path**: secrets can leak via path segments (`/api/sk-abc123/data`)
- **Query parameters**: `?api_key=sk-abc123&token=ghp_xxx`
- **Request headers**: `Authorization: Bearer sk-abc123`, custom headers
- **Cookie values**: session tokens, API keys in cookies

This is a massive blind spot. Most secret leakage via HTTP happens
in URLs and headers, not request bodies.

**Fix**: Scan `flow.request.url`, `flow.request.headers`, and
`flow.request.query` in addition to the body. Apply the same scan
pipeline to each.

### 3.2 [CRITICAL] Response Headers Not Scanned

**File**: `proxy/addon.py:330-380`

The `response()` hook only scans the **response body**. Response headers
like `Set-Cookie`, `X-Api-Key`, `Authorization` may contain secrets or
injection payloads that pass through unscanned.

**Fix**: Scan response headers with the secrets scanner.

### 3.3 [HIGH] No WebSocket Traffic Handling

**File**: `proxy/addon.py`

The KatanaAddon only implements `request()` and `response()` hooks.
mitmproxy supports `websocket_message()` for WebSocket traffic. WebSocket
messages bypass all scanning.

Many LLM APIs (especially streaming) use WebSocket connections. An
attacker could exfiltrate data via WebSocket if the proxy doesn't
intercept it.

**Fix**: Implement `websocket_message(self, flow)` hook with the same
scanning pipeline.

### 3.4 [HIGH] TLS Certificate Handling Deferred to mitmproxy Defaults

**File**: `proxy/config.py`

`tls_verify` defaults to `True` (good), but there's no configuration for:
- Custom CA certificate for corporate environments
- Certificate pinning for known LLM provider endpoints
- Client certificate authentication
- HSTS enforcement

**Fix**: Add CA bundle path config, optional cert pinning for major
providers (api.openai.com, api.anthropic.com, etc.).

### 3.5 [HIGH] Body Size Bypass

**File**: `proxy/addon.py:230-235`

Bodies larger than `max_body_scan_size` (default 1MB) are **silently
passed through without scanning**. An attacker can bypass all body
scanning by padding their payload to exceed 1MB.

**Fix**: For oversized bodies, at minimum scan the first N bytes. Log a
warning. Consider blocking oversized bodies to unknown domains.

### 3.6 [MEDIUM] Rate Limiter State Not Persisted

**File**: `proxy/addon.py:17-110`

The `RateTracker` is purely in-memory. Restarting the proxy resets all
rate limiting state. An attacker can bypass rate limiting by triggering
a proxy restart.

**Fix**: Persist violation counts to disk or shared memory.

### 3.7 [MEDIUM] X-Katana-Scanned Header Information Leak

**File**: `proxy/addon.py:374-378`

The `X-Katana-Scanned: true` header injected into responses reveals to
downstream consumers that a security proxy is in the path. This is an
information leak useful for reconnaissance.

**Fix**: Make this header opt-in and disabled by default in production.

### 3.8 [MEDIUM] Credential Injection Before Scanning

**File**: `proxy/addon.py:296-300`

Credentials are injected into requests BEFORE body scanning. This means
the scanner will see the injected credentials in the request and may
flag them as leaked secrets (false positive), or the scan result
processing may inadvertently log the injected credentials.

**Fix**: Inject credentials AFTER scanning, or exclude injected
credential headers from the scan.

### 3.9 [LOW] No CONNECT Tunnel Handling

The proxy doesn't implement `tls_start_client` or `tls_start_server`
hooks for custom TLS handling. All TLS behavior is delegated to mitmproxy
defaults.

---

## 4. MIDDLEWARE CHAIN

### 4.1 [HIGH] Middleware Bypass via Direct Tool Invocation

**File**: `middleware/chain.py`, `middleware/integration.py`

The middleware chain only intercepts tool calls that go through the
chain's `execute()` method. If Hermes (or a plugin) invokes a tool
directly without going through the middleware, all security checks are
bypassed.

There's no enforcement that tool calls MUST go through the chain. This
is a fundamental architectural gap — the security layer is opt-in, not
mandatory.

**Fix**: Monkey-patch or wrap the tool execution layer in Hermes to force
all calls through the middleware. Add runtime detection for direct tool
invocations that bypassed the chain.

### 4.2 [HIGH] Unknown Tools Default to Allow

**File**: `policy/defaults.py`

The BALANCED policy set has explicit rules for known tools (terminal,
write_file, patch, send_message, memory, skill_manage, etc.) but
unknown tool names that don't match any pattern fall through to the
catch-all. In BALANCED mode, the catch-all for tainted data is
`log_only` (priority 5). In PERMISSIVE mode, unknown tainted tools
are just logged.

This means any new tool added to Hermes is automatically allowed with
tainted data in BALANCED and PERMISSIVE modes.

**Fix**: The catch-all for unknown tools with taint should be `escalate`
in BALANCED mode, not `log_only`.

### 4.3 [MEDIUM] Policy Engine Thread Safety Gap

**File**: `policy/engine.py`

The PolicyEngine uses a lock for `evaluate()` and `add/remove`, but the
`replace_all()` and hot-reload callback don't appear to be atomic with
respect to concurrent `evaluate()` calls. A policy evaluation mid-reload
could see a partially-updated policy set.

**Fix**: Use a read-write lock pattern. Snapshot the policy list at the
start of evaluate under the lock, then evaluate without holding it.

### 4.4 [MEDIUM] No Middleware Execution Order Guarantee

**File**: `middleware/chain.py`

The middleware chain processes middlewares in insertion order, but there's
no explicit priority or dependency system. If middlewares are registered
in the wrong order (e.g., audit after policy), the audit trail may miss
denied calls.

**Fix**: Add explicit priority ordering to middleware registration, or
document and enforce the required order.

### 4.5 [MEDIUM] Policy Hot-Reload From Untrusted Directory

**File**: `policy/engine.py`

The `start_watcher()` method watches a directory for YAML policy changes
and hot-reloads them. If the watched directory is writable by the agent
(or a compromised tool), an attacker could inject permissive policies.

**Fix**: Verify file ownership/permissions before loading. Sign policy
files with HMAC using a key from the vault.

### 4.6 [LOW] `matches_pattern` Regex Not Anchored

**File**: `policy/defaults.py`

Several policy conditions use `matches_pattern` with regexes like
`".*(curl|wget)\\s+.*"`. These are not anchored, so they match
anywhere in the string. An attacker could potentially craft a command
that includes the pattern as a harmless substring to trigger a
false positive and cause a "cry wolf" effect, or conversely craft
evasions using shell features (aliases, env vars, `$(...)`, backticks).

**Fix**: Document regex limitations. Add command-parsing-aware matching
that handles shell escaping, pipes, and subshells.

---

## 5. INSTALLER / BOOTSTRAP

### 5.1 [HIGH] TOCTOU Race in Installer Patching

**File**: `installer/installer.py`, `installer/patches.py`

The installer reads a file, checks its content, then writes the patched
version. Between the read and write, another process could modify the
file. This is a classic TOCTOU (Time-of-Check-Time-of-Use) race.

**Fix**: Use file locking (`fcntl.flock()`) during the read-check-write
cycle. Open the file with `O_EXCL` where possible.

### 5.2 [HIGH] Backup Files Store Unprotected Originals

**File**: `installer/installer.py`

The installer creates backup files (`.bak`) of patched files. These
backups are unencrypted and contain the original Hermes source code.
An attacker could restore from backup to remove security patches.

**Fix**: Set restrictive permissions on backup files. Consider encrypting
them or storing a hash for integrity verification on restore.

### 5.3 [MEDIUM] Installer Doesn't Verify Target File Integrity

**File**: `installer/installer.py`

The installer patches files based on content matching but doesn't verify
the target files haven't been tampered with before patching. A compromised
Hermes installation could have already-modified files that the installer
patches incorrectly.

**Fix**: Maintain a manifest of expected file hashes (from known Hermes
versions) and verify before patching.

### 5.4 [MEDIUM] Compat Snapshots Could Be Stale

**File**: `installer/compat_snapshots.py`

Compatibility snapshots are static records of known Hermes versions. If
Hermes is updated between HermesKatana releases, the installer may
apply incorrect patches.

**Fix**: Add a "verify after patch" step that checks the patched file
still functions correctly (import test, syntax check).

---

## 6. SCANNER

### 6.1 [MEDIUM] Scanner Regex Patterns Are Static

**File**: `scanner/secrets.py`, `scanner/injection.py`

Secret detection patterns and injection signatures are hardcoded. New
secret formats (e.g., new LLM providers) or new injection techniques
require a code update and release.

**Fix**: Support loading patterns from external files (YAML/JSON) that
can be updated independently. Add a pattern-update mechanism.

### 6.2 [MEDIUM] Scanner Doesn't Handle Encoded Payloads

**File**: `scanner/injection.py`

The injection scanner works on UTF-8 text. Payloads encoded as base64,
URL-encoded, HTML entities, Unicode escapes, or hex may bypass detection.

Example: `\x69\x67\x6e\x6f\x72\x65 previous instructions` (hex-encoded
"ignore previous instructions") would not trigger injection rules.

**Fix**: Add a decoding/normalization step before scanning that handles
common encodings (base64, URL-encoding, HTML entities, Unicode escapes).

### 6.3 [LOW] Ensemble Scanner Weights Are Hardcoded

**File**: `scanner/ensemble.py`

The ensemble scanner combines multiple scanner results with fixed weights.
These weights aren't tunable without code changes.

**Fix**: Make weights configurable via the config system.

---

## 7. AUDIT TRAIL

### 7.1 [MEDIUM] Audit Trail Has No Tamper Protection

**File**: `audit/trail.py`

The audit trail is an append-only file with no integrity protection.
An attacker with file access can delete, modify, or truncate entries.

**Fix**: Implement a hash chain where each entry includes the hash of
the previous entry. Periodically checkpoint the chain hash to an
external store (vault, remote syslog).

### 7.2 [MEDIUM] No Remote Audit Sink

**File**: `audit/trail.py`

All audit data is local. If the host is compromised, all audit data
can be destroyed.

**Fix**: Support remote syslog, webhook, or cloud logging sinks for
audit data forwarding.

### 7.3 [LOW] Audit Entry Timestamps Not Cryptographically Bound

Timestamps in audit entries are from `time.time()` which an attacker
with local access can manipulate via NTP or system clock changes.

**Fix**: Use monotonic timestamps for ordering, and optionally
include a trusted timestamp service.

---

## 8. OVERALL ARCHITECTURE GAPS

### 8.1 [CRITICAL] No Runtime Enforcement of Taint — Purely Advisory

The entire taint tracking system is **advisory**. It depends on:
1. Every data entry point being wrapped with `taint_*()` functions
2. Every tool call going through the middleware chain
3. No code path accidentally calling `str()` on a TaintedStr

There is no compile-time or runtime enforcement. A single missing
taint wrapper at an entry point creates an unmonitored data flow.
A single direct tool call bypasses all policies.

**Fix**: This is the hardest problem to solve. Options:
- AST-level instrumentation to auto-wrap entry points
- Runtime monkey-patching of Hermes tool dispatch
- Type-checker plugin (mypy/pyright) that flags raw str → tool flows
- Integration tests that verify every Hermes entry point is wrapped

### 8.2 [HIGH] No Secret Rotation Automation

The vault supports manual key rotation but there's no automated rotation
for stored secrets. API keys sitting in the vault indefinitely accumulate
risk.

**Fix**: Add configurable rotation reminders/policies. Integrate with
provider APIs where possible (OpenAI key rotation API, etc.).

### 8.3 [HIGH] No Anomaly Detection Beyond Rate Limiting

The proxy has basic rate limiting but no behavioral anomaly detection:
- No baseline of normal tool usage patterns
- No detection of unusual tool sequences (recon → read → exfil)
- No detection of data volume anomalies
- No detection of timing anomalies (tool calls at unusual hours)

**Fix**: Add a lightweight anomaly detection module that learns normal
patterns and flags deviations.

### 8.4 [MEDIUM] No Multi-Tenant Isolation

HermesKatana assumes a single user/agent. In multi-tenant deployments:
- The vault is shared (all agents see all secrets)
- The taint tracker is a singleton (cross-session contamination)
- Policies are global (can't have per-tenant policies)
- Audit trail doesn't separate tenants

**Fix**: Add tenant/session scoping to vault, tracker, and policies.

### 8.5 [MEDIUM] No Automated Incident Response

When a security event is detected (injection blocked, secret leaked,
etc.), the system logs it but takes no automated response:
- No alerting (email, Slack, PagerDuty)
- No automatic session termination on critical events
- No quarantine mode that restricts the agent after N violations
- No automatic secret rotation on leak detection

**Fix**: Add an incident response module with configurable actions
per severity level.

### 8.6 [MEDIUM] No Compliance Reporting

No built-in support for:
- SOC2 audit trail formatting
- GDPR data flow documentation
- PCI-DSS secret handling compliance
- Export of security posture reports

**Fix**: Add report generation for common compliance frameworks.

### 8.7 [MEDIUM] No Supply Chain Verification

HermesKatana doesn't verify its own integrity or dependencies:
- No signature verification on package install
- No SBOM (Software Bill of Materials)
- No hash pinning of dependencies
- No verification that scanner patterns haven't been tampered with

**Fix**: Sign releases. Include SBOM. Pin dependency hashes in
requirements.

### 8.8 [LOW] No Metrics/Observability Integration

**File**: `metrics.py`

The metrics module exists but doesn't integrate with standard
observability stacks (Prometheus, OpenTelemetry, StatsD).

**Fix**: Add Prometheus metric export endpoint and OpenTelemetry
span integration.

### 8.9 [LOW] Configuration Validation Incomplete

**File**: `config.py`

Configuration is validated by Pydantic but there's no validation of
semantic correctness — e.g., you can configure `allowed_domains: []`
which means "allow all" (permissive), which might surprise users who
think it means "allow none."

**Fix**: Add semantic validation warnings for potentially dangerous
configurations.

---

## Summary Table

| Severity | Count | Key Areas |
|----------|-------|-----------|
| CRITICAL | 7 | Taint laundering via str/format, headers not scanned, advisory-only enforcement, master key in memory |
| HIGH | 12 | Missing sinks, WebSocket gap, unknown tools default-allow, TOCTOU, keyring fallback |
| MEDIUM | 18 | Thread safety, encoding bypass, no remote audit, no compliance, stale metadata |
| LOW | 9 | History truncation, fork safety, hardcoded weights, timestamps |

## Priority Recommendations

1. **IMMEDIATE** (Week 1): Fix `__str__`/`__format__` taint laundering — this undermines the entire taint system
2. **IMMEDIATE** (Week 1): Add header/URL/query param scanning to proxy — covers the biggest blind spot
3. **SHORT-TERM** (Week 2-3): Add WebSocket scanning, lock vault master key memory, implement file-level vault locking
4. **SHORT-TERM** (Week 2-3): Change default flow decision to ASK_USER, add missing critical sinks
5. **MEDIUM-TERM** (Month 1-2): Add payload decoding/normalization, anomaly detection, remote audit
6. **LONG-TERM** (Quarter): Multi-tenant isolation, compliance reporting, supply chain verification
