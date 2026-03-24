# Cryptography and Secret Management Research for HermesKatana

Sources: OWASP Cryptographic Storage Cheat Sheet, GitGuardian documentation (Shannon entropy), argon2-cffi docs, Python cryptography library docs, mitmproxy architecture docs, dev.to/veritaschain (hash-chained logs), deepwiki.com/infiligence/governed-rag.

---

## 1. Symmetric Encryption for Vault Storage

### 1.1 AES-128-CBC (Fernet) vs AES-256-GCM

HermesKatana already uses AES-256-GCM — this is the right choice. Here's why it matters:

| Property | AES-128-CBC (Fernet) | AES-256-GCM |
|----------|---------------------|-------------|
| Key length | 128-bit | 256-bit |
| Authentication | HMAC-SHA256 (separate) | Built-in GCM tag |
| Padding oracle vulnerability | Yes (CBC mode) | No |
| Parallelizable | No (CBC chaining) | Yes |
| Nonce reuse consequence | Predictable IV patterns | **Catastrophic** — key recovery possible |
| Quantum resistance | Grover's: 64-bit effective | Grover's: 128-bit effective |
| Python library | `cryptography.fernet` | `cryptography.hazmat.primitives.ciphers.aead.AESGCM` |

**Authenticated encryption matters**: Without authentication (MAC), an attacker who can write to the vault file can modify ciphertext bytes in ways that decrypt to attacker-chosen plaintext (padding oracle attacks for CBC, bit-flipping for CTR). AES-256-GCM's authentication tag makes any tampering detectable.

### 1.2 Nonce Management — The Critical Detail

AES-256-GCM requires a unique nonce per encryption. HermesKatana generates random 96-bit (12-byte) nonces — this is correct.

**What happens on nonce reuse** (the GCM catastrophe):
- Two ciphertexts encrypted with the same key and nonce can be XORed to recover the XOR of plaintexts
- Authentication keys can be recovered
- Full plaintext recovery with enough ciphertexts

For a vault that encrypts ~100 values, the birthday problem gives negligible collision probability with random 96-bit nonces. At 1 million values, it remains safe (~10^-12 collision probability).

**Recommended**: Keep random 96-bit nonces. Do NOT use counters for nonces unless implementing a proper counter management system.

### 1.3 Per-Value Encryption vs Bulk Encryption
HermesKatana encrypts each vault entry individually. This is correct for these reasons:
- Partial decryption: can read one key without decrypting the whole vault
- Granular integrity: tampering with one entry detected without affecting others
- Rotation: can re-encrypt individual values on key rotation without touching others

**Tradeoff**: Slightly larger storage (nonce + tag overhead per entry = 28 bytes). Negligible for a secret vault.

---

## 2. Key Derivation and Master Keys

### 2.1 PBKDF2 vs bcrypt vs Argon2id

| Algorithm | Memory-Hard | GPU-Resistant | OWASP Recommended | Standard |
|-----------|-------------|--------------|-------------------|---------|
| PBKDF2-SHA256 | No | No | Minimum (legacy) | NIST SP 800-132 |
| bcrypt | Partial (2KB RAM) | Partial | Yes (for passwords) | de facto |
| scrypt | Yes | Yes | Yes | RFC 7914 |
| **Argon2id** | **Yes** | **Yes** | **Preferred** | **RFC 9106 (2021)** |

**Argon2id** won the Password Hashing Competition (PHC) in 2015 and was standardized in RFC 9106. It combines:
- Argon2i's resistance to timing side-channels (data-independent memory access)
- Argon2d's GPU resistance (data-dependent memory access)

OWASP recommends Argon2id with:
- Memory: 64 MB minimum (production: 128 MB)
- Iterations: 3 minimum
- Parallelism: 4

**HermesKatana current state**: The vault uses OS keyring for master key storage (not derivation from password). This sidesteps the KDF problem entirely for interactive use. For headless/CI environments where keyring is unavailable, Argon2id should be used to derive the master key from a passphrase.

### 2.2 Key Hierarchy
A proper key hierarchy provides security boundaries:

```
Master Key (MK) — stored in OS keyring / derived via Argon2id
    ↓ HKDF-SHA256
Data Encryption Key (DEK) — one per vault or per key category
    ↓ AES-256-GCM per entry
Individual encrypted values
```

Benefits:
- DEK rotation doesn't require changing MK
- Compromise of one DEK doesn't affect others
- Category separation (LLM API keys, git tokens, etc.)

**HermesKatana improvement**: Implement DEK derivation using HKDF:
```python
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

def derive_dek(master_key: bytes, context: str) -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32,
                salt=None, info=context.encode())
    return hkdf.derive(master_key)

# Usage:
llm_keys_dek = derive_dek(master_key, "hermes-katana-llm-keys-v1")
git_tokens_dek = derive_dek(master_key, "hermes-katana-git-tokens-v1")
```

### 2.3 OS Keyring Systems

| Platform | System | Library | Notes |
|----------|--------|---------|-------|
| macOS | Keychain | `keyring` → macOS backend | Most secure; hardware-backed on Apple Silicon |
| Linux | Secret Service (GNOME) | `keyring` → `secretstorage` → libsecret | Requires DBus, unavailable in headless containers |
| Linux | KWallet | `keyring` → KDE backend | Less common |
| Windows | Credential Locker | `keyring` → Windows backend | DPAPI-backed |

**Headless/CI fallback**: When keyring is unavailable, encrypt master key with a passphrase using Argon2id, store the encrypted MK in `~/.hermes-katana/vault-key.enc`. Require passphrase on startup.

### 2.4 Zero-Downtime Key Rotation

Current HermesKatana approach: re-encrypt all values with new key, then swap. This creates a window where the vault is partially re-encrypted.

**Improved approach**: atomic key rotation
```python
def rotate_key_atomic(self, new_master_key: bytes) -> None:
    """Rotate all vault entries to new key atomically."""
    # 1. Read all current values with old key
    all_values = {k: self.get(k) for k in self.list_keys()}
    
    # 2. Write to a new vault file with new key
    new_vault = self._vault_path.with_suffix('.new')
    # ... encrypt all to new_vault ...
    
    # 3. Atomic swap (POSIX rename guarantee)
    new_vault.rename(self._vault_path)
    
    # 4. Update keyring
    self._set_master_key(new_master_key)
```

---

## 3. Secret Detection

### 3.1 Shannon Entropy as a Signal

Shannon entropy measures information density (bits per character):
```
H = -sum(p_i * log2(p_i))  for each character i
```

| String Type | Typical Entropy (bits/char) |
|-------------|---------------------------|
| English prose | 1.0 – 1.5 |
| Random lowercase | ~2.5 |
| Alphanumeric random | ~3.5 |
| Base64 random | ~4.0 |
| Hex random | ~3.5 |
| **API keys (base62/base64)** | **4.5 – 5.5** |
| Binary data as base64 | ~6.0 |

**GitGuardian threshold**: ≥ 4.5 bits/char with length ≥ 20 characters. At 4.5+ with proper length filter, human text almost never triggers (false positive rate near 0). HermesKatana's current 4.5 threshold matches industry practice.

**Calibration tips**:
- Apply minimum length filter (≥ 20 chars) before entropy check
- Use character set analysis alongside entropy: API keys tend to use specific character sets
- Combine with regex: entropy alone catches unknown formats; regex catches known ones

### 3.2 Pattern-Based Detection

HermesKatana has 20 patterns. Key additions for completeness:

| Missing Pattern | Regex | Example |
|-----------------|-------|---------|
| Slack bot token | `xoxb-[0-9]+-[0-9]+-[a-zA-Z0-9]+` | `xoxb-123-456-abc` |
| Stripe live key | `sk_live_[a-zA-Z0-9]{24,}` | `sk_live_abc123...` |
| SendGrid | `SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}` | `SG.xxxx.yyyy` |
| Twilio | `SK[a-f0-9]{32}` | `SKabc123...` |
| Vercel token | `vercel_[a-zA-Z0-9_-]{20,}` | — |
| Databricks | `dapi[a-f0-9]{32}` | `dapi...` |
| HuggingFace | `hf_[a-zA-Z0-9]{37}` | `hf_abc123...` |

### 3.3 Multi-Encoding Detection

HermesKatana detects: base64, hex, URL-encoded, reversed, ROT13.

**Missing encodings to add**:
- **Unicode escapes**: `\u0041\u0050\u0049` for "API"
- **HTML entity encoding**: `&#65;&#80;&#73;` 
- **Double base64**: base64 of base64 (used to bypass single-layer scanners)
- **Decimal encoding**: `65 80 73 95 75 69 89` for "API_KEY"

### 3.4 Chunked/Split Secret Detection

A secret split across multiple parameters:
```json
{"key_part1": "sk-ant-api03-", "key_part2": "ABCDEFGH..."}
```

Or split across consecutive requests:
```
Request 1: {"data": "sk-ant-api03-"}
Request 2: {"data": "ABCDEFGH_more_key_data"}
```

**Implementation approach**:
```python
class ChunkedSecretDetector:
    def __init__(self, window_size: int = 5):
        self.buffer: deque[str] = deque(maxlen=window_size)

    def scan(self, new_text: str) -> list[SecretFinding]:
        self.buffer.append(new_text)
        combined = "".join(self.buffer)
        # Scan combined text for patterns and entropy
        return scan_for_secrets(combined)
```

### 3.5 Vault Exact-Value Matching

The most reliable method — if an exact vault value appears in outbound traffic, it's definitionally a leak. HermesKatana already does this. Key improvements:
- Pre-compute all encoding variants of each vault value at vault-load time (not per-request)
- Cache the variant set, invalidate on vault write

---

## 4. MITM Proxy Security

### 4.1 mitmproxy Architecture
mitmproxy intercepts TLS traffic by acting as a transparent CA:
1. Client connects to proxy
2. Proxy generates a certificate for the target domain, signed by its root CA
3. Client must trust the proxy's CA (installed in system/app trust store)
4. Proxy decrypts traffic, runs addon code, re-encrypts to forward

**Coverage**: HTTP/1, HTTP/2, WebSockets. Does NOT cover: QUIC/HTTP3, raw TCP, UDP, stdio.

### 4.2 Secrets Window on Disk

HermesKatana's proxy startup sequence has a brief window where secrets exist in `proxy-config.json`:
```
write_text(config_with_secrets) → proxy starts → addon reads + overwrites config
```
Even with chmod 0600, this window is exploitable by a process running as the same user.

**Improvement**: Use a Unix domain socket or named pipe for secret transmission instead of file:
```python
# Parent process
import socket, os
sock_path = f"/tmp/katana-proxy-{os.getpid()}.sock"
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(sock_path)
server.listen(1)
# Start proxy with sock_path as argument
# ... proxy connects back to receive secrets ...
os.unlink(sock_path)  # Remove after transfer
```

### 4.3 Protocol Coverage Gaps

| Protocol | Coverage | Mitigation |
|----------|----------|------------|
| HTTP/HTTPS | Full | Proxy intercepts |
| WebSocket | Full (via HTTP upgrade) | Proxy intercepts |
| HTTP/3 (QUIC) | None | Block UDP 443 at firewall |
| Raw TCP | None | Docker network isolation |
| UDP | None | Docker network isolation |
| stdio (MCP) | None | Python-level hook in installer |
| Unix domain sockets | None | Process isolation |

### 4.4 Process Isolation
HermesKatana uses `os.setsid()` (POSIX only) to put the proxy in its own process group. On Windows, use `creationflags=subprocess.CREATE_NEW_PROCESS_GROUP`.

---

## 5. API Key Injection Architecture

### 5.1 Why Proxy-Level Injection is Superior

| Approach | API keys in agent memory | Keys in .env file | Proxy-level injection |
|----------|------------------------|-------------------|----------------------|
| Memory dump attack | Exposed | Exposed | **Not exposed** |
| Env var leak (tool output) | Exposed | Exposed | **Not exposed** |
| Log file leak | Depends | Depends | **Not exposed** |
| Screenshot/video exfil | Exposed | Depends | **Not exposed** |
| Legitimate agent use | Works | Works | **Works** |

With proxy injection, the agent process receives `"aegis-managed"` as the API key value. Real keys only exist in the vault (encrypted) and the proxy process (ephemeral, in memory only while proxy runs).

### 5.2 Provider Authentication Patterns

| Provider | Auth Header | Scheme |
|----------|------------|--------|
| OpenAI | `Authorization` | `Bearer sk-...` |
| Anthropic | `x-api-key` | `sk-ant-...` |
| Google AI | `x-goog-api-key` | Direct value |
| Groq | `Authorization` | `Bearer gsk_...` |
| Together | `Authorization` | `Bearer ...` |
| OpenRouter | `Authorization` | `Bearer sk-or-...` |
| Vercel AI | `Authorization` | `Bearer ...` |
| DeepSeek | `Authorization` | `Bearer sk-...` |
| Mistral | `Authorization` | `Bearer ...` |
| Cohere | `Authorization` | `Bearer ...` |
| HuggingFace | `Authorization` | `Bearer hf_...` |
| Replicate | `Authorization` | `Token r8_...` |

### 5.3 Hot-Reload vs Restart-to-Rotate

Current: vault key changes restart the proxy on the same port.
**Improvement**: Add a reload endpoint or signal handler:

```python
# In addon.py
def _reload_vault(self):
    """Called on SIGUSR1 — reload vault keys without restarting proxy."""
    new_keys = self.vault.get_all_values()
    with self._keys_lock:
        self._injected_keys = new_keys
    self.audit.log(CIRCUIT_BREAKER, "vault keys reloaded")
```

---

## 6. Hash-Chained Audit Logs

### 6.1 Design Review

HermesKatana's audit trail: SHA-256 hash chain, O(1) last-hash cache, file locking, auto-rotation at 10MB.

The chain structure:
```
Entry N:
  entry_hash = SHA256(timestamp + event_type + tool_name + args_hash + decision + prev_hash)
  prev_hash = entry_hash of Entry N-1
  first entry: prev_hash = "0" * 64
```

**What this detects**:
- Modification of any entry field → hash mismatch at that entry
- Deletion of any middle entry → chain break (prev_hash mismatch)
- Reordering entries → chain break
- Insertion of fake entries → chain break (unless appended to end)

**What this does NOT detect**:
- Deletion of the most recent N entries (chain still valid from the new tail)
- Complete log replacement (chain is consistent from genesis)

### 6.2 SHA-256 vs SHA-3 vs BLAKE3

| Algorithm | Speed | Security | FIPS | Notes |
|-----------|-------|----------|------|-------|
| SHA-256 | Baseline | 128-bit | Yes (FIPS 180-4) | Hardware acceleration on x86 |
| SHA-3-256 | 0.3–0.5× SHA-256 | 128-bit | Yes (FIPS 202) | Different construction, no length extension |
| BLAKE3 | 3–10× SHA-256 | 128-bit | No | Fastest software hash; parallel |
| BLAKE2b | 2–4× SHA-256 | 128-bit | No | Python stdlib (hashlib) |

For audit logs, the bottleneck is I/O not hashing. SHA-256 is fine and has the advantage of being in `hashlib` (no dependency). BLAKE3 could be a future option if hash throughput becomes a concern (very large audit trails).

### 6.3 Rotation Without Breaking the Chain

HermesKatana auto-rotates at 10MB. The current approach archives the old file and starts fresh with a new genesis block. This breaks the chain across rotation boundaries.

**Improved approach**: Cross-file chain linking
```python
def rotate(self) -> Path:
    archive = self._log_path.with_suffix(f".{timestamp}.jsonl")
    self._log_path.rename(archive)
    
    # New log's first entry contains hash of last entry in archived log
    last_hash_from_old = self._last_hash
    self._last_hash = "0" * 64  # Reset for new file
    
    # Write a ROTATION sentinel entry
    self.log(AuditEntry(
        event_type=AuditEventType.CONFIG_CHANGE,
        tool_name="rotation",
        details={
            "archived_to": str(archive),
            "last_hash_of_archive": last_hash_from_old,
        }
    ))
    return archive
```

### 6.4 Remote Log Shipping

For production deployments, shipping audit logs to a remote (attacker-inaccessible) location provides true tamper protection:

```python
class RemoteAuditShipper:
    def __init__(self, endpoint: str, api_key: str, batch_size: int = 10):
        self.endpoint = endpoint
        self.api_key = api_key
        self.batch: list[dict] = []

    def ship(self, entry: AuditEntry) -> None:
        self.batch.append(entry.model_dump())
        if len(self.batch) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        requests.post(
            self.endpoint,
            json={"entries": self.batch},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=5,
        )
        self.batch.clear()
```

Compatible with: AWS CloudWatch Logs, Datadog Logs, Loki, any SIEM.

---

## 7. Zero Trust for Agent Secrets

### 7.1 Core Principles Applied to HermesKatana

**Never trust, always verify**: Every vault access includes an integrity check (HMAC). Every tool call passes through middleware. No component is implicitly trusted.

**Micro-segmentation**: Different vault categories (LLM keys, git tokens, MCP credentials) use different derived encryption keys (DEK hierarchy). Compromise of one category doesn't expose others.

**Assume breach**: The circuit breaker is a "assume breach" control — when anomaly count exceeds threshold, vault is locked, proxy is killed, all activity stops. Recovery requires explicit human action.

**Least privilege**: The policy engine enforces minimum capabilities. Tools that don't need write access don't get it. Tainted data can't flow to high-privilege tools.

### 7.2 Circuit Breaker Design Improvements

Current: `vault.lock` sentinel file activates on reactive agent trigger.

**Improvements**:
```python
class VaultCircuitBreaker:
    LOCK_LEVELS = {
        0: "normal",       # All operations allowed
        1: "degraded",     # Read-only; no new key adds
        2: "locked",       # No reads or writes
        3: "sealed",       # Requires out-of-band recovery key
    }

    def trip(self, level: int, reason: str) -> None:
        """Trip circuit breaker to given level."""
        self._write_lock(level, reason, datetime.utcnow())
        # Kill proxy if level >= 2
        if level >= 2:
            self._kill_proxy()
        # Emit audit event
        self._audit.log(CIRCUIT_BREAKER, {"level": level, "reason": reason})
```

---

## 8. HermesKatana Improvements (22 specific items)

### vault/store.py
- **V1**: Argon2id for password-based master key derivation (headless mode): `pip install argon2-cffi`
- **V2**: HKDF-based DEK hierarchy — separate DEKs for LLM keys, git tokens, MCP credentials
- **V3**: Atomic key rotation using temp file + rename instead of in-place re-encryption
- **V4**: Vault integrity check on every `get()` call (not just on explicit verify)
- **V5**: `unix:// socket` secret delivery to proxy (eliminates config-file secrets window)
- **V6**: Multi-level circuit breaker (normal/degraded/locked/sealed)

### audit/trail.py
- **A1**: Cross-file chain linking on rotation (prev_log_last_hash in first entry)
- **A2**: Optional BLAKE3 hashing for high-throughput deployments (`pip install blake3`)
- **A3**: Remote log shipping adapter (CloudWatch, Loki, SIEM)
- **A4**: Merkle tree option for batch verification of large log files

### proxy/runner.py + addon.py
- **P1**: Unix socket secret delivery (V5 above) — proxy receives secrets via socket on startup
- **P2**: SIGUSR1 handler for hot vault key reload without proxy restart
- **P3**: HTTP/3 blocking via iptables/nftables rule injection (document this in installer)
- **P4**: Pre-screen LLM request bodies for IMPORTANT tags and zero-width chars

### scanner/secrets.py
- **S1**: Add 7 missing provider patterns (Slack, Stripe, SendGrid, Twilio, Vercel, Databricks, HuggingFace)
- **S2**: Double base64 decoding pass
- **S3**: Unicode escape decoding (`\uXXXX` sequences)
- **S4**: Pre-compute all encoding variants of vault values at load time (cache invalidation on vault write)
- **S5**: ChunkedSecretDetector class for split-secret detection across request sequence

### installer/patches.py
- **I1**: Add SIGTERM handler to hermes process that flushes audit trail before exit
- **I2**: Document iptables rule for blocking UDP 443 (HTTP/3) in Docker isolation mode

### General
- **G1**: Keyring unavailability detection with clear user guidance (headless mode fallback to Argon2id)
- **G2**: Quarterly key rotation reminder in `katana status` output

---

## References

- OWASP Cryptographic Storage Cheat Sheet — owasp.org/www-project-cheat-sheets
- RFC 9106 — Argon2 (IETF, 2021)
- RFC 7914 — scrypt (IETF, 2016)
- NIST SP 800-132 — PBKDF recommendations
- GitGuardian Shannon entropy documentation — docs.gitguardian.com/secrets-detection
- Python cryptography library — cryptography.io
- mitmproxy architecture — docs.mitmproxy.org/stable/concepts/how-mitmproxy-works
- dev.to/veritaschain — Hash-chained audit log implementation
- BLAKE3 spec — github.com/BLAKE3-team/BLAKE3
- argon2-cffi — pypi.org/project/argon2-cffi
