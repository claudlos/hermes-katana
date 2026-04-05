# HermesKatana Runbook

Day-2 operations guide for operators managing HermesKatana deployments.

---

## Table of Contents

- [Primary Operator Commands](#primary-operator-commands)
- [State Locations](#state-locations)
- [Rotating Vault Keys](#rotating-vault-keys)
- [Customizing Policies](#customizing-policies)
- [Adding Custom Scanner Patterns](#adding-custom-scanner-patterns)
- [Reading Audit Trails](#reading-audit-trails)
- [Incident Response Playbook](#incident-response-playbook)
- [Performance Tuning](#performance-tuning)
- [CI and Validation](#ci-and-validation)
- [Maintainer Workflows](#maintainer-workflows)

---

## Primary Operator Commands

| Task | Command |
|------|---------|
| Health check | `katana doctor` |
| Hermes checkout health | `katana doctor --target /path/to/hermes` |
| Local state overview | `katana status` |
| Checkout state | `katana status --target /path/to/hermes` |
| Install patches | `katana install --target /path/to/hermes --backup` |
| Remove patches | `katana uninstall --target /path/to/hermes` |
| Run Hermes with Katana | `katana run --target /path/to/hermes -- --task "hello"` |
| List policies | `katana policy list` |
| Switch policy preset | `katana policy use paranoid` |
| Export policies to YAML | `katana policy export policy.yaml` |
| List vault keys | `katana vault list` |
| Store a secret | `katana vault set KEY VALUE` |
| Remove a secret | `katana vault remove KEY` |
| Rotate master key | `katana vault rotate` |
| Lock vault (emergency) | `katana vault lock` |
| Unlock vault | `katana vault unlock` |
| Verify vault integrity | `katana vault verify` |
| Show audit entries | `katana audit show --limit 20` |
| Verify audit chain | `katana audit verify` |
| Audit statistics | `katana audit stats` |
| Clear audit trail | `katana audit clear` |
| Start proxy | `katana proxy start` |
| Stop proxy | `katana proxy stop` |
| Proxy status | `katana proxy status` |

---

## State Locations

| Item | Path |
|------|------|
| Global config | `~/.hermes-katana/config.yaml` |
| Vault file | `~/.config/hermes-katana/vault.json` |
| Vault lock sentinel | `~/.config/hermes-katana/vault.lock` |
| Audit trail | `~/.config/hermes-katana/audit/audit.jsonl` |
| Proxy PID file | `$TMPDIR/hermes_katana_proxy.pid` |
| Checkout config | `<hermes>/.katana/katana.yaml` |
| Checkout audit | `<hermes>/.katana/audit/` |
| Checkout certs | `<hermes>/.katana/certs/` |
| Checkout backups | `<hermes>/.katana-backups/` |

---

## Rotating Vault Keys

The vault uses AES-256-GCM encryption with the master key stored in the
OS keyring. Rotate keys periodically or after any suspected compromise.

### Routine Rotation

```bash
# 1. Verify current state
katana vault verify

# 2. Rotate -- re-encrypts all values with a new master key
katana vault rotate

# 3. Confirm the new state
katana vault verify
katana vault list
```

**What happens during rotation:**

1. A new 256-bit master key is generated
2. All values are decrypted with the old key
3. All values are re-encrypted with the new key (fresh nonces)
4. The new HMAC is computed over all entries
5. The vault file is written atomically (tmp + rename)
6. The old master key is replaced in the OS keyring
7. The old key material is zeroed from memory

### Recovery After Failed Rotation

If rotation is interrupted (power loss, crash), the vault has a recovery
mechanism:

```bash
# Check if a rotation was in progress
katana vault verify

# If verify reports a rotation recovery file, apply it:
# The vault will auto-recover on next access
katana vault list
```

### Programmatic Key Rotation

```python
from hermes_katana.vault.store import Vault

vault = Vault()
vault.verify_integrity()  # pre-check
vault.rotate_key()        # atomic rotation
vault.verify_integrity()  # post-check
```

---

## Customizing Policies

### Using a Built-in Preset

```bash
katana policy use balanced    # Smart defaults
katana policy use paranoid    # Maximum security
katana policy use permissive  # Monitoring only
```

### Writing Custom Policy YAML

Export the current preset as a starting point:

```bash
katana policy export my-policies.yaml
```

Edit the file:

```yaml
name: my-org-policies
version: "2.0.0"
extends: balanced
policies:
  # Block crypto mining commands
  - name: block_crypto_mining
    description: "Block mining-related terminal commands"
    tool_pattern: terminal
    conditions:
      - field: command
        operator: matches_pattern
        value: ".*(xmrig|minergate|cryptonight|ethminer).*"
    action: deny
    priority: 200
    tags: [crypto, critical]

  # Require approval for database tools with tainted input
  - name: escalate_tainted_db
    description: "Escalate when tainted data flows to DB operations"
    tool_pattern: "database_*"
    conditions:
      - field: query
        operator: contains_taint
        value: "true"
    action: escalate
    priority: 150
    tags: [database, taint]

  # Allow read-only file tools unconditionally
  - name: allow_reads
    description: "Read tools are always safe"
    tool_pattern: "read_file"
    action: allow
    priority: 100
```

Load it by editing `~/.hermes-katana/config.yaml`:

```yaml
policy_path: /path/to/my-policies.yaml
```

Or programmatically:

```python
from hermes_katana.policy import PolicyEngine
from hermes_katana.policy.yaml_loader import load_yaml_policies

policies = load_yaml_policies("/path/to/my-policies.yaml")
engine = PolicyEngine(policies=policies)
result = engine.evaluate(tool_name="terminal", args={"command": "xmrig"})
print(result.action)  # PolicyResult.DENY
```

### Condition Operators Reference

| Operator           | Description                                     |
|--------------------|-------------------------------------------------|
| `contains_taint`   | True when the field carries any taint label     |
| `source_is`        | True when taint source matches given value      |
| `reader_lacks`     | True when reader set lacks given capability     |
| `matches_pattern`  | True when field value matches a regex           |
| `argument_matches` | True when argument value matches a glob         |
| `taint_level_gte`  | True when taint severity >= threshold           |
| `has_label`        | True when a specific taint label is present     |

---

## Adding Custom Scanner Patterns

The scanner supports custom patterns for injection, secret, and command
detection. See [CONTRIBUTING.md](../CONTRIBUTING.md#adding-new-scanner-patterns)
for the full guide on adding patterns.

Quick example -- adding an injection pattern:

```python
# In src/hermes_katana/scanner/injection.py, add to PATTERNS:
(
    "custom_category",
    r"your_regex_pattern",
    0.85,  # confidence 0.0-1.0
    "Human-readable description",
)
```

Using the allowlist to suppress false positives:

```python
from hermes_katana.scanner.allowlist import Allowlist

al = Allowlist()
al.add(pattern_id="safe-123", reason="Known safe in our context")
```

---

## Reading Audit Trails

### CLI Access

```bash
# Show the last 20 entries
katana audit show --limit 20

# Verify the hash chain has not been tampered with
katana audit verify

# Get aggregate statistics
katana audit stats
```

### Understanding Audit Entries

Each entry in the JSONL audit trail contains:

```json
{
  "timestamp": "2025-01-15T10:30:45.123Z",
  "event_type": "tool_call",
  "tool_name": "terminal",
  "decision": "deny",
  "reasons": ["taint_flow_denied: WEB_CONTENT -> terminal"],
  "taint_context": {
    "tainted_fields": {"command": {"labels": ["WEB_CONTENT"]}}
  },
  "scan_results": [
    {"scanner": "injection", "category": "instruction_override", "confidence": 0.95}
  ],
  "prev_hash": "a1b2c3d4...",
  "hash": "e5f6a7b8..."
}
```

**Event types:** `tool_call`, `scan_hit`, `policy_eval`, `taint_flow`,
`vault_access`, `proxy_request`, `session_start`, `session_end`

### Programmatic Audit Access

```python
from hermes_katana.audit.trail import AuditTrail

trail = AuditTrail()

# Query recent entries
entries = trail.query(limit=50)
for entry in entries:
    print(f"{entry['timestamp']} {entry['event_type']}: {entry['decision']}")

# Verify chain integrity
is_valid = trail.verify()
print(f"Chain valid: {is_valid}")

# Statistics
stats = trail.stats()
print(f"Total entries: {stats['total_entries']}")
```

### Hash Chain Verification

The audit trail is a SHA-256 hash chain. Each entry includes:
- `hash`: SHA-256 of `prev_hash + json(entry_content)`
- `prev_hash`: hash of the previous entry (genesis entry uses zeros)

If any entry is modified, all subsequent hashes become invalid, making
tampering immediately detectable.

---

## Incident Response Playbook

### Suspected Prompt Injection Attack

1. **Lock down immediately:**

   ```bash
   katana vault lock          # Block all secret access
   katana proxy stop          # Stop outbound traffic
   ```

2. **Review the audit trail:**

   ```bash
   katana audit show --limit 100
   katana audit verify        # Ensure trail was not tampered
   ```

3. **Check for data exfiltration:**

   Look for `event_type: tool_call` entries where:
   - `tool_name` is `send_message`, `terminal`, or any network tool
   - `decision` is `allow` with tainted inputs
   - Outbound URLs in proxy logs point to unknown domains

4. **Rotate secrets:**

   ```bash
   katana vault rotate
   ```

5. **Tighten policy:**

   ```bash
   katana policy use paranoid
   ```

6. **Resume operations:**

   ```bash
   katana vault unlock
   katana proxy start
   ```

### Vault Compromise Suspected

1. Lock: `katana vault lock`
2. Rotate: `katana vault rotate`
3. Re-provision all secrets: `katana vault set KEY NEW_VALUE` for each key
4. Verify: `katana vault verify`
5. Review access log for anomalies

### Audit Chain Corruption

1. Run `katana audit verify` -- identifies the first broken link
2. Back up the corrupted file
3. `katana audit clear` to start fresh
4. Investigate the gap in the broken chain
5. Consider this a security incident -- someone may have tampered with logs

### Proxy Certificate Issues

1. `katana proxy stop`
2. Delete stale certs: `rm -rf <hermes>/.katana/certs/`
3. Reinstall: `katana install --target /path/to/hermes --backup`
4. `katana proxy start`

---

## Performance Tuning

### Scanner Performance

The scanner runs synchronously on every tool call. Typical latencies:

| Component | Latency |
|-----------|---------|
| Injection scan | ~2-5ms per input |
| Secret scan | ~1-3ms per input |
| Command scan | ~0.5-1ms per command |
| Unicode scan | ~1-2ms per input |
| Content scan | ~1-3ms per output |
| Full middleware chain | ~5-15ms per tool call |

To reduce overhead, disable specific scanners in config:

```yaml
# In ~/.hermes-katana/config.yaml
scan_inputs: true        # Disable if input scanning is too slow
scan_outputs: true       # Output scanning is cheaper
scan_commands: true      # Command scanning is fast (~1ms)
```

### Proxy Performance

The mitmproxy addon adds ~5-20ms per HTTP request for secret scrubbing.
To reduce overhead:

- Use `ignore_hosts` in proxy config for trusted internal endpoints
- Disable `inject_credentials` if not needed
- Set `scan_request_bodies: false` for high-bandwidth endpoints

### Vault Performance

Vault operations are I/O-bound (file read + keyring access):

| Operation | Latency |
|-----------|---------|
| `get` | ~1-5ms (file read + decrypt) |
| `set` | ~5-10ms (encrypt + atomic write) |
| `rotate` | ~50-200ms (re-encrypt all + write) |

The vault uses a reentrant lock -- concurrent access is serialized.

### Taint Tracker Memory

The taint tracker holds references to all registered values. For
long-running sessions, use scoped tracking:

```python
with TaintTracker.scoped() as tracker:
    # Values registered here are released when the scope exits
    pass
```

---

## CI and Validation

### Running Tests

```bash
# Full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src \
  python3 -m pytest -p pytest_asyncio.plugin -q

# Specific test modules
python3 -m pytest tests/unit/test_scanner.py -v
python3 -m pytest tests/unit/test_taint.py -v
python3 -m pytest tests/unit/test_policy.py -v
```

Use `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` with the explicit
`pytest_asyncio.plugin` entry point to avoid config warnings.

### Compatibility Validation

```bash
python3 -m pytest tests/unit/test_compat_snapshots.py \
  tests/unit/test_installer.py tests/unit/test_bootstrap.py -v
```

Pinned snapshots: `tests/fixtures/hermes_compat/`
Adversarial eval pack: `evals/adversarial_dispatch.yaml`

---

## Maintainer Workflows

### Refresh Hermes Compatibility Snapshots

When a new Hermes release ships:

```bash
# 1. Preview
python3 scripts/refresh_compat_snapshots.py \
  --source /path/to/hermes-release \
  --source-ref vX.Y.Z --dry-run

# 2. Refresh with archive verification
python3 scripts/refresh_compat_snapshots.py \
  --source /path/to/hermes-release \
  --source-archive /path/to/hermes-vX.Y.Z.tar.gz \
  --archive-sha256 <published_sha256> \
  --source-ref vX.Y.Z --replace-existing

# 3. Run compatibility tests
python3 -m pytest tests/unit/test_compat_snapshots.py \
  tests/unit/test_installer.py tests/unit/test_bootstrap.py -v

# 4. Update docs/compatibility.md if snapshot list changed
```

### Install Into a Hermes Checkout

```bash
katana doctor --target /path/to/hermes
katana install --target /path/to/hermes --dry-run   # preview
katana install --target /path/to/hermes --backup     # install
katana status --target /path/to/hermes               # confirm
katana run --target /path/to/hermes -- --task "test"  # run
```

### Recover From Install Mistakes

```bash
# Find the backup manifest
ls <hermes>/.katana-backups/

# Preview rollback
katana restore --manifest <manifest.json> --dry-run

# Apply rollback
katana restore --manifest <manifest.json>
katana status --target /path/to/hermes
```
