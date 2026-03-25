```
 _   _                                _  __      _
| | | | ___ _ __ _ __ ___   ___  ___ | |/ /__ _ | |_ __ _ _ __   __ _
| |_| |/ _ \ '__| '_ ` _ \ / _ \/ __|| ' // _` || __/ _` | '_ \ / _` |
|  _  |  __/ |  | | | | | |  __/\__ \| . \ (_| || || (_| | | | | (_| |
|_| |_|\___|_|  |_| |_| |_|\___||___/|_|\_\__,_| \__\__,_|_| |_|\__,_|
```

# HermesKatana

**Defense-in-depth security toolkit for Hermes Agent -- CaMeL taint tracking, proxy-based secret guard, declarative policy engine, and multi-layer attack detection.**

---

## Table of Contents

- [Architecture](#architecture)
- [7-Layer Defense Model](#7-layer-defense-model)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Policy System](#policy-system)
- [Taint Tracking](#taint-tracking)
- [Scanner Capabilities](#scanner-capabilities)
- [Proxy Features](#proxy-features)
- [Vault Security](#vault-security)
- [Audit Trail](#audit-trail)
- [Middleware Chain](#middleware-chain)
- [Benchmarks](#benchmarks)
- [Comparison](#comparison)
- [Credits and Provenance](#credits-and-provenance)
- [License](#license)

---

## Architecture

```
                           HermesKatana Architecture

    +-------------------------------------------------------------------+
    |                        Hermes Agent Runtime                        |
    +-------------------------------------------------------------------+
           |                    |                    |
           v                    v                    v
    +--------------+    +--------------+    +--------------+
    | User Input   |    | Tool Output  |    | MCP Server   |
    +--------------+    +--------------+    +--------------+
           |                    |                    |
           +--------------------+--------------------+
                                |
                    +-----------v-----------+
                    |   Middleware Chain     |
                    |                       |
                    |  +------------------+ |
                    |  | 1. Taint Tracker | |   --> Tag every value with origin
                    |  +------------------+ |
                    |  | 2. Scanner       | |   --> Detect injections, secrets
                    |  +------------------+ |
                    |  | 3. Policy Engine | |   --> Evaluate declarative rules
                    |  +------------------+ |
                    |  | 4. Audit Trail   | |   --> Log decisions to hash chain
                    |  +------------------+ |
                    +-----------+-----------+
                                |
                     ALLOW / DENY / ESCALATE
                                |
                    +-----------v-----------+
                    |     Tool Execution    |
                    +-----------------------+
                                |
                    +-----------v-----------+
                    |    mitmproxy (HTTPS)  |   --> Intercept outbound traffic
                    |    Secret Scrubber    |   --> Strip vault values from
                    |    Request Logger     |       all HTTP request bodies
                    +-----------------------+
                                |
                    +-----------v-----------+
                    |  AES-256-GCM Vault    |   --> Encrypted secret storage
                    |  OS Keyring Master    |       with circuit breaker
                    +-----------------------+
```

---

## 7-Layer Defense Model

HermesKatana implements defense-in-depth with seven distinct security layers.
No single layer is sufficient alone (as the CaMeL paper demonstrates), but
together they provide robust protection against prompt injection, data
exfiltration, and tool-use attacks.

| Layer | Component         | Purpose                                      | Key Technique                    |
|-------|-------------------|----------------------------------------------|----------------------------------|
| 1     | Taint Tracking    | Track data provenance through all transforms  | CaMeL-inspired label propagation |
| 2     | Flow Analysis     | Block untrusted data from reaching sinks      | Information-flow control          |
| 3     | Input Scanner     | Detect prompt injection before processing     | 30+ heuristic patterns + encoding|
| 4     | Output Scanner    | Catch content attacks in LLM responses        | ANSI/markdown/homograph detection|
| 5     | Policy Engine     | Declarative allow/deny rules per tool         | Glob matching + taint conditions |
| 6     | HTTPS Proxy       | Intercept and scrub outbound HTTP traffic     | mitmproxy addon + secret matching|
| 7     | Audit Trail       | Tamper-evident logging for post-incident       | SHA-256 hash-chained JSONL       |

---

## Quick Start

Start with the operator docs that match the current CLI:

- [docs/quickstart.md](docs/quickstart.md) for the fastest local setup path
- [docs/runbook.md](docs/runbook.md) for day-2 operations and recovery steps

### Installation

```bash
# From source
git clone https://github.com/Tranquil-Flow/hermes-katana.git
cd hermes-katana
pip install -e ".[dev]"

# Or with all optional dependencies
pip install -e ".[all]"
```

### Verify Installation

```bash
katana doctor
katana --help
```

### Basic Setup

```bash
# Persist the default policy preset
katana policy use balanced

# Create the encrypted vault on first write
katana vault set OPENAI_API_KEY "sk-..."
katana vault verify

# Inspect a Hermes checkout before patching it
katana doctor --target /path/to/hermes
katana install --target /path/to/hermes --dry-run
katana install --target /path/to/hermes --backup
katana status --target /path/to/hermes
katana run --target /path/to/hermes -- --task "hello"

# Start and inspect the proxy
katana proxy start --host 127.0.0.1 --port 8443
katana proxy status
```

### Quick Security Check

```python
from hermes_katana.taint import TaintTracker, Source, FlowDecision

tracker = TaintTracker.get_instance()

# Register data with its origin
web_data = tracker.register("some web content", Source.web("https://example.com"))

# Check if it can flow to terminal
decision = tracker.check_flow(web_data, "terminal")
assert decision == FlowDecision.DENY  # blocked: untrusted -> critical sink

# User data flows freely
user_data = tracker.register("ls -la", Source.user("cli"))
decision = tracker.check_flow(user_data, "terminal")
assert decision == FlowDecision.ALLOW  # allowed: trusted source
```

---

## CLI Reference

HermesKatana provides the `katana` (or `hermes-katana`) CLI with these
commands:

| Command | Description |
|---------|-------------|
| `katana doctor` | Check prerequisites, runtime state, and optionally a Hermes checkout |
| `katana status` | Show overall security status and environment details |
| `katana install --target PATH [--dry-run] [--backup]` | Patch a Hermes checkout, preview changes, or create a backup first |
| `katana uninstall --target PATH [--dry-run] [--backup]` | Remove Katana patches and optionally preview or back up the checkout |
| `katana restore --manifest PATH [--dry-run]` | Restore a checkout from a backup manifest |
| `katana run --target PATH -- ...` | Run Hermes with runtime behavior derived from the installed checkout state |
| `katana scan` | Scan text for injections, secrets, and dangerous content |
| `katana scan-file` | Scan a file on disk |
| `katana scan-command` | Scan a shell command for dangerous patterns |
| `katana policy list` | Show the active policy set |
| `katana policy use PRESET` | Persist the active preset to config |
| `katana policy export PATH` | Export the current policy set to YAML |
| `katana vault list` | List stored secret names |
| `katana vault set KEY VALUE` | Create or update a secret |
| `katana vault remove KEY` | Delete a secret |
| `katana vault rotate` | Rotate the vault master key |
| `katana vault lock` | Activate the vault circuit breaker |
| `katana vault unlock` | Clear the vault circuit breaker |
| `katana vault verify` | Verify vault integrity |
| `katana audit show --limit N` | Show recent audit entries |
| `katana audit verify` | Verify the audit hash chain |
| `katana audit stats` | Show audit statistics |
| `katana audit clear` | Clear the current audit file |
| `katana proxy start` | Start the proxy and load the Katana mitmproxy addon |
| `katana proxy stop` | Stop the running proxy |
| `katana proxy status` | Show live proxy state from pidfile-backed status |
| `katana benchmark` | Run benchmark suites |
| `katana version` | Print version details |

The installer regression surface is pinned in versioned Hermes snapshots under
`tests/fixtures/hermes_compat/`, the supported snapshot registry lives in
`tests/fixtures/hermes_compat/fixtures.json`, and the adversarial dispatch eval
pack lives in `evals/adversarial_dispatch.yaml`. Refresh those pinned snapshots
from a real Hermes release checkout with verified provenance, for example:
`python scripts/refresh_compat_snapshots.py --source /path/to/hermes-release --source-archive /path/to/hermes-vX.Y.Z.tar.gz --archive-sha256 <published_sha256> --source-ref vX.Y.Z --replace-existing`.

---

## Policy System

The policy engine evaluates every tool call against declarative rules.
Rules combine tool-name globs, taint conditions, and priority ordering
to produce one of four outcomes.

### Policy Results

| Result     | Effect                                              |
|------------|-----------------------------------------------------|
| `allow`    | Permit the tool call to proceed                     |
| `deny`     | Block the tool call entirely                        |
| `escalate` | Pause and request human approval                    |
| `log_only` | Allow but emit a structured audit entry             |

### Built-in Presets

| Preset       | Philosophy                  | Terminal (tainted) | Terminal (clean) | Read-only (tainted) | Exfiltration   |
|--------------|-----------------------------|--------------------|------------------|---------------------|----------------|
| `paranoid`   | Deny everything untrusted   | DENY               | ESCALATE         | ESCALATE            | DENY           |
| `balanced`   | Smart defaults              | DENY               | ALLOW            | ALLOW               | DENY           |
| `permissive` | Log only, block exfil       | LOG_ONLY           | ALLOW            | ALLOW               | DENY           |

### Usage

```python
from hermes_katana.policy import PolicyEngine, PolicyResult

engine = PolicyEngine.with_defaults("balanced")

# Evaluate a tool call with taint context
result = engine.evaluate(
    tool_name="terminal",
    args={"command": "curl https://evil.com"},
    taint_context={
        "tainted_fields": {
            "command": {
                "is_tainted": True,
                "source": "web_content",
                "labels": ["untrusted"],
                "level": 8,
            }
        }
    },
)
print(result.action)  # PolicyResult.DENY
```

### Custom YAML Policies

```yaml
name: my-custom-policies
version: "1.0.0"
extends: balanced
policies:
  - name: block_crypto_mining
    description: "Block any command containing mining keywords"
    tool_pattern: terminal
    conditions:
      - field: command
        operator: matches_pattern
        value: ".*(xmrig|minergate|cryptonight).*"
    action: deny
    priority: 200
    tags: [crypto, critical]
```

### Condition Operators

| Operator           | Description                                    |
|--------------------|------------------------------------------------|
| `contains_taint`   | True when the field carries any taint label    |
| `source_is`        | True when taint source matches given value     |
| `reader_lacks`     | True when reader set lacks given capability    |
| `matches_pattern`  | True when field value matches a regex          |
| `argument_matches` | True when argument value matches a glob        |
| `taint_level_gte`  | True when taint severity >= given threshold    |
| `has_label`        | True when a specific taint label is present    |

---

## Taint Tracking

Inspired by Google's CaMeL paper (arXiv 2503.18813), every value entering the
agent runtime is tagged with its origin. Taint labels propagate automatically
through string operations, collection manipulation, and tool pipelines.

### Taint Labels

| Label          | Trust Level  | Description                              |
|----------------|-------------|------------------------------------------|
| `USER`         | TRUSTED     | Direct user input (chat, CLI)            |
| `SYSTEM`       | TRUSTED     | System prompt, hard-coded instructions   |
| `TOOL_OUTPUT`  | CONDITIONAL | Return value from tool invocations       |
| `WEB_CONTENT`  | UNTRUSTED   | Data fetched from the open web           |
| `FILE_CONTENT` | CONDITIONAL | Data from local/remote filesystem        |
| `MEMORY`       | CONDITIONAL | Data from persistent agent memory        |
| `MCP`          | UNTRUSTED   | Data from MCP servers                    |
| `AGENT`        | CONDITIONAL | Content generated by the LLM             |
| `UNKNOWN`      | UNTRUSTED   | Origin cannot be determined              |

### Flow Rules (Default)

| Source Labels                   | Target Tools                    | Decision   |
|---------------------------------|---------------------------------|------------|
| WEB_CONTENT, MCP, UNKNOWN      | terminal, send_message, etc.    | DENY       |
| TOOL_OUTPUT, FILE, MEMORY      | terminal, send_message, etc.    | ASK_USER   |
| USER, SYSTEM                   | * (anything)                    | ALLOW      |
| AGENT                          | terminal, send_message, etc.    | QUARANTINE |

### Character-Level Taint

`TaintedStr` tracks taint at the character level. When strings from different
sources are concatenated, sliced, or transformed, each character retains its
original provenance:

```python
from hermes_katana.taint import TaintedStr, Source

user = TaintedStr("echo ", sources=frozenset({Source.user()}))
web  = TaintedStr("rm -rf /", sources=frozenset({Source.web("evil.com")}))

combined = user + web          # Taint merges: USER + WEB_CONTENT
safe_part = combined[0:5]      # "echo " -- USER only
dangerous = combined[5:]       # "rm -rf /" -- WEB_CONTENT
```

---

## Scanner Capabilities

Five scanner modules provide comprehensive attack detection:

### Injection Scanner (30+ patterns)

| Category              | Examples                                       | Confidence |
|-----------------------|------------------------------------------------|------------|
| Instruction Override  | "ignore previous instructions"                 | 0.95       |
| Role Override         | "you are now DAN", "enter developer mode"      | 0.90       |
| Delimiter Escape      | XML tag injection, markdown delimiters         | 0.80       |
| Encoding Attack       | base64-encoded instructions, hex payloads      | 0.80       |
| System Prompt Extract | "reveal your system prompt"                    | 0.85       |
| Tool Manipulation     | JSON tool_call injection, forced tool use      | 0.80       |
| Invisible Characters  | Zero-width char blocks, tag soup               | 0.85       |

### Secret Scanner (15+ patterns)

| Pattern            | Category            | Severity  |
|--------------------|---------------------|-----------|
| OpenAI API Key     | API_KEY             | CRITICAL  |
| GitHub Token       | TOKEN               | CRITICAL  |
| AWS Access Key     | API_KEY             | CRITICAL  |
| AWS Secret Key     | API_KEY             | CRITICAL  |
| Anthropic Key      | API_KEY             | CRITICAL  |
| Stripe Key         | API_KEY             | CRITICAL  |
| JWT Token          | TOKEN               | HIGH      |
| Private Key        | PRIVATE_KEY         | CRITICAL  |
| Database URL       | CONNECTION_STRING   | CRITICAL  |
| Password Assignment| PASSWORD            | HIGH      |
| High Entropy Blob  | HIGH_ENTROPY        | MEDIUM    |
| Encoded Secret     | ENCODED_SECRET      | HIGH      |

### Command Scanner (40+ patterns)

| Category               | Examples                                    | Severity  |
|------------------------|---------------------------------------------|-----------|
| Filesystem Destruction | `rm -rf /`, `mkfs`, `dd of=/dev/sda`        | CRITICAL  |
| SQL Injection          | `DROP TABLE`, `UNION SELECT`, `OR 1=1`      | HIGH      |
| Fork Bomb              | `:(){ :|:& };:`, Python os.fork()           | CRITICAL  |
| Pipe to Shell          | `curl | sh`, `wget | bash`                  | CRITICAL  |
| SSH Exfiltration       | `scp /etc/passwd`, SSH tunneling             | CRITICAL  |
| Container Escape       | `nsenter --target 1`, docker.sock mount      | CRITICAL  |
| Crypto Mining          | `xmrig`, `minergate`, `cryptonight`          | HIGH      |
| Privilege Escalation   | `sudo NOPASSWD`, SUID bit, LD_PRELOAD       | HIGH      |
| Reverse Shell          | `nc -e /bin/sh`, `bash -i >& /dev/tcp`      | CRITICAL  |
| Network Tunneling      | `ngrok`, `socat`, DNS tunneling              | HIGH      |

### Content Scanner

| Category            | Detection                                       |
|---------------------|------------------------------------------------|
| Homograph URLs      | Cyrillic/Greek chars mimicking Latin in URLs    |
| ANSI Injection      | CSI, OSC, DCS escape sequences                  |
| Code Injection      | Dangerous exec/eval patterns in LLM output      |
| Markdown Injection  | Image exfil, link disguise, HTML in markdown     |
| HTML/SVG Injection  | Script tags, event handlers, SVG payloads        |

### Unicode Scanner

| Category         | Detection                                          |
|------------------|----------------------------------------------------|
| Bidi Override    | Right-to-left overrides (Trojan Source CVE)        |
| Zero-Width Chars | ZWSP, ZWNJ, ZWJ, BOM (invisible payloads)         |
| Homoglyphs       | Cyrillic/Greek/fullwidth lookalikes                |
| Mixed Script     | Latin+Cyrillic in same word (visual spoofing)      |
| Control Chars    | Unexpected control characters in text              |

---

## Proxy Features

HermesKatana includes a mitmproxy-based HTTPS proxy that intercepts all
outbound HTTP traffic from the agent:

| Feature             | Description                                       |
|---------------------|---------------------------------------------------|
| Secret Scrubbing    | Strips vault values from request bodies/headers   |
| Request Logging     | Logs all outbound requests to the audit trail     |
| Domain Allowlisting | Restrict requests to approved domains only        |
| Header Injection    | Adds security headers to outbound requests        |
| TLS Interception    | Full HTTPS visibility with auto-generated CA cert |

### Configuration

```python
from hermes_katana.proxy.config import ProxyConfig

config = ProxyConfig(
    listen_port=8080,
    allowed_domains=["api.openai.com", "api.anthropic.com"],
    block_unknown_domains=True,
    scrub_secrets=True,
)
```

---

## Vault Security

The vault provides AES-256-GCM encrypted storage for secrets with defense-
grade security properties:

| Property                | Implementation                                |
|-------------------------|-----------------------------------------------|
| Encryption              | AES-256-GCM (256-bit key, 96-bit nonce)       |
| Key Storage             | OS keyring (never on disk)                    |
| Nonce Management        | Per-value random nonces (no reuse)            |
| Integrity               | HMAC-SHA256 over all entries                  |
| Atomicity               | tmp file + rename (no partial writes)         |
| Circuit Breaker         | Sentinel file locks all operations            |
| Key Rotation            | Re-encrypt all values with new master key     |
| Thread Safety           | Reentrant lock on all operations              |

### Vault Operations

```python
from hermes_katana.vault.store import Vault

vault = Vault()
vault.set("OPENAI_API_KEY", "sk-abc123...")
value = vault.get("OPENAI_API_KEY")
keys = vault.list_keys()

# Emergency: lock vault if breach suspected
vault.lock()      # blocks all operations
vault.unlock()    # restore access
vault.rotate_key()  # rotate master key
```

---

## Audit Trail

Every security decision is logged to a SHA-256 hash-chained append-only
audit trail. Tampering with any entry invalidates all subsequent hashes.

### Properties

| Property          | Value                                           |
|-------------------|-------------------------------------------------|
| Format            | JSONL (one JSON entry per line)                 |
| Hash Algorithm    | SHA-256                                         |
| Chain Integrity   | Each entry hashes prev_hash + content           |
| Concurrency       | File locking for multi-process writes           |
| Rotation          | Automatic at 10MB, configurable                 |
| Query             | Filter by event type, tool, decision, time range|

### Event Types

| Event Type           | Description                                    |
|----------------------|------------------------------------------------|
| `tool_call`          | A tool was invoked by the agent                |
| `scan_result`        | Scanner produced a finding                     |
| `policy_decision`    | Policy engine made allow/deny/escalate         |
| `flow_analysis`      | Taint flow analysis result                     |
| `secret_blocked`     | A secret was blocked from transmission         |
| `injection_detected` | Prompt injection detected                      |
| `circuit_breaker`    | Vault circuit breaker activated/deactivated    |
| `config_change`      | Configuration modified                         |
| `session_start`      | Agent session began                            |
| `session_end`        | Agent session ended                            |

### Verification

```bash
katana audit verify   # verify entire hash chain
katana audit stats    # show entry counts and breakdown
katana audit show --event-type policy_decision --limit 20
```

---

## Middleware Chain

Tool calls flow through an ordered middleware chain with short-circuit
semantics:

```
Incoming Call --> [Taint MW] --> [Scan MW] --> [Policy MW] --> [Audit MW] --> Execute
                  pri=100        pri=80        pri=60         pri=20
```

| Middleware | Priority | Role                                            |
|------------|----------|-------------------------------------------------|
| Taint      | 100      | Check taint flows, build taint context          |
| Scanner    | 80       | Detect injections, secrets, dangerous content   |
| Policy     | 60       | Evaluate declarative policy rules               |
| Audit      | 20       | Log decision to audit trail (never blocks)      |

- **DENY** short-circuits the chain immediately
- **ESCALATE** is sticky but does not short-circuit
- **Post-dispatch** runs in reverse order after tool execution

```python
from hermes_katana.middleware.integration import create_default_chain

chain = create_default_chain({
    "policy.preset": "paranoid",
    "scan.block_threshold": 0.5,
})

ctx = chain.execute("terminal", {"command": "ls"})
print(ctx.decision)  # DispatchDecision.ALLOW
```

---

## Benchmarks

> Benchmarks are under active development. Initial measurements below
> are from a development machine (Apple M2, 16GB RAM, Python 3.12).

| Operation                    | Time      | Throughput |
|------------------------------|-----------|------------|
| Taint register + flow check  | <0.1 ms   | 10k+ ops/s |
| Injection scan (1KB input)   | <0.5 ms   | 2k+ ops/s  |
| Secret scan (1KB input)      | <0.3 ms   | 3k+ ops/s  |
| Command scan (single cmd)    | <0.1 ms   | 10k+ ops/s |
| Unicode scan (1KB input)     | <0.5 ms   | 2k+ ops/s  |
| Policy evaluation            | <0.1 ms   | 10k+ ops/s |
| Full middleware chain         | <2 ms     | 500+ ops/s |
| Audit trail append            | <1 ms     | 1k+ ops/s  |
| Vault get (AES-256-GCM)      | <0.5 ms   | 2k+ ops/s  |

All scanners use precompiled regex patterns loaded at import time.
Zero allocation overhead in the hot path for taint label checks.

---

## Comparison

How HermesKatana compares to existing LLM security tools:

| Feature                          | HermesKatana | Lakera Guard | NeMo Guardrails | LLM Guard | hermes-aegis |
|----------------------------------|:------------:|:------------:|:----------------:|:---------:|:------------:|
| Taint tracking (CaMeL)          | Y            | --           | --               | --        | --           |
| Character-level taint            | Y            | --           | --               | --        | --           |
| Information flow control         | Y            | --           | --               | --        | --           |
| Prompt injection detection       | Y            | Y            | Y                | Y         | Y            |
| Encoding attack detection        | Y            | --           | --               | Partial   | --           |
| Secret scanning (15+ patterns)   | Y            | --           | --               | Partial   | Y            |
| Multi-encoding secret detection  | Y            | --           | --               | --        | --           |
| Dangerous command detection      | Y            | --           | --               | --        | Y            |
| Unicode attack detection         | Y            | --           | --               | --        | --           |
| Content/ANSI injection           | Y            | --           | --               | --        | --           |
| Homograph URL detection          | Y            | --           | --               | --        | --           |
| Declarative policy engine        | Y            | --           | Y                | --        | --           |
| YAML policy hot-reload           | Y            | --           | Y                | --        | --           |
| HTTPS proxy (secret scrubbing)   | Y            | --           | --               | --        | Y            |
| AES-256-GCM vault                | Y            | --           | --               | --        | Fernet       |
| Hash-chained audit trail         | Y            | --           | --               | --        | Partial      |
| Middleware chain architecture    | Y            | --           | Y                | --        | --           |
| MCP server taint support         | Y            | --           | --               | --        | --           |
| Per-tool policy granularity      | Y            | --           | Partial          | --        | --           |
| Open source                      | Y            | --           | Y                | Y         | Y            |
| Self-hosted (no API calls)       | Y            | --           | Y                | Y         | Y            |

Legend: Y = full support, Partial = limited, -- = not available

---

## Credits and Provenance

HermesKatana builds on research and prior work:

### CaMeL Paper

The taint tracking system is inspired by Google DeepMind's CaMeL paper:

> **CaMeL: CApabilities for Machine Learning**
> arXiv:2503.18813 (2025)
>
> The paper demonstrates that detection-based defenses alone are
> insufficient against prompt injection. Instead, a data-flow tracking
> approach (taint labels + information-flow control) provides provable
> security guarantees. HermesKatana implements the core CaMeL concepts
> with practical extensions for character-level tracking, policy
> declaration, and middleware integration.

### hermes-aegis

The proxy-based secret scrubbing, vault design, and command scanner
patterns were originally developed in
[hermes-aegis](https://github.com/claudlos/hermes-aegis), the predecessor
project. HermesKatana upgrades these with:

- AES-256-GCM (from Fernet/AES-128-CBC)
- O(1) audit trail hash tracking (from O(n) file reads)
- 40+ command patterns (from ~15)
- Multi-encoding secret detection
- Cross-platform file locking

### camelup

The [camelup](https://github.com/jkminder/camelup) project provided
reference implementation insights for CaMeL taint tracking in Python.

---

## License

MIT License

Copyright (c) 2025 Carlos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
