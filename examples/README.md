# HermesKatana Examples

Runnable examples demonstrating the core security features.

## Prerequisites

```bash
cd /path/to/hermes-katana
pip install -e .          # or: pip install -e ".[dev]"
```

## Examples

| File | What It Shows |
|------|---------------|
| `basic_scanning.py` | Detect prompt injection, dangerous commands, and leaked secrets |
| `taint_tracking.py` | Track data provenance — taint labels, merging, flow decisions |
| `policy_engine.py` | Evaluate tool calls against max/balanced/permissive presets |
| `middleware_chain.py` | Full security pipeline — chain scanners + policy + audit |
| `custom_policy.yaml` | Example YAML policy file with comments explaining each field |
| `vault_usage.py` | Encrypted secret storage — store, retrieve, rotate keys |

## Running

Each Python example is self-contained:

```bash
python3 examples/basic_scanning.py
python3 examples/taint_tracking.py
python3 examples/policy_engine.py
python3 examples/middleware_chain.py
python3 examples/vault_usage.py
```

## What to Try

1. **Start with `basic_scanning.py`** — see how the scanner catches injection,
   dangerous commands, and leaked API keys.

2. **Explore `taint_tracking.py`** — understand how data provenance flows through
   string operations and how flow analysis blocks untrusted data from reaching
   critical tools like `terminal`.

3. **Compare presets in `policy_engine.py`** — the same tool call gets different
   verdicts depending on the security level.

4. **See the full pipeline in `middleware_chain.py`** — this is how HermesKatana
   works end-to-end: middleware chain → scanner → policy → allow/deny.

5. **Edit `custom_policy.yaml`** — write your own rules and load them with
   `PolicyEngine.from_file()` to see them take effect.

6. **Try `vault_usage.py`** — secrets are AES-256-GCM encrypted with OS keyring
   master key, HMAC integrity verification, and key rotation support.
