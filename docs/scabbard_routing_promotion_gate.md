# Scabbard Routing and MiniLM Promotion Gate

This document defines the operational gate for enabling Scabbard as a Hermes tool-call guard with the v15 MiniLM ONNX artifact.

## Why routing exists

Scabbard is a prompt/content classifier. Hermes tool arguments are mixed JSON schemas: text, paths, URLs, integers, booleans, enum values, shell commands, and content payloads. Feeding every raw argument to the ML classifier causes false positives on normal structural values such as:

- `sandbox/input.txt`
- `https://example.com`
- `printf benchmark`

Balanced routing sends natural-language content to Scabbard and leaves structural data to the scanners/policy layers that are designed for it.

## Route modes

Plugin config:

```yaml
plugins:
  katana:
    scabbard_enabled: true
    scabbard_profile: katana_v15_minilm
    scabbard_backend: onnx
    scabbard_route_mode: balanced     # off | content_only | balanced | paranoid
    scabbard_scan_outputs: true
    scabbard_audit_routes: true
```

Modes:

- `off`: Scabbard route checks skip classification.
- `content_only`: scan explicit content fields such as `content`, `text`, `prompt`, `message`, `html`, `markdown`, `body`.
- `balanced`: scan content/query/prose fields; skip paths, URLs, controls, commands, enums, booleans, and numbers.
- `paranoid`: scan most string-like values, including structural strings, for diagnostic/security stress runs.

## Specialized detector split

Balanced Scabbard routing intentionally skips shell-command fields (`command`, `cmd`, `script`) for ML prompt classification. Those strings still pass through the command scanner, policy engine, taint layer, and audit middleware in the full stack. This avoids treating commands as prose while preserving command-specific security checks.

URL/path/control fields are similarly skipped by Scabbard but remain available to URL/path/policy/sandbox controls.

## Audit extras

When `scabbard_audit_routes` is enabled, middleware records:

- `scabbard_routes`: per top-level argument scan/skip decisions
- `scabbard_skipped_args`: skipped argument details with route kind/reason
- `scabbard_results_by_arg`: per scanned argument result
- `scabbard_route_counts`: scanned/skipped counts
- `scabbard_output_routes`: post-dispatch output fragments selected for scanning
- `scabbard_output_results_by_path`: per output fragment result

Values are not embedded in route audit entries; only field names, route kind, and reason are recorded.

## Promotion gate for MiniLM as default

Before making `katana_v15_minilm` the default Scabbard tool gate, all of these must pass:

1. Routing unit tests:
   ```bash
   .venv/bin/pytest tests/unit/test_scabbard_routing.py -q
   ```
2. Profile/plugin tests:
   ```bash
   .venv/bin/pytest tests/unit/test_katana_model_profiles.py tests/unit/test_hermes_plugin.py -q
   ```
3. Full unit suite:
   ```bash
   .venv/bin/pytest tests/unit/ -q
   ```
4. Verification triangle:
   ```bash
   python3 test_false_positives.py
   python3 test_evasion.py
   .venv/bin/pytest tests/integration/test_adversarial_eval_pack.py -q
   ```
5. Routing benchmark sanity:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/benchmark_hermes_katana_tool_sandbox.py \
     --variants base,minilm \
     --stack scabbard-only \
     --suites benign_tool_args,content_payloads,tool_outputs \
     --route-mode balanced \
     --warmups 1 \
     --iterations 2
   ```

Benchmark acceptance:

- `benign_tool_args` structural FP blocks = 0
- content scanned/expected = expected count
- tool output content scan rows > 0 for `tool_outputs`
- MiniLM latency remains in the low-ms/tool-call range on the local CPU path

## Benchmark scripts

Middleware/sandbox benchmark:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_hermes_katana_tool_sandbox.py \
  --variants base,minilm,v15-deberta \
  --stack scabbard-only \
  --route-mode balanced \
  --isolate-variants
```

Real Hermes CLI e2e scaffold:

```bash
PYTHONPATH=src .venv/bin/python scripts/benchmark_hermes_katana_cli_e2e.py
```

The CLI e2e script is dry-run by default. Pass `--execute` only when you want real provider calls and are prepared for provider latency/noise.
