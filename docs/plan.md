# HermesKatana v0.2.0 — Completed

Date: 2026-03-26

## Track 1: Native Hermes Integration — COMPLETE

- `hermes_katana.hermes_plugin`: Native plugin with pre/post tool call
  hooks, session lifecycle, katana_status tool. Registers via pip entry
  point `hermes_agent.plugins`.
- `hermes_katana.exceptions`: Exception hierarchy (KatanaSecurityError,
  EscalationRequired, TaintFlowDenied, ScanBlocked, PolicyDenied).
- `hermes_katana.taint.registrar`: Tainting at entry points (user input,
  tool output, web, file, MCP, LLM, memory, delegated tasks).
- 63 tests.

## Track 2: Scanner Improvements — COMPLETE

- `scanner/ensemble.py`: TF-IDF + logistic regression injection classifier
  with hand-crafted feature fallback. 60 labeled examples.
- `scanner/context_analyzer.py`: Multi-turn conversation analyzer (topic
  drift, instruction density, persona shifts, cumulative risk with decay).
- `scanner/allowlist.py`: FP suppression with regex/glob patterns, expiry,
  hit counts. 6 built-in suppressions.
- `scanner/__init__.py`: scan_with_context() combining all layers.
- 80 tests.

## Track 3: Runtime Hardening — COMPLETE

- `metrics.py`: Thread-safe runtime metrics (tool calls, scan hits, taint
  flows, policy evals, latency). Prometheus export.
- `vault/access_log.py`: JSONL access audit for all vault operations.
- `vault/expiry.py`: TTL-based secret expiry management.
- `proxy/config.py`: Body size limits, graceful shutdown timeout.
- `proxy/runner.py`: Runtime counters, graceful shutdown.
- 43 tests.

## Totals

- 3 commits, 4,387 lines added
- 186 new tests (443 total passing)
- Version bumped to 0.2.0
