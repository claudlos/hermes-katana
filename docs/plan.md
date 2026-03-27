# HermesKatana v0.2.0 Plan

Date: 2026-03-26

## Track 1: Native Hermes Integration (COMPLETE)

Katana now integrates with Hermes as a first-class plugin using the
existing plugin system, eliminating the need for source-patching.

### Delivered

- `hermes_katana.hermes_plugin`: Native plugin with `pre_tool_call`,
  `post_tool_call`, `on_session_start`, `on_session_end` hooks.
  Registers via pip entry point `hermes_agent.plugins`.
- `hermes_katana.exceptions`: Exception hierarchy for security decisions
  (KatanaSecurityError, EscalationRequired, TaintFlowDenied, ScanBlocked,
  PolicyDenied).
- `hermes_katana.taint.registrar`: Convenience functions for tainting data
  at agent entry points (user input, tool output, web content, file content,
  MCP results, LLM responses, memory, delegated tasks).
- `katana_status` tool registered by the plugin for runtime introspection.
- 63 new tests covering all new modules.

## Track 2: Scanner Improvements (PLANNED)

- Ensemble classifier (TF-IDF + regex confidence boosting)
- Context-aware multi-turn analysis
- False-positive suppression and allowlisting

## Track 3: Runtime Hardening (PLANNED)

- Proxy connection pooling, graceful shutdown, body size limits
- Vault secret expiry, access auditing
- Runtime metrics collection and reporting
