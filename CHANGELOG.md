# Changelog

All notable changes to HermesKatana will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-04-04

### Added
- CaMeL-inspired taint tracking with character-level provenance
- Information flow control with configurable source/sink policies
- Injection scanner with 30+ heuristic patterns and encoding detection
- Secret scanner with 15+ patterns (API keys, tokens, private keys)
- Command scanner with 40+ dangerous command patterns
- Content scanner (ANSI injection, homograph URLs, markdown attacks)
- Unicode scanner (bidi overrides, zero-width chars, homoglyphs)
- Declarative policy engine with YAML configuration and three presets
- Middleware chain architecture with priority-ordered processing
- AES-256-GCM encrypted vault with OS keyring integration
- SHA-256 hash-chained audit trail with tamper detection
- mitmproxy-based HTTPS proxy with secret scrubbing
- `katana` CLI with doctor, scan, policy, vault, audit, and proxy commands
- Hermes agent installer with dry-run and backup support
- 1214 tests with property-based testing via Hypothesis
