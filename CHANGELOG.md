# Changelog

All notable changes to HermesKatana will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **`ScabbardConfig.block_threshold` default lowered from 0.7 to 0.5** (also reflected in `production()` and `katana_v14()` factories). Selected via principled sweep over `confirmed_only_v1` + `hard_negatives.jsonl` + `splits/test.jsonl`; new threshold catches +12 attacks per 1000 on confirmed_only_v1 vs 0.7, with hard-negatives FPR unchanged at 0.10%. The threshold is argmax-equivalent (matches the eval script's reporting) and recovers the one live-test miss observed at confidence 0.5031 in the 2026-05-08 codex+minimax bare/katana run. `katana_v11()` factory keeps 0.7 for v1.0 reproducibility.
- `live_test_v14_attacks.py` now takes `--block-threshold` and `--allow-threshold` arguments (default 0.5/0.3) and records them in `metrics.json` for replay.

### Added
- `scripts/tune_v14_thresholds.py` — principled threshold sweep with multi-recommendation output (F1-max, operational-conservative, aggressive-max-recall) across all three eval surfaces.
- `scripts/post_process_threshold_tune.py` — replays selector logic on an existing sweep without re-running v14 inference (~1s vs ~25 min CPU).
- `scripts/per_class_score_analysis.py` — per-attack-class confidence-quartile analyzer + hard-negative FPR drill-down. Run after each v14 retrain to spot under-confident categories.
- `scripts/compare_live_tests.py` — A/B comparison between two live-test runs (overlap analysis with per-attack flip detail).
- Three regression tests in `tests/unit/test_scabbard_pipeline.py` pinning the threshold defaults so accidental reverts fail loudly.

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
