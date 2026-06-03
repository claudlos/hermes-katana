# Changelog

All notable changes to HermesKatana will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.1.0] - 2026-06-03

### Added
- v3.1 preprint under `paper/` (*Cross-Platform Transferability of Prompt Injection Attacks: Universal Attack Surfaces and an Origin-Aware Defense*), self-contained (LaTeX source, figures, and bibliography), buildable with `pdflatex`/`bibtex` via the included Makefile.
- Released the v17 origin-aware prompt-injection classifiers on Hugging Face under MIT: `Carlosian/hermes-katana-17` (DeBERTa-v3-large teacher) and `Carlosian/hermes-katana-90` (distilled MiniLM-L6 CPU scanner).
- Registered those models in the artifact CLI: `katana artifacts download v17_large` / `v17_minilm` (commit-pinned, integrity-verified via `artifact_manifest.json`; kept out of the managed `setup --all` as research models).
- GitHub Pages static manual at `docs/index.html`.
- Generated policy documentation check via `scripts/generate_policy_assets.py`.

### Changed
- Built-in policy YAML files are now the source of truth for runtime defaults and README preset documentation.
- The strict built-in policy preset is now named `max`; users with older configs should reinstall or upgrade and run `katana policy use max`.
- Proving Ground helper entry points now use packaged module paths instead of repository-root compatibility shims.

### Fixed
- Proving Ground analysis scripts (`factorial_decompose.py`, `harness_matrix.py`) no longer double-count `shard_*.fp.jsonl` false-positive sidecar files when globbing `shard_*.jsonl`.

### Removed
- Legacy root compatibility shims and duplicated Proving Ground research trees from the public repository root.
- Stale machine-specific Proving Ground runbooks that referenced private fleet specs.

## [3.0.0] - 2026-05-19

### Added
- V3 production middleware profiles: `fast_cpu`, `balanced`, and `max`.
- Fast CPU Scabbard profile using the distilled v15 MiniLM ONNX runtime with route-aware scanning defaults.
- Readiness and latency diagnostics in Katana plugin status output.
- Scanner-change release gate covering ruff, false-positive smoke, evasion, and adversarial integration checks.
- `katana artifacts` registry and guided setup for the default MiniLM ONNX artifact and optional large local model.
- Three regression tests in `tests/unit/test_scabbard_pipeline.py` pinning the threshold defaults so accidental reverts fail loudly.

### Changed
- **`ScabbardConfig.block_threshold` default lowered from 0.7 to 0.5** (also reflected in `production()` and `katana_v14()` factories). Selected via principled sweep over `confirmed_only_v1` + `hard_negatives.jsonl` + `splits/test.jsonl`; new threshold catches +12 attacks per 1000 on confirmed_only_v1 vs 0.7, with hard-negatives FPR unchanged at 0.10%. The threshold is argmax-equivalent (matches the eval script's reporting) and recovers the one live-test miss observed at confidence 0.5031 in the 2026-05-08 codex+minimax bare/katana run. `katana_v11()` factory keeps 0.7 for v1.0 reproducibility.
- `live_test_v14_attacks.py` now takes `--block-threshold` and `--allow-threshold` arguments (default 0.5/0.3) and records them in `metrics.json` for replay.
- `ScabbardConfig.katana_v15_minilm()` now resolves ONNX artifacts through `KATANA_MINILM_ONNX_DIR` or the artifact cache instead of `training/checkpoints`.
- Release metadata now reports `3.0.0` across package, CLI, installer marker, plugin metadata, README, and operations docs.

### Fixed
- Codec-taint propagation now survives base64, hex, and JSON round trips.
- Batch 1 scanner gates now include decoder findings and fail closed on semantic recall backend errors.
- Removed the broken top-level `validators` shim that imported a nonexistent module.

### Removed
- Research PDFs and private release debris from the public release branch.

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
