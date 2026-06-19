# Changelog

All notable changes to HermesKatana will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Cosine-similarity false-positive softener (`hermes_katana.scabbard.similarity_allowlist`). On a Scabbard or pattern-scanner BLOCK, the verdict is softened to ALLOW when the classified text is cosine-close to a vetted benign exemplar (`policies/scabbard_benign_exemplars.yaml`) and carries no concrete-exploit finding (secret / dangerous command / unicode-evasion / binary payload). It generalises past the hash allowlist to the security-domain content a research agent writes (documentation that *quotes* attack strings). Torch-free: runs on an ONNX all-MiniLM-L6-v2 encoder via `onnxruntime` (install with `scripts/setup_similarity_embedder.py`); fails closed (no softening) when the encoder is absent. The threshold sits above the adversarial-corpus attack ceiling, so it never softens an attack — enforced by the evasion gate and `tests/smoke/test_similarity_allowlist_safety.py`. Untrusted-origin (tainted) content is never softened.
- `scabbard.audit_blocked_text` config flag: records a truncated (≤200 char) preview of the exact classified text for softened and denied Scabbard blocks, so live false positives can be reviewed and allowlisted by call_id. Off by default (stores tool-argument plaintext).
- `scabbard_block_threshold` plugin config: lets operators tune the Scabbard ML classifier BLOCK threshold independently from `scan_block_threshold`, including named Scabbard profiles, deployment profiles, and runtime-default selection.

### Changed
- Capability-aware Scabbard backend selection. The v17/v14 production checkpoints are PyTorch models; in a torch-free deployment they failed to load and Scabbard ran DEGRADED — every call fail-closed to BLOCK and unsoftenable, so the security tool blocked its own benign tool calls. `ScabbardConfig.runtime_default()` and the `fast_cpu` profile now fall back to the v15 ONNX MiniLM (run via `onnxruntime`) when torch is unavailable, instead of degrading.
- Behavioral spike detector: default `spike_threshold` raised 5 → 10 (an active agent legitimately bursts several file/exec calls a minute; the spike is observe-only), and the finding is now de-duplicated so a saturated window logs once per new high instead of on every call.
- Scabbard `block_threshold` default raised 0.5 → 0.7 (fewer raw false positives on the deployed v15-ONNX model); the cosine softener and hash allowlist provide additional, surgical FP relief without lowering attack recall (evasion gate stays at 0 evasions).
- v17 origin-aware MiniLM-L6 (HF `Carlosian/hermes-katana-90`, PyTorch) is now wired in as a selectable Scabbard backend and is preferred by `runtime_default()` *when PyTorch is available*. It is NOT the forced default: v15-ONNX remains the default deployable backend (the production gateway is torch-free and pins `katana_v15_minilm` + `onnx`), and the capability-aware fallback selects v17-torch only when torch can actually load. Select v17 explicitly with `scabbard_profile: katana_v17_minilm`.

### Fixed
- `_handle_katana_status` accepts the Hermes tool-registry dispatch contract `handler(args, **kwargs)` (was `**kwargs`-only, which raised a live TypeError when katana_status was invoked through the registry).
- Regenerated `policies/scabbard_known_fps.yaml` against the deployed v15-ONNX backend (the prior hashes were generated against a different model/text and no longer matched), restoring the false-positive gate to zero blocks on the 154-case benign corpus.
- Short-text FP softening now recognizes benign `persona-shift` detector documentation without softening direct persona-shift attack instructions. The imperative-attack gate was hardened to catch inflected/determiner-light attack verbs (`ignores … rules`, `exfiltrates`, `discloses`, `leak the system prompt`, `reveal hidden config`) so third-person attack phrasing around the new `persona-shift` security-context term cannot ride the descriptive-note softener.

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
