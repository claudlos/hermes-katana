# Hermes Katana v3 Audit Status

Last reconciled: 2026-05-23

Source audit: an external v3 audit report (the original HTML was authored
off-tree against this repository's `c3f3db2` snapshot — findings are mirrored
into the tables below).

This file is a working tracker, not a substitute for tests. Each status is based
on the current tree plus spot checks run during reconciliation.

## Verification Snapshot

- `dd84f65 fix(security): close v3 audit gaps` committed the first passing fix tranche.
- Full test suite after that tranche: `4176 passed, 88 skipped, 22 xfailed, 2 xpassed`.
- CLI smoke in isolated temp HOME:
  - `katana scan "rm -rf /"` returned the security exit code and BLOCK verdict.
  - `katana audit verify` passed before and after `katana audit clear --yes`.
  - `policies/max.yaml` loaded through `load_policy_file`.
- Packaging smoke:
  - Built `dist/hermes_katana-3.0.0.tar.gz` and wheel in a temporary build venv.
  - `twine check dist/*` passed.
  - Installed the wheel in a clean venv; installed `katana scan "rm -rf /"` blocked.
- Current local verification after policy/source cleanup and repo publication cleanup:
  - `ruff check` and `ruff format --check` passed for `src/`, `tests/`, and retained root helper scripts.
  - `mypy` CI smoke command passed.
  - `python scripts/generate_policy_assets.py --check` passed.
  - `scripts/verify_scanner_change.sh --skip-lint` passed: smoke gates plus `365 passed, 15 xfailed`.
  - Full suite: `4186 passed, 88 skipped, 22 xfailed, 2 xpassed`.
  - Static manual link check and Playwright desktop/mobile screenshot smoke passed.
  - Built sdist/wheel in a clean build venv; `twine check` passed; wheel install/version/policy/CLI smoke passed.
- Current local verification after release-gate rerun and proxy hardening tranche:
  - Created repo-local `.venv` and installed `.[dev,security,fast-cpu,proving-ground]` for a clean local release-gate path.
  - `scripts/release_gate.sh --allow-missing-gitleaks` passed end-to-end.
  - Full suite: `4257 passed, 75 skipped`.
  - `scripts/verify_scanner_change.sh --skip-lint` passed: smoke gates plus `380 passed`.
  - `python -m build` and `python -m twine check` passed from the release-gate dist output directory.
  - `gitleaks detect --source . --redact --no-banner --config .gitleaks.toml` passed with `no leaks found`.

## P0 Findings

| ID | Finding | Status | Notes |
|---:|---|---|---|
| 1 | `TaintedStr` laundering through string methods | Fixed in `dd84f65` | Added propagation overrides and laundering regression tests. |
| 2 | `TaintedBytes` laundering | Fixed in `dd84f65` | Added byte transform propagation and tests. |
| 3 | Reader-set merge used union | Fixed in `dd84f65` | Reader merge now uses intersection semantics. |
| 4 | `replace` / `format` / `join` / `%` / `__format__` erased char taint | Fixed in `dd84f65` | Added targeted tests for char-taint preservation. |
| 5 | Mutable `sources=` accepted despite `frozenset` annotation | Fixed in `dd84f65` | Constructors now defensively freeze metadata. |
| 6 | `TaintedValue` bypassed policy cache canonicalizer | Fixed in `dd84f65` | Added explicit `TaintedValue` fingerprinting and collision test. |
| 7 | Same-value/different-source tainted values hash-collided | Fixed in `dd84f65` | Fingerprints include source/reader metadata. |
| 8 | `taint_level_lte` rejected by YAML validator | Fixed in `dd84f65` | Validator accepts the operator. |
| 9 | README preset table mismatches current behavior | Fixed locally | README table is covered by a runtime drift test. |
| 10 | Parallel balanced/max policy sets disagree | Fixed locally | Built-in defaults now load from top-level `policies/*.yaml`; wheel packaging includes those files. |
| 11 | Direct `PolicyEngine()` defaulted to ALLOW | Fixed in `dd84f65` | Direct construction now defaults to DENY. |
| 12 | Audit chain broke at rotation | Fixed in `dd84f65` | Chain verification covers rotated files and active file. |
| 13 | `audit clear` destroyed history without sentinel | Fixed in `dd84f65` | Clear now appends `trail_cleared` and preserves history. |
| 14 | JSONL append is not crash-atomic | Open | Writes fsync, but a partial trailing line can still poison verification. |
| 15 | Proxy keeps vault secrets in plaintext set | Fixed locally | Proxy now collects vault values per scan instead of keeping a long-lived `KatanaAddon._vault_values` plaintext cache. |
| 16 | Proxy scrubbing bypassed by compression/multipart/binary | Fixed locally | Request/response scanning now decodes gzip/deflate bodies and parses `multipart/*` payloads part-by-part before scanning; binary bodies still route through `scan_bytes`. |
| 17 | `HERMES_KATANA_VAULT_KEY` is popped | Needs decision | Current tests assert consumption; decide whether this is intended security behavior or a usability bug. |
| 18 | `katana run` launches without Katana protection | Fixed before this tranche | Current CLI composes runtime env and tests assert `KATANA_ACTIVE`/policy state. |
| 19 | CA private-key passphrase derived from on-disk salt | Open | Needs key handling redesign or documentation downgrade. |
| 20 | `pip install hermes-katana` does not resolve | Partially fixed | README/quickstart now use source install; PyPI publish remains open. |
| 21 | README quickstart output was fictional | Fixed | README/quickstart now describe actual Rich output instead of a fake line report. |
| 22 | README links to missing `docs/research/` | Fixed | Missing research links removed from README. |
| 23 | Operator's home dir hard-coded in shipped scripts | Verified obsolete | No remaining tracked matches; the depersonalize pass replaced the last fixture path with `/home/user/`. |
| 24 | `zvec` dependency in `[ml]` resolves wrong package | Fixed | Removed the PyPI `zvec` dependency from the `ml` extra. |
| 25 | Zero false-positive guarantee is false | Partially fixed | README guarantee removed; scanner FP tuning remains product work. |
| 26 | `<0.5 ms / 1KB` performance claim is false | Fixed | README now says to benchmark locally instead of publishing fixed latency claims. |
| 27 | `rm -rf /` long-form variants bypass | Fixed in `dd84f65` | Added long-option and root-boundary coverage. |
| 28 | Bash `$()` command substitution not expanded | Fixed in `dd84f65` | Added `$()` scanning and tests. |
| 29 | Decoder does not re-scan decoded plaintext for commands/secrets | Fixed locally | Decoder now re-scans decoded payloads for injection, dangerous commands, and secrets. |
| 30 | IFS / shell variable evasion unhandled | Fixed in `dd84f65` | Added simple variable, IFS, and ANSI-C quote normalization. |

## P1 Findings

| ID | Finding | Status | Notes |
|---:|---|---|---|
| 31 | HMAC-over-vault allows rollback | Open | Needs monotonic version/counter or external anchor. |
| 32 | HMAC key derived with string-prefix SHA-256 | Open | Replace with HKDF or equivalent KDF separation. |
| 33 | AES-GCM uses no AAD | Open | Add stable metadata as AAD. |
| 34 | Audit `compute_hash` uses `json.dumps(..., default=str)` | Open | Still present in `audit/trail.py`. |
| 35 | Headless Linux degrades to in-memory master key | Open | Needs explicit fail/opt-in behavior. |
| 36 | `vault.set()` lost-update race | Open | Needs file lock around read-modify-write. |
| 37 | `tls_verify=False` still injects vault credentials | Fixed locally | Proxy startup/request handling now refuses credential injection when `tls_verify` is false. |
| 38 | mitmproxy subprocess inherits full env | Fixed locally | Proxy runner now builds an allowlisted child environment instead of inheriting `os.environ.copy()`. |
| 39 | ProtectAI middleware fails open on scan exception/stub | Open | Current middleware returns ALLOW on exception and unavailable model. |
| 40 | No regex timeout in `_safe_compile` | Open | No central regex timeout guard found. |
| 41 | ESCALATE has no built-in approval handler | Needs decision | Current plugin raises escalation; interactive approval flow is still a product decision. |
| 42 | `KatanaTaintMiddleware._find_tainted` ignores dict keys | Fixed locally | Recursive taint search now checks mapping keys and values. |
| 43 | Sentinel duplicates Scabbard with different classify call | Open | Needs consolidation or documented rationale. |
| 44 | Metrics `_call_starts` leaks on exception | Verified fixed | Current `_record()` pops starts on post-dispatch and short-circuit paths. |
| 45 | `*.katana-backup` files left forever | Open | Installer cleanup/retention policy needed. |
| 46 | Patch revert breaks on user edit; `--backup` not default | Open | Needs installer UX and three-way/manifest strategy. |
| 47 | Non-atomic patch writes can corrupt checkout | Open | Needs atomic write path for installer patches. |
| 48 | `Path.home()` in `artifacts.py` bypasses safe-home helper | Fixed locally | `default_artifact_cache_dir` now uses the shared safe-home fallback. |
| 49 | Version string duplicated | Fixed locally | Runtime code now uses `hermes_katana._version`; tests compare it to `pyproject.toml`. |
| 50 | `katana doctor` can report all OK while ML is drifted | Needs verification | Preflight has stricter checks; doctor behavior still needs targeted smoke. |
| 51 | No `--json` for scan commands | Open | `preflight` has JSON, scan/scan-file/scan-command do not. |
| 52 | Mistral tokenizer warning on CLI invocation | Needs verification | Requires optional model/runtime repro. |
| 53 | `semantic_recall` and `deberta` print to stdout | Open | Direct `print()` calls remain in scanner modules. |
| 54 | `katana scan ""` and `katana scan "rm -rf /"` behavior | Partially fixed | `rm -rf /` blocks; empty input behavior still needs product decision. |
| 55 | `proving-ground` and `preflight` exposed but undocumented | Fixed | README CLI reference now lists both. |
| 56 | `dist/` tracked at repo root | Verified fixed | `dist/` is ignored and not tracked. |
| 57 | Internal one-shot scripts ship publicly | Fixed locally | Root `scripts/` now keeps only public maintenance/benchmark helpers; proving-ground research helpers live under the package namespace. |
| 58 | Three different copyright names | Open | README and LICENSE still disagree. |
| 59 | Top-level repo clutter duplicates `src/` | Fixed locally | Removed root compatibility shims plus duplicate `sandbox/` and `synthdata/` trees. |
| 60 | CI `mypy` step uses `|| true` | Verified fixed | CI and release-gate now use `scripts/mypy_smoke.sh`; the smoke is fail-closed and covers the public CLI/proxy release path in addition to policy/version modules. |
| 61 | CI workflow triggers only on main/master | Fixed locally | CI and release-gate now cover `release/**` branches. |
| 62 | `SECURITY.md` lacks 3.0.x row | Verified fixed | `SECURITY.md` lists 3.0.x as supported. |
| 63 | CHANGELOG 3.0.0 references missing scripts | Fixed locally | Removed references to unavailable one-off scripts and added current unreleased cleanup notes. |

## Next Batch Queue

1. Vault cryptography/concurrency: rollback protection, HKDF separation, AAD, and locked read-modify-write.
2. Proxy/platform security follow-up: CA private-key handling, remaining key-management decisions, and any additional content-encoding coverage beyond gzip/deflate if needed.
3. CLI/operator polish: JSON scan output, stdout/stderr cleanup, copyright cleanup, and broader full-tree mypy cleanup beyond the enforced smoke gate.
