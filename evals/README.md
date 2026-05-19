# `evals/` — Adversarial evaluation suite

This directory holds the adversarial evaluation corpus for the Hermes dispatch
path. The main artifact is [`adversarial_dispatch.yaml`](adversarial_dispatch.yaml),
a YAML file of case definitions consumed by
`tests/integration/test_adversarial_eval_pack.py` and by the convenience
runner [`run_adversarial.sh`](run_adversarial.sh).

Each case is evaluated end-to-end through the middleware chain
(`create_default_chain`) against its `expected_decision` (`allow` / `deny` /
`gap`). This README documents the case-ID naming convention, the four tiers,
and how to add new cases.

---

## Case-ID naming

Every case has a short, stable `id`. IDs are structured as
`<tier_prefix>_<short_topic>[_<variant>]`:

| Prefix            | Tier               | Meaning                                                                                                            |
|-------------------|--------------------|--------------------------------------------------------------------------------------------------------------------|
| `pi_*`            | wild_attacks       | Historical prompt-injection cases (direct, role-hijack, XML, etc.).                                                |
| `dc_*`            | wild_attacks       | Dangerous-command cases (`rm -rf /`, fork bomb, disk wipe, …).                                                     |
| `pts_*`           | wild_attacks       | Pipe-to-shell cases (`curl … \| sh`, `wget … \| bash`, …).                                                         |
| `rs_*`            | wild_attacks       | Reverse-shell cases (`nc -e`, `bash -i >& /dev/tcp/…`).                                                            |
| `exfil_*`         | wild_attacks       | Data-exfiltration cases.                                                                                           |
| `priv_*`          | wild_attacks       | Privilege-escalation cases.                                                                                        |
| `cont_*`          | wild_attacks       | Container-escape cases.                                                                                            |
| `crypto_*`        | wild_attacks       | Crypto-miner payload cases.                                                                                        |
| `taint_*`         | wild_attacks       | Taint-tracking cases (web/mcp tainted → sensitive sink).                                                           |
| `ctx_*` / `multi_*` / `tc_*` | wild_attacks | Context-dependent, multi-stage, and tool-chain evasion cases.                                                      |
| `gap_*`           | **gap**            | One-to-one regression fixtures for the failing cases in [`EVAL_GAPS.md`](EVAL_GAPS.md). **Expected to fail** until PLAN_A scanners close the gap. |
| `provenance_*`    | **provenance**     | Origin-routing fence cases. Each pair shares an identical payload, delivered once as `user_input` (allow) and once as `mcp_tool_result` (deny). |
| `canary_*`        | **canary**         | Runtime-compromise signals (bet #3). Canary values planted in tool descriptions / memory / honeytokens surfaced in a later tool call → `deny` with reason `canary_echo`. |

A case ID should be unique across the file and match `^[a-z][a-z0-9_]*$`.

### Tier prefixes in this README’s four-tier structure

The four-tier taxonomy for *new* work in this suite is:

1. **`wild_attacks_*`** — historical attack corpus (all legacy prefixes above).
   These cases must keep passing; they are the regression floor.
2. **`gap_*`** — known detection gaps. These are currently failing and are
   tracked in `EVAL_GAPS.md`. Fixing them is PLAN_A scanner work.
3. **`provenance_*`** — origin-aware enforcement. These prove the scanner
   stack routes on *origin*, not on content alone (plan.md bet #2).
4. **`canary_*`** — runtime-compromise signals. Planted canary values must
   never echo into a subsequent tool call (plan.md bet #3).

---

## Case schema (YAML)

```yaml
- id: <unique_case_id>
  description: >-
    Human-readable sentence describing the attack or the benign baseline.
  tool_name: <notes|terminal|…>             # which middleware-chain tool to hit
  args:                                     # optional: untainted args block
    <arg_name>: <value>
  tainted_args:                             # optional: tainted args block
    <arg_name>:
      value: <string>
      source: <web|mcp|tool|user>
      origin: <string describing upstream origin>
  vault_values: []                          # optional: strings the vault sink guards
  output: …                                 # optional: post-execution tool output
  expected_output_findings: false           # optional: should post-scan fire?
  expected_decision: <allow|deny|gap>       # REQUIRED
  expected_reason: <snake_case>             # optional, recommended for gap/canary
  tier: <wild_attacks|gap|provenance|canary>
  # Tier-specific metadata:
  gap_ref: "EVAL_GAPS.md#…"                 # gap_* only
  pair: <pair_name>                         # provenance_* only
  origin_hint: <user_input|mcp_tool_result> # provenance_* _user member only
  canary_source: <where the canary was planted> # canary_* only
```

`expected_decision: gap` marks a case as a known gap — it is `pytest.xfail`ed
by the harness. Use this sparingly; prefer `gap_*`-tier cases with
`expected_decision: deny` so regressions surface when the scanner improves.

The harness reads `args` and `tainted_args` via
`tests/integration/test_adversarial_eval_pack.py::_run_case`. Extra keys
(`expected_reason`, `tier`, `gap_ref`, `pair`, `canary_source`,
`origin_hint`) are currently informational — they do not affect pass/fail,
but they are preserved in reports.

---

## How to add a new case

1. Pick the right tier and prefix (`gap_*`, `provenance_*`, `canary_*`, or a
   `wild_attacks_*` family prefix).
2. Append the case under the matching section of
   `adversarial_dispatch.yaml`. Keep sections in numeric order; add a new
   section only if none of the existing ones fit.
3. Write a one-sentence `description`. Include the attack class so
   humans scanning the failures report can map failing IDs back to
   scanner features.
4. Set `expected_decision` deliberately. For `gap_*` cases, **always**
   `deny` (even if the scanner currently returns `allow`) — the whole
   point of elevating a gap to a named regression case is to *fail loudly*
   when the scanner drifts back.
5. For `provenance_*` cases, always add both halves of the pair in the same
   commit and link them with `pair: <name>`.
6. For `canary_*` cases, pick a distinctive prefix (`HKCANARY_*`) so the
   canary detector only needs one substring match.
7. Run `bash evals/run_adversarial.sh` locally and update
   `EVAL_GAPS.md` if the new case introduces a new documented gap.

---

## Running the suite

```bash
bash evals/run_adversarial.sh
```

This executes the suite through pytest against
`tests/integration/test_adversarial_eval_pack.py` and writes a human-readable
summary to [`latest-results.md`](latest-results.md). See
`run_adversarial.sh` for the exact invocation; override the output path
with the first positional argument if needed.

### Expected state on `HEAD` of `feature/saw-b4-adversarial-expansion`

All 8 `gap_*` cases are **expected to fail** on first run. They are owned by
PLAN_A scanner workers (see `/tmp/saw-build/PLAN_A.md`). Worker-b4 only
authors and documents the cases — it does not modify scanners.

The 10 `provenance_*` cases and 8 `canary_*` cases may also fail until the
origin-aware enforcement (bet #2) and canary-echo detection (bet #3) land.
Those failures should be tracked as new rows in `EVAL_GAPS.md` when they
are first surfaced.
