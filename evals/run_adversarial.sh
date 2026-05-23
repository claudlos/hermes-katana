#!/usr/bin/env bash
# run_adversarial.sh — one-shot runner for evals/adversarial_dispatch.yaml.
#
# Executes every case in adversarial_dispatch.yaml through the
# middleware chain (via tests/integration/test_adversarial_eval_pack.py)
# and writes a tier-aware pass/fail summary to evals/latest-results.md.
#
# Spec note: PLAN_B worker-b4 originally called for running the suite
# through training/evaluate.py in "adversarial mode", but evaluate.py is
# a model-vs-model benchmark runner (reads the wild-attacks corpus) and
# has no such mode. Instead of cross-wiring into evaluate.py (which is
# worker B2's territory, per PLAN_B), this script drives the existing
# pytest harness that already consumes adversarial_dispatch.yaml.
#
# Usage:
#   bash evals/run_adversarial.sh              # writes evals/latest-results.md
#   bash evals/run_adversarial.sh some/out.md  # custom output path
#
# Exit code is always 0 on a successful *run*; the report itself captures
# pass/fail. Use the report (or pytest's own exit code, printed in it) to
# gate CI.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

OUT="${1:-$HERE/latest-results.md}"
JUNIT_XML="$(mktemp -t adversarial-junit-XXXXXX.xml)"
PYTEST_LOG="$(mktemp -t adversarial-pytest-XXXXXX.log)"
trap 'rm -f "$JUNIT_XML" "$PYTEST_LOG"' EXIT

PYTHON="${PYTHON:-python3}"

echo "[run_adversarial] Running adversarial suite via pytest ..."
set +e
"$PYTHON" -m pytest \
    tests/integration/test_adversarial_eval_pack.py \
    -v -p no:randomly \
    --junitxml="$JUNIT_XML" \
    --tb=line \
    2>&1 | tee "$PYTEST_LOG"
PYTEST_RC=${PIPESTATUS[0]}
set -e

echo "[run_adversarial] pytest exit=$PYTEST_RC. Generating $OUT ..."

"$PYTHON" - "$JUNIT_XML" "$OUT" "$PYTEST_RC" "$HERE/adversarial_dispatch.yaml" <<'PY'
"""Parse the JUnit XML produced by pytest and the adversarial YAML, then
write a tier-aware markdown summary to the output path."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

junit_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
pytest_rc = int(sys.argv[3])
yaml_path = Path(sys.argv[4])

# ---------------------------------------------------------------------------
# Load the case index so we can tier / annotate results even for cases that
# pytest didn't emit individual <testcase> entries for.
# ---------------------------------------------------------------------------


def _tier_of(case_id: str) -> str:
    if case_id.startswith("gap_"):
        return "gap"
    if case_id.startswith("provenance_"):
        return "provenance"
    if case_id.startswith("canary_"):
        return "canary"
    return "wild_attacks"


cases_by_id: dict[str, dict] = {}
if yaml is not None and yaml_path.exists():
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    for c in raw.get("cases", []):
        cases_by_id[c["id"]] = c


# ---------------------------------------------------------------------------
# Parse JUnit XML
# ---------------------------------------------------------------------------

results: dict[str, dict] = {}
if junit_path.exists() and junit_path.stat().st_size > 0:
    tree = ET.parse(junit_path)
    root = tree.getroot()
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        # pytest parametrize format: test_eval_case[<case_id>]
        if "[" in name and name.endswith("]"):
            case_id = name[name.index("[") + 1 : -1]
        else:
            # Skip non-parametrized entries (summary tests etc.)
            continue
        status = "pass"
        message = ""
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if failure is not None:
            status = "fail"
            message = (failure.get("message") or "").splitlines()[0] if failure.get("message") else ""
        elif error is not None:
            status = "error"
            message = (error.get("message") or "").splitlines()[0] if error.get("message") else ""
        elif skipped is not None:
            msg = skipped.get("message") or ""
            status = "xfail" if "xfail" in msg.lower() or "gap" in msg.lower() else "skip"
            message = msg.splitlines()[0] if msg else ""
        # test_eval_case is the authoritative run; prefer it over the legacy
        # TestAdversarialEvalPack.test_cases parametrization which reports the
        # same case a second time.
        is_legacy = "TestAdversarialEvalPack" in (tc.get("classname") or "")
        prior = results.get(case_id)
        if prior is None or (prior.get("is_legacy") and not is_legacy):
            results[case_id] = {
                "status": status,
                "message": message,
                "is_legacy": is_legacy,
            }


# ---------------------------------------------------------------------------
# Bucket by tier
# ---------------------------------------------------------------------------

tiers = ["wild_attacks", "gap", "provenance", "canary"]
tier_buckets: dict[str, dict[str, list[str]]] = {
    t: {"pass": [], "fail": [], "error": [], "xfail": [], "skip": [], "missing": []}
    for t in tiers
}

for case_id, case in cases_by_id.items():
    tier = _tier_of(case_id)
    r = results.get(case_id)
    if r is None:
        tier_buckets[tier]["missing"].append(case_id)
    else:
        tier_buckets[tier][r["status"]].append(case_id)

# Cases pytest reported but YAML doesn't know about (shouldn't happen, but
# keep them visible).
orphans = [cid for cid in results if cid not in cases_by_id]

# ---------------------------------------------------------------------------
# Write markdown report
# ---------------------------------------------------------------------------

now = datetime.now(timezone.utc).isoformat(timespec="seconds")
total = len(cases_by_id)
total_passed = sum(len(tier_buckets[t]["pass"]) for t in tiers)
total_failed = sum(len(tier_buckets[t]["fail"]) + len(tier_buckets[t]["error"]) for t in tiers)
total_xfail = sum(len(tier_buckets[t]["xfail"]) for t in tiers)
total_skip = sum(len(tier_buckets[t]["skip"]) for t in tiers)
total_missing = sum(len(tier_buckets[t]["missing"]) for t in tiers)

lines: list[str] = [
    "# Adversarial Eval — Latest Results",
    "",
    f"- Generated: `{now}`",
    f"- Source: `evals/adversarial_dispatch.yaml` ({total} cases)",
    f"- Harness: `tests/integration/test_adversarial_eval_pack.py`",
    f"- pytest exit code: `{pytest_rc}`",
    "",
    "## Totals",
    "",
    "| Tier          | Total | Pass | Fail | xfail | Skip | Missing |",
    "|---------------|------:|-----:|-----:|------:|-----:|--------:|",
]

for t in tiers:
    b = tier_buckets[t]
    total_t = sum(len(b[k]) for k in ("pass", "fail", "error", "xfail", "skip", "missing"))
    lines.append(
        f"| {t:<13} | {total_t:>5} | {len(b['pass']):>4} | "
        f"{len(b['fail']) + len(b['error']):>4} | {len(b['xfail']):>5} | "
        f"{len(b['skip']):>4} | {len(b['missing']):>7} |"
    )

lines += [
    f"| **OVERALL**   | **{total:>5}** | **{total_passed:>4}** | **{total_failed:>4}** | "
    f"**{total_xfail:>5}** | **{total_skip:>4}** | **{total_missing:>7}** |",
    "",
]

# Gap tier — expected to fail until PLAN_A fixes scanners.
lines += [
    "## `gap_*` tier — expected to fail until PLAN_A closes the scanner gaps",
    "",
]
gap_b = tier_buckets["gap"]
if gap_b["fail"] or gap_b["error"]:
    lines.append("Currently failing (this is the expected baseline):")
    lines.append("")
    for cid in sorted(gap_b["fail"] + gap_b["error"]):
        case = cases_by_id.get(cid, {})
        gap_ref = case.get("gap_ref", "")
        lines.append(f"- `{cid}` — {gap_ref}")
    lines.append("")
if gap_b["pass"]:
    lines.append(":tada: Previously-failing gaps that now PASS (promote out of `gap_*`):")
    lines.append("")
    for cid in sorted(gap_b["pass"]):
        lines.append(f"- `{cid}`")
    lines.append("")

# Provenance tier — per-pair breakdown.
lines += [
    "## `provenance_*` tier — origin-routing fence (bet #2)",
    "",
    "Each pair has an identical payload with two different origins. The",
    "`_user` member must ALLOW; the `_mcp` member must DENY. If both members",
    "of a pair agree on the decision, origin routing is not active.",
    "",
]

pair_members: dict[str, list[str]] = defaultdict(list)
for cid, case in cases_by_id.items():
    if _tier_of(cid) != "provenance":
        continue
    pair = case.get("pair", cid)
    pair_members[pair].append(cid)

lines.append("| Pair | `_user` | `_mcp` | Verdict |")
lines.append("|------|:-------:|:------:|---------|")
for pair in sorted(pair_members):
    members = pair_members[pair]
    user_id = next((m for m in members if m.endswith("_user")), None)
    mcp_id = next((m for m in members if m.endswith("_mcp")), None)

    def _symbol(cid: str | None) -> str:
        if cid is None or cid not in results:
            return "?"
        st = results[cid]["status"]
        return {"pass": "PASS", "fail": "FAIL", "error": "ERR", "xfail": "xfail", "skip": "skip"}.get(st, st)

    user_sym = _symbol(user_id)
    mcp_sym = _symbol(mcp_id)
    if user_sym == "PASS" and mcp_sym == "PASS":
        verdict = "routing active"
    elif user_sym == "PASS" and mcp_sym == "FAIL":
        verdict = "mcp under-blocked"
    elif user_sym == "FAIL" and mcp_sym == "PASS":
        verdict = "user over-blocked"
    else:
        verdict = "needs review"
    lines.append(f"| `{pair}` | {user_sym} | {mcp_sym} | {verdict} |")

lines.append("")

# Canary tier
canary_b = tier_buckets["canary"]
lines += [
    "## `canary_*` tier — runtime-compromise signals (bet #3)",
    "",
    f"- pass: {len(canary_b['pass'])}",
    f"- fail: {len(canary_b['fail']) + len(canary_b['error'])}",
    "",
]
if canary_b["fail"] or canary_b["error"]:
    lines.append("Failing canary cases (no canary_echo scanner yet — see PLAN_A):")
    lines.append("")
    for cid in sorted(canary_b["fail"] + canary_b["error"]):
        lines.append(f"- `{cid}`")
    lines.append("")

# Wild attacks — just counts + top failures.
wa_b = tier_buckets["wild_attacks"]
lines += [
    "## `wild_attacks_*` tier — historical regression floor",
    "",
    f"- pass: {len(wa_b['pass'])}",
    f"- fail: {len(wa_b['fail']) + len(wa_b['error'])}",
    f"- xfail (known gaps): {len(wa_b['xfail'])}",
    "",
]
if wa_b["fail"] or wa_b["error"]:
    lines.append("**REGRESSIONS in `wild_attacks_*` tier — fix before merge:**")
    lines.append("")
    for cid in sorted(wa_b["fail"] + wa_b["error"]):
        msg = results.get(cid, {}).get("message", "")
        suffix = f" — {msg}" if msg else ""
        lines.append(f"- `{cid}`{suffix}")
    lines.append("")

if total_missing:
    lines += [
        "## Missing from pytest output",
        "",
        "Cases in the YAML but not reported by pytest — investigate the",
        "harness filter:",
        "",
    ]
    for t in tiers:
        for cid in sorted(tier_buckets[t]["missing"]):
            lines.append(f"- `{cid}` (tier={t})")
    lines.append("")

if orphans:
    lines += [
        "## Orphan pytest cases (not in YAML)",
        "",
    ]
    for cid in sorted(orphans):
        lines.append(f"- `{cid}`")
    lines.append("")

lines += [
    "---",
    "",
    f"_Generated by `evals/run_adversarial.sh`. pytest log available in harness stdout._",
    "",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"[run_adversarial] Wrote {out_path}")
PY

echo "[run_adversarial] Done. Summary at: $OUT"
