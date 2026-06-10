#!/usr/bin/env python3
"""Pre- and post-dispatch property gate for the adversarial-dispatch eval set.

Two checks, both must pass (exit 0):

1. Pre-dispatch false positives: no ``expected_decision: allow`` case is denied.
2. Output scrubbing (audit finding E3): every case with
   ``expected_output_findings`` has its tool output handled as declared —
   secret-bearing outputs must be redacted, clean outputs must pass through.
   The gate previously ignored this property entirely even though validating
   output scrubbing is its reason to exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Add src to path if not installed
sys.path.insert(0, str(ROOT / "src"))

from hermes_katana.middleware import CallContext, DispatchDecision  # noqa: E402
from hermes_katana.middleware.integration import create_default_chain  # noqa: E402

_EVAL_PATH = ROOT / "evals" / "adversarial_dispatch.yaml"


def _check_false_positives(cases: list[dict]) -> list[str]:
    benign = [c for c in cases if c.get("expected_decision") == "allow"]
    chain = create_default_chain()
    false_positives: list[str] = []
    for case in benign:
        tool_name = case.get("tool_name", "unknown")
        args = case.get("args", {})
        ctx = CallContext(tool_name=tool_name, args=args)
        decision = chain.execute_pre(ctx)
        if decision == DispatchDecision.DENY:
            false_positives.append(f"  FP: {case['id']} ({tool_name}) blocked — {'; '.join(ctx.deny_reasons)}")
    print(
        f"OK: 0 false positives across {len(benign)} benign cases"
        if not false_positives
        else f"FALSE POSITIVES ({len(false_positives)}/{len(benign)} benign cases blocked):"
    )
    for fp in false_positives:
        print(fp)
    return false_positives


def _check_output_scrubbing(cases: list[dict]) -> list[str]:
    output_cases = [c for c in cases if "expected_output_findings" in c]
    failures: list[str] = []
    for case in output_cases:
        expected = bool(case["expected_output_findings"])
        vault_values = set(case.get("vault_values") or [])
        chain = create_default_chain({"scan.vault_values": vault_values})
        ctx = CallContext(tool_name=case.get("tool_name", "unknown"), args=case.get("args", {}))
        chain.execute_pre(ctx)
        ctx.tool_output = case.get("output", "")
        chain.execute_post(ctx)
        scrubbed = bool(ctx.extras.get("output_redacted") or ctx.extras.get("scabbard_output_redacted"))
        if scrubbed != expected:
            failures.append(
                f"  OUTPUT: {case['id']} expected_output_findings={expected} but scrubbed={scrubbed}"
            )
    print(
        f"OK: output scrubbing correct on {len(output_cases)} cases"
        if not failures
        else f"OUTPUT-SCRUB FAILURES ({len(failures)}/{len(output_cases)}):"
    )
    for f in failures:
        print(f)
    return failures


def main() -> int:
    raw = yaml.safe_load(_EVAL_PATH.read_text(encoding="utf-8"))
    cases = raw["cases"]
    failures = _check_false_positives(cases) + _check_output_scrubbing(cases)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
