#!/usr/bin/env python3
"""False-positive check: verify that benign eval cases are not blocked.

Exit code 0 if FP count == 0, else 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

# Add src to path if not installed
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hermes_katana.middleware import CallContext, DispatchDecision
from hermes_katana.middleware.integration import create_default_chain

_EVAL_PATH = Path(__file__).parent / "evals" / "adversarial_dispatch.yaml"


def main() -> int:
    raw = yaml.safe_load(_EVAL_PATH.read_text(encoding="utf-8"))
    cases = raw["cases"]

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

    total_benign = len(benign)
    fp_count = len(false_positives)

    if false_positives:
        print(f"FALSE POSITIVES ({fp_count}/{total_benign} benign cases blocked):")
        for fp in false_positives:
            print(fp)
    else:
        print(f"OK: 0 false positives across {total_benign} benign cases")

    return 1 if false_positives else 0


if __name__ == "__main__":
    sys.exit(main())
