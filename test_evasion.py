#!/usr/bin/env python3
"""Evasion check: verify that adversarial eval cases are correctly blocked.

Exit code 0 if evasion count == 0, else 1.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

# Add src to path if not installed
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hermes_katana.audit import AuditTrail
from hermes_katana.middleware import CallContext, DispatchDecision
from hermes_katana.middleware.integration import create_default_chain
from hermes_katana.taint import Source, TaintTracker

_EVAL_PATH = Path(__file__).parent / "evals" / "adversarial_dispatch.yaml"
_GAP_DECISION = "gap"


def _make_source(source_name: str, origin: str | None):
    if source_name == "web":
        return Source.web(origin or "https://example.invalid")
    if source_name == "mcp":
        return Source.mcp(origin or "untrusted-mcp")
    if source_name == "tool":
        return Source.tool(origin or "tool-output")
    if source_name == "user":
        return Source.user(origin or "eval-user")
    raise ValueError(
        f"Unsupported taint source in eval case: {source_name!r} "
        "(expected one of: web, mcp, tool, user)"
    )


def _run_case(case: dict, tmp_path: Path) -> DispatchDecision:
    tracker = TaintTracker()
    audit_trail = AuditTrail(path=tmp_path / f"{case['id']}.jsonl")
    args = dict(case.get("args", {}))

    for arg_name, spec in case.get("tainted_args", {}).items():
        source = _make_source(spec.get("source", "user"), spec.get("origin"))
        args[arg_name] = tracker.register(spec["value"], source)

    chain = create_default_chain(
        {
            "taint.tracker": tracker,
            "scan.vault_values": set(case.get("vault_values", [])),
            "audit.trail": audit_trail,
        }
    )
    ctx = CallContext(tool_name=case["tool_name"], args=args)
    return chain.execute_pre(ctx)


def main() -> int:
    raw = yaml.safe_load(_EVAL_PATH.read_text(encoding="utf-8"))
    cases = raw["cases"]

    adversarial = [
        c for c in cases
        if c.get("expected_decision") == "deny"
    ]

    evasions: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for case in adversarial:
            decision = _run_case(case, tmp_path)
            if decision != DispatchDecision.DENY:
                evasions.append(
                    f"  EVASION: {case['id']} ({case.get('tool_name', '?')}) "
                    f"not blocked — decision={decision}"
                )

    total_adversarial = len(adversarial)
    evasion_count = len(evasions)
    blocked_count = total_adversarial - evasion_count

    if evasions:
        print(f"EVASIONS ({evasion_count}/{total_adversarial} adversarial cases not blocked):")
        for ev in evasions:
            print(ev)
        print(f"Blocked {blocked_count}/{total_adversarial} adversarial cases")
    else:
        print(f"OK: all {total_adversarial} adversarial cases correctly blocked (0 evasions)")

    return 1 if evasions else 0


if __name__ == "__main__":
    sys.exit(main())
