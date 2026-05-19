"""Adversarial end-to-end eval pack for the Hermes dispatch path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from hermes_katana.audit import AuditTrail
from hermes_katana.middleware import CallContext, DispatchDecision
from hermes_katana.middleware.integration import create_default_chain
from hermes_katana.taint import Source, TaintTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_YAML_PATH = Path(__file__).resolve().parents[2] / "evals" / "adversarial_dispatch.yaml"

# Known-gap decisions: cases where expected_decision is "gap" or the YAML
# comment says KNOWN GAP with an `allow` expectation we know is a gap.
_GAP_DECISION = "gap"


def _load_eval_cases() -> list[dict]:
    raw = yaml.safe_load(_YAML_PATH.read_text(encoding="utf-8"))
    return list(raw["cases"])


def _case_ids(cases: list[dict]) -> list[str]:
    return [c["id"] for c in cases]


def _make_source(source_name: str, origin: str | None):
    if source_name == "web":
        return Source.web(origin or "https://example.invalid")
    if source_name == "mcp":
        return Source.mcp(origin or "untrusted-mcp")
    if source_name == "tool":
        return Source.tool(origin or "tool-output")
    return Source.user(origin or "eval-user")


@pytest.fixture(scope="module")
def shared_protectai_gate():
    """Reuse the heavyweight ProtectAI model across all parametrized cases."""
    from hermes_katana.scanner.protectai_gate import ProtectAIGate

    return ProtectAIGate()


def _run_case(case: dict, tmp_path: Path, shared_protectai_gate: Any) -> dict[str, Any]:
    """Execute a single eval case and return a result dict."""
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
            "protectai.gate": shared_protectai_gate,
        }
    )

    ctx = CallContext(tool_name=case["tool_name"], args=args)
    decision = chain.execute_pre(ctx)

    result: dict[str, Any] = {
        "id": case["id"],
        "tool_name": case["tool_name"],
        "args": args,
        "expected_decision": case["expected_decision"],
        "actual_decision": decision.value,
        "pre_pass": decision.value == case["expected_decision"],
        "output_pass": True,
        "audit_pass": False,
    }

    if case.get("output") is not None and decision != DispatchDecision.DENY:
        ctx.tool_output = case["output"]
        chain.execute_post(ctx)
        output_scan = ctx.extras.get("output_scan_result")
        expected_findings = bool(case.get("expected_output_findings", False))
        actual_findings = bool(output_scan and output_scan.has_findings)
        result["output_pass"] = actual_findings is expected_findings

    result["audit_pass"] = audit_trail.stats()["total_entries"] >= 1
    return result


# ---------------------------------------------------------------------------
# Load cases once at module level for parametrize
# ---------------------------------------------------------------------------

_ALL_CASES = _load_eval_cases()


def _case_id_func(case: dict) -> str:
    return case["id"]


# ---------------------------------------------------------------------------
# Parametrized test: one test item per YAML case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _ALL_CASES, ids=_case_id_func)
def test_eval_case(case, tmp_dir, shared_protectai_gate):
    """Run a single adversarial eval case."""
    expected = case["expected_decision"]

    # Support 'expected_decision: gap' → known gap, mark xfail
    if expected == _GAP_DECISION:
        pytest.xfail(f"Known gap: {case['id']} — no scanner coverage yet")

    result = _run_case(case, tmp_dir, shared_protectai_gate)

    # Build a rich error message
    msg = (
        f"\n  Case ID : {result['id']}"
        f"\n  Tool    : {result['tool_name']}"
        f"\n  Args    : {result['args']}"
        f"\n  Expected: {result['expected_decision']}"
        f"\n  Actual  : {result['actual_decision']}"
    )

    assert result["pre_pass"], f"Decision mismatch!{msg}"
    assert result["output_pass"], f"Output scan mismatch!{msg}"
    assert result["audit_pass"], f"Audit trail missing entries!{msg}"
