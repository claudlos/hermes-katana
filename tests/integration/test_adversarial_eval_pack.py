"""Adversarial end-to-end eval pack for the Hermes dispatch path."""

from __future__ import annotations

from pathlib import Path

import yaml

from hermes_katana.audit import AuditTrail
from hermes_katana.middleware import CallContext, DispatchDecision
from hermes_katana.middleware.integration import create_default_chain
from hermes_katana.taint import Source, TaintTracker


def _load_eval_cases() -> list[dict]:
    path = Path(__file__).resolve().parents[2] / "evals" / "adversarial_dispatch.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return list(raw["cases"])


def _make_source(source_name: str, origin: str | None):
    if source_name == "web":
        return Source.web(origin or "https://example.invalid")
    if source_name == "mcp":
        return Source.mcp(origin or "untrusted-mcp")
    if source_name == "tool":
        return Source.tool(origin or "tool-output")
    return Source.user(origin or "eval-user")


class TestAdversarialEvalPack:
    def test_cases(self, tmp_dir):
        for case in _load_eval_cases():
            tracker = TaintTracker()
            audit_trail = AuditTrail(path=tmp_dir / f"{case['id']}.jsonl")
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
            decision = chain.execute_pre(ctx)
            assert decision.value == case["expected_decision"], case["id"]

            if case.get("output") is not None and decision != DispatchDecision.DENY:
                ctx.tool_output = case["output"]
                chain.execute_post(ctx)
                output_scan = ctx.extras.get("output_scan_result")
                assert bool(output_scan and output_scan.has_findings) is bool(
                    case.get("expected_output_findings", False)
                ), case["id"]

            assert audit_trail.stats()["total_entries"] >= 1, case["id"]
