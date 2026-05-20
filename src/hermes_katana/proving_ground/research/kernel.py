"""ResearchKernel — composes the rest of the package.

Single orchestrator that holds:

  event_stream      — append-only record of everything the kernel did
  hypothesis_dag    — long-term memory of hypotheses / refinements / claims
  budget            — BudgetLedger with hard caps
  doom              — DoomLoopDetector
  verifier          — separate critic (evidence-only)
  router            — ToolRouter with lint-gated tools
  registry          — HypothesisRegistry backed by YAML files

Typical usage (library):

    from hermes_katana.proving_ground.research.kernel import ResearchKernel
    k = ResearchKernel.build(run_id="apr22-r1")
    k.propose_hypothesis(spec)           # register + add to DAG
    k.call("query_corpus", {...})        # tool call, gated
    k.submit_claim(claim, supporting_obs)  # rigor-gate + verifier → Result or Downgrade
    k.save()

This is NOT an LLM-driven loop. Adding an LLM planner is a thin layer
above: see `scripts/intern.py`. The kernel is deliberately small and
dep-free so it can be wired into any planner (rule-based, LLM-driven,
human-driven).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hermes_katana.proving_ground.research.budget import BudgetLedger
from hermes_katana.proving_ground.research.dag import HypothesisDAG, Node
from hermes_katana.proving_ground.research.doom import DoomLoopDetector
from hermes_katana.proving_ground.research.events import (
    EventStream,
    Observation,
    Hypothesis,
    Result,
    ClaimDowngrade,
    BudgetTick,
    DoomLoopFired,
)
from hermes_katana.proving_ground.research.registry import HypothesisRegistry
from hermes_katana.proving_ground.research.rigor import Claim, STANDARD, RigorRules
from hermes_katana.proving_ground.research.tools import ToolRouter, build_default_router, ExecutionResult
from hermes_katana.proving_ground.research.verifier import Verifier


ROOT = Path(__file__).resolve().parents[1]
KERNEL_ROOT = ROOT / "results" / "research_runs"


@dataclass
class ResearchKernel:
    run_id: str
    event_stream: EventStream
    dag: HypothesisDAG
    budget: BudgetLedger
    doom: DoomLoopDetector
    verifier: Verifier
    router: ToolRouter
    registry: HypothesisRegistry
    rules: RigorRules = field(default_factory=lambda: STANDARD)

    # --------------------------------------------------------------- factory
    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        budget: BudgetLedger | None = None,
        rules: RigorRules = STANDARD,
        auto_approve_human_gates: bool = False,
    ) -> "ResearchKernel":
        base = KERNEL_ROOT / run_id
        base.mkdir(parents=True, exist_ok=True)
        stream = EventStream(base / "events.jsonl")
        dag = HypothesisDAG(base / "hypothesis_dag.json")
        budget = budget or BudgetLedger()
        doom = DoomLoopDetector()
        verifier = Verifier(rules=rules)
        router = build_default_router(
            budget=budget,
            doom=doom,
            run_id=run_id,
            auto_approve_human_gates=auto_approve_human_gates,
        )
        registry = HypothesisRegistry()
        return cls(
            run_id=run_id,
            event_stream=stream,
            dag=dag,
            budget=budget,
            doom=doom,
            verifier=verifier,
            router=router,
            registry=registry,
            rules=rules,
        )

    # --------------------------------------------------------------- hypothesis flow
    def propose_hypothesis(self, spec: dict) -> Hypothesis:
        """Register a hypothesis + mirror into the DAG + emit event."""
        h = self.registry.register(spec)
        if not self.dag.has(h.id):
            self.dag.add_node(
                Node(
                    id=h.id,
                    title=h.title,
                    status="preregistered",
                    parent_id=spec.get("parent_hypothesis_id"),
                )
            )
        evt = Hypothesis(
            title=h.title,
            statement=h.statement,
            predicted_direction=h.predicted_direction,
            hypothesis_id=h.id,
            run_id=self.run_id,
            parent_hypothesis_id=spec.get("parent_hypothesis_id"),
        )
        self.event_stream.append(evt)
        self.dag.save()
        return evt

    def submit_claim(
        self,
        claim: Claim,
        supporting_observations: list[Observation] | None = None,
    ) -> Result | ClaimDowngrade:
        """Verifier audits the claim → emit Result or ClaimDowngrade."""
        verdict = self.verifier.audit(claim, supporting_observations or [])
        if verdict.passed:
            evt = Result(
                primary_outcome=claim.primary_outcome,
                value=claim.value,
                n_samples=claim.n_samples,
                ci=claim.ci,
                test_kind=claim.test_kind,
                p_value=claim.p_value,
                effect_size=claim.effect_size,
                baseline_run_id=claim.baseline_run_id,
                comparison_run_id=claim.comparison_run_id,
                meta=claim.meta,
                hypothesis_id=claim.hypothesis_id,
                run_id=self.run_id,
            )
            self.event_stream.append(evt)
            if claim.hypothesis_id and self.dag.has(claim.hypothesis_id):
                self.dag.attach_claim(claim.hypothesis_id, evt.event_id)
                self.dag.save()
            return evt
        else:
            evt = ClaimDowngrade(
                attempted_outcome=claim.primary_outcome,
                reasons=verdict.reasons,
                rule_set="STANDARD",
                hypothesis_id=claim.hypothesis_id,
                run_id=self.run_id,
            )
            self.event_stream.append(evt)
            return evt

    def resolve_hypothesis(
        self,
        hyp_id: str,
        *,
        resolved_run_id: str,
        p_value: float,
        effect_size: dict,
        verdict: str,
        notes: str = "",
    ) -> None:
        self.registry.resolve(
            hyp_id,
            run_id=resolved_run_id,
            p_value=p_value,
            effect_size=effect_size,
            verdict=verdict,
            notes=notes,
        )
        if self.dag.has(hyp_id):
            self.dag.set_status(hyp_id, "resolved")
            self.dag.save()

    # ----------------------------------------------------------------- tools
    def call(self, tool: str, args: dict, *, rationale: str = "") -> ExecutionResult:
        """Gated tool call. Always appends an Action event and (either an
        Observation on success, a GateRejection on failure, or a HumanGate
        event if requires_human_approval).

        If the doom detector fires during observation, emits a DoomLoopFired
        event too.
        """
        res = self.router.execute(tool, args, rationale=rationale)
        self.event_stream.append(res.action)
        if res.observation:
            self.event_stream.append(res.observation)
        if res.gate_rejection:
            self.event_stream.append(res.gate_rejection)
        if res.human_gate:
            self.event_stream.append(res.human_gate)
        fires, details = self.doom.fires()
        if fires:
            self.event_stream.append(
                DoomLoopFired(
                    window=details["window"],
                    fingerprint=details["fingerprint"],
                    recent_actions=details["recent"],
                    run_id=self.run_id,
                )
            )
        return res

    # -------------------------------------------------------------- bookkeeping
    def tick_budget(self) -> None:
        self.event_stream.append(
            BudgetTick(
                state=self.budget.snapshot(),
                run_id=self.run_id,
            )
        )

    def save(self) -> None:
        self.dag.save()

    def summary(self) -> dict:
        evs = list(self.event_stream)
        return {
            "run_id": self.run_id,
            "n_events": len(evs),
            "by_kind": _count_kinds(evs),
            "n_hypotheses": len(self.dag),
            "budget": self.budget.snapshot(),
        }


def _count_kinds(events) -> dict:
    out: dict[str, int] = {}
    for e in events:
        out[e.kind] = out.get(e.kind, 0) + 1
    return out
