"""Verifier — separate critic with no tool access.

Reflexion / AI Scientist v2 / multi-agent-validation literature converges
on one lesson: the planner model, left alone, over-claims. It will say
"our method improved accuracy by 5%" when the evidence actually shows
noise. Verbalized chain-of-thought doesn't expose this; the agent simply
isn't self-critical enough.

Fix: a SEPARATE model with the following constraints:
  - No tool access at all. Cannot run commands, cannot fetch data, cannot
    edit files.
  - Receives ONLY the raw Observations and Results from the event stream
    plus the claim being audited. Not the chat history.
  - Sole job: downgrade claims that exceed their evidence, flag
    hallucinated numbers, flag missing baselines.

For now this is implemented as a rule-based verifier (rigor contract +
some extra checks). A future version can delegate to an actual LLM call
against Claude Haiku or GPT-4.1-mini with a locked prompt.

Usage:

    from hermes_katana.proving_ground.research.verifier import Verifier
    v = Verifier(report_query=lambda run_id: query_report(run_id))
    verdict = v.audit(claim, supporting_observations=[...])
    if verdict.passed:
        # emit Result event
    else:
        # emit ClaimDowngrade event with verdict.reasons
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from hermes_katana.proving_ground.research.events import Observation
from hermes_katana.proving_ground.research.rigor import Claim, enforce_rigor, STANDARD, RigorRules


@dataclass
class Verdict:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    downgraded_claim: Claim | None = None


class Verifier:
    """Separate critic. Stateless; called once per claim."""

    def __init__(
        self,
        *,
        rules: RigorRules = STANDARD,
        report_query: Callable[[str], dict] | None = None,
    ):
        """`report_query` is an optional callable that, given a run_id,
        returns the canonical summary for that campaign. Used to spot-check
        numbers in claims against recomputed figures from source data."""
        self.rules = rules
        self.report_query = report_query

    def audit(
        self,
        claim: Claim,
        supporting_observations: list[Observation] | None = None,
    ) -> Verdict:
        reasons: list[str] = []

        # Layer 1: rigor contract
        enforced = enforce_rigor(claim, self.rules)
        if isinstance(enforced, Claim):
            pass  # passed rigor
        else:
            reasons.extend(f"rigor: {r}" for r in enforced.reasons)

        # Layer 2: numbers must appear in at least one supporting observation
        supporting_observations = supporting_observations or []
        if supporting_observations:
            found = _value_grounded(claim.value, supporting_observations)
            if not found:
                reasons.append(
                    f"ungrounded: claim value {claim.value!r} does not appear in any supporting observation payload"
                )

        # Layer 3: if a run_id is cited, the report layer must confirm
        # the approximate magnitude. Skipped if report_query is None.
        if self.report_query and claim.comparison_run_id:
            try:
                summary = self.report_query(claim.comparison_run_id)
                if not _magnitude_matches(claim, summary):
                    reasons.append(
                        f"report mismatch: run {claim.comparison_run_id} "
                        f"reports metrics inconsistent with claimed value {claim.value}"
                    )
            except Exception as e:
                reasons.append(f"report_query failed: {e}")

        if not reasons:
            return Verdict(passed=True)
        return Verdict(passed=False, reasons=reasons, downgraded_claim=claim)


def _value_grounded(value: float, obs: list[Observation], tol: float = 1e-3) -> bool:
    """Search observation payloads for a numeric equal to `value` within tol."""

    def _walk(o):
        if isinstance(o, (int, float)):
            return abs(o - value) <= tol
        if isinstance(o, dict):
            return any(_walk(v) for v in o.values())
        if isinstance(o, (list, tuple)):
            return any(_walk(v) for v in o)
        return False

    return any(_walk(o.data) for o in obs)


def _magnitude_matches(claim: Claim, summary: dict) -> bool:
    """Very rough sanity: if the run's summary has the same primary_outcome,
    the claim value should be within the CI the summary reports."""
    po = summary.get(claim.primary_outcome)
    if po is None:
        return True  # unknown → don't block; let it pass
    if isinstance(po, dict) and "value" in po:
        reported = po["value"]
        ci = po.get("ci")
        if ci and len(ci) == 2:
            return ci[0] - 0.05 <= claim.value <= ci[1] + 0.05
        return abs(reported - claim.value) <= 0.1
    return True
