"""Statistical-rigor contract.

Every scientific claim emitted by the research engine must satisfy a
minimal rigor contract or be downgraded to a plain observation. This
enforces — at the type level, not by prompting — that we never ship a
point-estimate without an uncertainty interval, never claim a cross-
condition effect without a paired test, never report a rate on N < k.

Design note: chat-style "please report CIs" prompting fails 5-10% of the
time even for capable models. Type-level enforcement pushes the failure
rate toward zero because the `Claim` object literally cannot be emitted
without the fields.

Usage:

    from hermes_katana.proving_ground.research.rigor import Claim, ObservationDowngrade, enforce_rigor

    claim = Claim(
        hypothesis_id="H-20260422-harness-dominates-model",
        primary_outcome="effective_rate_delta",
        value=-0.42,
        n_samples=412,
        ci=(-0.48, -0.36),
        baseline_run_id="hermes-run-abc",
        comparison_run_id="ccli-run-xyz",
        test_kind="mcnemar",
        p_value=0.003,
        effect_size={"kind": "cohens_h", "value": -0.58},
    )
    result = enforce_rigor(claim)
    assert isinstance(result, Claim)   # passed the contract

    weak_claim = Claim(
        hypothesis_id="...", primary_outcome="rate", value=0.53,
        n_samples=8, ci=None, baseline_run_id=None, comparison_run_id=None,
        test_kind=None, p_value=None, effect_size=None,
    )
    result = enforce_rigor(weak_claim)
    assert isinstance(result, ObservationDowngrade)
    print(result.reasons)  # ['n_samples < 30', 'missing CI', ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    """A scientific claim with statistical structure.

    A Claim that PASSES enforce_rigor may be appended to the event stream
    as a `Result`. A Claim that FAILS is downgraded to Observation.
    """

    hypothesis_id: str | None  # None for exploratory claims (see note)
    primary_outcome: str  # e.g. "effective_rate", "auc_transfer"
    value: float  # point estimate
    n_samples: int  # N backing the estimate
    ci: tuple[float, float] | None  # (low, high), same units as value
    # Comparison context (None for single-condition claims):
    baseline_run_id: str | None
    comparison_run_id: str | None
    test_kind: str | None  # "mcnemar", "bootstrap", ...
    p_value: float | None
    effect_size: dict | None  # {"kind": "cohens_h", "value": -0.58}
    # Free-form metadata (model, channel, language, timestamps, etc.)
    meta: dict = field(default_factory=dict)

    def is_comparison(self) -> bool:
        return self.comparison_run_id is not None

    # Note on hypothesis_id=None: claims without a preregistered hypothesis
    # are EXPLORATORY by definition. enforce_rigor still applies but the
    # downgrade policy permits them as long as they're clearly tagged.


@dataclass
class ObservationDowngrade:
    """Result of enforce_rigor when a Claim fails rigor.

    Carries the failing claim + why it failed so the event stream has a
    full audit trail of "we looked at this, didn't earn the Result tag."
    """

    claim: Claim
    reasons: list[str]
    rule_set: str


# ---------------------------------------------------------------------------
# Configurable rule sets
# ---------------------------------------------------------------------------


@dataclass
class RigorRules:
    """Thresholds used by enforce_rigor. Tune per publication target."""

    min_n_samples: int = 30
    require_ci: bool = True
    max_ci_half_width: float = 0.15  # point ± 0.15 acceptable
    require_test_for_comparison: bool = True
    require_effect_size_for_comparison: bool = True
    max_p_for_claim: float = 0.05  # preregistered claims must beat α
    allow_exploratory: bool = True  # permit hypothesis_id=None with tag


# Pre-tuned rule sets:
STRICT = RigorRules(
    min_n_samples=100,
    max_ci_half_width=0.10,
    max_p_for_claim=0.01,
)
STANDARD = RigorRules()  # defaults above
LOOSE = RigorRules(
    min_n_samples=10,
    require_ci=True,
    max_ci_half_width=0.25,
    max_p_for_claim=0.10,
)


# ---------------------------------------------------------------------------
# Contract enforcer
# ---------------------------------------------------------------------------


def enforce_rigor(
    claim: Claim,
    rules: RigorRules = STANDARD,
    *,
    rule_set_name: str = "STANDARD",
) -> Claim | ObservationDowngrade:
    reasons: list[str] = []

    # Sample size
    if claim.n_samples < rules.min_n_samples:
        reasons.append(f"n_samples ({claim.n_samples}) < {rules.min_n_samples}")

    # CI presence + width
    if rules.require_ci:
        if claim.ci is None:
            reasons.append("missing CI")
        else:
            half = (claim.ci[1] - claim.ci[0]) / 2
            if half > rules.max_ci_half_width:
                reasons.append(f"CI too wide (half-width={half:.3f} > {rules.max_ci_half_width})")

    # Comparison-claim specifics
    if claim.is_comparison():
        if rules.require_test_for_comparison and not claim.test_kind:
            reasons.append("comparison missing statistical test")
        if rules.require_effect_size_for_comparison and not claim.effect_size:
            reasons.append("comparison missing effect size")
        if claim.p_value is None:
            reasons.append("comparison missing p-value")

    # Significance (only enforced for preregistered claims)
    if claim.hypothesis_id is not None and claim.p_value is not None:
        if claim.p_value > rules.max_p_for_claim:
            reasons.append(f"p ({claim.p_value:.4f}) > alpha ({rules.max_p_for_claim})")

    # Exploratory allowance
    if claim.hypothesis_id is None and not rules.allow_exploratory:
        reasons.append("exploratory claim not permitted by rule set")

    if reasons:
        return ObservationDowngrade(claim=claim, reasons=reasons, rule_set=rule_set_name)
    return claim


# ---------------------------------------------------------------------------
# Convenience: construct a Claim directly from rate + stats.Proportion
# ---------------------------------------------------------------------------


def claim_from_proportion(
    *,
    hypothesis_id: str | None,
    primary_outcome: str,
    k: int,
    n: int,
    baseline_run_id: str | None = None,
    comparison_run_id: str | None = None,
    meta: dict | None = None,
    conf: float = 0.95,
) -> Claim:
    """Helper: wrap a binomial proportion as a Claim with Wilson CI."""
    from hermes_katana.proving_ground.research.statistics import wilson_ci

    low, hi = wilson_ci(k, n, conf=conf)
    return Claim(
        hypothesis_id=hypothesis_id,
        primary_outcome=primary_outcome,
        value=(k / n) if n else 0.0,
        n_samples=n,
        ci=(low, hi),
        baseline_run_id=baseline_run_id,
        comparison_run_id=comparison_run_id,
        test_kind=None,
        p_value=None,
        effect_size=None,
        meta=meta or {},
    )
