"""Statistics primitives for rigorous reporting.

Every claim the research engine emits must carry a confidence interval,
and cross-condition comparisons must carry a paired test. The field has
been sloppy about this (JailbreakBench leaderboard reports point estimates
only); our differentiator is that every number is reported with
uncertainty.

Kept intentionally dependency-light: plain math + stdlib. No scipy.
When scipy is desired for chi-square / exact tests, see `scipy_adapters`
at the bottom (optional import).

Usage:
    from hermes_katana.proving_ground.research.statistics import wilson_ci, paired_bootstrap_ci, mcnemar
    low, hi = wilson_ci(k=12, n=125, conf=0.95)
    delta_low, delta_hi = paired_bootstrap_ci(a_bits, b_bits, iters=5000)
    p = mcnemar(b=7, c=19)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Single-proportion confidence intervals
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over normal approximation because it behaves well for small n
    and for proportions near 0 or 1 (where "normal" CIs bleed past [0,1]).
    Agresti & Coull (1998), Brown, Cai, DasGupta (2001).
    """
    if n == 0:
        return (0.0, 1.0)
    z = _z_for_conf(conf)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    low = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (low, hi)


def clopper_pearson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson exact binomial CI.

    More conservative than Wilson; guaranteed coverage ≥ nominal. Slow for
    huge n. Use when a reviewer asks for "exact" intervals.
    """
    if n == 0:
        return (0.0, 1.0)
    alpha = 1 - conf
    low = 0.0 if k == 0 else _beta_quantile(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_quantile(1 - alpha / 2, k + 1, n - k)
    return (low, hi)


# ---------------------------------------------------------------------------
# Paired-sample inference
# ---------------------------------------------------------------------------


def mcnemar(b: int, c: int, *, correction: bool = True) -> float:
    """McNemar's test for paired binary outcomes.

    For a 2x2 contingency of paired successes:
        b = A-success & B-failure count
        c = A-failure & B-success count
    Returns a two-sided p-value.

    Uses the binomial exact test when b+c < 25 (recommended by Fagerland &
    Lydersen 2013); otherwise the chi-square approximation with Yates'
    continuity correction (on by default).
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        # Exact two-sided binomial on min(b,c) observations at p=0.5
        m = min(b, c)
        # P(X ≤ m) * 2 for two-sided
        tail = sum(math.comb(n, i) for i in range(m + 1)) / (2**n)
        return min(1.0, 2.0 * tail)
    diff = abs(b - c) - (1 if correction else 0)
    chi2 = (diff * diff) / n
    return _chi2_sf(chi2, df=1)


def paired_bootstrap_ci(
    a: Sequence[int | float],
    b: Sequence[int | float],
    iters: int = 10_000,
    conf: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for the mean of (a_i - b_i), paired samples.

    Use when both conditions run against the same N items (same attack on
    condition A and condition B) and you want a CI on the difference in
    mean effectiveness.
    """
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_ci requires len(a) == len(b)")
    n = len(a)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    diffs = [a[i] - b[i] for i in range(n)]
    boots: list[float] = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    alpha = 1 - conf
    lo = boots[int(alpha / 2 * iters)]
    hi = boots[int((1 - alpha / 2) * iters) - 1]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    values: Sequence[float],
    iters: int = 5000,
    conf: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of a single sample.

    Use for repeated-measures aggregation: collapse repeats to per-attack
    rates first, then bootstrap the cell-level mean across attacks. This
    correctly accounts for clustering — Wilson CIs on row-counted (k, n)
    are over-confident when rows are repeated measures of the same
    attack.

    Returns (point_estimate, ci_low, ci_high).
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)
    if n == 1:
        v = float(values[0])
        return (v, v, v)
    rng = random.Random(seed)
    boots: list[float] = []
    arr = list(values)
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += arr[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    alpha = 1 - conf
    lo = boots[int(alpha / 2 * iters)]
    hi = boots[int((1 - alpha / 2) * iters) - 1]
    return (sum(arr) / n, lo, hi)


def sample_size_single_proportion(
    *,
    half_width: float,
    conf: float = 0.95,
    p: float = 0.5,
) -> int:
    """Approximate independent units needed for a proportion CI half-width.

    This is for planning, not final inference. Use ``p=0.5`` when unknown;
    it is the conservative/worst-case binomial variance. For repeated LLM
    trials, the independent unit should normally be attack family/cluster,
    not physical output row.
    """
    if not (0 < half_width < 1):
        raise ValueError("half_width must be between 0 and 1")
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1")
    z = _z_for_conf(conf)
    return math.ceil((z * z * p * (1 - p)) / (half_width * half_width))


def sample_size_paired_delta(
    *,
    delta: float,
    discordance: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Approximate paired-family N for detecting a binary ASR delta.

    Let D_i = Y_iA - Y_iB in {-1, 0, 1}. ``delta`` is E[D_i] and
    ``discordance`` is Pr(D_i != 0). Then Var(D_i) ≈ q - delta².
    This is a normal approximation suitable for preregistration/budgeting.
    """
    if not (0 < abs(delta) < 1):
        raise ValueError("delta must be non-zero and between -1 and 1")
    if not (0 < discordance <= 1):
        raise ValueError("discordance must be between 0 and 1")
    if discordance <= delta * delta:
        raise ValueError("discordance must exceed delta^2 for positive variance")
    if not (0 < alpha < 1) or not (0 < power < 1):
        raise ValueError("alpha and power must be probabilities")
    z_alpha = _norm_inv(1 - alpha / 2)
    z_power = _norm_inv(power)
    var_d = discordance - delta * delta
    return math.ceil(((z_alpha + z_power) ** 2 * var_d) / (delta * delta))


def cluster_bootstrap_mean_ci(
    clusters: dict[str, Sequence[int | float]],
    *,
    iters: int = 5000,
    conf: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster bootstrap over primary units after collapsing repeats.

    ``clusters`` maps an attack-family/primary-unit id to repeated outcomes.
    Repeats are averaged within cluster first, then clusters are resampled.
    This avoids pretending stochastic repeats are independent rows.
    """
    rates: list[float] = []
    for values in clusters.values():
        vals = [float(v) for v in values]
        if vals:
            rates.append(sum(vals) / len(vals))
    return bootstrap_mean_ci(rates, iters=iters, conf=conf, seed=seed)


def paired_cluster_bootstrap_ci(
    a_clusters: dict[str, Sequence[int | float]],
    b_clusters: dict[str, Sequence[int | float]],
    *,
    iters: int = 5000,
    conf: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, int]:
    """Paired cluster-bootstrap CI for the mean delta A-B.

    Only primary units present in both conditions are used, preserving the
    matched design. Returns (delta, ci_low, ci_high, n_common_clusters).
    """
    common = sorted(set(a_clusters) & set(b_clusters))
    if not common:
        return (0.0, 0.0, 0.0, 0)
    a_rates = []
    b_rates = []
    for key in common:
        av = [float(v) for v in a_clusters[key]]
        bv = [float(v) for v in b_clusters[key]]
        if not av or not bv:
            continue
        a_rates.append(sum(av) / len(av))
        b_rates.append(sum(bv) / len(bv))
    if not a_rates:
        return (0.0, 0.0, 0.0, 0)
    point = sum(a - b for a, b in zip(a_rates, b_rates)) / len(a_rates)
    lo, hi = paired_bootstrap_ci(a_rates, b_rates, iters=iters, conf=conf, seed=seed)
    return (point, lo, hi, len(a_rates))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for difference of two proportions.

    |h| conventions: 0.2 small, 0.5 medium, 0.8 large.
    Preferred over raw Δp when comparing near-boundary rates (0.05 vs 0.15
    is "larger" than 0.45 vs 0.55, and h reflects that).
    """
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def auc_from_scores(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """Mann-Whitney-U AUC. For detector evaluation."""
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Combine + rank with average ranks for ties
    merged = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    merged.sort()
    ranks = [0.0] * len(merged)
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-indexed
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    sum_ranks_pos = sum(ranks[k] for k, (_, y) in enumerate(merged) if y == 1)
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


@dataclass
class Proportion:
    """A single proportion with its CI — the canonical way to report rates."""

    k: int
    n: int
    ci_low: float
    ci_high: float
    conf: float = 0.95

    @classmethod
    def of(cls, k: int, n: int, conf: float = 0.95) -> "Proportion":
        low, hi = wilson_ci(k, n, conf)
        return cls(k=k, n=n, ci_low=low, ci_high=hi, conf=conf)

    @property
    def rate(self) -> float:
        return 0.0 if self.n == 0 else self.k / self.n

    def format(self, pct: bool = True) -> str:
        if pct:
            return (
                f"{100 * self.rate:.1f}% "
                f"({int(self.conf * 100)}% CI: "
                f"{100 * self.ci_low:.1f}%-{100 * self.ci_high:.1f}%)"
            )
        return f"{self.rate:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]"


# ---------------------------------------------------------------------------
# Internals — normal / chi-square / beta quantiles (no scipy)
# ---------------------------------------------------------------------------


def _z_for_conf(conf: float) -> float:
    # Inverse normal CDF via rational approximation (Peter Acklam).
    alpha = (1 - conf) / 2
    return _norm_inv(1 - alpha)


def _norm_inv(p: float) -> float:
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    # Acklam's approximation, |err| < 1.15e-9
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5] / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
    )


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of chi-square, df ∈ {1,2,3,...}. Used for McNemar."""
    if x <= 0:
        return 1.0
    # Use the regularized upper incomplete gamma: Q(df/2, x/2)
    return _gamma_upper_reg(df / 2.0, x / 2.0)


def _gamma_upper_reg(a: float, x: float) -> float:
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0:
        return 1.0
    if x < a + 1.0:
        # Series expansion for lower; subtract from 1
        return 1.0 - _gamma_lower_series(a, x)
    return _gamma_upper_cf(a, x)


def _gamma_lower_series(a: float, x: float) -> float:
    term = 1.0 / a
    s = term
    for n in range(1, 200):
        term *= x / (a + n)
        s += term
        if abs(term) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_upper_cf(a: float, x: float) -> float:
    # Lentz's algorithm for continued-fraction form of Q(a,x).
    b = x + 1.0 - a
    c = 1e30
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse regularized incomplete beta via bisection.

    Adequate for CIs up to ~1e-6 precision; not hot-path code.
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _beta_regularized(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _beta_regularized(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_bt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(ln_bt) * _beta_cf(x, a, b) / a
    return 1.0 - math.exp(ln_bt) * _beta_cf(1 - x, b, a) / b


def _beta_cf(x: float, a: float, b: float) -> float:
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h
