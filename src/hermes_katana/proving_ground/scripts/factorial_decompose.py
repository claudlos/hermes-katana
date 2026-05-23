"""Factorial decomposition — the exact math on what each harness element does.

HONESTY DISCLAIMER built into the tool:

  The current corpus is OBSERVATIONAL. Harness features (instruction_hier,
  bash_allowlist, permission_gate, untrusted_marker) are deterministically
  entailed by harness_type in our fleet — e.g. every `cli_coding_agent`
  has instruction_hier=true AND bash_allowlist=strict AND untrusted_marker=
  true. A regression cannot identify independent effects of colinear
  features; the MLE blows up to [0, ∞] CIs and fits on boundary cases.

  To cleanly decompose WHICH harness component matters, we need FACTORIAL
  DATA COLLECTION — runs that VARY each feature independently. This tool
  flags that requirement and fits two models:

  1. "main":  effective ~ C(model_family) + C(harness_type) + C(channel)
                       + C(model_size)
             Orthogonal axes. Statsmodels with HC3 SEs. Reports log-odds,
             odds ratios, p-values, 95% CIs.

  2. "full":  same plus all binary harness features, fit with sklearn
             L2-regularized logistic. No p-values, but regularization keeps
             coefficients finite under colinearity, so we get INTERPRETABLE
             relative magnitudes. Use for descriptive "which features
             correlate with effective-rate holding others constant."

Output: results/factorial.json with both models. Console prints the main
model (publishable numbers) plus a short note about the full model's
identifiability issues.

Usage:
    python scripts/factorial_decompose.py --apply-exclusion
    python scripts/factorial_decompose.py --subsample 80000
    python scripts/factorial_decompose.py --interactions
    python scripts/factorial_decompose.py --out results/factorial.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

from hermes_katana.proving_ground.research.harness_profile import load_profiles  # noqa: E402

SHARD_RUNS = ROOT / "results" / "agent_shard_runs"
EXCLUSION_LIST = ROOT / "results" / "exclusion_list.json"


def _exclusion_keys() -> set[tuple]:
    if not EXCLUSION_LIST.exists():
        return set()
    d = json.loads(EXCLUSION_LIST.read_text())
    return {(r.get("agent_id"), r.get("shard"), r.get("channel"), r.get("attack_id")) for r in d.get("rows", [])}


def _stream(apply_exclusion: bool):
    excl = _exclusion_keys() if apply_exclusion else set()
    for p in sorted(SHARD_RUNS.glob("shard_*.jsonl")):
        if "_broken" in str(p):
            continue
        with p.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if excl:
                    k = (
                        row.get("agent_id"),
                        row.get("shard"),
                        row.get("channel"),
                        row.get("attack_id"),
                    )
                    if k in excl:
                        continue
                yield row


def build_frame(apply_exclusion: bool, subsample: int | None = None):
    import pandas as pd

    profiles = load_profiles()
    recs = []
    for row in _stream(apply_exclusion):
        agent = row.get("agent_id")
        p = profiles.get(agent)
        if p is None:
            continue
        recs.append(
            {
                "effective": int(bool(row.get("effective"))),
                "channel": row.get("channel") or "?",
                "model_family": p.model_family,
                "model_size": p.model_size,
                "harness_type": p.harness_type,
                "bash_allowlist": p.bash_allowlist,
                "instruction_hier": int(p.instruction_hier),
                "untrusted_marker": int(p.untrusted_marker),
                "permission_gate": int(p.permission_gate),
                "scanner_layer": int(p.scanner_layer),
                "tool_auto_exec": int(p.tool_auto_exec),
                "multi_turn": int(p.multi_turn),
                "agent": agent,
            }
        )
    df = pd.DataFrame(recs)
    if subsample and len(df) > subsample:
        df = df.sample(subsample, random_state=42).reset_index(drop=True)
    return df


def fit_main_model(df, include_interactions: bool) -> dict:
    import statsmodels.formula.api as smf

    parts = [
        "C(model_family, Treatment(reference='claude'))",
        "C(harness_type, Treatment(reference='hermes_agent'))",
        "C(channel, Treatment(reference='file_content'))",
        "C(model_size, Treatment(reference='medium'))",
    ]
    formula = "effective ~ " + " + ".join(parts)
    if include_interactions:
        formula += (
            " + C(harness_type, Treatment(reference='hermes_agent')):C(channel, Treatment(reference='file_content'))"
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.logit(formula, data=df).fit(disp=False, method="lbfgs", maxiter=200, cov_type="HC3")

    params, conf, pvals = model.params, model.conf_int(alpha=0.05), model.pvalues
    terms = []
    for name in params.index:
        beta = float(params[name])
        lo = float(conf.loc[name][0])
        hi = float(conf.loc[name][1])
        terms.append(
            {
                "term": name,
                "log_odds": round(beta, 4),
                "odds_ratio": round(float(np.exp(beta)), 4) if np.isfinite(beta) else None,
                "or_ci_low": round(float(np.exp(lo)), 4) if np.isfinite(lo) else None,
                "or_ci_high": round(float(np.exp(hi)), 4) if np.isfinite(hi) else None,
                "p_value": round(float(pvals[name]), 6),
            }
        )
    return {
        "model": "main",
        "formula": formula,
        "n_obs": int(model.nobs),
        "llf": round(float(model.llf), 2),
        "pseudo_r2_mcfadden": round(float(model.prsquared), 4),
        "converged": bool(model.mle_retvals.get("converged", False)),
        "terms": terms,
    }


def fit_full_model_l2(df) -> dict:
    """L2-regularized full model via sklearn — finite coefficients under colinearity."""
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    binary_features = [
        "instruction_hier",
        "untrusted_marker",
        "permission_gate",
        "scanner_layer",
        "tool_auto_exec",
        "multi_turn",
    ]
    live_binary = [c for c in binary_features if df[c].nunique() > 1]
    dropped = [c for c in binary_features if c not in live_binary]

    X_parts = []
    # One-hot categoricals (drop first for reference coding)
    X_parts.append(pd.get_dummies(df["model_family"], prefix="mf", drop_first=True).astype(float))
    X_parts.append(pd.get_dummies(df["harness_type"], prefix="ht", drop_first=True).astype(float))
    X_parts.append(pd.get_dummies(df["channel"], prefix="ch", drop_first=True).astype(float))
    X_parts.append(pd.get_dummies(df["model_size"], prefix="ms", drop_first=True).astype(float))
    X_parts.append(pd.get_dummies(df["bash_allowlist"], prefix="ba", drop_first=True).astype(float))
    if live_binary:
        X_parts.append(df[live_binary].astype(float))
    X = pd.concat(X_parts, axis=1)
    y = df["effective"].values

    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",  # our positive class is ~11% — balance it
    )
    lr.fit(X.values, y)
    coefs = lr.coef_[0]

    terms = []
    for name, c in zip(X.columns, coefs):
        terms.append(
            {
                "term": name,
                "log_odds": round(float(c), 4),
                "odds_ratio": round(float(np.exp(c)), 4),
            }
        )
    terms.sort(key=lambda t: -abs(t["log_odds"]))

    return {
        "model": "full_l2",
        "n_obs": int(len(df)),
        "intercept": round(float(lr.intercept_[0]), 4),
        "dropped_binary_features": dropped,
        "C": 1.0,
        "class_weight": "balanced",
        "n_features": int(X.shape[1]),
        "terms": terms,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--apply-exclusion", action="store_true")
    p.add_argument("--subsample", type=int, default=80000)
    p.add_argument("--interactions", action="store_true")
    p.add_argument("--out", default="results/factorial.json")
    args = p.parse_args()

    t0 = time.time()
    df = build_frame(apply_exclusion=args.apply_exclusion, subsample=args.subsample)
    print(f"[factorial] built frame: {len(df):,} rows in {time.time() - t0:.1f}s")

    print("[factorial] fitting main model (orthogonal axes) ...")
    main_fit = fit_main_model(df, include_interactions=args.interactions)

    print("[factorial] fitting full L2 model (descriptive magnitudes) ...")
    full_fit = fit_full_model_l2(df)

    out = {
        "schema_version": 2,
        "apply_exclusion": args.apply_exclusion,
        "subsample": args.subsample,
        "interactions": args.interactions,
        "main": main_fit,
        "full_l2": full_fit,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    # Console summary — main model (publishable numbers)
    mf = main_fit
    print("\n=== MAIN model — orthogonal axes (publishable CIs) ===")
    print(f"N={mf['n_obs']:,}  converged={mf['converged']}  pseudo-R²={mf['pseudo_r2_mcfadden']:.3f}")
    print(f"\n{'term':<72} {'OR':>7}  {'95% CI':>16}  {'p':>8}")
    sig = [t for t in mf["terms"] if t["term"] != "Intercept"]
    sig.sort(key=lambda t: t["p_value"])
    for t in sig[:30]:
        ci = f"[{t['or_ci_low']:.2f}, {t['or_ci_high']:.2f}]" if t["or_ci_low"] is not None else "[nan]"
        if t["p_value"] < 0.001:
            mark = " ***"
        elif t["p_value"] < 0.01:
            mark = " **"
        elif t["p_value"] < 0.05:
            mark = " *"
        else:
            mark = ""
        print(f"  {t['term']:<70} {t['odds_ratio']:>7.3f}  {ci:>16}  {t['p_value']:>8.5f}{mark}")

    ff = full_fit
    print("\n=== FULL L2 model — top 15 log-odds by magnitude (no p-values) ===")
    print(f"N={ff['n_obs']:,}  intercept={ff['intercept']}  n_features={ff['n_features']}")
    if ff["dropped_binary_features"]:
        print(f"dropped constant: {ff['dropped_binary_features']}")
    print(f"\n{'term':<30} {'log_odds':>10} {'OR':>7}")
    for t in ff["terms"][:15]:
        print(f"  {t['term']:<28} {t['log_odds']:>+10.3f} {t['odds_ratio']:>7.3f}")

    print(f"\nfull report: {out_path}")
    print("\nINTERPRETATION NOTES:")
    print("  - Main model uses only orthogonal axes (model_family × harness_type × channel × model_size).")
    print("  - Harness-COMPONENT features (instruction_hier, bash_allowlist, etc) are DETERMINISTICALLY")
    print("    entailed by harness_type in the current corpus. To identify their independent effect,")
    print("    a factorial campaign (where these features vary INDEPENDENTLY of harness_type) is required.")
    print("  - The full L2 model gives descriptive magnitudes but NOT causal estimates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
