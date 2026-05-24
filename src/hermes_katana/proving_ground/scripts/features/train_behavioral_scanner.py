"""Train the novel "behavioural-signature" scanner.

Scanner #3. Input: the 33-dim behavioural signature vector from
`sandbox/analyzers/semantic_fingerprint.py`. Output: P(attacked | signature).

Why this is novel — the existing hermes-katana stack (Aho-Corasick, bloom,
DeBERTa binary, zvec) all score the *attack text*. This scanner scores the
*model response* instead. It catches attacks that succeed even though their
surface text looks novel, because a compromised model's response shape
(refusal drop, attack-reflection rise, length collapse, semantic drift)
tends to lie on a narrow manifold regardless of what triggered it.

Intended deployment:
- Offline research tool now — scores signatures from session DBs.
- In-scanner use later — once Tier 1 telemetry is routinely captured in
  production, run this over the live response to flag "the model is
  behaving like it was successfully injected". Orthogonal to input-side
  scanners.

Evaluation: with a small session corpus (tens of samples) we use k-fold CV
with stratification. Metrics reported: ROC-AUC, precision, recall, feature
importance. We keep the model simple (logistic regression with L2) so
weights are directly interpretable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_signatures(path: str):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows


def _vector_for(sig: dict, feature_names: list[str]):
    return [float(sig.get(f, 0.0) or 0.0) for f in feature_names]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signatures", default="results/behavioral_signatures.jsonl")
    p.add_argument("--out-dir", default="results/scanner_feeds")
    p.add_argument("--cv-folds", type=int, default=5)
    args = p.parse_args()

    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        StratifiedKFold,
        cross_val_score,
        cross_val_predict,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score, recall_score, f1_score

    # Import feature_names from the analyzer so we stay in lockstep.

    from hermes_katana.proving_ground.sandbox.analyzers.semantic_fingerprint import Signature

    feature_names = Signature.feature_names()

    sigs = load_signatures(args.signatures)
    # Drop sessions with no assistant messages (nothing to score).
    sigs = [s for s in sigs if s.get("n_messages", 0) > 0 and s.get("effective", -1) >= 0]
    print(f"Loaded {len(sigs)} usable signatures")

    X = np.array([_vector_for(s, feature_names) for s in sigs], dtype="float64")
    y = np.array([int(s["effective"]) for s in sigs], dtype="int64")
    print(f"Class balance — effective: {int(y.sum())}, clean: {int((1 - y).sum())}")

    if y.sum() < 2 or (1 - y).sum() < 2:
        print("Not enough labeled data in one class to train. Aborting.")
        return

    # Pipeline: standardise first, then L2-regularised logistic regression
    # with class_weight="balanced" to handle the mild class imbalance.
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )

    n_folds = min(args.cv_folds, max(2, min(int(y.sum()), int((1 - y).sum()))))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    try:
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
    except Exception as e:
        print(f"  WARN: roc_auc unavailable: {e}")
        auc = float("nan")

    # OOF predictions for precision/recall at threshold 0.5.
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    print(f"\n{n_folds}-fold cross-validation")
    print(f"  ROC-AUC   : {auc:.3f}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")

    # Fit on full data for the shipped model.
    pipe.fit(X, y)

    # Feature importances via standardised logistic coefficients.
    coefs = pipe.named_steps["clf"].coef_[0]
    importances = sorted(
        zip(feature_names, coefs),
        key=lambda kv: -abs(kv[1]),
    )

    print("\nTop features by |coef| (positive = pushes toward 'attacked'):")
    for name, c in importances[:10]:
        print(f"  {c:+.3f}  {name}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Persist the full sklearn pipeline for later inference.
    import joblib

    joblib.dump(pipe, out / "behavioral_signature_scanner.joblib")

    # Plus a human-readable JSON summary. Weights are AFTER scaling, so for
    # inference callers must either reuse the joblib pipeline (preferred) or
    # apply the scaler params below.
    scaler = pipe.named_steps["scaler"]
    summary = {
        "model": "sklearn.linear_model.LogisticRegression (L2, balanced)",
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "cv_folds": int(n_folds),
        "cv_metrics": {
            "roc_auc": None if np.isnan(auc) else round(float(auc), 4),
            "precision_at_0.5": round(float(prec), 4),
            "recall_at_0.5": round(float(rec), 4),
            "f1_at_0.5": round(float(f1), 4),
        },
        "feature_names": feature_names,
        "coefficients_standardised": {n: round(float(c), 4) for n, c in zip(feature_names, coefs)},
        "intercept": round(float(pipe.named_steps["clf"].intercept_[0]), 4),
        "scaler_mean": [round(float(m), 5) for m in scaler.mean_],
        "scaler_scale": [round(float(s), 5) for s in scaler.scale_],
    }
    (out / "behavioral_signature_scanner.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote scanner to {out}/behavioral_signature_scanner.{{joblib,json}}")


if __name__ == "__main__":
    main()
