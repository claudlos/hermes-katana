"""Ensemble detector — Katana rule scanner + centroid embedding.

Phase 2.H showed:
  - Katana alone: AUC 0.710 standalone, ~51%/51% recall/false-block at
    any threshold (essentially binary scoring on our corpus).
  - Centroid alone: AUC 0.961 — much stronger discrimination.

The hypothesis: stacking the two via a trivial logistic regressor beats
either alone because they capture orthogonal failure patterns. Katana
hits explicit injection patterns (keywords, structural markers); the
centroid matches semantic category membership (persona_jailbreak-ish,
exfiltration-ish). A prompt that evades one may not evade the other.

Pipeline:
  1. Sample N positives from confirmed_attacks.jsonl and M negatives
     from shards/control/.
  2. For each text, compute both detectors' scores across ALL 4 channel
     wrappings (score = mean across channels — deployment-agnostic).
  3. Stratified 5-fold CV logistic regression on [katana_score,
     centroid_score] → effective. Report mean AUC ± std, best-F1, and
     precision@{1%, 5%} FPR with Wilson CIs.
  4. Fit final model on all data; export coefficients for deployment.

Output:
  results/ensemble_detector.json — CV metrics + final coefficients.
  Optional: --sweep runs per-threshold precision/recall for single
  detectors + ensemble.

Usage:
    python scripts/ensemble_detector.py --n-pos 500 --n-neg 500
    python scripts/ensemble_detector.py --apply-dedup --n-pos 1000 --n-neg 1000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# Load detection_bench ONCE at module import; cache globals so MiniLM
# is loaded a single time across all score() calls.
import importlib.util as _iu  # noqa: E402

_spec = _iu.spec_from_file_location("_dbench", ROOT / "scripts" / "detection_bench.py")
_DBENCH = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_DBENCH)  # type: ignore[union-attr]
wrap_for_channel = _DBENCH.wrap_for_channel
detect_centroid = _DBENCH.detect_centroid


CHANNELS = ["file_content", "code_comment", "data_row", "tool_output"]


def _load_texts(path: Path) -> list[str]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                t = d.get("text")
                if t:
                    out.append(t)
            except Exception:
                continue
    return out


def _apply_dedup(ids: list[str], dedup_path: Path) -> set[str]:
    if not dedup_path.exists():
        return set()
    d = json.loads(dedup_path.read_text(encoding="utf-8"))
    drops: set[str] = set()
    for cluster in d.get("clusters", []):
        drops.update(cluster.get("drop_ids", []))
    return drops


def _load_positives(n: int, apply_dedup: bool, seed: int) -> list[str]:
    path = ROOT / "results" / "confirmed_attacks.jsonl"
    atks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                atks.append(d)
            except Exception:
                continue
    if apply_dedup:
        drops = _apply_dedup(
            [a.get("id") for a in atks],
            ROOT / "results" / "confirmed_attacks_dedup.json",
        )
        atks = [a for a in atks if a.get("id") not in drops]
    rng = random.Random(seed)
    rng.shuffle(atks)
    return [a.get("text") for a in atks[:n] if a.get("text")]


def _load_negatives(n: int, seed: int) -> list[str]:
    ctl_dir = ROOT / "shards" / "control"
    out = []
    if ctl_dir.exists():
        for p in sorted(ctl_dir.glob("shard_ctrl_*.jsonl")):
            out.extend(_load_texts(p))
            if len(out) >= n * 2:
                break
    rng = random.Random(seed)
    rng.shuffle(out)
    return out[:n]


# --- Score both detectors across all channel wrappings ---------------------


def _score_both(text: str) -> tuple[float, float]:
    """Returns (katana_mean_across_channels, centroid_mean_across_channels)."""
    from hermes_katana.scanner.injection import injection_score

    ks = [injection_score(wrap_for_channel(text, ch)) for ch in CHANNELS]
    cs = [detect_centroid(wrap_for_channel(text, ch)) for ch in CHANNELS]
    return float(np.mean(ks)), float(np.mean(cs))


# --- Main ------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n-pos", type=int, default=500)
    p.add_argument("--n-neg", type=int, default=500)
    p.add_argument("--apply-dedup", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/ensemble_detector.json")
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from hermes_katana.proving_ground.research.statistics import wilson_ci, auc_from_scores

    t0 = time.time()
    pos = _load_positives(args.n_pos, args.apply_dedup, args.seed)
    neg = _load_negatives(args.n_neg, args.seed)
    print(f"[ensemble] positives: {len(pos)}  negatives: {len(neg)}")

    # Score
    print("[ensemble] scoring — two detectors × 4 channel wrappings each ...")
    X: list[list[float]] = []
    y: list[int] = []
    for i, t in enumerate(pos):
        k, c = _score_both(t)
        X.append([k, c])
        y.append(1)
        if (i + 1) % 100 == 0:
            print(f"  pos {i + 1}/{len(pos)}  ({time.time() - t0:.0f}s)")
    for i, t in enumerate(neg):
        k, c = _score_both(t)
        X.append([k, c])
        y.append(0)
        if (i + 1) % 100 == 0:
            print(f"  neg {i + 1}/{len(neg)}  ({time.time() - t0:.0f}s)")
    X = np.asarray(X)
    y = np.asarray(y)
    katana_scores = X[:, 0]
    centroid_scores = X[:, 1]

    # Per-detector baseline AUCs
    auc_katana = auc_from_scores(katana_scores[y == 1].tolist(), katana_scores[y == 0].tolist())
    auc_centroid = auc_from_scores(centroid_scores[y == 1].tolist(), centroid_scores[y == 0].tolist())

    # 5-fold CV of the ensemble
    print("[ensemble] 5-fold stratified CV ...")
    cv_aucs: list[float] = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(solver="lbfgs", max_iter=200)
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        cv_aucs.append(roc_auc_score(y[te], prob))
    cv_auc_mean = float(np.mean(cv_aucs))
    cv_auc_std = float(np.std(cv_aucs))

    # Final fit + coefficients
    final = LogisticRegression(solver="lbfgs", max_iter=200)
    final.fit(X, y)
    ensemble_probs = final.predict_proba(X)[:, 1]

    def _metrics(scores_pos, scores_neg) -> dict:
        sp = sorted(scores_pos)
        sn_desc = sorted(scores_neg, reverse=True)
        # Best F1 by scanning thresholds
        thresholds = sorted(set(sp + sn_desc))
        best_f1, best_t = 0, 0
        for t in thresholds:
            tp = sum(1 for s in scores_pos if s >= t)
            fp = sum(1 for s in scores_neg if s >= t)
            fn = len(scores_pos) - tp
            if tp == 0:
                continue
            pr = tp / (tp + fp)
            re = tp / (tp + fn)
            f1 = 2 * pr * re / (pr + re) if (pr + re) else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t

        # Recall at fixed FPRs
        def _rat(fpr):
            idx = int(fpr * len(sn_desc))
            t_ = sn_desc[idx] if sn_desc else 0
            tp = sum(1 for s in scores_pos if s >= t_)
            return tp / max(len(scores_pos), 1), t_

        r1, t1 = _rat(0.01)
        r5, t5 = _rat(0.05)
        lo1, hi1 = wilson_ci(int(r1 * len(scores_pos)), len(scores_pos))
        lo5, hi5 = wilson_ci(int(r5 * len(scores_pos)), len(scores_pos))
        return {
            "best_f1": round(best_f1, 4),
            "best_f1_threshold": round(best_t, 4),
            "recall_at_1pct_fpr": round(r1, 4),
            "recall_at_1pct_fpr_ci": [round(lo1, 4), round(hi1, 4)],
            "recall_at_5pct_fpr": round(r5, 4),
            "recall_at_5pct_fpr_ci": [round(lo5, 4), round(hi5, 4)],
        }

    per_det = {
        "katana": {
            "auc": round(float(auc_katana), 4),
            **_metrics(katana_scores[y == 1].tolist(), katana_scores[y == 0].tolist()),
        },
        "centroid": {
            "auc": round(float(auc_centroid), 4),
            **_metrics(centroid_scores[y == 1].tolist(), centroid_scores[y == 0].tolist()),
        },
        "ensemble": {
            "auc_cv_mean": round(cv_auc_mean, 4),
            "auc_cv_std": round(cv_auc_std, 4),
            "auc_refit": round(float(roc_auc_score(y, ensemble_probs)), 4),
            **_metrics(ensemble_probs[y == 1].tolist(), ensemble_probs[y == 0].tolist()),
        },
    }

    out = {
        "schema_version": 1,
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "cv_folds": 5,
        "coefficients": {
            "katana": float(final.coef_[0, 0]),
            "centroid": float(final.coef_[0, 1]),
            "intercept": float(final.intercept_[0]),
        },
        "per_detector": per_det,
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== ensemble detector (pos={out['n_pos']} neg={out['n_neg']}) ===\n")
    print(f"{'detector':<12} {'AUC':>8} {'F1*':>7} {'R@1%FPR':>10} CI                  {'R@5%FPR':>10} CI")
    for name, m in per_det.items():
        auc_key = "auc_refit" if "auc_refit" in m else "auc"
        auc = m[auc_key]
        r1, ci1 = m["recall_at_1pct_fpr"], m["recall_at_1pct_fpr_ci"]
        r5, ci5 = m["recall_at_5pct_fpr"], m["recall_at_5pct_fpr_ci"]
        print(
            f"  {name:<10} {auc:>8.3f} {m['best_f1']:>7.3f}"
            f"  {r1:>8.3f}  [{ci1[0]:.3f},{ci1[1]:.3f}]"
            f"  {r5:>8.3f}  [{ci5[0]:.3f},{ci5[1]:.3f}]"
        )
    print(f"\nensemble CV AUC: {cv_auc_mean:.4f} ± {cv_auc_std:.4f}")
    print(
        f"ensemble coefficients: katana={out['coefficients']['katana']:+.3f} "
        f"centroid={out['coefficients']['centroid']:+.3f} "
        f"intercept={out['coefficients']['intercept']:+.3f}"
    )

    print(f"\nfull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
