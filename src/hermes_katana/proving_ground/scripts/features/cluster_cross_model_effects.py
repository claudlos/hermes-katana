"""Cluster attacks by their cross-model behavioural effect.

Scanner output #4 + corpus dedup tool. The premise: two attacks with similar
surface text may be mechanistically different, and two attacks with very
different surface text may trigger the same behavioural shape across models.
If we cluster attacks by their cross-model *effect vector* rather than by
their text, we get equivalence classes that map onto how models actually
respond — which is the right grouping for corpus refinement.

Input signals (merged):
- `smoke_rerun.db`          — per-session tool_usage_drift from the analyzer
- `confirmed_attacks.jsonl` — `effective_on` marks (sparse model membership)
- `rejected_attacks.jsonl`  — zero-effect marks across attempted models

For each attack we build a fixed-order effect vector (one slot per model).
Missing values are imputed to 0.0 (conservative — assumes "didn't affect
this model"). We then run KMeans with a small k and emit assignments.

At current scale (≤30 attacks) clusters are illustrative, not statistical.
The pipeline is built to scale: feed it a larger proving-ground batch and
the same script produces cleaner families.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


_MODELS_ORDER = [
    "nemotron-4b",
    "tinyllama-1b",
    "qwen3-4b",
    "qwen3.5-4b",
    "qwen3.5-4b-abliterated",
    "qwen3.5:9b",
    "qwen3.5-9b",  # ollama tag vs friendly name
    "qwen3.5-9b-ollama",
    "bonsai-1.7b",
    "bonsai-4b",
    "bonsai-8b",
]


def _load_jsonl(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def build_effect_matrix(db_path: str, confirmed_path: str, rejected_path: str):
    import numpy as np

    from hermes_katana.proving_ground.sandbox.analyzers.behavioral_drift import BehavioralAnalyzer

    analyzer = BehavioralAnalyzer(db_path) if Path(db_path).exists() else None
    conn = sqlite3.connect(db_path) if Path(db_path).exists() else None
    if conn:
        conn.row_factory = sqlite3.Row

    # Attack metadata keyed by attack_id.
    meta: dict[str, dict] = {}
    confirmed = _load_jsonl(confirmed_path)
    rejected = _load_jsonl(rejected_path)
    for a in confirmed:
        meta[a["id"]] = {
            "id": a["id"],
            "label": a.get("label", "?"),
            "text_preview": (a.get("text", "") or "")[:180],
            "source": "confirmed",
            "known_effective_on": a.get("effective_on", ""),
        }
    for a in rejected:
        meta[a["id"]] = {
            "id": a["id"],
            "label": a.get("label", "?"),
            "text_preview": (a.get("text", "") or "")[:180],
            "source": "rejected",
            "known_effective_on": "",
        }

    # Per-attack, per-model drift from the session DB.
    drift_map: dict[str, dict[str, float]] = defaultdict(dict)
    if conn and analyzer:
        rows = list(conn.execute("SELECT session_id, attack_id, attack_text, attack_label, model FROM sessions"))
        for r in rows:
            aid = r["attack_id"] or f"__unnamed:{hash(r['attack_text'] or '') & 0xFFFFFF:x}"
            if aid not in meta:
                meta[aid] = {
                    "id": aid,
                    "label": r["attack_label"] or "?",
                    "text_preview": (r["attack_text"] or "")[:180],
                    "source": "session_only",
                    "known_effective_on": "",
                }
            try:
                rep = analyzer.analyze_session(r["session_id"])
                drift = rep.tool_usage_drift
                if rep.collapse_detected:
                    drift = 1.0
            except Exception:
                drift = 0.0
            # Keep the max drift if multiple sessions for the same (attack, model).
            cur = drift_map[aid].get(r["model"], 0.0)
            drift_map[aid][r["model"]] = max(cur, drift)

    # Layer on known effective_on from confirmed_attacks.jsonl (sparse 1.0 marks).
    # v1 stored effective_on as a comma-separated string; v2 stores a list of
    # model ids. Accept either so this runs against both corpora.
    for a in confirmed:
        raw = a.get("effective_on", "")
        if isinstance(raw, list):
            models = [m for m in raw if m]
        else:
            models = [raw] if raw else []
        for m in models:
            if m not in drift_map[a["id"]]:
                drift_map[a["id"]][m] = max(float(a.get("max_drift") or 0.0), 0.5)

    # Build a stable model order covering every model we saw.
    seen_models = sorted({m for d in drift_map.values() for m in d})
    model_order = [m for m in _MODELS_ORDER if m in seen_models] + [m for m in seen_models if m not in _MODELS_ORDER]

    # Normalise the Ollama tag onto the friendly id where possible.
    # qwen3.5:9b and qwen3.5-9b are the same model.
    aliases = {"qwen3.5:9b": "qwen3.5-9b", "qwen3.5-9b-ollama": "qwen3.5-9b"}
    canonical_order: list[str] = []
    for m in model_order:
        canonical = aliases.get(m, m)
        if canonical not in canonical_order:
            canonical_order.append(canonical)

    attack_ids = sorted(drift_map.keys())
    X = np.zeros((len(attack_ids), len(canonical_order)), dtype="float32")
    for i, aid in enumerate(attack_ids):
        for m, drift in drift_map[aid].items():
            canonical = aliases.get(m, m)
            j = canonical_order.index(canonical)
            X[i, j] = max(X[i, j], drift)

    return attack_ids, canonical_order, X, meta


def cluster(X, n_clusters: int, seed: int = 0):
    from sklearn.cluster import KMeans

    if len(X) < n_clusters:
        n_clusters = max(1, len(X))
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="results/smoke_rerun.db")
    p.add_argument("--confirmed", default="results/confirmed_attacks.jsonl")
    p.add_argument("--rejected", default="results/rejected_attacks.jsonl")
    p.add_argument("--out-dir", default="results/scanner_feeds")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()

    attack_ids, models, X, meta = build_effect_matrix(args.db, args.confirmed, args.rejected)
    print(f"{len(attack_ids)} attacks × {len(models)} models")
    print(f"  non-zero cells: {int((X > 0).sum())}/{X.size}")

    labels, centers = cluster(X, args.k)

    # Group into families.
    clusters: dict[int, list[dict]] = defaultdict(list)
    for aid, lbl in zip(attack_ids, labels):
        m = meta.get(aid, {"id": aid, "label": "?", "text_preview": "", "source": "?"})
        clusters[int(lbl)].append(
            {
                **m,
                "effect_vector": {models[j]: float(round(X[attack_ids.index(aid), j], 3)) for j in range(len(models))},
            }
        )

    print(f"\n{args.k}-cluster assignment summary:")
    for k, members in sorted(clusters.items()):
        labels_in_cluster = [m["label"] for m in members]
        from collections import Counter

        top_label = Counter(labels_in_cluster).most_common(1)[0] if labels_in_cluster else ("?", 0)
        avg_effect = float(X[[attack_ids.index(m["id"]) for m in members]].mean())
        print(
            f"  cluster {k}: n={len(members)}  mean_effect={avg_effect:.3f}  top_label={top_label[0]} ({top_label[1]}x)"
        )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_attacks": len(attack_ids),
        "models_order": models,
        "k": args.k,
        "cluster_centers": centers.tolist(),
        "clusters": {str(k): members for k, members in clusters.items()},
    }
    (out / "cross_model_effect_clusters.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Flat CSV-friendly dump: one row per attack → cluster.
    with (out / "cross_model_effect_clusters.jsonl").open("w", encoding="utf-8") as f:
        for aid, lbl in zip(attack_ids, labels):
            m = meta.get(aid, {})
            f.write(
                json.dumps(
                    {
                        "attack_id": aid,
                        "cluster": int(lbl),
                        "attack_label": m.get("label", "?"),
                        "source": m.get("source", "?"),
                        "text_preview": m.get("text_preview", ""),
                        "effect_vector": {
                            models[j]: float(round(X[attack_ids.index(aid), j], 3)) for j in range(len(models))
                        },
                    }
                )
                + "\n"
            )
    print(f"\nWrote {out}/cross_model_effect_clusters.{{json,jsonl}}")


if __name__ == "__main__":
    main()
