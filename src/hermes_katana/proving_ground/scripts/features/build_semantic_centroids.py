"""Build per-family semantic centroids from confirmed attacks.

Scanner feed #2. The hermes-katana zvec scanner already ships with a centroid
bundle at `training/models/attack_centroids_128d.npz`, trained on the curator-
labeled corpus. What we add here is a complementary bundle built from *attacks
that were confirmed effective against real LLMs in the proving ground*.

We emit 384-dim MiniLM embeddings (pre-projection) so the zvec scanner can
pipe them through its INT8 projector to get the 128-dim representation it
actually uses for similarity search. Projecting 384→128 is the zvec project's
responsibility; we stop at 384 so the artifact is also usable by any other
embedding pipeline that wants to run its own projector.

Outputs:
- `semantic_centroids.json`     — human-readable: per-label centroid, member count
- `semantic_centroids.npz`      — numpy-loadable: labels[], centroids[N,384], member_vectors[M,384], member_labels[M]
- `attack_vectors.jsonl`        — one line per attack with {id, label, vector, effective}

Effective attacks get their own centroids; rejected attacks are embedded too
so downstream analysis can measure the distance gap between effective and
rejected clusters.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _embed(texts: list[str]):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = model.encode(
        texts,
        normalize_embeddings=True,  # L2-normalised → cosine == dot product
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return vecs.astype("float32")


def build_centroids(
    confirmed_path: str,
    rejected_path: str,
    out_dir: str,
):
    import numpy as np

    confirmed = _load_jsonl(confirmed_path)
    rejected = _load_jsonl(rejected_path)
    all_attacks = [{**a, "_effective": True} for a in confirmed] + [{**a, "_effective": False} for a in rejected]
    print(f"Embedding {len(all_attacks)} attacks ({len(confirmed)} confirmed, {len(rejected)} rejected)")

    texts = [a.get("text", "") for a in all_attacks]
    vecs = _embed(texts)  # (N, 384), L2-normalised

    # Per-label centroid — ONLY over confirmed attacks; rejected kept for
    # distance-gap analysis but don't pollute the centroid.
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, a in enumerate(all_attacks):
        if a["_effective"]:
            by_label[a.get("label", "unknown")].append(i)

    centroids_list: list[tuple[str, list[float], list[str]]] = []
    for label, idxs in sorted(by_label.items()):
        sub = vecs[idxs]
        centroid = sub.mean(axis=0)
        # Re-normalise so cosine comparisons are a plain dot product.
        n = np.linalg.norm(centroid)
        if n > 0:
            centroid = centroid / n
        members = [all_attacks[i].get("id", f"idx_{i}") for i in idxs]
        centroids_list.append((label, centroid.tolist(), members))
        print(f"  {label:<25} n={len(idxs)}  centroid_norm={n:.3f}")

    # Effective/rejected gap — average max-cos between each rejected attack
    # and the closest confirmed centroid. If this is low, the confirmed
    # centroids discriminate well; if high, they overlap with rejected text.
    if centroids_list and rejected:
        c_matrix = np.array([c for _, c, _ in centroids_list], dtype="float32")
        rejected_idxs = [i for i, a in enumerate(all_attacks) if not a["_effective"]]
        rej_vecs = vecs[rejected_idxs]
        sims = rej_vecs @ c_matrix.T  # (n_rejected, n_labels)
        max_sim_per_rejected = sims.max(axis=1)
        effective_idxs = [i for i, a in enumerate(all_attacks) if a["_effective"]]
        eff_vecs = vecs[effective_idxs]
        eff_sims = eff_vecs @ c_matrix.T
        max_sim_per_effective = eff_sims.max(axis=1)
        print("\nDistance gap (max-cos to closest centroid):")
        print(f"  effective   mean={max_sim_per_effective.mean():.3f}  min={max_sim_per_effective.min():.3f}")
        print(f"  rejected    mean={max_sim_per_rejected.mean():.3f}  max={max_sim_per_rejected.max():.3f}")
        print(f"  separation  = {max_sim_per_effective.mean() - max_sim_per_rejected.mean():+.3f}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Human-readable JSON (truncated centroid values for inspection).
    json_payload = {
        "dim": 384,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "n_attacks": len(all_attacks),
        "n_confirmed": len(confirmed),
        "n_rejected": len(rejected),
        "centroids": [
            {
                "label": label,
                "n_members": len(members),
                "member_ids": members,
                "vector_preview": [round(x, 4) for x in vec[:8]],
            }
            for label, vec, members in centroids_list
        ],
    }
    (out / "semantic_centroids.json").write_text(json.dumps(json_payload, indent=2))

    # Full numeric bundle — consumable by the zvec projector.
    np.savez(
        out / "semantic_centroids.npz",
        labels=np.array([label for label, _, _ in centroids_list]),
        centroids=np.array([c for _, c, _ in centroids_list], dtype="float32"),
        member_vectors=vecs,
        member_labels=np.array(
            [
                (a.get("label", "unknown") if a["_effective"] else f"_rejected:{a.get('label', '?')}")
                for a in all_attacks
            ]
        ),
        member_ids=np.array([a.get("id", "") for a in all_attacks]),
        effective=np.array([a["_effective"] for a in all_attacks], dtype="bool"),
    )

    # Per-attack vectors JSONL (for projection / downstream ML).
    with (out / "attack_vectors.jsonl").open("w") as f:
        for i, a in enumerate(all_attacks):
            f.write(
                json.dumps(
                    {
                        "id": a.get("id", ""),
                        "label": a.get("label", ""),
                        "effective": bool(a["_effective"]),
                        "text_preview": (a.get("text", "") or "")[:180],
                        "vector": [round(float(x), 5) for x in vecs[i]],
                    }
                )
                + "\n"
            )

    print(f"\nWrote artifacts to {out}/")
    print(f"  semantic_centroids.npz  {len(centroids_list)} centroids, {len(all_attacks)} member vectors")
    print("  semantic_centroids.json")
    print(f"  attack_vectors.jsonl    {len(all_attacks)} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--confirmed", default="results/confirmed_attacks.jsonl")
    p.add_argument("--rejected", default="results/rejected_attacks.jsonl")
    p.add_argument("--out-dir", default="results/scanner_feeds")
    args = p.parse_args()
    build_centroids(args.confirmed, args.rejected, args.out_dir)
