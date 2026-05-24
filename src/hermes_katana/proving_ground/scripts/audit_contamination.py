"""Contamination audit — how much of our confirmed corpus is a near-duplicate
of already-public jailbreak data or of another attack in our own corpus.

The question a reviewer will ask: "are your effectiveness rates measuring
recall of attacks that are in the model's training data, or are you
measuring true novel attack success?"

We can't see training data. But we CAN bound contamination risk two ways:

  INTRA-CORPUS   Near-duplicate pairs WITHIN confirmed_attacks.jsonl.
                 If attack A and attack B are 90%+ identical and both
                 confirmed, they inflate the reported novel-attack count.

  PUBLIC-CORPUS  Overlap with known public jailbreak datasets (HarmBench,
                 JailbreakBench, AdvBench, deepset/prompt-injections).
                 High overlap = high probability the pretraining cutoff
                 included this text; effectiveness partially reflects recall.

Matching is character 5-gram Jaccard: robust to minor rewordings, fast.
A pair with Jaccard ≥ 0.80 is a near-duplicate (tunable).

Usage:

    python scripts/audit_contamination.py                      # intra only
    python scripts/audit_contamination.py \\
        --external /path/to/harmbench_behaviors.jsonl:text     # +external
    python scripts/audit_contamination.py --threshold 0.75     # looser match
    python scripts/audit_contamination.py --out results/contamination_audit.json

External-corpus format: path:field where `field` is the JSONL key containing
the attack text (default "text"). TSV/CSV with a header row is also
supported via path:csv:field.

Output: per-attack max-similarity scores, top pairs for each corpus, and
an overall "potential contamination rate" at the configured threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# --- Near-duplicate matching via character n-gram Jaccard ------------------


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Keep punctuation — prompt-injection attacks are sensitive to it.
    return s


def _char_shingles(s: str, n: int = 5) -> frozenset[str]:
    s = _normalize(s)
    if len(s) < n:
        return frozenset({s})
    return frozenset(s[i : i + n] for i in range(len(s) - n + 1))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# --- External-corpus loaders -----------------------------------------------


def _load_external(spec: str) -> list[str]:
    """spec formats:
    path/to/file.jsonl:field
    path/to/file.csv:csv:field
    path/to/file.txt             (one attack per line)
    """
    parts = spec.split(":")
    path = Path(parts[0])
    if not path.exists():
        raise FileNotFoundError(f"external corpus not found: {path}")
    if path.suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(parts) >= 3 and parts[1] == "csv":
        field = parts[2]
        out = []
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                v = row.get(field)
                if v:
                    out.append(v)
        return out
    # JSONL with a named field
    field = parts[1] if len(parts) > 1 else "text"
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                v = d.get(field)
                if v:
                    out.append(v)
            except Exception:
                continue
    return out


# --- Nearest-neighbor computation ------------------------------------------


def _max_match_per_attack(
    our_attacks: list[dict],
    their_texts: list[str],
) -> list[dict]:
    """For each of our attacks, find the best-matching text in the external list."""
    their_shingles = [_char_shingles(t) for t in their_texts]
    results: list[dict] = []
    for atk in our_attacks:
        txt = atk.get("text") or ""
        our_sh = _char_shingles(txt)
        best_j = 0.0
        best_idx = -1
        for i, sh in enumerate(their_shingles):
            j = _jaccard(our_sh, sh)
            if j > best_j:
                best_j = j
                best_idx = i
        results.append(
            {
                "id": atk.get("id"),
                "label": atk.get("label"),
                "best_match_jaccard": round(best_j, 4),
                "best_match_idx": best_idx,
                "best_match_preview": (their_texts[best_idx][:200] if best_idx >= 0 else ""),
            }
        )
    return results


def _intra_corpus_pairs(
    our_attacks: list[dict],
    threshold: float,
) -> list[dict]:
    """Find near-duplicate PAIRS (i<j) inside our own corpus."""
    shingles = [(a, _char_shingles(a.get("text") or "")) for a in our_attacks]
    pairs: list[dict] = []
    # Quadratic — fine for 3.8k attacks (~7M comparisons). Pre-index by
    # first-few-shingle signature for a future speedup if it matters.
    for i in range(len(shingles)):
        a_i, s_i = shingles[i]
        for j in range(i + 1, len(shingles)):
            a_j, s_j = shingles[j]
            # Quick Jaccard upper-bound gate: |s_i & s_j| / min(|s_i|, |s_j|)
            # always ≥ Jaccard, so if the upper bound < threshold, skip exact.
            if len(s_i) == 0 or len(s_j) == 0:
                continue
            if min(len(s_i), len(s_j)) * 1.0 / max(len(s_i), len(s_j)) < threshold * 0.5:
                continue  # sizes too different for any high-Jaccard match
            jac = _jaccard(s_i, s_j)
            if jac >= threshold:
                pairs.append(
                    {
                        "id_a": a_i.get("id"),
                        "label_a": a_i.get("label"),
                        "id_b": a_j.get("id"),
                        "label_b": a_j.get("label"),
                        "jaccard": round(jac, 4),
                    }
                )
    pairs.sort(key=lambda r: -r["jaccard"])
    return pairs


# --- Main audit -------------------------------------------------------------


def audit(
    corpus_path: Path,
    threshold: float = 0.80,
    external_specs: list[str] | None = None,
    max_top_pairs: int = 50,
) -> dict:
    atks: list[dict] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                atks.append(json.loads(line))
            except Exception:
                continue
    n = len(atks)

    # Intra-corpus near-duplicates
    intra = _intra_corpus_pairs(atks, threshold)
    # Count attacks that participate in at least one near-dup pair
    flagged_ids: set[str] = set()
    for p in intra:
        flagged_ids.add(p["id_a"])
        flagged_ids.add(p["id_b"])

    out: dict = {
        "schema_version": 1,
        "corpus": str(corpus_path),
        "n_attacks": n,
        "threshold": threshold,
        "intra": {
            "n_pairs_above_threshold": len(intra),
            "n_attacks_in_pairs": len(flagged_ids),
            "rate_attacks_in_pairs": round(len(flagged_ids) / max(n, 1), 4),
            "top_pairs": intra[:max_top_pairs],
        },
        "external": {},
    }

    # External-corpus comparisons
    for spec in external_specs or []:
        label = spec.split(":")[0]
        try:
            their = _load_external(spec)
        except Exception as e:
            out["external"][label] = {"error": str(e)}
            continue
        matches = _max_match_per_attack(atks, their)
        hits = [m for m in matches if m["best_match_jaccard"] >= threshold]
        out["external"][label] = {
            "n_external_texts": len(their),
            "n_hits_above_threshold": len(hits),
            "contamination_rate": round(len(hits) / max(n, 1), 4),
            "top_hits": sorted(matches, key=lambda r: -r["best_match_jaccard"])[:max_top_pairs],
        }

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--corpus", default="results/confirmed_attacks.jsonl")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="Jaccard threshold for near-duplicate; lower = looser",
    )
    p.add_argument(
        "--external",
        action="append",
        default=[],
        help="External corpus spec: 'path.jsonl:field' or 'path.csv:csv:field' or 'path.txt'. Repeatable.",
    )
    p.add_argument("--top-n", type=int, default=50, dest="max_top_pairs")
    p.add_argument("--out", default="results/contamination_audit.json")
    p.add_argument(
        "--write-dedup-list",
        action="store_true",
        help="also write results/confirmed_attacks_dedup.json — "
        "one keep_id per near-duplicate cluster, plus the drop_ids "
        "the keep_id represents. Downstream consumers that want "
        "a dedup-reduced corpus should filter to the keep_ids.",
    )
    args = p.parse_args()

    corpus = ROOT / args.corpus
    if not corpus.exists():
        print(f"ERROR: corpus not found: {corpus}")
        return 2

    out = audit(
        corpus_path=corpus,
        threshold=args.threshold,
        external_specs=args.external,
        max_top_pairs=args.max_top_pairs,
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if args.write_dedup_list:
        # Union-find over intra-corpus near-duplicate pairs → one cluster id
        # per near-dup equivalence class. Keep the lex-smallest attack_id
        # as the canonical representative; the rest are drop candidates.
        parent: dict[str, str] = {}

        def _find(x: str) -> str:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for pair in out["intra"]["top_pairs"]:
            _union(pair["id_a"], pair["id_b"])
        # Expand beyond top_pairs (we currently cap top_pairs to max_top_pairs);
        # the full list is needed for a correct dedup. Recompute from full data:
        # we pass the already-computed pairs via `out["intra"]` which kept only
        # top_n. Fall back to re-running against the corpus at threshold if the
        # pair count was capped.
        if out["intra"]["n_pairs_above_threshold"] > len(out["intra"]["top_pairs"]):
            # Re-run quickly — cheap relative to n^2 we already paid.
            all_pairs = _intra_corpus_pairs(
                [json.loads(line) for line in corpus.open(encoding="utf-8")],
                threshold=args.threshold,
            )
            for pair in all_pairs:
                _union(pair["id_a"], pair["id_b"])

        # Build clusters
        from collections import defaultdict as dd

        clusters: dict[str, list[str]] = dd(list)
        for nid in list(parent):
            clusters[_find(nid)].append(nid)
        dedup_rows = []
        for keep, members in clusters.items():
            if len(members) <= 1:
                continue
            drops = sorted(m for m in members if m != keep)
            dedup_rows.append({"keep_id": keep, "drop_ids": drops, "cluster_size": len(members)})
        dedup_rows.sort(key=lambda r: -r["cluster_size"])
        dedup_path = ROOT / "results" / "confirmed_attacks_dedup.json"
        dedup_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_from": str(out_path),
                    "threshold": args.threshold,
                    "n_clusters": len(dedup_rows),
                    "n_keep_after_dedup": out["n_attacks"] - sum(len(r["drop_ids"]) for r in dedup_rows),
                    "clusters": dedup_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"dedup list: {dedup_path}  ({len(dedup_rows)} clusters, "
            f"reduces {out['n_attacks']} → "
            f"{out['n_attacks'] - sum(len(r['drop_ids']) for r in dedup_rows)} unique)"
        )

    # Console summary
    print(f"\n=== contamination audit — {corpus.name} (n={out['n_attacks']:,}) ===\n")
    intra = out["intra"]
    print(f"intra-corpus near-duplicates (Jaccard ≥ {args.threshold}):")
    print(f"  pairs: {intra['n_pairs_above_threshold']:,}")
    print(
        f"  attacks in pairs: {intra['n_attacks_in_pairs']:,} ({intra['rate_attacks_in_pairs'] * 100:.1f}% of corpus)"
    )
    if intra["top_pairs"]:
        print("\n  top 3 pairs:")
        for pair in intra["top_pairs"][:3]:
            print(
                f"    {pair['jaccard']:.3f}  {pair['id_a']} ({pair['label_a']})  ↔  {pair['id_b']} ({pair['label_b']})"
            )

    for label, ext in (out.get("external") or {}).items():
        if "error" in ext:
            print(f"\nexternal [{label}]: ERROR {ext['error']}")
            continue
        print(f"\nexternal [{label}]:  n_external={ext['n_external_texts']:,}")
        print(
            f"  hits above threshold: {ext['n_hits_above_threshold']:,} "
            f"({ext['contamination_rate'] * 100:.2f}% of our corpus)"
        )
        if ext.get("top_hits"):
            print("  top 3 most-similar:")
            for hit in ext["top_hits"][:3]:
                print(
                    f"    {hit['best_match_jaccard']:.3f}  "
                    f"our={hit['id']} ({hit['label']})  ≈  "
                    f"their={hit['best_match_preview'][:80]!r}"
                )

    print(f"\nfull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
