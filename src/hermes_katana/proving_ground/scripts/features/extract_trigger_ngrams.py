"""Extract distinctive n-gram triggers from confirmed vs rejected attacks.

Scanner feed #1. Produces three artifacts:

- `trigger_ngrams.json` — raw research output with per-n-gram statistics
  (doc-frequency in confirmed vs rejected, distinctiveness score). Used for
  inspection and for training the Aho-Corasick / bloom filter seed.
- `fast_patterns_extension.json` — drop-in addition for hermes-katana's
  `src/hermes_katana/scanner/data/fast_patterns.json`, grouped by "injection_phrase",
  "jailbreak_phrase", etc. depending on which attack labels the n-gram appeared in.
- `attack_seed_phrases_extension.json` — per-label list in the same shape as
  `src/hermes_katana/scabbard/data/attack_seed_phrases.json`.

Distinctiveness score for an n-gram g:
    score(g) = df_confirmed(g)/N_confirmed - alpha * df_rejected(g)/N_rejected
where alpha defaults to 1.0 — a gram that appears in every rejected attack as
often as in every confirmed one has score 0 and is useless.

The corpus we train on is tiny (15 confirmed, 13 rejected). That's fine because
the goal is *complementary* coverage — we're not trying to replace the curated
seed list, we're contributing phrases the curators didn't think of. Every phrase
proposed here should be human-reviewed before production use.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_WORD = re.compile(r"[\w']+", re.UNICODE)
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "you",
    "your",
    "yours",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "they",
    "them",
    "their",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def extract_ngrams(tokens: list[str], n_min: int = 3, n_max: int = 5) -> set[str]:
    """All n-grams of length n_min..n_max. Lowercased, whitespace-joined."""
    grams: set[str] = set()
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        for i in range(L - n + 1):
            grams.add(" ".join(tokens[i : i + n]))
    return grams


def _is_informative(ngram: str) -> bool:
    """Drop n-grams that are all stopwords or mostly non-alpha."""
    toks = ngram.split()
    if all(t in _STOPWORDS for t in toks):
        return False
    alpha = sum(1 for c in ngram if c.isalpha())
    return alpha >= max(5, len(ngram) // 2)


def load_attacks(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_triggers(
    confirmed: list[dict],
    rejected: list[dict],
    alpha: float = 1.0,
    min_score: float = 0.1,
    min_df_confirmed: int = 2,
) -> list[dict]:
    """Return list of n-grams with df stats + score, sorted high-to-low.

    Parameters chosen so that an n-gram must:
    - appear in at least `min_df_confirmed` confirmed attacks AND
    - have score >= `min_score` (distinctiveness threshold)
    """
    nc, nr = max(len(confirmed), 1), max(len(rejected), 1)

    df_conf: Counter[str] = Counter()
    df_rej: Counter[str] = Counter()
    label_by_ngram: dict[str, Counter[str]] = defaultdict(Counter)

    for atk in confirmed:
        toks = tokenize(atk.get("text", ""))
        grams = extract_ngrams(toks)
        for g in grams:
            if _is_informative(g):
                df_conf[g] += 1
                label_by_ngram[g][atk.get("label", "unknown")] += 1

    for atk in rejected:
        toks = tokenize(atk.get("text", ""))
        grams = extract_ngrams(toks)
        for g in grams:
            if _is_informative(g):
                df_rej[g] += 1

    out: list[dict] = []
    for g, dc in df_conf.items():
        if dc < min_df_confirmed:
            continue
        dr = df_rej.get(g, 0)
        score = dc / nc - alpha * dr / nr
        if score < min_score:
            continue
        # Pick the label this n-gram is most associated with.
        labels = label_by_ngram[g]
        dominant_label = labels.most_common(1)[0][0] if labels else "unknown"
        out.append(
            {
                "ngram": g,
                "length": len(g.split()),
                "df_confirmed": dc,
                "df_rejected": dr,
                "confirmed_rate": round(dc / nc, 4),
                "rejected_rate": round(dr / nr, 4),
                "score": round(score, 4),
                "dominant_label": dominant_label,
                "label_distribution": dict(labels),
            }
        )

    out.sort(key=lambda r: (-r["score"], -r["df_confirmed"]))
    return out


# Mapping from proving_ground attack labels to hermes-katana seed-file
# categories. When we ship a phrase, we bucket it into the right list.
_LABEL_TO_SEED_BUCKET = {
    "content_injection": "content_injection",
    "jailbreak": "jailbreak",
    "persona_jailbreak": "persona_jailbreak",
    "semantic_manipulation": "semantic_manipulation",
    "behavioral_control": "behavioral_control",
    "encoding_evasion": "encoding_evasion",
    "cognitive_state_attack": "cognitive_state_attack",
    "exfiltration_attempt": "exfiltration_attempt",
}
_LABEL_TO_FAST_BUCKET = {
    "content_injection": "injection_phrase",
    "jailbreak": "jailbreak_phrase",
    "persona_jailbreak": "jailbreak_phrase",
    "semantic_manipulation": "injection_phrase",
    "behavioral_control": "injection_phrase",
    "encoding_evasion": "injection_phrase",
    "cognitive_state_attack": "injection_phrase",
    "exfiltration_attempt": "injection_phrase",
}


def to_fast_patterns_extension(triggers: list[dict]) -> dict[str, list[str]]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for row in triggers:
        bucket = _LABEL_TO_FAST_BUCKET.get(row["dominant_label"], "injection_phrase")
        buckets[bucket].add(row["ngram"])
    return {k: sorted(v) for k, v in buckets.items()}


def to_attack_seed_phrases_extension(triggers: list[dict]) -> dict[str, list[str]]:
    buckets: dict[str, set[str]] = defaultdict(set)
    for row in triggers:
        bucket = _LABEL_TO_SEED_BUCKET.get(row["dominant_label"], row["dominant_label"])
        buckets[bucket].add(row["ngram"])
    return {k: sorted(v) for k, v in buckets.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--confirmed", default="results/confirmed_attacks.jsonl")
    p.add_argument("--rejected", default="results/rejected_attacks.jsonl")
    p.add_argument("--out-dir", default="results/scanner_feeds")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--min-score", type=float, default=0.1)
    p.add_argument("--min-df-confirmed", type=int, default=2)
    args = p.parse_args()

    confirmed = load_attacks(args.confirmed)
    rejected = load_attacks(args.rejected)
    print(f"Loaded {len(confirmed)} confirmed, {len(rejected)} rejected attacks")

    triggers = extract_triggers(
        confirmed,
        rejected,
        alpha=args.alpha,
        min_score=args.min_score,
        min_df_confirmed=args.min_df_confirmed,
    )
    print(f"Extracted {len(triggers)} distinctive n-grams")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Raw research output
    (out_dir / "trigger_ngrams.json").write_text(
        json.dumps(
            {
                "n_confirmed": len(confirmed),
                "n_rejected": len(rejected),
                "ngrams": triggers,
            },
            indent=2,
        )
    )
    # Drop-in for hermes-katana fast_patterns
    (out_dir / "fast_patterns_extension.json").write_text(json.dumps(to_fast_patterns_extension(triggers), indent=2))
    # Drop-in for hermes-katana attack_seed_phrases
    (out_dir / "attack_seed_phrases_extension.json").write_text(
        json.dumps(to_attack_seed_phrases_extension(triggers), indent=2)
    )
    print(f"Wrote artifacts to {out_dir}/")

    # Quick human-readable summary
    print("\nTop 15 triggers by score:")
    for row in triggers[:15]:
        print(
            f"  [{row['score']:>5.2f}] {row['dominant_label']:<25} df={row['df_confirmed']}c/{row['df_rejected']}r  '{row['ngram']}'"
        )


if __name__ == "__main__":
    main()
