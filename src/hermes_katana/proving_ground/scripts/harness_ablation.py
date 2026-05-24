"""Harness-ablation framework — paired same-model-different-harness analysis.

The S-tier publishable claim (per Phase 2 research audit):

  "Holding the underlying model constant, injection effectiveness varies
   more by harness than by model family."

A proper test is PAIRED: same attack_id, same channel, run under two
harnesses. Per-attack outcome on each harness → McNemar's test on the
binary discordance.

This script:
  1. Streams results/agent_shard_runs/*.jsonl.
  2. Builds: (attack_id, channel) → {harness_A: effective, harness_B: effective}.
  3. Keeps only keys present under BOTH harnesses.
  4. Aggregates per-harness outcomes with ANY-effective-across-runs (logical OR).
     This is intentionally conservative — if an attack fired even once under
     harness X, count it as "effective under X" for pairing purposes.
  5. Computes McNemar's exact / chi-square test (b, c discordants) with Wilson
     CIs on rates, Cohen's h on the paired difference.
  6. Optionally emits a kernel Claim that can resolve the preregistered
     hypothesis H-20260422-harness-dominates-model.

Usage:

    # Default harness pair: claude_code_cli vs hermes_claude_haiku
    python scripts/harness_ablation.py

    # Custom harness pair
    python scripts/harness_ablation.py \\
        --harness-a 'claude_cli,claude_cli_haiku,claude_cli_sonnet' \\
        --harness-b 'hermes_claude_haiku' \\
        --label-a claude_code_cli --label-b hermes

    # Write a kernel Claim (resolves the hypothesis if evidence is strong)
    python scripts/harness_ablation.py --submit-to-kernel --run-id ablation-001
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from hermes_katana.proving_ground.research.statistics import mcnemar, wilson_ci, cohens_h, paired_bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# Pair assembly
# ---------------------------------------------------------------------------


def _iter_rows(agent_ids: set[str]):
    """Stream only rows where agent_id is in the given set."""
    shard_dir = ROOT / "results" / "agent_shard_runs"
    if not shard_dir.exists():
        return
    for p in sorted(shard_dir.glob("shard_*.jsonl")):
        if "_broken" in str(p):
            continue
        # Filename fast-filter: skip file if no agent substring matches.
        name = p.name
        if not any(f"_{a}_" in name or name.endswith(f"_{a}.jsonl") for a in agent_ids):
            continue
        try:
            with p.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("agent_id") in agent_ids:
                        yield row
        except FileNotFoundError:
            continue


def _build_per_harness_outcomes(agent_ids: set[str]) -> dict[tuple[str, str], bool]:
    """(attack_id, channel) -> True if ever effective under this harness."""
    out: dict[tuple[str, str], bool] = {}
    for row in _iter_rows(agent_ids):
        aid = row.get("attack_id")
        ch = row.get("channel")
        if not aid or not ch:
            continue
        key = (aid, ch)
        if row.get("effective"):
            out[key] = True
        elif key not in out:
            out[key] = False
    return out


def _pair(
    out_a: dict[tuple[str, str], bool],
    out_b: dict[tuple[str, str], bool],
) -> list[tuple[str, str, bool, bool]]:
    both = set(out_a) & set(out_b)
    return [(aid, ch, out_a[(aid, ch)], out_b[(aid, ch)]) for (aid, ch) in sorted(both)]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def _mcnemar_table(pairs: list[tuple[str, str, bool, bool]]) -> dict:
    """Returns the full 2x2 + McNemar stats."""
    a_only = b_only = both = neither = 0
    for _aid, _ch, aa, bb in pairs:
        if aa and bb:
            both += 1
        elif aa and not bb:
            a_only += 1
        elif bb and not aa:
            b_only += 1
        else:
            neither += 1
    n = len(pairs)
    rate_a = (both + a_only) / n if n else 0.0
    rate_b = (both + b_only) / n if n else 0.0
    ci_a = wilson_ci(both + a_only, n)
    ci_b = wilson_ci(both + b_only, n)
    # McNemar: discordants are (A-only, B-only) = (b, c)
    p_val = mcnemar(a_only, b_only) if (a_only + b_only) else 1.0
    h = cohens_h(rate_a, rate_b)
    # Paired bootstrap CI on the per-pair delta (a_eff - b_eff)
    deltas = [int(aa) - int(bb) for _aid, _ch, aa, bb in pairs]
    b_vals = [0] * n
    # We want CI on mean(deltas). paired_bootstrap_ci takes (a, b) and bootstraps
    # a_i - b_i; so pass (deltas, zeros).
    ci_delta_lo, ci_delta_hi = paired_bootstrap_ci(deltas, b_vals, iters=5000, seed=7)
    return {
        "n_pairs": n,
        "table": {"both": both, "a_only": a_only, "b_only": b_only, "neither": neither},
        "rate_a": round(rate_a, 4),
        "rate_a_ci": [round(ci_a[0], 4), round(ci_a[1], 4)],
        "rate_b": round(rate_b, 4),
        "rate_b_ci": [round(ci_b[0], 4), round(ci_b[1], 4)],
        "delta_ab": round(rate_a - rate_b, 4),
        "delta_ci": [round(ci_delta_lo, 4), round(ci_delta_hi, 4)],
        "cohens_h": round(h, 4),
        "mcnemar_b": a_only,
        "mcnemar_c": b_only,
        "mcnemar_p": round(p_val, 6),
    }


# ---------------------------------------------------------------------------
# Channel stratification
# ---------------------------------------------------------------------------


def _per_channel_breakdown(pairs: list[tuple[str, str, bool, bool]]) -> dict:
    by_channel: dict[str, list[tuple[str, str, bool, bool]]] = defaultdict(list)
    for aid, ch, aa, bb in pairs:
        by_channel[ch].append((aid, ch, aa, bb))
    return {ch: _mcnemar_table(prs) for ch, prs in sorted(by_channel.items()) if len(prs) >= 30}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--harness-a",
        default="claude_cli,claude_cli_haiku,claude_cli_sonnet",
        help="comma-sep agent_ids making up harness A",
    )
    p.add_argument(
        "--harness-b",
        default="hermes_claude_haiku",
        help="comma-sep agent_ids making up harness B",
    )
    p.add_argument("--label-a", default="claude_code_cli", help="human label for harness A")
    p.add_argument("--label-b", default="hermes", help="human label for harness B")
    p.add_argument("--out", default="results/harness_ablation.json")
    p.add_argument(
        "--submit-to-kernel",
        action="store_true",
        help="if evidence is strong, submit a Claim to ResearchKernel and resolve H-harness-dominates-model",
    )
    p.add_argument("--run-id", default=None, help="kernel run_id (if --submit-to-kernel)")
    args = p.parse_args()

    a_ids = {a.strip() for a in args.harness_a.split(",") if a.strip()}
    b_ids = {b.strip() for b in args.harness_b.split(",") if b.strip()}
    print(f"[harness-ablation] A={args.label_a} = {sorted(a_ids)}")
    print(f"[harness-ablation] B={args.label_b} = {sorted(b_ids)}")

    print("[harness-ablation] building outcome maps ...")
    out_a = _build_per_harness_outcomes(a_ids)
    out_b = _build_per_harness_outcomes(b_ids)
    print(f"  A: {len(out_a):,} (attack_id, channel) pairs with ≥1 row")
    print(f"  B: {len(out_b):,} (attack_id, channel) pairs with ≥1 row")

    pairs = _pair(out_a, out_b)
    print(f"  paired both-sides:  {len(pairs):,}")

    if not pairs:
        print(
            "NO PAIRED DATA. Neither harness has overlap with the other on "
            "(attack_id, channel) cells. Widen your harness sets or run more "
            "matched campaigns."
        )
        return 1

    overall = _mcnemar_table(pairs)
    by_channel = _per_channel_breakdown(pairs)

    result = {
        "schema_version": 1,
        "harness_a": {"label": args.label_a, "agent_ids": sorted(a_ids)},
        "harness_b": {"label": args.label_b, "agent_ids": sorted(b_ids)},
        "overall": overall,
        "per_channel": by_channel,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\n=== harness-ablation — paired n={overall['n_pairs']:,} ===\n")
    t = overall["table"]
    print(f"  both-effective     : {t['both']:>6}")
    print(f"  {args.label_a}-only : {t['a_only']:>6}   (McNemar b)")
    print(f"  {args.label_b}-only : {t['b_only']:>6}   (McNemar c)")
    print(f"  neither           : {t['neither']:>6}")
    print("")
    print(
        f"  rate({args.label_a}) = {overall['rate_a'] * 100:5.2f}%  "
        f"CI [{overall['rate_a_ci'][0] * 100:.2f}%, {overall['rate_a_ci'][1] * 100:.2f}%]"
    )
    print(
        f"  rate({args.label_b}) = {overall['rate_b'] * 100:5.2f}%  "
        f"CI [{overall['rate_b_ci'][0] * 100:.2f}%, {overall['rate_b_ci'][1] * 100:.2f}%]"
    )
    print(
        f"  delta (A-B)      = {overall['delta_ab'] * 100:+5.2f}pp  "
        f"CI [{overall['delta_ci'][0] * 100:+.2f}pp, {overall['delta_ci'][1] * 100:+.2f}pp]"
    )
    print(f"  Cohen's h         = {overall['cohens_h']:+.3f}")
    print(f"  McNemar p-value   = {overall['mcnemar_p']:.6f}")

    if by_channel:
        print("\n--- per-channel (min-n=30) ---")
        print(f"{'channel':<14} {'n':>6} {'rate_A':>7} {'rate_B':>7} {'delta':>8} {'h':>6} {'p':>9}")
        for ch, st in by_channel.items():
            print(
                f"{ch:<14} {st['n_pairs']:>6} "
                f"{st['rate_a'] * 100:>6.2f}% "
                f"{st['rate_b'] * 100:>6.2f}% "
                f"{st['delta_ab'] * 100:+7.2f}pp "
                f"{st['cohens_h']:+5.2f} "
                f"{st['mcnemar_p']:.6f}"
            )

    # Submit to ResearchKernel if requested and evidence warrants
    if args.submit_to_kernel:
        from hermes_katana.proving_ground.research.kernel import ResearchKernel
        from hermes_katana.proving_ground.research.rigor import Claim
        from hermes_katana.proving_ground.research.events import Observation

        run_id = args.run_id or f"ablation-{args.label_a}-vs-{args.label_b}"
        k = ResearchKernel.build(run_id=run_id)

        claim = Claim(
            hypothesis_id="H-20260422-harness-dominates-model",
            primary_outcome="effective_rate_delta",
            value=overall["delta_ab"],
            n_samples=overall["n_pairs"],
            ci=tuple(overall["delta_ci"]),
            baseline_run_id=f"harness-b:{args.label_b}",
            comparison_run_id=f"harness-a:{args.label_a}",
            test_kind="mcnemar",
            p_value=overall["mcnemar_p"],
            effect_size={"kind": "cohens_h", "value": overall["cohens_h"]},
            meta={
                "harness_a": args.label_a,
                "harness_b": args.label_b,
                "agent_ids_a": sorted(a_ids),
                "agent_ids_b": sorted(b_ids),
                "table": overall["table"],
            },
        )
        supporting = [Observation(source="harness_ablation", data=result)]
        res = k.submit_claim(claim, supporting)
        print(f"\n[kernel] claim → {res.kind} (run_id={run_id})")

        # If p < 0.05 AND the direction matches the hypothesis prediction, resolve.
        from hermes_katana.proving_ground.research.registry import HypothesisRegistry

        reg = HypothesisRegistry()
        h = reg.load("H-20260422-harness-dominates-model")
        if h.status == "preregistered":
            predicted_less = h.predicted_direction == "less"
            direction_ok = (overall["delta_ab"] < 0) if predicted_less else (overall["delta_ab"] > 0)
            if overall["mcnemar_p"] < 0.05 and direction_ok and overall["n_pairs"] >= h.min_n_per_condition:
                reg.resolve(
                    h.id,
                    run_id=run_id,
                    p_value=overall["mcnemar_p"],
                    effect_size={"kind": "cohens_h", "value": overall["cohens_h"]},
                    verdict="supported",
                    notes=f"paired n={overall['n_pairs']}, delta={overall['delta_ab'] * 100:+.2f}pp",
                )
                print("[kernel] hypothesis RESOLVED: supported")
            elif overall["mcnemar_p"] < 0.05 and not direction_ok:
                reg.resolve(
                    h.id,
                    run_id=run_id,
                    p_value=overall["mcnemar_p"],
                    effect_size={"kind": "cohens_h", "value": overall["cohens_h"]},
                    verdict="rejected",
                    notes=f"significant but WRONG direction: delta={overall['delta_ab'] * 100:+.2f}pp",
                )
                print("[kernel] hypothesis RESOLVED: rejected (wrong direction)")

    print(f"\nfull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
