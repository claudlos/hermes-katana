"""Simulate Hermes Katana as defense-in-harness.

Instead of treating Katana as a post-hoc classifier (which we did in
detection_bench.py), this simulates what happens if Katana sits BETWEEN
the environment and the agent: every incoming attack text is scanned, and
if Katana's score ≥ threshold, the agent NEVER SEES the attack. The
attack is counted as blocked (effective=False for the defended harness).

Input:  every (attack_id, agent_id, channel, effective) row in
        results/agent_shard_runs/*.jsonl
Output: counterfactual effective rate under defense, per-agent + per-
        channel breakdown, and a deployable "defense block rate" at
        configurable thresholds.

For each row:
  1. Reconstruct the channel-wrapped text the agent would have seen.
  2. Score via hermes_katana.scanner.injection.injection_score.
  3. If score ≥ threshold, count as BLOCKED regardless of original
     effective status. Otherwise, keep original effective.

Metrics:
  base_rate         = rate(effective=True) in the corpus
  defended_rate     = rate(effective=True AND score<threshold)
  block_rate        = rate(score>=threshold)  — incl. benign → FPR proxy
  defense_recall    = rate(score>=threshold | effective=True)
                       = chance Katana would have caught each real attack
  unnecessary_block = rate(score>=threshold | effective=False)
                       = chance Katana would have blocked a benign call
                         (proxy FPR)

Output: results/katana_defense_simulation.json

Usage:
    python scripts/simulate_katana_defense.py
    python scripts/simulate_katana_defense.py --threshold 0.7
    python scripts/simulate_katana_defense.py --apply-exclusion
    python scripts/simulate_katana_defense.py --max-rows 50000  # speed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from hermes_katana.proving_ground.research.statistics import wilson_ci, cohens_h  # noqa: E402

# Import channel wrapping from detection_bench (single source of truth)
_spec = None
try:
    from hermes_katana.proving_ground.scripts.detection_bench import wrap_for_channel
except ImportError:
    # Allow direct execution: scripts/ is not a package, so load by path
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("_dbench", ROOT / "scripts" / "detection_bench.py")
    _mod = _iu.module_from_spec(spec)
    spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    wrap_for_channel = _mod.wrap_for_channel


SHARD_RUNS = ROOT / "results" / "agent_shard_runs"
EXCLUSION_LIST = ROOT / "results" / "exclusion_list.json"


def _exclusion_keys() -> set[tuple]:
    if not EXCLUSION_LIST.exists():
        return set()
    d = json.loads(EXCLUSION_LIST.read_text(encoding="utf-8"))
    return {(r.get("agent_id"), r.get("shard"), r.get("channel"), r.get("attack_id")) for r in d.get("rows", [])}


def _load_attack_texts() -> dict[str, str]:
    """Map attack_id → attack text. Uses confirmed + rejected + provisional."""
    out: dict[str, str] = {}
    for fname in (
        "confirmed_attacks.jsonl",
        "rejected_attacks.jsonl",
        "provisional_attacks.jsonl",
    ):
        p = ROOT / "results" / fname
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                aid = d.get("id")
                txt = d.get("text")
                if aid and txt and aid not in out:
                    out[aid] = txt
    # Fallback: try to pull text from shards/ if missing
    for p in (ROOT / "shards").glob("shard_*.jsonl"):
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                aid = d.get("id")
                txt = d.get("text")
                if aid and txt and aid not in out:
                    out[aid] = txt
    return out


def _stream_rows(apply_exclusion: bool):
    excl = _exclusion_keys() if apply_exclusion else set()
    for p in sorted(SHARD_RUNS.glob("shard_*.jsonl")):
        if "_broken" in str(p):
            continue
        try:
            with p.open(encoding="utf-8") as f:
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
        except FileNotFoundError:
            continue


def simulate(threshold: float, apply_exclusion: bool, max_rows: int | None) -> dict:
    from hermes_katana.scanner.injection import injection_score

    attack_texts = _load_attack_texts()
    if not attack_texts:
        raise SystemExit("No attack texts found. Regenerate confirmed_attacks.jsonl first.")

    print(f"[sim] attack text lookup: {len(attack_texts):,} attacks")

    # Cache (attack_id, channel) → score so we scan each unique wrapping once
    score_cache: dict[tuple[str, str], float] = {}

    def _get_score(aid: str, channel: str) -> float:
        k = (aid, channel)
        s = score_cache.get(k)
        if s is not None:
            return s
        text = attack_texts.get(aid)
        if text is None:
            score_cache[k] = 0.0
            return 0.0
        wrapped = wrap_for_channel(text, channel)
        s = float(injection_score(wrapped))
        score_cache[k] = s
        return s

    t0 = time.time()
    rows_seen = 0
    missing_text = 0

    # Aggregate by (agent, channel) cells
    cell_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {
            "n": 0,
            "eff": 0,
            "blocked": 0,
            "eff_unblocked": 0,
            "eff_blocked": 0,
            "ineff_blocked": 0,
        }
    )
    overall = {
        "n": 0,
        "eff": 0,
        "blocked": 0,
        "eff_unblocked": 0,
        "eff_blocked": 0,
        "ineff_blocked": 0,
    }

    for row in _stream_rows(apply_exclusion):
        aid = row.get("attack_id")
        ch = row.get("channel")
        ag = row.get("agent_id")
        if not (aid and ch and ag):
            continue
        if aid not in attack_texts:
            missing_text += 1
            continue
        rows_seen += 1
        if max_rows and rows_seen > max_rows:
            break

        s = _get_score(aid, ch)
        blocked = s >= threshold
        eff = bool(row.get("effective"))

        cell = cell_stats[(ag, ch)]
        cell["n"] += 1
        cell["eff"] += int(eff)
        cell["blocked"] += int(blocked)
        if eff and not blocked:
            cell["eff_unblocked"] += 1
        if eff and blocked:
            cell["eff_blocked"] += 1
        if not eff and blocked:
            cell["ineff_blocked"] += 1

        overall["n"] += 1
        overall["eff"] += int(eff)
        overall["blocked"] += int(blocked)
        if eff and not blocked:
            overall["eff_unblocked"] += 1
        if eff and blocked:
            overall["eff_blocked"] += 1
        if not eff and blocked:
            overall["ineff_blocked"] += 1

        if rows_seen % 20000 == 0:
            print(f"  {rows_seen:>6} rows scanned  ({time.time() - t0:.0f}s)")

    print(
        f"[sim] scanned {rows_seen:,} rows, "
        f"unique (attack,channel) scored: {len(score_cache):,}, "
        f"missing-text skipped: {missing_text:,}"
    )

    def _mk(c: dict) -> dict:
        n = c["n"]
        eff_rate = c["eff"] / n if n else 0
        def_rate = c["eff_unblocked"] / n if n else 0
        block_rate = c["blocked"] / n if n else 0
        lo_eff, hi_eff = wilson_ci(c["eff"], n) if n else (0, 0)
        lo_def, hi_def = wilson_ci(c["eff_unblocked"], n) if n else (0, 0)
        recall = c["eff_blocked"] / c["eff"] if c["eff"] else 0
        unnecessary = c["ineff_blocked"] / (n - c["eff"]) if (n - c["eff"]) else 0
        return {
            "n": n,
            "n_effective": c["eff"],
            "n_blocked": c["blocked"],
            "base_rate": round(eff_rate, 4),
            "base_rate_ci": [round(lo_eff, 4), round(hi_eff, 4)],
            "defended_rate": round(def_rate, 4),
            "defended_rate_ci": [round(lo_def, 4), round(hi_def, 4)],
            "block_rate": round(block_rate, 4),
            "defense_recall": round(recall, 4),  # TP rate on real attacks
            "unnecessary_block_rate": round(unnecessary, 4),
            "delta": round(eff_rate - def_rate, 4),
            "cohens_h": round(cohens_h(eff_rate, def_rate), 4),
        }

    return {
        "schema_version": 1,
        "threshold": threshold,
        "apply_exclusion": apply_exclusion,
        "overall": _mk(overall),
        "per_cell": {f"{a}|{ch}": _mk(c) for (a, ch), c in cell_stats.items()},
        "rows_scanned": rows_seen,
        "unique_wrappings_scored": len(score_cache),
    }


def simulate_sweep(thresholds: list[float], apply_exclusion: bool, max_rows: int | None) -> dict:
    """Efficient multi-threshold sweep: score each (attack_id, channel) ONCE,
    then derive per-threshold metrics by varying the cutoff."""
    from hermes_katana.scanner.injection import injection_score

    attack_texts = _load_attack_texts()
    print(f"[sweep] attack text lookup: {len(attack_texts):,}")
    score_cache: dict[tuple[str, str], float] = {}

    t0 = time.time()
    # Collect all (score, eff) pairs once
    scored: list[tuple[float, bool]] = []
    rows_seen = 0
    for row in _stream_rows(apply_exclusion):
        aid = row.get("attack_id")
        ch = row.get("channel")
        if not (aid and ch):
            continue
        if aid not in attack_texts:
            continue
        rows_seen += 1
        if max_rows and rows_seen > max_rows:
            break
        k = (aid, ch)
        s = score_cache.get(k)
        if s is None:
            wrapped = wrap_for_channel(attack_texts[aid], ch)
            s = float(injection_score(wrapped))
            score_cache[k] = s
        scored.append((s, bool(row.get("effective"))))
        if rows_seen % 20000 == 0:
            print(f"  {rows_seen:>6} rows scanned  ({time.time() - t0:.0f}s)")
    print(f"[sweep] rows={rows_seen:,}  unique_scored={len(score_cache):,}")

    results_per_thr: list[dict] = []
    for t in thresholds:
        n = len(scored)
        eff = sum(1 for s, e in scored if e)
        sum(1 for s, e in scored if s >= t)
        eff_unblocked = sum(1 for s, e in scored if e and s < t)
        eff_blocked = sum(1 for s, e in scored if e and s >= t)
        ineff_blocked = sum(1 for s, e in scored if not e and s >= t)
        ineff_n = n - eff
        base_rate = eff / n if n else 0
        def_rate = eff_unblocked / n if n else 0
        lo_def, hi_def = wilson_ci(eff_unblocked, n) if n else (0, 0)
        recall = eff_blocked / eff if eff else 0
        fpr = ineff_blocked / ineff_n if ineff_n else 0
        results_per_thr.append(
            {
                "threshold": t,
                "n": n,
                "base_rate": round(base_rate, 4),
                "defended_rate": round(def_rate, 4),
                "defended_rate_ci": [round(lo_def, 4), round(hi_def, 4)],
                "delta_pp": round((base_rate - def_rate) * 100, 2),
                "defense_recall": round(recall, 4),
                "false_block_rate": round(fpr, 4),  # proxy FPR
                "cohens_h": round(cohens_h(base_rate, def_rate), 4),
            }
        )
    return {
        "schema_version": 1,
        "sweep": results_per_thr,
        "rows_scanned": rows_seen,
        "unique_wrappings_scored": len(score_cache),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--apply-exclusion", action="store_true")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--out", default="results/katana_defense_simulation.json")
    p.add_argument(
        "--sweep",
        action="store_true",
        help="threshold sweep (0.1..0.9) to find the deployable operating point (defense recall vs. false block rate)",
    )
    args = p.parse_args()

    if args.sweep:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
        res = simulate_sweep(
            thresholds=thresholds,
            apply_exclusion=args.apply_exclusion,
            max_rows=args.max_rows,
        )
        out_path = ROOT / args.out.replace(".json", "_sweep.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"\n=== Katana defense threshold sweep (n={res['rows_scanned']:,}) ===")
        print(f"{'thresh':>7} {'base':>7} {'defended':>9} {'Δpp':>7} {'recall':>8} {'false_block':>12}")
        for r in res["sweep"]:
            print(
                f"  {r['threshold']:>5.2f} {r['base_rate'] * 100:>6.2f}% "
                f"{r['defended_rate'] * 100:>8.2f}% {r['delta_pp']:+7.2f} "
                f"{r['defense_recall'] * 100:>7.2f}% "
                f"{r['false_block_rate'] * 100:>11.2f}%"
            )
        print(f"\nfull: {out_path}")
        return 0

    res = simulate(
        threshold=args.threshold,
        apply_exclusion=args.apply_exclusion,
        max_rows=args.max_rows,
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")

    o = res["overall"]
    print("\n=== Katana defense-in-harness simulation ===")
    print(f"threshold={args.threshold}  rows={res['rows_scanned']:,}")
    print("")
    print(
        f"  base_rate         (without defense):   {o['base_rate'] * 100:5.2f}%  "
        f"CI [{o['base_rate_ci'][0] * 100:.2f}%, {o['base_rate_ci'][1] * 100:.2f}%]"
    )
    print(
        f"  defended_rate     (with Katana):       {o['defended_rate'] * 100:5.2f}%  "
        f"CI [{o['defended_rate_ci'][0] * 100:.2f}%, {o['defended_rate_ci'][1] * 100:.2f}%]"
    )
    print(f"  delta            = {o['delta'] * 100:+5.2f}pp   Cohen's h = {o['cohens_h']:+.3f}")
    print(f"  block_rate                                    {o['block_rate'] * 100:5.2f}%")
    print(f"  defense_recall   (real attacks Katana caught): {o['defense_recall'] * 100:5.2f}%")
    print(f"  unnecessary_block (non-attack blocks):        {o['unnecessary_block_rate'] * 100:5.2f}%")

    # Top 10 (agent, channel) cells with biggest defense delta (>= 30 rows)
    cells = [(k, v) for k, v in res["per_cell"].items() if v["n"] >= 30]
    cells.sort(key=lambda kv: -kv[1]["delta"])
    print("\n--- top 10 cells by defense improvement (Δ = base - defended) ---")
    print(f"{'agent|channel':<46} {'n':>6} {'base':>7} {'defended':>9} {'Δ':>7} {'recall':>8}")
    for k, v in cells[:10]:
        print(
            f"  {k:<44} {v['n']:>6} "
            f"{v['base_rate'] * 100:>6.2f}% {v['defended_rate'] * 100:>8.2f}% "
            f"{v['delta'] * 100:+6.2f}pp {v['defense_recall'] * 100:>7.2f}%"
        )

    print(f"\nfull report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
