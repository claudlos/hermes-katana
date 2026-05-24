"""Harness coverage matrix — who we've tested and what's missing.

Reads results/agent_shard_runs/*.jsonl, joins on research/harness_profiles.yaml,
aggregates by (model_family, harness_type) and (model_family × harness_type ×
channel). Reports:

  - Rate-table with Wilson CIs for every cell with n ≥ MIN_N.
  - Coverage gaps: (model_family, harness_type) cells present in one family
    but NOT this one, OR with n < MIN_N.
  - Suggestion for the next fleet campaign — which (model, harness, channel)
    gaps to prioritize.

Usage:
    python scripts/harness_matrix.py
    python scripts/harness_matrix.py --min-n 50 --apply-exclusion
    python scripts/harness_matrix.py --propose-fleet fleet_v13_gaps.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from hermes_katana.proving_ground.research.harness_profile import load_profiles  # noqa: E402
from hermes_katana.proving_ground.research.statistics import wilson_ci  # noqa: E402


SHARD_RUNS = ROOT / "results" / "agent_shard_runs"
EXCLUSION_LIST = ROOT / "results" / "exclusion_list.json"


def _load_exclusion() -> set[tuple]:
    if not EXCLUSION_LIST.exists():
        return set()
    d = json.loads(EXCLUSION_LIST.read_text(encoding="utf-8"))
    return {(r.get("agent_id"), r.get("shard"), r.get("channel"), r.get("attack_id")) for r in d.get("rows", [])}


def _stream_rows(apply_exclusion: bool):
    excl = _load_exclusion() if apply_exclusion else set()
    for p in sorted(SHARD_RUNS.glob("shard_*.jsonl")):
        if "_broken" in str(p):
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if excl:
                    k = (
                        r.get("agent_id"),
                        r.get("shard"),
                        r.get("channel"),
                        r.get("attack_id"),
                    )
                    if k in excl:
                        continue
                yield r


def build_matrix(apply_exclusion: bool) -> dict:
    profiles = load_profiles()
    # Aggregate
    by_cell: dict[tuple[str, str], Counter] = defaultdict(Counter)
    by_triple: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    by_agent: dict[str, Counter] = defaultdict(Counter)

    unknown_agents: set[str] = set()
    for r in _stream_rows(apply_exclusion):
        agent = r.get("agent_id")
        if not agent:
            continue
        ch = r.get("channel") or "?"
        p = profiles.get(agent)
        if p is None:
            unknown_agents.add(agent)
            continue
        cell = (p.model_family, p.harness_type)
        trip = (p.model_family, p.harness_type, ch)
        by_cell[cell]["n"] += 1
        by_triple[trip]["n"] += 1
        by_agent[agent]["n"] += 1
        if r.get("effective"):
            by_cell[cell]["eff"] += 1
            by_triple[trip]["eff"] += 1
            by_agent[agent]["eff"] += 1

    def _mk(c: Counter) -> dict:
        n = c["n"]
        eff = c["eff"]
        lo, hi = wilson_ci(eff, n) if n else (0, 0)
        return {
            "n": n,
            "eff": eff,
            "rate": round(eff / n, 4) if n else None,
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
        }

    result = {
        "schema_version": 1,
        "apply_exclusion": apply_exclusion,
        "by_cell": {f"{mf}|{ht}": _mk(c) for (mf, ht), c in by_cell.items()},
        "by_triple": {f"{mf}|{ht}|{ch}": _mk(c) for (mf, ht, ch), c in by_triple.items()},
        "by_agent": {a: {**_mk(c), "profile": profiles[a].__dict__} for a, c in by_agent.items() if a in profiles},
        "unknown_agents": sorted(unknown_agents),
    }
    return result


def identify_gaps(result: dict, min_n: int = 50) -> list[dict]:
    """A gap is (model_family, harness_type) combination that has n < min_n."""
    profiles = load_profiles()
    present: dict[str, set[str]] = defaultdict(set)  # mf -> {ht, ...}
    coverage = {}
    for key, stats in result["by_cell"].items():
        mf, ht = key.split("|", 1)
        coverage[(mf, ht)] = stats["n"]
        if stats["n"] >= min_n:
            present[mf].add(ht)

    all_mfs = {p.model_family for p in profiles.values()}
    all_hts = {p.harness_type for p in profiles.values()}

    gaps: list[dict] = []
    for mf in all_mfs:
        for ht in all_hts:
            n = coverage.get((mf, ht), 0)
            if n < min_n:
                # Is there any agent in our profiles that could fill this cell?
                candidates = [p.agent_id for p in profiles.values() if p.model_family == mf and p.harness_type == ht]
                gaps.append(
                    {
                        "model_family": mf,
                        "harness_type": ht,
                        "n_current": n,
                        "candidate_agents": candidates,
                        "covered": bool(candidates),
                    }
                )
    gaps.sort(key=lambda g: (not g["candidate_agents"], g["model_family"], g["harness_type"]))
    return gaps


def _print_matrix(result: dict, min_n: int) -> None:
    profiles = load_profiles()
    mfs = sorted({p.model_family for p in profiles.values()})
    hts = sorted({p.harness_type for p in profiles.values()})

    # Header
    print(f"\n{'mf/ht':<14}", end="")
    for ht in hts:
        print(f"{ht[:16]:>18}", end="")
    print()
    for mf in mfs:
        print(f"{mf:<14}", end="")
        for ht in hts:
            key = f"{mf}|{ht}"
            s = result["by_cell"].get(key)
            if not s or s["n"] < min_n:
                cell = f"{'-':>18}"
            else:
                cell = f"{s['rate'] * 100:>5.1f}% ({s['n']:>4})"
                cell = f"{cell:>18}"
            print(cell, end="")
        print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--min-n", type=int, default=50)
    p.add_argument(
        "--apply-exclusion",
        action="store_true",
        help="drop rows in results/exclusion_list.json (parser-audit flagged)",
    )
    p.add_argument("--out", default="results/harness_matrix.json")
    p.add_argument(
        "--propose-fleet",
        default=None,
        help="write a fleet spec JSON targeting high-priority gaps",
    )
    args = p.parse_args()

    print("[harness-matrix] streaming rows ...")
    result = build_matrix(args.apply_exclusion)
    gaps = identify_gaps(result, min_n=args.min_n)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**result, "gaps": gaps}, indent=2), encoding="utf-8")

    print(f"\n=== harness coverage matrix (min_n={args.min_n}, exclusion={args.apply_exclusion}) ===")
    _print_matrix(result, args.min_n)
    if result["unknown_agents"]:
        print("\n⚠ agents in data but NOT in research/harness_profiles.yaml:")
        for a in result["unknown_agents"]:
            print(f"  - {a}")

    print("\n--- coverage gaps ---")
    actionable = [g for g in gaps if g["candidate_agents"]]
    missing = [g for g in gaps if not g["candidate_agents"]]
    print(f"{len(actionable)} cells are actionable (we have agent drivers; just need runs)")
    print(f"{len(missing)} cells have NO agent driver — would need new integration")
    for g in actionable[:10]:
        print(
            f"  {g['model_family']:<10} × {g['harness_type']:<18} "
            f"n={g['n_current']:<5} candidates={g['candidate_agents']}"
        )

    if args.propose_fleet and actionable:
        spec = _propose_fleet_spec(actionable)
        fp = ROOT / args.propose_fleet
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"\nproposed fleet spec: {fp}")

    print(f"\nfull report: {out_path}")
    return 0


def _propose_fleet_spec(actionable_gaps: list[dict]) -> dict:
    """Emit a fleet.json-compatible spec covering the highest-priority gaps.

    Strategy: one worker per candidate agent across all 4 channels × a sample
    of shards (001-010 English + 101-110 multilingual). max_attacks=50 keeps
    cost modest while establishing coverage.
    """
    workers = []
    for gap in actionable_gaps[:15]:  # top 15 gaps
        for agent in gap["candidate_agents"][:1]:  # first candidate per cell
            workers.append(
                {
                    "agent": agent,
                    "shards": list(range(1, 11)) + list(range(101, 111)),
                    "channels": [
                        "file_content",
                        "code_comment",
                        "tool_output",
                        "data_row",
                    ],
                    "max_attacks": 50,
                    "instances": 1,
                }
            )
    return {
        "_comment": (
            "Auto-generated gap-filling fleet. Targets (model_family, "
            "harness_type) cells below min_n. Edit before launching."
        ),
        "max_concurrency": 20,
        "workers": workers,
    }


if __name__ == "__main__":
    sys.exit(main())
