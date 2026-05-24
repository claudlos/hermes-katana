"""Analyze a fleet run against synth shards (200..222).

Joins per-attack effective signals from `results/agent_shard_runs/*.jsonl`
with the synth provenance carried in `shards/shard_2NN.jsonl` (each row's
`source` field names the synth run: katana_synth_v1, ..._v4_persona, etc).

Outputs:
  - per-(source) effective rate
  - per-(label) effective rate
  - per-(source, label) breakdown (small grid)
  - top-severity attacks (likely best training rows)
  - "dead" rows (severity == 0 across all observers): trim candidates

Usage:
    python scripts/analyze_synth_fleet.py --run-id smoke_2026_04_26
    python scripts/analyze_synth_fleet.py                        # all rows
    python scripts/analyze_synth_fleet.py --json                 # machine output

Notes:
  This is a one-model summary unless multiple agents ran the same shards
  in the same run. For per-attack confirmation across models / platforms
  use scripts/cross_reference_confirm.py instead — that's the canonical
  "is this attack real" pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
SHARDS = ROOT / "shards"


def _load_shard_provenance() -> dict[str, dict]:
    """attack_id → {source, label, text_length, language?, original_atk_id?}.

    Loads ALL shard ranges:
      - 1-122   empirical (data_v3)
      - 200-222 synth (Simula-style)
      - 300-321 multilingual factory (random sample)
      - 400-523 recovery (English supplement)
      - 600+    multilingual back-trace
    Originally only loaded shard_2*; widened 2026-05-01 to include
    recovery + back-trace ranges so analyze runs on the new corpus.
    """
    out: dict[str, dict] = {}
    for shard_file in sorted(SHARDS.glob("shard_*.jsonl")):
        with shard_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                aid = r.get("id")
                if not aid:
                    continue
                out[aid] = {
                    "source": r.get("source", "unknown"),
                    "label": r.get("label", ""),
                    "text_length": r.get("text_length", 0),
                    # New extension fields — present on multilingual / recovery
                    # rows, absent on classic shards 1-222. Carriers for
                    # downstream per-language / per-recovery-source analysis.
                    "language": r.get("language", ""),
                    "original_atk_id": r.get("original_atk_id", ""),
                    "recovery_origin": r.get("recovery_origin", ""),
                    "shard_origin": r.get("shard_origin", ""),
                }
    return out


def _is_broken_runner_row(r: dict) -> bool:
    """Detect the CCLI broken-runner pattern: exit≠0, ≤6kB stdout, <5s, 0 tools.

    Caused by Claude Max quota burn during fleet runs; the CLI emits just
    the system-init event and exits. Rows matching this pattern carry no
    real signal — exclude from analysis.
    """
    ar = r.get("attack_run") or {}
    return (
        ar.get("exit_code", 0) != 0
        and ar.get("output_chars", 0) <= 6000
        and ar.get("tool_call_count", 0) == 0
        and (ar.get("duration_sec") or 0) < 5.0
    )


def _iter_results(run_id: str | None, include_broken: bool = False):
    for f in sorted(AGENT_RUNS.glob("shard_*.jsonl")):
        with f.open(encoding="utf-8") as fp:
            for line in fp:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if run_id and r.get("run_id") != run_id:
                    continue
                if (not include_broken) and _is_broken_runner_row(r):
                    continue
                yield r


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  -  "
    return f"{100.0 * num / denom:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--agent", default=None, help="restrict to one agent_id")
    ap.add_argument("--top", type=int, default=10, help="how many top-severity rows to print")
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--include-broken",
        action="store_true",
        help="include rows matching the CCLI broken-runner pattern. "
        "Default skips them (they reflect quota burns, not real drift).",
    )
    args = ap.parse_args()

    prov = _load_shard_provenance()

    by_source = defaultdict(lambda: {"n": 0, "eff": 0, "sev_total": 0, "sev_max": 0})
    by_label = defaultdict(lambda: {"n": 0, "eff": 0, "sev_total": 0})
    by_source_label = defaultdict(lambda: {"n": 0, "eff": 0})
    rows: list[dict] = []

    for r in _iter_results(args.run_id, include_broken=args.include_broken):
        aid = r.get("attack_id")
        if not aid or aid == "__baseline__":
            continue
        if args.agent and r.get("agent_id") != args.agent:
            continue
        info = prov.get(aid)
        if not info:
            continue
        source = info["source"]
        label = info["label"]
        eff = bool(r.get("effective"))
        sev = int(r.get("severity") or 0)

        by_source[source]["n"] += 1
        by_source[source]["eff"] += int(eff)
        by_source[source]["sev_total"] += sev
        by_source[source]["sev_max"] = max(by_source[source]["sev_max"], sev)

        by_label[label]["n"] += 1
        by_label[label]["eff"] += int(eff)
        by_label[label]["sev_total"] += sev

        key = (source, label)
        by_source_label[key]["n"] += 1
        by_source_label[key]["eff"] += int(eff)

        rows.append(
            {
                "attack_id": aid,
                "label": label,
                "source": source,
                "agent": r.get("agent_id"),
                "channel": r.get("channel"),
                "effective": eff,
                "severity": sev,
                "reasons": r.get("reasons", []),
                "duration_sec": r.get("attack_run", {}).get("duration_sec"),
            }
        )

    if args.json:
        print(
            json.dumps(
                {
                    "by_source": {k: dict(v) for k, v in by_source.items()},
                    "by_label": {k: dict(v) for k, v in by_label.items()},
                    "by_source_label": {f"{s}::{lbl}": dict(v) for (s, lbl), v in by_source_label.items()},
                    "rows": rows,
                },
                indent=2,
                default=list,
            )
        )
        return 0

    print(f"\n=== Fleet analysis (run_id={args.run_id or 'ALL'} agent={args.agent or 'ALL'}) ===")
    print(f"Total rows: {len(rows)}")

    print("\nBy synth source:")
    print(f"  {'source':<45} {'n':>6} {'eff':>6} {'rate':>7} {'avg_sev':>8} {'max_sev':>8}")
    for src, d in sorted(by_source.items(), key=lambda x: -x[1]["n"]):
        avg = d["sev_total"] / d["n"] if d["n"] else 0.0
        print(f"  {src:<45} {d['n']:>6} {d['eff']:>6} {_pct(d['eff'], d['n']):>7} {avg:>8.1f} {d['sev_max']:>8}")

    print("\nBy label:")
    print(f"  {'label':<28} {'n':>6} {'eff':>6} {'rate':>7} {'avg_sev':>8}")
    for lbl, d in sorted(by_label.items(), key=lambda x: -x[1]["n"]):
        avg = d["sev_total"] / d["n"] if d["n"] else 0.0
        print(f"  {lbl:<28} {d['n']:>6} {d['eff']:>6} {_pct(d['eff'], d['n']):>7} {avg:>8.1f}")

    print("\nSource × label (eff/n):")
    sources = sorted({s for s, _ in by_source_label.keys()})
    labels = sorted({lbl for _, lbl in by_source_label.keys()})
    head = "  " + "source/label".ljust(35) + " ".join(f"{lbl[:10]:>10}" for lbl in labels)
    print(head)
    for s in sources:
        cells = []
        for lbl in labels:
            d = by_source_label.get((s, lbl), {"n": 0, "eff": 0})
            cells.append(f"{d['eff']:>3}/{d['n']:<3}".rjust(10) if d["n"] else f"{'-':>10}")
        print(f"  {s[:35]:<35} {' '.join(cells)}")

    print(f"\nTop {args.top} by severity (likely strongest training rows):")
    for r in sorted(rows, key=lambda x: -x["severity"])[: args.top]:
        print(
            f"  sev={r['severity']:>3}  {r['attack_id']}  [{r['label']:<25}] {r['source'][-25:]:<25}  reasons={'|'.join(r['reasons'])[:60]}"
        )

    dead_rows = [r for r in rows if r["severity"] == 0]
    print(f"\nDead rows (severity=0): {len(dead_rows)} / {len(rows)}  ({_pct(len(dead_rows), len(rows))})")
    if dead_rows[:5]:
        print("  Sample dead-row attack_ids (drop candidates):")
        for r in dead_rows[:5]:
            print(f"    {r['attack_id']}  [{r['label']}]  {r['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
