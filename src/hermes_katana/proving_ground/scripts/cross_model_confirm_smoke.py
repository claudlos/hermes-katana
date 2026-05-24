"""Quick cross-model confirmation across the 3 smoke fleets.

Joins haiku + sonnet + minimax effective signals per attack_id. Reports:
  - attacks effective on N=1, 2, 3 models
  - per-source confirmed-rate (eff_on >= 2 models / tested by >= 2 models)
  - per-label confirmed-rate
  - the actual confirmed attack_ids (write to a JSONL for downstream training)

Skips rows matching the broken-runner pattern (CCLI quota burn).

Output:
  results/synth_smoke_confirmed.jsonl  -- one row per attack confirmed on >=2 models
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
SHARDS = ROOT / "shards"
RUN_IDS = {"smoke_2026_04_26", "minimax_2026_04_26", "sonnet_2026_04_27"}
OUT = ROOT / "results" / "synth_smoke_confirmed.jsonl"


def _is_broken(r: dict) -> bool:
    ar = r.get("attack_run") or {}
    return (
        ar.get("exit_code", 0) != 0
        and ar.get("output_chars", 0) <= 6000
        and ar.get("tool_call_count", 0) == 0
        and (ar.get("duration_sec") or 0) < 5.0
    )


def _provenance() -> dict[str, dict]:
    out = {}
    for f in sorted(SHARDS.glob("shard_2*.jsonl")):
        with f.open(encoding="utf-8") as fp:
            for line in fp:
                r = json.loads(line)
                out[r["id"]] = {
                    "label": r["label"],
                    "source": r["source"],
                    "text": r.get("text", ""),
                }
    return out


def main() -> int:
    prov = _provenance()
    per_attack = defaultdict(
        lambda: {
            "tested_by": set(),
            "effective_on": set(),
            "max_severity": 0,
            "labels": set(),
        }
    )

    for f in sorted(AGENT_RUNS.glob("shard_*.jsonl")):
        with f.open(encoding="utf-8") as fp:
            for line in fp:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("run_id") not in RUN_IDS or r.get("attack_id") == "__baseline__":
                    continue
                if _is_broken(r):
                    continue
                aid = r["attack_id"]
                model = r["agent_id"]
                rec = per_attack[aid]
                rec["tested_by"].add(model)
                rec["max_severity"] = max(rec["max_severity"], int(r.get("severity") or 0))
                if r.get("effective"):
                    rec["effective_on"].add(model)
                if r.get("attack_label"):
                    rec["labels"].add(r["attack_label"])

    coverage = defaultdict(int)
    confirmed_rows = []
    for aid, rec in per_attack.items():
        n_models = len(rec["effective_on"])
        coverage[n_models] += 1
        if n_models >= 2:
            info = prov.get(aid, {})
            confirmed_rows.append(
                {
                    "id": aid,
                    "label": info.get("label", ""),
                    "source": info.get("source", ""),
                    "text": info.get("text", "")[:500],
                    "n_models_effective": n_models,
                    "n_models_tested": len(rec["tested_by"]),
                    "effective_on": sorted(rec["effective_on"]),
                    "max_severity": rec["max_severity"],
                }
            )

    print(f"\nTotal unique attacks tested: {len(per_attack)}")
    print("By #models effective:")
    for n in sorted(coverage):
        print(f"  on {n} model(s): {coverage[n]:>4}")

    print(f"\nConfirmed (effective on >=2 models): {len(confirmed_rows)}")

    if confirmed_rows:
        from collections import Counter

        by_src = Counter(r["source"] for r in confirmed_rows)
        by_lbl = Counter(r["label"] for r in confirmed_rows)
        print("\nConfirmed by source:")
        for s, n in by_src.most_common():
            print(f"  {s:<45} {n}")
        print("\nConfirmed by label:")
        for label, n in by_lbl.most_common():
            print(f"  {label:<28} {n}")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w", encoding="utf-8") as f:
            for row in sorted(
                confirmed_rows,
                key=lambda r: (-r["n_models_effective"], -r["max_severity"]),
            ):
                f.write(json.dumps(row) + "\n")
        print(f"\nWrote {len(confirmed_rows)} rows -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
