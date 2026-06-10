#!/usr/bin/env python3
"""Build the ``confirmed_only_v2`` benchmark files from ``data_v9``."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HERMES_KATANA_ROOT = Path(
    os.environ.get("HERMES_KATANA_ROOT", str(Path(__file__).resolve().parents[3]))
)
DEFAULT_DATA = DEFAULT_HERMES_KATANA_ROOT / "training" / "data_v9"
DEFAULT_OUT = Path(__file__).resolve().parent

GOLD_TIERS = {
    "v8_agentic_confirmed_cross_model",
    "v9_agentic_confirmed_cross_model",
    "v9_upgraded_confirmed_cross_model",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def is_confirmed_attack(row: dict[str, Any]) -> bool:
    tier = str(row.get("quality_tier") or "")
    return tier.startswith("confirmed_") or tier in GOLD_TIERS


# Per-model success metadata in proving_ground tells an attacker which models
# each payload already defeated (attacker > defender asymmetry). Strip it on
# build so published rows carry only the prompt + labels (audit finding F3,
# matching the v1 release policy).
_PG_STRIP_PREFIXES = ("effective_",)
_PG_STRIP_KEYS = {"agreement_class"}


def scrub_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return *row* with attacker-advantage metadata removed from proving_ground."""
    pg = row.get("proving_ground")
    if isinstance(pg, dict):
        row = dict(row)
        row["proving_ground"] = {
            k: v
            for k, v in pg.items()
            if k not in _PG_STRIP_KEYS and not any(k.startswith(p) for p in _PG_STRIP_PREFIXES)
        }
    return row


def write_leaderboard(path: Path, summary: dict[str, Any]) -> None:
    strict = summary["strict"]
    stress = summary["all_gold_stress"]
    lines = [
        "# confirmed_only_v2 - katana benchmark",
        "",
        "Built from `training/data_v9`. The primary `test.jsonl` benchmark keeps the "
        "family-disjoint test policy: clean rows and confirmed trainable attacks from "
        "`data_v9/splits/test.jsonl`, plus held-out gold rows whose family assignment is `test`.",
        "",
        "| slice | rows |",
        "|---|---:|",
        f"| clean test rows | {strict['clean']} |",
        f"| confirmed trainable test attacks | {strict['confirmed_trainable_test_attacks']} |",
        f"| test-family held-out gold attacks | {strict['test_family_gold_attacks']} |",
        f"| **primary total** | **{strict['rows']}** |",
        "",
        "## All-Gold Stress Eval",
        "",
        "`all_gold_stress.jsonl` keeps the same clean test rows but includes every held-out "
        "gold attack, regardless of family split. It is an attack-recall stress slice, not "
        "a family-disjoint leaderboard replacement.",
        "",
        "| slice | rows |",
        "|---|---:|",
        f"| clean test rows | {stress['clean']} |",
        f"| all held-out gold attacks | {stress['gold_attacks']} |",
        f"| **stress total** | **{stress['rows']}** |",
        "",
        "## Attack Labels - Primary",
        "",
    ]
    for label, count in strict["attack_labels"].items():
        lines.append(f"- `{label}`: {count}")
    lines += ["", "## Attack Labels - All-Gold Stress", ""]
    for label, count in stress["attack_labels"].items():
        lines.append(f"- `{label}`: {count}")
    lines += [
        "",
        "## Rebuild",
        "",
        "```bash",
        "python evals/benchmarks/confirmed_only_v2/build.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    test_rows = read_jsonl(args.data / "splits" / "test.jsonl")
    gold_rows = read_jsonl(args.data / "eval" / "gold_confirmed.jsonl")

    clean_test = [r for r in test_rows if not r.get("is_attack")]
    confirmed_test = [r for r in test_rows if r.get("is_attack") and is_confirmed_attack(r)]
    gold_test = [r for r in gold_rows if str(r.get("split")) == "test"]

    strict = [scrub_row(r) for r in (clean_test + confirmed_test + gold_test)]
    stress = [scrub_row(r) for r in (clean_test + gold_rows)]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "test.jsonl", strict)
    write_jsonl(args.out / "all_gold_stress.jsonl", stress)

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strict": {
            "rows": len(strict),
            "clean": len(clean_test),
            "confirmed_trainable_test_attacks": len(confirmed_test),
            "test_family_gold_attacks": len(gold_test),
            "attack_labels": dict(
                Counter(str(r.get("label")) for r in strict if r.get("is_attack")).most_common()
            ),
        },
        "all_gold_stress": {
            "rows": len(stress),
            "clean": len(clean_test),
            "gold_attacks": len(gold_rows),
            "attack_labels": dict(Counter(str(r.get("label")) for r in gold_rows).most_common()),
        },
    }
    (args.out / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_leaderboard(args.out / "LEADERBOARD.md", summary)

    print(f"wrote {summary['strict']['rows']} rows -> {args.out / 'test.jsonl'}")
    print(f"wrote {summary['all_gold_stress']['rows']} rows -> {args.out / 'all_gold_stress.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
