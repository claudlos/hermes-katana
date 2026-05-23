#!/usr/bin/env python3
"""Build the ``confirmed_only_v1`` benchmark file.

Filters ``data_v5_1/splits/test.jsonl`` to:

    * all rows where ``is_attack`` is False (the 572 clean controls)
    * attack rows where ``quality_tier`` begins with ``confirmed_``

That gives empirically-validated attacks plus the full clean set, which is the
denominator the binary metrics need. Synthetic critic-passed attack rows are
excluded so the leaderboard number is measured only on real attacks.

Deterministic — re-running on an unchanged ``data_v5_1/splits/test.jsonl``
produces a byte-identical file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DEFAULT_HERMES_KATANA_ROOT = os.environ.get(
    "HERMES_KATANA_ROOT",
    str(Path(__file__).resolve().parents[3]),
)
DEFAULT_SRC = Path(DEFAULT_HERMES_KATANA_ROOT) / "training" / "data_v5_1" / "splits" / "test.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent / "test.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.src.exists():
        print(f"source not found: {args.src}")
        return 2

    kept: list[dict] = []
    with args.src.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("is_attack"):
                if str(row.get("quality_tier", "")).startswith("confirmed_"):
                    kept.append(row)
            else:
                kept.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n_attack = sum(1 for r in kept if r.get("is_attack"))
    n_clean = len(kept) - n_attack
    print(f"wrote {len(kept)} rows -> {args.out}")
    print(f"  clean: {n_clean}")
    print(f"  confirmed attacks: {n_attack}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
