#!/usr/bin/env python3
"""Sanitize the v5.1 corpus for public release.

Reads ``hermes-katana/training/data_v5_1/`` and writes a sanitized companion
directory (``training/data_v5_1_public/`` by default). Operations:

1. Replace personal-domain emails (gmail / yahoo / hotmail / outlook / icloud /
   protonmail / aol / live / me) with example.com substitutes that preserve
   the local part where useful.
2. Replace `anthropic.com` references with `example.com` — the dataset is
   adversarial; references to a real LLM vendor in attack contexts read as
   targeted even when the original was synthetic.
3. Replace 9-digit SSN patterns (NNN-NN-NNNN) with `XXX-XX-XXXX`.
4. Replace 16-digit credit-card patterns with `XXXX-XXXX-XXXX-XXXX`.

Family hashes / split assignment / labels are NOT changed. Sanitization
operates on text only. Resulting rows keep the same `family_sha256` so the
splits stay aligned across the original and public versions.

The original `data_v5_1/` is untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

DEFAULT_PG_ROOT = Path(
    os.environ.get(
        "KATANA_PROVING_GROUND_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
)
DEFAULT_HERMES_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(DEFAULT_PG_ROOT.parent / "hermes-katana"),
    )
)


PERSONAL_EMAIL_DOMAINS = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "aol.com",
    "protonmail.com",
    "live.com",
    "msn.com",
    "me.com",
    "googlemail.com",
    "ymail.com",
    "rocketmail.com",
}

EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
# US SSN format with dashes only (avoid corrupting random 9-digit numbers).
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# 16-digit credit card with optional separators (- or space).
CC_RE = re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b")
ANTHROPIC_RE = re.compile(r"\banthropic\.com\b", re.IGNORECASE)


def sanitize_text(text: str, stats: Counter) -> str:
    def email_sub(m: re.Match) -> str:
        local, domain = m.group(1), m.group(2)
        if domain.lower() in PERSONAL_EMAIL_DOMAINS:
            stats["email_personal_redacted"] += 1
            # Preserve the local part for context but anchor to example.com.
            return f"{local}@example.com"
        return m.group(0)

    text = EMAIL_RE.sub(email_sub, text)

    def anthropic_sub(_: re.Match) -> str:
        stats["anthropic_redacted"] += 1
        return "example.com"

    text = ANTHROPIC_RE.sub(anthropic_sub, text)

    def ssn_sub(_: re.Match) -> str:
        stats["ssn_redacted"] += 1
        return "XXX-XX-XXXX"

    text = SSN_RE.sub(ssn_sub, text)

    def cc_sub(_: re.Match) -> str:
        stats["cc_redacted"] += 1
        return "XXXX-XXXX-XXXX-XXXX"

    text = CC_RE.sub(cc_sub, text)

    return text


def sanitize_row(row: dict, stats: Counter) -> dict:
    out = dict(row)
    text_before = out.get("text", "")
    text_after = sanitize_text(text_before, stats)
    if text_after != text_before:
        stats["rows_sanitized"] += 1
    out["text"] = text_after
    return out


def process_file(src: Path, dst: Path, stats: Counter) -> int:
    rows = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            row = sanitize_row(row, stats)
            fout.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows += 1
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_HERMES_ROOT / "training" / "data_v5_1")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_HERMES_ROOT / "training" / "data_v5_1_public")
    args = ap.parse_args()

    if not args.src_dir.exists():
        print(f"source not found: {args.src_dir}")
        return 2

    files_to_copy = ["combined.jsonl", "attacks.jsonl", "controls.jsonl", "hard_negatives.jsonl"]
    splits = ["train.jsonl", "val.jsonl", "test.jsonl"]

    stats: Counter = Counter()
    total_rows = 0

    for name in files_to_copy:
        src = args.src_dir / name
        if not src.exists():
            print(f"  skip (missing): {name}")
            continue
        dst = args.out_dir / name
        n = process_file(src, dst, stats)
        total_rows += n
        print(f"  wrote {n:>6,} -> {dst}")

    for name in splits:
        src = args.src_dir / "splits" / name
        if not src.exists():
            print(f"  skip (missing): splits/{name}")
            continue
        dst = args.out_dir / "splits" / name
        n = process_file(src, dst, stats)
        print(f"  wrote {n:>6,} -> {dst}")

    # Carry metadata.json forward unchanged (provenance fields are accurate).
    meta_src = args.src_dir / "metadata.json"
    if meta_src.exists():
        meta = json.loads(meta_src.read_text())
        meta.setdefault("public_release", {})
        meta["public_release"] = {
            "sanitization_applied": True,
            "stats": dict(stats),
            "notes": (
                "Personal-domain emails (gmail/yahoo/hotmail/outlook/icloud/"
                "protonmail) replaced with example.com. anthropic.com replaced. "
                "SSN-like and credit-card-like patterns redacted with X. "
                "Family hashes and split assignment unchanged."
            ),
        }
        (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        print(f"  wrote metadata -> {args.out_dir / 'metadata.json'}")

    print(f"\nTotal rows scanned: {total_rows:,}")
    print("Sanitization stats:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
