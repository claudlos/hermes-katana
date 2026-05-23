"""Build the focused v5.1 Katana training corpus.

This keeps Hermes-Katana's focused data_v4 as the spine, then:

1. enriches thin attack labels from the proving-ground v5 balanced pool,
2. adds a small schema-clean benign top-up from Hermes data_v3,
3. adds curated hard negatives from Hermes data_v3 for false-positive control.

The script writes a new Hermes-Katana data directory and a metrics report. It
does not print raw attack text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PG_ROOT = Path(
    os.environ.get(
        "KATANA_PROVING_GROUND_ROOT",
        str(Path(__file__).resolve().parents[1]),
    )
)
HERMES_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(PG_ROOT.parent / "hermes-katana"),
    )
)

DEFAULT_BASELINE = HERMES_ROOT / "training" / "data_v4" / "combined.jsonl"
DEFAULT_V5 = PG_ROOT / "synthdata" / "incoming" / "v5_balanced_synthdata_final.jsonl"
DEFAULT_BENIGN = HERMES_ROOT / "training" / "data_v3" / "benign.jsonl"
DEFAULT_HARD_NEG = HERMES_ROOT / "training" / "data_v3" / "hard_negatives.jsonl"
DEFAULT_OUT_DIR = HERMES_ROOT / "training" / "data_v5_1"
DEFAULT_REPORT_DIR = PG_ROOT / "results" / "reports" / "focused_v5_1_20260506"

LABELS = [
    "clean",
    "content_injection",
    "semantic_manipulation",
    "behavioral_control",
    "exfiltration_attempt",
    "jailbreak",
    "cognitive_state_attack",
    "encoding_evasion",
    "persona_jailbreak",
]

ATTACK_LABELS = [label for label in LABELS if label != "clean"]

# V4 already has healthy coverage for most labels. These floors pull the thin
# labels up without bloating the release set.
ATTACK_TARGETS = {
    "jailbreak": 800,
    "cognitive_state_attack": 800,
    "encoding_evasion": 900,
    "persona_jailbreak": 800,
}

ALLOWED_ORIGINS = {
    "user_input",
    "retrieved_web",
    "mcp_tool_description",
    "mcp_tool_result",
    "prior_session_memory",
    "delegated_agent_output",
}

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff\u202a-\u202e]")
WS_RE = re.compile(r"\s+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    return WS_RE.sub(" ", text).strip()


def family_text(text: str) -> str:
    return normalize_text(text).lower()


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_for_family(family_sha: str) -> str:
    bucket = int(family_sha[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def text_from(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("attack_text") or row.get("prompt") or "").strip()


def normal_origin(value: object) -> str:
    origin = str(value or "user_input")
    return origin if origin in ALLOWED_ORIGINS else "user_input"


def base_hashes(row: dict[str, Any], text: str) -> tuple[str, str, str]:
    raw_sha = str(row.get("text_sha256") or sha(text))
    norm_sha = str(row.get("text_sha256_normalized") or sha(family_text(text)))
    family_sha = str(row.get("family_sha256") or norm_sha)
    return raw_sha, norm_sha, family_sha


def normalize_row(
    row: dict[str, Any],
    *,
    label: str | None = None,
    source: str | None = None,
    source_family: str | None = None,
    quality_tier: str | None = None,
    release_tier: str,
    force_split: str | None = None,
) -> dict[str, Any] | None:
    text = normalize_text(text_from(row))
    label = str(label or row.get("label") or row.get("attack_label") or "")
    if not text or label not in LABELS:
        return None
    if len(text) < 8 or len(text) > 10_000:
        return None

    raw_sha, norm_sha, family_sha = base_hashes(row, text)
    is_attack = label != "clean"
    source = str(source or row.get("source") or "unknown")
    source_family = str(source_family or row.get("source_family") or source)
    split = force_split or str(row.get("split") or split_for_family(family_sha))
    if split == "validation":
        split = "val"
    if split not in {"train", "val", "test"}:
        split = split_for_family(family_sha)

    out = {
        "id": str(row.get("id") or f"{label[:6]}_{raw_sha[:16]}"),
        "text": text,
        "label": label,
        "source": source,
        "source_family": source_family,
        "origin": normal_origin(row.get("origin")),
        "is_attack": is_attack,
        "binary_label": "attack" if is_attack else "benign",
        "quality_tier": str(quality_tier or row.get("quality_tier") or "unknown"),
        "release_tier": release_tier,
        "text_sha256": raw_sha,
        "text_sha256_normalized": norm_sha,
        "family_sha256": family_sha,
        "text_length": len(text),
        "split": split,
    }
    if row.get("technique"):
        out["technique"] = row["technique"]
    return out


def stable_rank(row: dict[str, Any], seed: int) -> str:
    basis = f"{seed}:{row.get('family_sha256') or row.get('text_sha256_normalized') or row.get('id')}"
    return sha(basis)


def add_row(
    row: dict[str, Any],
    *,
    out_rows: list[dict[str, Any]],
    seen_family: set[str],
    stats: Counter,
) -> bool:
    family = row["family_sha256"]
    if family in seen_family:
        stats["skipped_duplicate_family"] += 1
        return False
    seen_family.add(family)
    out_rows.append(row)
    stats["added"] += 1
    stats[f"added_release_tier:{row['release_tier']}"] += 1
    stats[f"added_label:{row['label']}"] += 1
    return True


def select_attack_enrichment(
    *,
    v5_rows: list[dict[str, Any]],
    out_rows: list[dict[str, Any]],
    seen_family: set[str],
    seed: int,
    stats: Counter,
) -> dict[str, int]:
    selected_by_label: dict[str, int] = {}
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for raw in v5_rows:
        label = str(raw.get("label") or "")
        if label not in ATTACK_LABELS:
            continue
        row = normalize_row(
            raw,
            source=f"proving_ground_v5:{raw.get('source') or raw.get('origin') or 'v5_balanced'}",
            source_family="proving_ground_v5",
            quality_tier=str(raw.get("quality_tier") or "v5_balanced_enrichment"),
            release_tier="silver_v5_enrichment",
        )
        if row is None:
            stats["v5_rejected_schema_or_length"] += 1
            continue
        row["split"] = split_for_family(row["family_sha256"])
        if not (20 <= row["text_length"] <= 4_500):
            stats["v5_rejected_length_window"] += 1
            continue
        by_label[label].append(row)

    for label, target in ATTACK_TARGETS.items():
        current = sum(1 for row in out_rows if row["label"] == label)
        needed = max(0, target - current)
        selected = 0
        candidates = sorted(by_label[label], key=lambda row: stable_rank(row, seed))
        for row in candidates:
            if selected >= needed:
                break
            if add_row(row, out_rows=out_rows, seen_family=seen_family, stats=stats):
                selected += 1
        selected_by_label[label] = selected
        if selected < needed:
            stats[f"v5_shortfall:{label}"] = needed - selected
    return selected_by_label


def source_diverse_sample(rows: list[dict[str, Any]], *, target: int, seed: int) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source_family") or row.get("source") or "unknown")].append(row)
    for bucket in by_source.values():
        bucket.sort(key=lambda row: stable_rank(row, seed))

    selected: list[dict[str, Any]] = []
    while len(selected) < target:
        progress = False
        for source in sorted(by_source):
            bucket = by_source[source]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break
    return selected


def select_controls(
    *,
    benign_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
    out_rows: list[dict[str, Any]],
    seen_family: set[str],
    seed: int,
    extra_clean_target: int,
    hard_negative_target: int,
    stats: Counter,
) -> dict[str, int]:
    clean_candidates: list[dict[str, Any]] = []
    clean_source_priority = {
        "awesome_prompts",
        "deepset",
        "synthetic_benign_extra",
        "synthetic_benign",
        "benign",
    }
    for raw in benign_rows:
        raw_source = str(raw.get("source") or "")
        if raw_source not in clean_source_priority:
            continue
        row = normalize_row(
            raw,
            label="clean",
            source=f"hermes_v3_benign:{raw_source}",
            source_family=raw_source,
            quality_tier="curated_schema_clean_benign",
            release_tier="benign_control",
        )
        if row is None:
            stats["benign_rejected_schema_or_length"] += 1
            continue
        if not (20 <= row["text_length"] <= 4_500):
            stats["benign_rejected_length_window"] += 1
            continue
        clean_candidates.append(row)

    hard_candidates: list[dict[str, Any]] = []
    for raw in hard_negative_rows:
        row = normalize_row(
            raw,
            label="clean",
            source=f"hermes_v3_hard_negative:{raw.get('source') or raw.get('source_family') or 'unknown'}",
            source_family=str(raw.get("source_family") or raw.get("source") or "hard_negative"),
            quality_tier="curated_hard_negative_benign",
            release_tier="hard_negative_control",
        )
        if row is None:
            stats["hard_negative_rejected_schema_or_length"] += 1
            continue
        if not (60 <= row["text_length"] <= 5_500):
            stats["hard_negative_rejected_length_window"] += 1
            continue
        hard_candidates.append(row)

    selected_clean = 0
    for row in source_diverse_sample(clean_candidates, target=len(clean_candidates), seed=seed):
        if add_row(row, out_rows=out_rows, seen_family=seen_family, stats=stats):
            selected_clean += 1
            if selected_clean >= extra_clean_target:
                break

    selected_hard = 0
    for row in source_diverse_sample(hard_candidates, target=len(hard_candidates), seed=seed + 17):
        if add_row(row, out_rows=out_rows, seen_family=seen_family, stats=stats):
            selected_hard += 1
            if selected_hard >= hard_negative_target:
                break

    return {
        "extra_clean_selected": selected_clean,
        "hard_negative_selected": selected_hard,
        "clean_candidate_pool": len(clean_candidates),
        "hard_negative_candidate_pool": len(hard_candidates),
    }


def lengths(rows: list[dict[str, Any]]) -> dict[str, int]:
    vals = sorted(int(row["text_length"]) for row in rows)
    if not vals:
        return {"min": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}

    def pct(q: float) -> int:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * q))))
        return vals[idx]

    return {
        "min": vals[0],
        "median": int(statistics.median(vals)),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": vals[-1],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_families": len({row["family_sha256"] for row in rows}),
        "by_label": dict(Counter(row["label"] for row in rows).most_common()),
        "by_binary_label": dict(Counter(row["binary_label"] for row in rows).most_common()),
        "by_release_tier": dict(Counter(row["release_tier"] for row in rows).most_common()),
        "by_quality_tier": dict(Counter(row["quality_tier"] for row in rows).most_common()),
        "by_source_family": dict(Counter(row["source_family"] for row in rows).most_common(30)),
        "by_split": dict(Counter(row["split"] for row in rows).most_common()),
        "text_length": lengths(rows),
    }


def write_report(report_dir: Path, metadata: dict[str, Any]) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = metadata["summary"]["combined"]
    lines = [
        "# Focused v5.1 Dataset Build",
        "",
        f"Generated: {metadata['generated_at']}",
        "",
        "## Headline",
        "",
        f"- Output: `{metadata['output_dir']}`",
        f"- Combined rows: {summary['rows']:,}; unique families: {summary['unique_families']:,}.",
        f"- Binary mix: {summary['by_binary_label'].get('attack', 0):,} attack / {summary['by_binary_label'].get('benign', 0):,} benign-control.",
        f"- Added from proving-ground v5: {metadata['selection']['v5_attack_enrichment_total']:,} attack rows.",
        f"- Added controls: {metadata['selection']['extra_clean_selected']:,} schema-clean benign rows and {metadata['selection']['hard_negative_selected']:,} hard-negative rows.",
        "",
        "## Label Mix",
        "",
        "| label | rows |",
        "| --- | ---: |",
    ]
    for label in LABELS:
        lines.append(f"| `{label}` | {summary['by_label'].get(label, 0):,} |")
    lines.extend(
        [
            "",
            "## Release Tiers",
            "",
            "| release tier | rows |",
            "| --- | ---: |",
        ]
    )
    for tier, count in summary["by_release_tier"].items():
        lines.append(f"| `{tier}` | {count:,} |")
    lines.extend(
        [
            "",
            "## Splits",
            "",
            "| split | rows |",
            "| --- | ---: |",
        ]
    )
    for split in ["train", "val", "test"]:
        lines.append(f"| `{split}` | {summary['by_split'].get(split, 0):,} |")
    lines.extend(
        [
            "",
            "## Enrichment",
            "",
            "| label | added from v5 | target |",
            "| --- | ---: | ---: |",
        ]
    )
    for label, target in ATTACK_TARGETS.items():
        lines.append(
            f"| `{label}` | {metadata['selection']['v5_attack_enrichment_by_label'].get(label, 0):,} | {target:,} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `data_v4` was preserved and used as the baseline spine.",
            "- New attack rows came only from the proving-ground v5 balanced pool, selected for under-covered labels.",
            "- New controls came from Hermes `data_v3/benign.jsonl` and `data_v3/hard_negatives.jsonl`.",
            "- Rows are deduped by `family_sha256`; no raw prompt examples are printed in this report.",
        ]
    )
    (report_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--v5", type=Path, default=DEFAULT_V5)
    ap.add_argument("--benign", type=Path, default=DEFAULT_BENIGN)
    ap.add_argument("--hard-negatives", type=Path, default=DEFAULT_HARD_NEG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--extra-clean-target", type=int, default=500)
    ap.add_argument("--hard-negative-target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stats: Counter = Counter()
    out_rows: list[dict[str, Any]] = []
    seen_family: set[str] = set()

    baseline_rows = read_jsonl(args.baseline)
    for raw in baseline_rows:
        row = normalize_row(raw, release_tier="focused_v4_baseline")
        if row is None:
            stats["baseline_rejected_schema_or_length"] += 1
            continue
        add_row(row, out_rows=out_rows, seen_family=seen_family, stats=stats)

    v5_selected_by_label = select_attack_enrichment(
        v5_rows=read_jsonl(args.v5),
        out_rows=out_rows,
        seen_family=seen_family,
        seed=args.seed,
        stats=stats,
    )

    control_selection = select_controls(
        benign_rows=read_jsonl(args.benign),
        hard_negative_rows=read_jsonl(args.hard_negatives),
        out_rows=out_rows,
        seen_family=seen_family,
        seed=args.seed,
        extra_clean_target=args.extra_clean_target,
        hard_negative_target=args.hard_negative_target,
        stats=stats,
    )

    out_rows.sort(key=lambda row: (row["split"], row["binary_label"], row["label"], row["release_tier"], row["id"]))
    attacks = [row for row in out_rows if row["is_attack"]]
    controls = [row for row in out_rows if not row["is_attack"]]
    hard_negatives = [row for row in controls if row["release_tier"] == "hard_negative_control"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "combined.jsonl", out_rows)
    write_jsonl(args.out_dir / "attacks.jsonl", attacks)
    write_jsonl(args.out_dir / "controls.jsonl", controls)
    write_jsonl(args.out_dir / "hard_negatives.jsonl", hard_negatives)
    split_dir = args.out_dir / "splits"
    for split in ["train", "val", "test"]:
        write_jsonl(split_dir / f"{split}.jsonl", [row for row in out_rows if row["split"] == split])

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    metadata = {
        "generated_at": generated_at,
        "inputs": {
            "baseline": str(args.baseline),
            "v5": str(args.v5),
            "benign": str(args.benign),
            "hard_negatives": str(args.hard_negatives),
        },
        "output_dir": str(args.out_dir),
        "seed": args.seed,
        "attack_targets": ATTACK_TARGETS,
        "selection": {
            "v5_attack_enrichment_by_label": v5_selected_by_label,
            "v5_attack_enrichment_total": sum(v5_selected_by_label.values()),
            **control_selection,
        },
        "summary": {
            "combined": summarize(out_rows),
            "attacks": summarize(attacks),
            "controls": summarize(controls),
            "hard_negatives": summarize(hard_negatives),
        },
        "stats": dict(stats),
        "schema": {
            "fields": [
                "id",
                "text",
                "label",
                "source",
                "source_family",
                "origin",
                "is_attack",
                "binary_label",
                "quality_tier",
                "release_tier",
                "text_sha256",
                "text_sha256_normalized",
                "family_sha256",
                "text_length",
                "split",
            ],
            "labels": LABELS,
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(args.report_dir, metadata)

    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "report": str(args.report_dir / "report.md"),
                "rows": metadata["summary"]["combined"]["rows"],
                "attack_rows": metadata["summary"]["combined"]["by_binary_label"].get("attack", 0),
                "benign_rows": metadata["summary"]["combined"]["by_binary_label"].get("benign", 0),
                "v5_added": metadata["selection"]["v5_attack_enrichment_total"],
                "hard_negatives_added": metadata["selection"]["hard_negative_selected"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
