"""Create a dataset metrics and release strategy report.

The report summarizes corpus size, label balance, confirmed evidence, run
telemetry, and a recommended organization for DeBERTa/scanner training and
public release. It intentionally avoids printing attack text.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = ROOT.parent / "hermes-katana"
OUT_DIR = ROOT / "results" / "reports" / "dataset_strategy_20260506"

CONFIRMED = ROOT / "results" / "confirmed_attacks.jsonl"
LEGACY_CONFIRMED = ROOT / "results" / "confirmed_attacks.v1_legacy.jsonl"
SMOKE_CONFIRMED = ROOT / "results" / "synth_smoke_confirmed.jsonl"
V5_BALANCED = ROOT / "synthdata" / "incoming" / "v5_balanced_synthdata_final.jsonl"
HERMES_DATASETS = {
    "hermes_v4_focused_combined": HERMES_ROOT / "training" / "data_v4" / "combined.jsonl",
    "hermes_v3_combined": HERMES_ROOT / "training" / "data_v3" / "combined.jsonl",
    "hermes_v3_attacks": HERMES_ROOT / "training" / "data_v3" / "attacks.jsonl",
    "hermes_v3_benign": HERMES_ROOT / "training" / "data_v3" / "benign.jsonl",
    "hermes_v3_hard_negatives": HERMES_ROOT / "training" / "data_v3" / "hard_negatives.jsonl",
    "hermes_v3_binary_balanced": HERMES_ROOT / "training" / "data_v3" / "balanced" / "binary_balanced.jsonl",
    "hermes_merged_refined_combined": HERMES_ROOT / "training" / "data" / "merged_refined" / "combined.jsonl",
    "hermes_merged_refined_attacks": HERMES_ROOT / "training" / "data" / "merged_refined" / "attacks.jsonl",
    "hermes_merged_refined_benign": HERMES_ROOT / "training" / "data" / "merged_refined" / "benign.jsonl",
    "hermes_merged_refined_hard_negatives": HERMES_ROOT
    / "training"
    / "data"
    / "merged_refined"
    / "hard_negatives.jsonl",
    "hermes_eval_gap_fixed": HERMES_ROOT / "eval-corpus-expansion" / "combined-gap-corpus-fixed.jsonl",
    "hermes_wild_normalized_clean": HERMES_ROOT / "research" / "wild-attacks-2026-04-05" / "normalized-clean.jsonl",
    "hermes_wild_strict_heldout": HERMES_ROOT
    / "research"
    / "wild-attacks-2026-04-05"
    / "normalized-strict-heldout.jsonl",
    "hermes_still_missed_attacks": HERMES_ROOT / "research" / "still-missed-attacks.jsonl",
}
HERMES_TEXT_CONTROLS = {
    "hermes_benign_corpus_extended_txt": HERMES_ROOT
    / "research"
    / "wild-attacks-2026-04-05"
    / "benign_corpus_extended.txt",
    "hermes_benign_corpus_txt": HERMES_ROOT / "research" / "wild-attacks-2026-04-05" / "benign_corpus.txt",
}

SOURCE_RUNS_RECENT = [
    "free_reliable_discovery_20260506",
    "free_booster_mix_20260506",
    "copilot_gpt5mini_tail_20260506",
    "confirm_free_discovery_20260506",
    "quota_mix_fresh_20260505_2158",
    "confirm_quota_mix_priority_20260506",
    "confirm_quota_mix_remaining_20260506",
]


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def primary_id(row: dict) -> str:
    value = (
        row.get("primary_unit_id")
        or row.get("family_sha256")
        or row.get("text_sha256_normalized")
        or row.get("attack_text_sha256")
        or row.get("text_sha256")
        or row.get("id")
        or row.get("attack_id")
        or row.get("hash")
        or ""
    )
    if not value:
        text = str(row.get("text") or row.get("attack_text") or row.get("prompt") or row.get("content") or "")
        if text:
            value = hashlib.sha256(text.strip().encode("utf-8", errors="ignore")).hexdigest()
    value = str(value)
    return value if value.startswith("family:") else f"family:{value}" if value else "unknown"


def label(row: dict) -> str:
    return str(
        row.get("attack_label") or row.get("label") or row.get("category") or row.get("clean_label") or "unknown"
    )


def binary_label(row: dict) -> str:
    def normalize(value: object) -> str | None:
        v = str(value or "").lower()
        if v in {"attack", "unsafe", "malicious"}:
            return "attack"
        if v in {"benign", "clean", "safe"}:
            return "benign"
        if v == "1":
            return "attack"
        if v == "0":
            return "benign"
        return None

    if "binary_label" in row:
        return normalize(row.get("binary_label")) or str(row.get("binary_label") or "unknown")
    if row.get("is_attack") is True:
        return "attack"
    if row.get("is_attack") is False:
        return "benign"
    label_binary = normalize(row.get("label"))
    if label_binary:
        return label_binary
    return "unknown"


def text_len(row: dict) -> int:
    if isinstance(row.get("text_length"), int):
        return int(row["text_length"])
    text = str(row.get("text") or row.get("attack_text") or "")
    return len(text)


def pct(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, math.ceil((q / 100) * len(values)) - 1))
    return int(values[idx])


def summarize_rows(rows: list[dict]) -> dict:
    lengths = [text_len(r) for r in rows if text_len(r) > 0]
    return {
        "rows": len(rows),
        "unique_primary_units": len({primary_id(r) for r in rows if primary_id(r) != "unknown"}),
        "by_label": dict(Counter(label(r) for r in rows).most_common()),
        "by_binary_label": dict(Counter(binary_label(r) for r in rows).most_common()),
        "by_quality_tier": dict(Counter(str(r.get("quality_tier") or "unknown") for r in rows).most_common()),
        "by_source": dict(Counter(str(r.get("source") or "unknown") for r in rows).most_common(25)),
        "by_provenance": dict(Counter(str(r.get("provenance") or "unknown") for r in rows).most_common()),
        "text_length": {
            "min": min(lengths) if lengths else 0,
            "median": int(statistics.median(lengths)) if lengths else 0,
            "p90": pct(lengths, 90),
            "p95": pct(lengths, 95),
            "p99": pct(lengths, 99),
            "max": max(lengths) if lengths else 0,
        },
    }


def is_valid(row: dict) -> bool:
    return not bool(row.get("invalid_run")) and row.get("row_valid") is not False


def summarize_agent_runs(run_ids: list[str] | None = None) -> dict:
    files = sorted((ROOT / "results" / "agent_shard_runs").glob("shard_*__run_*.jsonl"))
    rows_total = 0
    valid_total = 0
    invalid_total = 0
    effective_total = 0
    effective_units: set[str] = set()
    by_run: dict[str, Counter] = defaultdict(Counter)
    by_agent: dict[str, Counter] = defaultdict(Counter)
    by_label_eff = Counter()
    run_filter = set(run_ids or [])

    for path in files:
        for row in read_jsonl(path) or []:
            run_id = str(row.get("run_id") or "unknown")
            if run_filter and run_id not in run_filter:
                continue
            agent = str(row.get("agent_id") or "unknown")
            rows_total += 1
            by_run[run_id]["rows"] += 1
            by_agent[agent]["rows"] += 1
            if not is_valid(row):
                invalid_total += 1
                by_run[run_id]["invalid"] += 1
                by_agent[agent]["invalid"] += 1
                continue
            valid_total += 1
            by_run[run_id]["valid"] += 1
            by_agent[agent]["valid"] += 1
            if row.get("effective"):
                effective_total += 1
                effective_units.add(primary_id(row))
                by_run[run_id]["effective"] += 1
                by_agent[agent]["effective"] += 1
                by_label_eff[label(row)] += 1

    return {
        "rows": rows_total,
        "valid": valid_total,
        "invalid": invalid_total,
        "effective_rows": effective_total,
        "effective_unique_units": len(effective_units),
        "by_run": {k: dict(v) for k, v in sorted(by_run.items())},
        "by_agent_top": {
            k: dict(v) for k, v in sorted(by_agent.items(), key=lambda kv: (-kv[1]["effective"], kv[0]))[:30]
        },
        "effective_by_label": dict(by_label_eff.most_common()),
    }


def shard_id(path: Path) -> int | None:
    try:
        return int(path.stem.split("_", 1)[1])
    except Exception:
        return None


def summarize_shards() -> dict:
    files = sorted((ROOT / "shards").glob("shard_*.jsonl"))
    buckets = {
        "base_shards_lt9000": [],
        "generated_campaign_shards_ge9000": [],
        "confirmed_shard_9500": [],
        "recent_followup_shards_9600_9609": [],
    }
    all_rows: list[dict] = []
    bucket_summaries = {}
    for path in files:
        sid = shard_id(path)
        rows = list(read_jsonl(path) or [])
        all_rows.extend(rows)
        if sid == 9500:
            buckets["confirmed_shard_9500"].extend(rows)
        elif sid is not None and 9600 <= sid <= 9609:
            buckets["recent_followup_shards_9600_9609"].extend(rows)
        elif sid is not None and sid >= 9000:
            buckets["generated_campaign_shards_ge9000"].extend(rows)
        else:
            buckets["base_shards_lt9000"].extend(rows)
    for name, rows in buckets.items():
        bucket_summaries[name] = summarize_rows(rows)
    return {
        "files": len(files),
        "all_shard_rows": summarize_rows(all_rows),
        "buckets": bucket_summaries,
    }


def summarize_synthdata() -> dict:
    paths = {
        "v5_balanced": V5_BALANCED,
        "v5_incoming_raw": ROOT / "synthdata" / "incoming" / "v5_synthdata_final_20260502T154816Z.jsonl",
        "v1_final": ROOT / "synthdata" / "checkpoints" / "katana_synth_v1" / "synthdata_final.jsonl",
        "v2_final": ROOT / "synthdata" / "checkpoints" / "katana_synth_v2_opus_elite" / "synthdata_final.jsonl",
        "v3_final": ROOT / "synthdata" / "checkpoints" / "katana_synth_v3_gap6" / "synthdata_final.jsonl",
        "v4_encoding_raw": ROOT / "synthdata" / "checkpoints" / "katana_synth_v4_encoding" / "examples_raw.jsonl",
        "v4_persona_raw": ROOT / "synthdata" / "checkpoints" / "katana_synth_v4_persona" / "examples_raw.jsonl",
    }
    out = {}
    for name, path in paths.items():
        rows = list(read_jsonl(path) or [])
        out[name] = {
            "path": str(path.relative_to(ROOT)),
            **summarize_rows(rows),
        }
    return out


def summarize_hermes_datasets() -> dict:
    out = {}
    for name, path in HERMES_DATASETS.items():
        rows = list(read_jsonl(path) or [])
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            **summarize_rows(rows),
        }
    controls = {}
    for name, path in HERMES_TEXT_CONTROLS.items():
        lines = 0
        chars = 0
        if path.exists():
            with path.open(encoding="utf-8", errors="ignore") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    lines += 1
                    chars += len(text)
        controls[name] = {
            "path": str(path),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "nonempty_lines": lines,
            "chars": chars,
        }
    return {
        "root": str(HERMES_ROOT),
        "datasets": out,
        "text_controls": controls,
    }


def load_design_summaries() -> dict:
    out = {}
    for path in sorted((ROOT / "results" / "designs").glob("*/trial_plan_summary.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        out[path.parent.name] = {
            "planned_trials": data.get("planned_trials"),
            "selected_primary_units": data.get("selected_primary_units"),
            "by_agent": data.get("by_agent", {}),
            "by_channel": data.get("by_channel", {}),
            "by_label": data.get("by_label", {}),
        }
    return out


def write_report(summary: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s_confirmed = summary["confirmed"]
    v5 = summary["synthdata"]["v5_balanced"]
    recent = summary["agent_runs_recent"]
    all_runs = summary["agent_runs_all"]
    shards = summary["shards"]
    synthdata = summary["synthdata"]
    hermes = summary["hermes_katana"]
    hermes_sets = hermes["datasets"]

    def rate(part: int, whole: int) -> str:
        return f"{(part / whole * 100):.1f}%" if whole else "0.0%"

    def append_count_table(lines: list[str], counts: dict, total: int, value_label: str = "rows") -> None:
        lines.extend(["| item | " + value_label + " | share |", "| --- | ---: | ---: |"])
        for item, n in counts.items():
            lines.append(f"| `{item}` | {int(n):,} | {rate(int(n), total)} |")

    def append_agent_table(lines: list[str], agents: dict, limit: int = 12) -> None:
        lines.extend(
            [
                "| agent | rows | valid | invalid | effective | valid effective rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for agent, stats in list(agents.items())[:limit]:
            valid = int(stats.get("valid", 0))
            effective = int(stats.get("effective", 0))
            lines.append(
                f"| `{agent}` | {int(stats.get('rows', 0)):,} | {valid:,} | "
                f"{int(stats.get('invalid', 0)):,} | {effective:,} | {rate(effective, valid)} |"
            )

    lines: list[str] = [
        "# Dataset strategy report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "## Executive Readout",
        "",
        f"- Current confirmed attack set: {s_confirmed['rows']:,} rows, {s_confirmed['unique_primary_units']:,} unique primary units.",
        f"- Latest balanced training pool: {v5['rows']:,} rows, {v5['unique_primary_units']:,} unique primary units.",
        f"- Hermes-Katana focused v4 set: {hermes_sets['hermes_v4_focused_combined']['rows']:,} rows ({hermes_sets['hermes_v4_focused_combined']['by_binary_label'].get('attack', 0):,} attack / {hermes_sets['hermes_v4_focused_combined']['by_binary_label'].get('benign', 0):,} benign).",
        f"- Hermes-Katana reusable controls: {hermes_sets['hermes_v3_benign']['rows']:,} schema-clean benign rows, {hermes_sets['hermes_v3_hard_negatives']['rows']:,} hard-negative rows, {hermes_sets['hermes_merged_refined_benign']['rows']:,} broader benign rows.",
        f"- Shard/candidate inventory: {shards['files']:,} shard files, {shards['all_shard_rows']['rows']:,} rows, {shards['all_shard_rows']['unique_primary_units']:,} unique primary units.",
        f"- Total agent-run telemetry on disk: {all_runs['rows']:,} rows, {all_runs['effective_rows']:,} effective rows, {all_runs['effective_unique_units']:,} effective unique units.",
        f"- Recent May 5-6 telemetry: {recent['rows']:,} rows, {recent['effective_rows']:,} effective rows, {recent['effective_unique_units']:,} effective unique units.",
        f"- Recent valid-run effective rate: {rate(recent['effective_rows'], recent['valid'])}; all-time valid-run effective rate: {rate(all_runs['effective_rows'], all_runs['valid'])}.",
        "- The bottleneck is no longer raw generation. It is high-confidence validation, family dedupe, benign/hard-negative design, and stable evaluation splits.",
        "",
        "## Confirmed Attack Set",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| rows | {s_confirmed['rows']:,} |",
        f"| unique primary units | {s_confirmed['unique_primary_units']:,} |",
        f"| median chars | {s_confirmed['text_length']['median']:,} |",
        f"| p95 chars | {s_confirmed['text_length']['p95']:,} |",
        f"| max chars | {s_confirmed['text_length']['max']:,} |",
        "",
        "### Confirmed Labels",
        "",
    ]
    append_count_table(lines, s_confirmed["by_label"], s_confirmed["rows"])

    lines.extend(["", "### Confirmed Provenance", "", "| provenance | rows |", "| --- | ---: |"])
    for prov, n in s_confirmed["by_provenance"].items():
        lines.append(f"| `{prov}` | {n:,} |")
    lines.extend(
        [
            "",
            "Interpretation: the confirmed set is high precision and label-diverse, but its legacy provenance is incomplete. Treat it as the gold benchmark seed and calibration set, not as the main training volume.",
        ]
    )

    lines.extend(
        [
            "",
            "## Candidate And Shard Inventory",
            "",
            "| bucket | rows | unique units | median chars | p95 chars |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| all shards | {shards['all_shard_rows']['rows']:,} | {shards['all_shard_rows']['unique_primary_units']:,} | {shards['all_shard_rows']['text_length']['median']:,} | {shards['all_shard_rows']['text_length']['p95']:,} |",
        ]
    )
    for bucket, stats in shards["buckets"].items():
        lines.append(
            f"| `{bucket}` | {stats['rows']:,} | {stats['unique_primary_units']:,} | "
            f"{stats['text_length']['median']:,} | {stats['text_length']['p95']:,} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: we already have enough candidate volume for multiple release iterations. More months of broad validation would mostly add cost and delay unless it targets a specific coverage gap.",
        ]
    )

    lines.extend(
        [
            "",
            "## Training Pool Snapshot",
            "",
            f"Primary candidate for DeBERTa/scanner training: `{V5_BALANCED.relative_to(ROOT)}`.",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| rows | {v5['rows']:,} |",
            f"| unique primary units | {v5['unique_primary_units']:,} |",
            f"| median chars | {v5['text_length']['median']:,} |",
            f"| p95 chars | {v5['text_length']['p95']:,} |",
            f"| max chars | {v5['text_length']['max']:,} |",
            "",
            "### V5 Binary Balance",
            "",
        ]
    )
    append_count_table(lines, v5["by_binary_label"], v5["rows"])

    lines.extend(["", "### V5 Attack Taxonomy", "", "| label | rows |", "| --- | ---: |"])
    for lab, n in v5["by_label"].items():
        lines.append(f"| `{lab}` | {n:,} |")

    lines.extend(
        [
            "",
            "### Synthdata Checkpoints",
            "",
            "| set | rows | unique units | median chars | p95 chars | top binary label |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, stats in synthdata.items():
        top_binary = next(iter(stats["by_binary_label"].items()), ("unknown", 0))
        lines.append(
            f"| `{name}` | {stats['rows']:,} | {stats['unique_primary_units']:,} | "
            f"{stats['text_length']['median']:,} | {stats['text_length']['p95']:,} | "
            f"`{top_binary[0]}` ({int(top_binary[1]):,}) |"
        )
    lines.extend(
        [
            "",
            "Important: the v5 balanced pool is balanced across attack families, but it is not a binary-safe training set by itself because every row is labeled `attack`. The DeBERTa scanner needs a matched benign/control pool before binary metrics mean anything.",
        ]
    )

    lines.extend(
        [
            "",
            "## Hermes-Katana Data Inventory",
            "",
            f"Additional corpus root checked: `{hermes['root']}`.",
            "",
            "| set | rows | attack | benign | unique units | median chars | p95 chars |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, stats in hermes_sets.items():
        binary = stats["by_binary_label"]
        lines.append(
            f"| `{name}` | {stats['rows']:,} | {int(binary.get('attack', 0)):,} | "
            f"{int(binary.get('benign', 0)):,} | {stats['unique_primary_units']:,} | "
            f"{stats['text_length']['median']:,} | {stats['text_length']['p95']:,} |"
        )
    lines.extend(
        [
            "",
            "### Hermes V4 Focused Labels",
            "",
        ]
    )
    append_count_table(
        lines, hermes_sets["hermes_v4_focused_combined"]["by_label"], hermes_sets["hermes_v4_focused_combined"]["rows"]
    )
    lines.extend(
        [
            "",
            "### Hermes V3 Binary-Balanced Labels",
            "",
        ]
    )
    append_count_table(
        lines, hermes_sets["hermes_v3_binary_balanced"]["by_label"], hermes_sets["hermes_v3_binary_balanced"]["rows"]
    )
    lines.extend(
        [
            "",
            "Text-only control side files:",
            "",
            "| file | nonempty lines | bytes |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, stats in hermes["text_controls"].items():
        lines.append(f"| `{name}` | {stats['nonempty_lines']:,} | {stats['bytes']:,} |")
    lines.extend(
        [
            "",
            "Interpretation: I was missing important Hermes-Katana data. The best immediate benign/control sources are `training/data_v4/combined.jsonl` for a focused release baseline, `training/data_v3/benign.jsonl` for schema-clean controls, and `training/data_v3/hard_negatives.jsonl` for false-positive pressure. The broader `merged_refined` files are useful for mining, but should be deduped and sampled rather than copied wholesale.",
        ]
    )

    lines.extend(
        [
            "",
            "## Validation Telemetry",
            "",
            "| scope | rows | valid | invalid | effective rows | effective units | invalid rate | valid effective rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| all runs | {all_runs['rows']:,} | {all_runs['valid']:,} | {all_runs['invalid']:,} | {all_runs['effective_rows']:,} | {all_runs['effective_unique_units']:,} | {rate(all_runs['invalid'], all_runs['rows'])} | {rate(all_runs['effective_rows'], all_runs['valid'])} |",
            f"| recent runs | {recent['rows']:,} | {recent['valid']:,} | {recent['invalid']:,} | {recent['effective_rows']:,} | {recent['effective_unique_units']:,} | {rate(recent['invalid'], recent['rows'])} | {rate(recent['effective_rows'], recent['valid'])} |",
            "",
            "### Recent Run Outcomes",
            "",
            "| run | rows | valid | invalid | effective |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run, stats in recent["by_run"].items():
        lines.append(
            f"| `{run}` | {stats.get('rows', 0):,} | {stats.get('valid', 0):,} | "
            f"{stats.get('invalid', 0):,} | {stats.get('effective', 0):,} |"
        )
    lines.extend(["", "### Top Effective Agents", ""])
    append_agent_table(lines, all_runs["by_agent_top"])
    lines.extend(
        [
            "",
            "Interpretation: telemetry is strong enough to support a reproducible validation story, but the reportable public claims should be framed as measured against this harness and these validators, not as universal model-breaking guarantees.",
        ]
    )

    lines.extend(
        [
            "",
            "## Recommended Dataset Organization",
            "",
            "Use a tiered release instead of one monolithic file:",
            "",
            "1. `gold/confirmed_attacks.jsonl`: high-precision confirmed attacks. This is the public benchmark seed and scanner stress set.",
            "2. `silver/judged_synthetic.jsonl`: synthetic and teacher/critic accepted examples, deduped by normalized family. This is the main DeBERTa training volume.",
            "3. `bronze/candidates_unconfirmed.jsonl`: plausible but unconfirmed attacks for community triage and decentralized validation.",
            "4. `benign/controls.jsonl`: matched benign task text, README/log/code/data controls, and near-miss negatives.",
            "5. `eval/heldout_public.jsonl`: frozen public eval with no train overlap by family hash.",
            "6. `eval/heldout_private_manifest.json`: hashes, labels, and scoring protocol for private leaderboard/audit use.",
            "7. `telemetry/agent_runs_manifest.jsonl`: model/channel/task/run metadata, validity, and effectiveness signals without needing to expose every raw agent output.",
            "",
            "Recommended row contract:",
            "",
            "- Required identity: `id`, `family_sha256`, `text_sha256_normalized`, `release_tier`, `source`, `provenance`.",
            "- Required labels: `binary_label`, `label`, `technique`, `quality_tier`.",
            "- Required split fields: `split`, `split_group`, `dedupe_basis`, `benchmark_eligible`.",
            "- Required evidence fields for gold/telemetry: `validated_by`, `n_validators`, `effective_count`, `invalid_count`, `first_confirmed_at`, `last_confirmed_at`.",
            "- Optional safety fields: `redacted_text`, `raw_text_available`, `public_release_ok`, `risk_notes`.",
            "",
            "## Recommended Sizes",
            "",
            "For DeBERTa, the useful public v1 target is not millions of rows. A cleaner focused release is stronger than a huge noisy dump, and it does not have to be exactly 50/50 attack/benign:",
            "",
            "- Train: 14k-24k rows total, attack-heavy is acceptable if the benign/control rows are high quality and threshold calibration uses a realistic eval set.",
            "- Validation: 2k-4k rows total, stratified by label/source/language/channel and including hard negatives.",
            "- Public test: 3k-5k rows total, family-disjoint and source-disjoint where possible.",
            "- Gold stress benchmark: 750-1,000 confirmed attacks plus matched controls, never mixed into train for headline eval.",
            "- Community queue: all remaining unconfirmed candidates with metadata and reproducible harness instructions.",
            "",
            "A practical v1 composition is 10k-14k attack rows sampled across all eight labels, 3k-6k curated benign/control rows, 1k-2k hard negatives, and a separate confirmed-only benchmark. Hermes-Katana already has enough benign and hard-negative material for this; the key is quality sampling and family-disjoint splits, not forcing class parity.",
            "",
            "## Split Policy",
            "",
            "- Split by `family_sha256` / `text_sha256_normalized`, never by row. Near-duplicates must stay in the same split.",
            "- Reserve all 2026-05-06 newly confirmed examples for eval or a small gold calibration set unless you explicitly create a later v2 model.",
            "- Keep multilingual and encoded attacks represented in every split, but make the public benchmark label-balanced.",
            "- Add near-miss negatives: prompts that look like attacks but were ineffective, benign task docs with scary words, and refusal-policy discussion that is not an attack.",
            "- Track every row with `source`, `quality_tier`, `provenance`, `family_sha256`, `label`, `binary_label`, and `release_tier`.",
            "",
            "## Public Release Package",
            "",
            "The release should make the research reproducible without requiring every contributor to trust our private logs:",
            "",
            "1. Dataset files organized by tier, with a data card describing generation, validation, dedupe, and known limitations.",
            "2. A scanner package with normalization, lexical prefilter, DeBERTa inference, thresholds, and batch-scanning CLI.",
            "3. A benchmark harness with fixtures, scoring code, and a small public eval set that anyone can run locally.",
            "4. A contribution protocol for decentralized validators: submit candidate, normalize/dedupe, run validator profile, emit signed result JSONL, promote by threshold.",
            "5. A private or delayed-release eval manifest for preventing overfitting while still allowing independent audits.",
            "",
            "## Scanner Product Shape",
            "",
            "Ship three layers, not just a classifier:",
            "",
            "1. Fast lexical/regex/canonicalization prefilter for cheap normalization and obvious trigger families.",
            "2. DeBERTa classifier for semantic attack probability and taxonomy label.",
            "3. Policy/harness evaluator that can run a small local or remote challenge suite against a model, producing comparable metrics.",
            "",
            "The public value proposition should be: protect now, measure continuously, contribute new attacks safely, and rerun the same benchmark against updated scanners.",
            "",
            "Recommended training objective:",
            "",
            "- Binary head: `attack` vs `benign/control` for scanner blocking or warning.",
            "- Taxonomy head: the eight attack labels for analyst routing and benchmark reporting.",
            "- Hard-negative score: separately monitor false positives on benign security writing, policies, logs, code, and model-evaluation prompts.",
            "- Threshold policy: ship default conservative thresholds plus a calibration script for teams with different risk tolerance.",
            "",
            "## Best Way Forward",
            "",
            "Stop broad fleet validation for now. Run targeted validation only when it answers a release-blocking question. The next high-leverage work is:",
            "",
            "1. Freeze a v1 release schema and family-dedup split script.",
            "2. Build `gold`, `silver`, `bronze`, `benign`, and `eval` manifests.",
            "3. Train DeBERTa on v5 balanced + curated controls; evaluate on held-out confirmed and hard negatives.",
            "4. Write a reproducible harness guide: how contributors submit candidates, how validators run fleets, how results are promoted.",
            "5. Publish dataset cards/model cards with limitations: agent-specific scorer signals, canary-vs-collapse distinction, invalid-run handling, and no claim that every attack generalizes to every model.",
            "",
            "## Risks",
            "",
            "- Confirmed rows are high precision but not yet huge. Do not overstate universal effectiveness.",
            "- A canary leak, collapse, and tool-delta are different failure modes. Train separate labels or at least preserve signal metadata.",
            "- Broad validation at current throughput is quota-bound and will take months. More scale should come from decentralized community validation, not a single local fleet.",
            "- Public release should avoid leaking unnecessary agent output logs; release row-level metadata and reproducible scripts instead.",
            "",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    confirmed_rows = list(read_jsonl(CONFIRMED) or [])
    summary = {
        "generated_at_unix": int(time.time()),
        "confirmed": summarize_rows(confirmed_rows),
        "confirmed_legacy": summarize_rows(list(read_jsonl(LEGACY_CONFIRMED) or [])),
        "confirmed_smoke": summarize_rows(list(read_jsonl(SMOKE_CONFIRMED) or [])),
        "synthdata": summarize_synthdata(),
        "shards": summarize_shards(),
        "hermes_katana": summarize_hermes_datasets(),
        "agent_runs_all": summarize_agent_runs(),
        "agent_runs_recent": summarize_agent_runs(SOURCE_RUNS_RECENT),
        "design_summaries": load_design_summaries(),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_report(summary)
    print(
        json.dumps(
            {
                "report": str((OUT_DIR / "report.md").relative_to(ROOT)),
                "summary": str((OUT_DIR / "summary.json").relative_to(ROOT)),
                "confirmed_rows": summary["confirmed"]["rows"],
                "confirmed_unique": summary["confirmed"]["unique_primary_units"],
                "v5_rows": summary["synthdata"]["v5_balanced"]["rows"],
                "agent_run_rows": summary["agent_runs_all"]["rows"],
                "recent_effective_units": summary["agent_runs_recent"]["effective_unique_units"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
