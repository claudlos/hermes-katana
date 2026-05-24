#!/usr/bin/env python3
"""Audit the frozen v8 proving-ground data for Hermes Katana use.

The output intentionally projects rows down to Katana-safe input fields:
attack text and metadata about the proving-ground confirmation. It does not
export agent stdout, run traces, canary values, or baseline outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SNAPSHOT = Path("results/snapshots/v8_freeze_20260515_final_post_minimax")
HK_LABELS = {
    "clean",
    "content_injection",
    "semantic_manipulation",
    "behavioral_control",
    "exfiltration_attempt",
    "jailbreak",
    "cognitive_state_attack",
    "encoding_evasion",
    "persona_jailbreak",
}
HK_ORIGINS = {
    "user_input",
    "retrieved_web",
    "mcp_tool_description",
    "mcp_tool_result",
    "prior_session_memory",
    "delegated_agent_output",
}
GOLD_AGREEMENT = {
    "confirmed_codex_and_router",
    "confirmed_codex_spark",
    "confirmed_router_free",
}
SILVER_AGREEMENT = {
    "minimax_only",
    "cross_only_effective",
}
QUARANTINE_AGREEMENT = {
    "not_reproduced",
}

VALUE_LIKE_SECRET_PATTERNS = [
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:OPENAI|ANTHROPIC|MINIMAX|AWS|GOOGLE)_API_KEY\s*=\s*[A-Za-z0-9_./+=-]{12,}\b"),
    re.compile(r"\bKATANA_CANARY[_:-][A-Za-z0-9_.:-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bCANARY[_:-][A-Za-z0-9_.:-]{12,}\b"),
]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(errors="ignore", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def split_for_family(family_sha256: str) -> str:
    digest = family_sha256 or sha256_text("missing-family")
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def group_key(row: dict[str, Any]) -> str:
    return f"{row.get('attack_id')}|{row.get('channel')}|{row.get('task')}"


def compact_group(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "agreement_class": row.get("agreement_class"),
        "attack_id": row.get("attack_id"),
        "canary_rows": int(row.get("canary_rows") or 0),
        "channel": row.get("channel"),
        "codex_spark_effective": bool(row.get("codex_spark_effective")),
        "cross_effective": bool(row.get("cross_effective")),
        "effective_agents": sorted(row.get("effective_agents") or []),
        "effective_cohorts": sorted(row.get("effective_cohorts") or []),
        "effective_model_families": sorted(row.get("effective_model_families") or []),
        "effective_rows": int(row.get("effective_rows") or 0),
        "effective_runs": sorted(row.get("effective_runs") or []),
        "families": sorted(row.get("families") or []),
        "group_key": row.get("group_key"),
        "invalid_reasons": row.get("invalid_reasons") or {},
        "invalid_rows": int(row.get("invalid_rows") or 0),
        "labels": sorted(row.get("labels") or []),
        "minimax_effective": bool(row.get("minimax_effective")),
        "planned_trials": int(row.get("planned_trials") or 0),
        "router_or_free_effective": bool(row.get("router_or_free_effective")),
        "task": row.get("task"),
        "valid_agents": sorted(row.get("valid_agents") or []),
        "valid_model_families": sorted(row.get("valid_model_families") or []),
        "valid_rows": int(row.get("valid_rows") or 0),
        "valid_runs": sorted(row.get("valid_runs") or []),
    }


def representative_score(row: dict[str, Any]) -> tuple[int, int, int]:
    family = str(row.get("_model_family") or "")
    family_bonus = {
        "codex_spark": 50,
        "free3": 40,
        "router2": 30,
        "minimax": 20,
    }.get(family, 10)
    effective_bonus = 1000 if row.get("effective") else 0
    text = str(row.get("attack_text") or "")
    return (effective_bonus + family_bonus, 1 if row.get("canary_leaked") else 0, -len(text))


def infer_origin(row: dict[str, Any], group: dict[str, Any]) -> tuple[str, str, str]:
    attack_id = str(row.get("attack_id") or group.get("attack_id") or "").lower()
    channel = str(row.get("channel") or group.get("channel") or "")

    if "tool_description" in attack_id or "mcp_tool_description" in attack_id:
        return (
            "mcp_tool_description",
            "attack_id_keyword",
            "high",
        )
    if "memory" in attack_id:
        return ("prior_session_memory", "attack_id_keyword", "medium")
    if "delegate" in attack_id or "delegated" in attack_id:
        return ("delegated_agent_output", "attack_id_keyword", "medium")
    if "mcp_tool" in attack_id or channel == "tool_output":
        return ("mcp_tool_result", "channel_or_attack_id_keyword", "high")
    if channel == "file_content":
        return ("retrieved_web", "channel_default_file_content", "medium")
    return ("user_input", "fallback", "low")


def text_flag_reasons(text: str) -> list[str]:
    reasons: list[str] = []
    for pattern in VALUE_LIKE_SECRET_PATTERNS:
        if pattern.search(text):
            reasons.append(pattern.pattern)
    return reasons


def safe_id(prefix: str, group: dict[str, Any], text_hash: str) -> str:
    raw = "|".join(
        [
            str(group.get("attack_id") or ""),
            str(group.get("channel") or ""),
            str(group.get("task") or ""),
            text_hash[:16],
        ]
    )
    suffix = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}_{suffix}"


def quality_tier(agreement_class: str) -> str:
    if agreement_class in GOLD_AGREEMENT:
        return "v8_agentic_confirmed_cross_model"
    if agreement_class == "minimax_only":
        return "v8_agentic_minimax_only"
    if agreement_class == "cross_only_effective":
        return "v8_agentic_cross_only"
    if agreement_class == "not_reproduced":
        return "v8_agentic_not_reproduced_quarantine"
    return "v8_agentic_not_effective_or_invalid"


def katana_row(
    rep: dict[str, Any],
    group: dict[str, Any],
    tier: str,
    flag_reasons: list[str],
) -> dict[str, Any]:
    text = str(rep.get("attack_text") or "")
    text_hash = sha256_text(text)
    normalized_hash = sha256_text(normalize_text(text))
    family_sha = ""
    families = group.get("families") or []
    if families:
        family_sha = str(families[0])
    if not family_sha:
        family_sha = normalized_hash
    origin, origin_source, origin_confidence = infer_origin(rep, group)
    agreement_class = str(group.get("agreement_class") or "")

    return {
        "text": text,
        "label": rep.get("attack_label") or (group.get("labels") or [None])[0],
        "source": f"proving_ground_v8_final_post_minimax:{tier}:{agreement_class}",
        "source_family": "proving_ground_v8",
        "is_attack": True,
        "binary_label": "attack",
        "origin": origin,
        "split": split_for_family(family_sha),
        "id": safe_id("pgv8", group, text_hash),
        "quality_tier": quality_tier(agreement_class),
        "family_sha256": family_sha,
        "text_sha256": text_hash,
        "text_sha256_normalized": normalized_hash,
        "text_length": len(text),
        "proving_ground": {
            "agreement_class": agreement_class,
            "attack_id": group.get("attack_id"),
            "channel": group.get("channel"),
            "task": group.get("task"),
            "group_key": group.get("group_key"),
            "effective_rows": group.get("effective_rows"),
            "effective_agents": group.get("effective_agents"),
            "effective_model_families": group.get("effective_model_families"),
            "effective_runs": group.get("effective_runs"),
            "canary_rows": group.get("canary_rows"),
            "valid_rows": group.get("valid_rows"),
            "invalid_rows": group.get("invalid_rows"),
            "representative_agent": rep.get("agent_id"),
            "representative_run_id": rep.get("run_id"),
            "representative_source_file": rep.get("_source_file"),
            "representative_source_line_no": rep.get("_source_line_no"),
            "origin_mapping_source": origin_source,
            "origin_mapping_confidence": origin_confidence,
            "text_flag_reasons": flag_reasons,
        },
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(r.get("label")) for r in rows)
    origins = Counter(str(r.get("origin")) for r in rows)
    splits = Counter(str(r.get("split")) for r in rows)
    quality = Counter(str(r.get("quality_tier")) for r in rows)
    agreement = Counter(str((r.get("proving_ground") or {}).get("agreement_class")) for r in rows)
    text_hashes = Counter(str(r.get("text_sha256_normalized")) for r in rows)
    family_hashes = Counter(str(r.get("family_sha256")) for r in rows)
    lengths = sorted(int(r.get("text_length") or 0) for r in rows)

    def pctile(pct: float) -> int:
        if not lengths:
            return 0
        idx = min(len(lengths) - 1, int((len(lengths) - 1) * pct))
        return lengths[idx]

    return {
        "rows": len(rows),
        "labels": dict(labels),
        "origins": dict(origins),
        "splits": dict(splits),
        "quality_tiers": dict(quality),
        "agreement_classes": dict(agreement),
        "unique_normalized_texts": len(text_hashes),
        "duplicate_normalized_texts": sum(1 for n in text_hashes.values() if n > 1),
        "unique_families": len(family_hashes),
        "families_with_multiple_rows": sum(1 for n in family_hashes.values() if n > 1),
        "text_length": {
            "min": lengths[0] if lengths else 0,
            "p50": pctile(0.50),
            "p90": pctile(0.90),
            "p99": pctile(0.99),
            "max": lengths[-1] if lengths else 0,
        },
    }


def write_markdown(
    path: Path,
    snapshot: Path,
    summary: dict[str, Any],
) -> None:
    source = summary["source_counts"]
    exports = summary["exports"]
    audit = summary["audit_counts"]
    lines = [
        "# Hermes Katana Data Audit - V8 Final Freeze",
        "",
        f"Snapshot: `{snapshot}`",
        "",
        "## Decision",
        "",
        "- Use `hermes_katana_gold_confirmed.jsonl` as the high-confidence v8 set.",
        "- Keep `hermes_katana_silver_candidates.jsonl` out of evaluation unless manually reviewed; it is useful for training augmentation only with family-based splits.",
        "- Do not use `hermes_katana_quarantine.jsonl` or invalid rows for Katana training metrics.",
        "- Exported Katana rows contain attack input text only; agent stdout, baseline output, and run traces are not exported.",
        "",
        "## Source Counts",
        "",
        f"- Valid worker rows: {source['valid_worker_rows']:,}",
        f"- Invalid current rows: {source['invalid_current_rows']:,}",
        f"- Invalid rows with backups: {source['invalid_with_backups_rows']:,}",
        f"- Effective worker rows: {source['effective_worker_rows']:,}",
        f"- Cross-validation groups: {source['cross_model_groups']:,}",
        f"- Focused retry candidates: {source['retry_candidates']:,}",
        f"- Batch-refused but agent-effective unique attacks: {source['batch_refused_agent_effective_unique']:,}",
        "",
        "## Export Counts",
        "",
        f"- Gold confirmed rows: {exports['gold']['rows']:,}",
        f"- Silver candidate rows: {exports['silver']['rows']:,}",
        f"- All usable rows: {exports['usable']['rows']:,}",
        f"- Quarantine rows: {exports['quarantine']['rows']:,}",
        f"- Missing-text effective groups: {audit['missing_text_groups']:,}",
        f"- Out-of-vocab label groups: {audit['out_of_vocab_label_groups']:,}",
        f"- Text secret/canary-value flags: {audit['text_value_like_secret_flags']:,}",
        "",
        "## Gold Label Distribution",
        "",
    ]
    for label, count in sorted(exports["gold"]["labels"].items()):
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Origin Projection", ""])
    for origin, count in sorted(exports["usable"]["origins"].items()):
        lines.append(f"- `{origin}`: {count}")
    lines.extend(
        [
            "",
            "Origin mapping is deterministic but partly heuristic: `tool_output` maps to "
            "`mcp_tool_result`; `file_content` maps to `retrieved_web` unless the attack id "
            "names memory, delegation, or tool-description surfaces.",
            "",
            "## Split Guidance",
            "",
            "The exports include a deterministic `split` field derived from family hash "
            "(80/10/10 train/val/test). Keep rows with the same `family_sha256` in the same "
            "split to avoid leakage across near-duplicate attacks and origin variants.",
            "",
            "## Use/Discard Guidance",
            "",
            "- Use gold rows for an adversarial benchmark or for high-confidence training positives.",
            "- Silver `minimax_only` rows are real agentic positives but lack cross-family confirmation; treat them as lower-weight training augmentation.",
            "- Silver `cross_only_effective` rows are useful but came from targeted cross-validation lanes; review before using for claims.",
            "- `not_reproduced` rows belong in quarantine because other valid model-family attempts contradicted the MiniMax signal.",
            "- Non-effective valid rows are attack attempts, not clean benign examples. Do not label them `clean` without a separate benign-data design.",
            "- Invalid rows are infrastructure artifacts and should only drive retries, not training or evaluation labels.",
            "",
            "## Batch Data",
            "",
            "`batch_refused_but_agent_effective_unique.jsonl` is useful as a static-vs-agentic "
            "gap slice: those prompts looked refuse-worthy to the batch classifier but still "
            "worked in agent context. Use it for diagnostics, not as a replacement for the "
            "agentic gold/silver labels.",
            "",
            "## Next Collection",
            "",
            "No broad OpenAI batch, free-router, or MiniMax wave is needed before first Katana "
            "integration. If more data is needed, run only the focused retry candidates from "
            "the freeze, prioritizing cells that can upgrade silver or contradicted groups into "
            "cross-model confirmed gold.",
            "",
            "## Files",
            "",
            "- `hermes_katana_gold_confirmed.jsonl`",
            "- `hermes_katana_silver_candidates.jsonl`",
            "- `hermes_katana_all_usable.jsonl`",
            "- `hermes_katana_quarantine.jsonl`",
            "- `hermes_katana_audit_summary.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    snapshot = args.snapshot
    derived = snapshot / "derived"
    out_dir = args.out_dir or (derived / "hermes_katana_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    cross_path = derived / "cross_model_confirmations.jsonl"
    valid_path = derived / "worker_valid_rows_all.jsonl"
    invalid_current_path = derived / "worker_invalid_rows_current.jsonl"
    invalid_with_backups_path = derived / "worker_invalid_rows_with_backups.jsonl"
    effective_path = derived / "worker_effective_rows_all.jsonl"
    retry_path = derived / "retry_candidates.jsonl"
    batch_refused_unique_path = derived / "batch_refused_but_agent_effective_unique.jsonl"

    groups: dict[str, dict[str, Any]] = {}
    agreement_counts: Counter[str] = Counter()
    for row in read_jsonl(cross_path):
        compact = compact_group(row)
        key = str(compact["group_key"])
        groups[key] = compact
        agreement_counts[str(compact.get("agreement_class"))] += 1

    reps: dict[str, dict[str, Any]] = {}
    rep_scores: dict[str, tuple[int, int, int]] = {}
    valid_rows = 0
    valid_effective_rows = 0
    valid_labels: Counter[str] = Counter()
    valid_channels: Counter[str] = Counter()
    valid_agents: Counter[str] = Counter()
    valid_run_ids: Counter[str] = Counter()

    for row in read_jsonl(valid_path):
        valid_rows += 1
        if row.get("effective"):
            valid_effective_rows += 1
        valid_labels[str(row.get("attack_label"))] += 1
        valid_channels[str(row.get("channel"))] += 1
        valid_agents[str(row.get("agent_id"))] += 1
        valid_run_ids[str(row.get("run_id"))] += 1

        key = group_key(row)
        if key not in groups:
            continue
        if not row.get("attack_text"):
            continue
        score = representative_score(row)
        if key not in reps or score > rep_scores[key]:
            reps[key] = row
            rep_scores[key] = score

    invalid_current_rows = count_jsonl(invalid_current_path)
    invalid_with_backups_rows = count_jsonl(invalid_with_backups_path)
    effective_rows = count_jsonl(effective_path)
    retry_candidates = count_jsonl(retry_path)
    batch_refused_agent_effective_unique = count_jsonl(batch_refused_unique_path)

    gold: list[dict[str, Any]] = []
    silver: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    missing_text_groups: list[dict[str, Any]] = []
    out_of_vocab_label_groups: list[dict[str, Any]] = []
    text_flag_groups: list[dict[str, Any]] = []

    for key, group in sorted(groups.items()):
        agreement_class = str(group.get("agreement_class") or "")
        if agreement_class == "not_effective":
            continue
        rep = reps.get(key)
        if not rep:
            missing_text_groups.append(group)
            continue
        label = str(rep.get("attack_label") or (group.get("labels") or [""])[0])
        if label not in HK_LABELS or label == "clean":
            out_of_vocab_label_groups.append(group)
            continue

        text = str(rep.get("attack_text") or "")
        flag_reasons = text_flag_reasons(text)
        if flag_reasons:
            text_flag_groups.append(group)

        if agreement_class in GOLD_AGREEMENT:
            tier = "gold"
        elif agreement_class in SILVER_AGREEMENT:
            tier = "silver"
        else:
            tier = "quarantine"
        projected = katana_row(rep, group, tier, flag_reasons)

        if flag_reasons or agreement_class in QUARANTINE_AGREEMENT:
            quarantine.append(projected)
        elif tier == "gold":
            gold.append(projected)
        elif tier == "silver":
            silver.append(projected)
        else:
            quarantine.append(projected)

    usable = gold + silver

    counts = {
        "gold": write_jsonl(out_dir / "hermes_katana_gold_confirmed.jsonl", gold),
        "silver": write_jsonl(out_dir / "hermes_katana_silver_candidates.jsonl", silver),
        "usable": write_jsonl(out_dir / "hermes_katana_all_usable.jsonl", usable),
        "quarantine": write_jsonl(out_dir / "hermes_katana_quarantine.jsonl", quarantine),
        "missing_text_groups": write_jsonl(out_dir / "missing_text_groups.jsonl", missing_text_groups),
        "out_of_vocab_label_groups": write_jsonl(
            out_dir / "out_of_vocab_label_groups.jsonl", out_of_vocab_label_groups
        ),
        "text_flag_groups": write_jsonl(out_dir / "text_value_like_secret_flag_groups.jsonl", text_flag_groups),
    }

    summary = {
        "snapshot": str(snapshot),
        "out_dir": str(out_dir),
        "source_counts": {
            "valid_worker_rows": valid_rows,
            "invalid_current_rows": invalid_current_rows,
            "invalid_with_backups_rows": invalid_with_backups_rows,
            "effective_worker_rows": effective_rows,
            "valid_effective_rows_recomputed": valid_effective_rows,
            "cross_model_groups": len(groups),
            "retry_candidates": retry_candidates,
            "batch_refused_agent_effective_unique": batch_refused_agent_effective_unique,
            "agreement_counts": dict(sorted(agreement_counts.items())),
            "valid_labels": dict(sorted(valid_labels.items())),
            "valid_channels": dict(sorted(valid_channels.items())),
            "valid_agents": dict(valid_agents.most_common()),
            "valid_run_ids": dict(sorted(valid_run_ids.items())),
        },
        "audit_counts": {
            "missing_text_groups": len(missing_text_groups),
            "out_of_vocab_label_groups": len(out_of_vocab_label_groups),
            "text_value_like_secret_flags": len(text_flag_groups),
            "groups_with_representative_text": len(reps),
            "hk_label_vocab": sorted(HK_LABELS),
            "hk_origin_vocab": sorted(HK_ORIGINS),
        },
        "exports": {
            "gold": summarize_rows(gold),
            "silver": summarize_rows(silver),
            "usable": summarize_rows(usable),
            "quarantine": summarize_rows(quarantine),
            "write_counts": counts,
        },
    }

    summary_path = out_dir / "hermes_katana_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(out_dir / "HERMES_KATANA_DATA_AUDIT.md", snapshot, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
