"""Build follow-on discovery fleets after the quota-mix confirmation queue.

Outputs:
  Item 6:
    - shards/shard_9606.jsonl
    - scripts/fleet_free_reliable_discovery_20260506.json
    - results/designs/D-free-reliable-discovery-20260506/trial_plan.jsonl

  Item 7:
    - shards/shard_9607.jsonl
    - scripts/fleet_targeted_labels_fresh_20260506.json
    - results/designs/D-targeted-labels-fresh-20260506/trial_plan.jsonl

Both shards exclude prior design plans, confirmed attacks, and observed result
rows. Generated campaign shards (>=9000) are not scanned as source material.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "shards"
DESIGNS = ROOT / "results" / "designs"
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
CONFIRMED = [
    ROOT / "results" / "confirmed_attacks.jsonl",
    ROOT / "results" / "confirmed_attacks.v1_legacy.jsonl",
    ROOT / "results" / "synth_smoke_confirmed.jsonl",
]

LABELS = [
    "behavioral_control",
    "cognitive_state_attack",
    "content_injection",
    "encoding_evasion",
    "exfiltration_attempt",
    "jailbreak",
    "persona_jailbreak",
    "semantic_manipulation",
]
TARGET_LABELS = [
    "semantic_manipulation",
    "cognitive_state_attack",
    "content_injection",
    "behavioral_control",
]
QUALITY_ORDER = {
    "high_confidence_attack": 0,
    "medium_high_confidence_attack": 1,
    "multilingual_backtrace": 2,
}
CHANNEL_TASK = {
    "file_content": "readme_summarize",
    "tool_output": "triage_log",
}


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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
        ).strip()
    except Exception:
        return None


def label(row: dict) -> str:
    return str(row.get("attack_label") or row.get("label") or "unknown")


def text(row: dict) -> str:
    return str(row.get("text") or row.get("attack_text") or "")


def primary_id(row: dict) -> str:
    value = (
        row.get("primary_unit_id")
        or row.get("family_sha256")
        or row.get("text_sha256_normalized")
        or row.get("attack_text_sha256")
        or row.get("text_sha256")
        or row.get("id")
        or row.get("attack_id")
        or sha256_text(text(row))
    )
    value = str(value)
    return value if value.startswith("family:") else f"family:{value}"


def generated_shard(path: Path) -> bool:
    stem = path.stem
    if not stem.startswith("shard_"):
        return True
    try:
        sid = int(stem.split("_", 1)[1])
    except ValueError:
        return True
    return sid >= 9000


def collect_excluded() -> set[str]:
    excluded: set[str] = set()
    for plan_path in DESIGNS.glob("*/trial_plan.jsonl"):
        for row in read_jsonl(plan_path) or []:
            excluded.add(primary_id(row))
    for path in CONFIRMED:
        for row in read_jsonl(path) or []:
            excluded.add(primary_id(row))
    for path in AGENT_RUNS.glob("shard_*.jsonl"):
        if "_broken" in path.name:
            continue
        for row in read_jsonl(path) or []:
            excluded.add(primary_id(row))
    return excluded


def load_candidates(excluded: set[str]) -> list[dict]:
    seen: set[str] = set()
    candidates: list[dict] = []
    for path in sorted(SHARDS.glob("shard_*.jsonl")):
        if generated_shard(path):
            continue
        try:
            source_shard = int(path.stem.split("_", 1)[1])
        except ValueError:
            continue
        for idx, row in enumerate(read_jsonl(path) or []):
            if not row.get("is_attack", True):
                continue
            lab = label(row)
            if lab not in LABELS:
                continue
            pid = primary_id(row)
            if pid in excluded or pid in seen:
                continue
            t = text(row)
            if not t:
                continue
            seen.add(pid)
            out = dict(row)
            out["id"] = out.get("id") or out.get("attack_id") or f"atk_{pid.split(':', 1)[-1][:16]}"
            out["text"] = t
            out["text_sha256"] = out.get("text_sha256") or sha256_text(t)
            family = pid.split(":", 1)[-1]
            out["family_sha256"] = family
            out["text_sha256_normalized"] = out.get("text_sha256_normalized") or family
            out["label"] = lab
            out["attack_label"] = lab
            out["binary_label"] = "attack"
            out["is_attack"] = True
            out["origin"] = out.get("origin") or "user_input"
            out["source"] = out.get("source") or "unknown"
            out["source_version"] = out.get("source_version") or "unknown"
            out["quality_tier"] = out.get("quality_tier") or "unknown"
            out["language"] = out.get("language") or "unknown"
            out["_source_file"] = str(path.relative_to(ROOT))
            out["_source_row_idx"] = idx
            out["_source_shard"] = source_shard
            candidates.append(out)
    return candidates


def select_balanced(
    candidates: list[dict],
    *,
    labels: list[str],
    per_label: int,
    seed: int,
    used: set[str] | None = None,
) -> list[dict]:
    used = used or set()
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        pid = primary_id(row)
        if pid in used:
            continue
        lab = label(row)
        if lab in labels:
            by_label[lab].append(row)
    selected: list[dict] = []
    for lab in labels:
        rows = by_label[lab]
        rng.shuffle(rows)
        rows.sort(
            key=lambda r: (
                QUALITY_ORDER.get(str(r.get("quality_tier") or ""), 99),
                len(text(r)),
                str(r.get("id") or ""),
            )
        )
        selected.extend(rows[:per_label])
    return selected


def prepare_shard(rows: list[dict], shard_id: int, tag: str) -> list[dict]:
    now = int(time.time())
    out: list[dict] = []
    for idx, row in enumerate(rows):
        item = dict(row)
        item["shard"] = shard_id
        item[f"{tag}_selected_at_unix"] = now
        item[f"{tag}_shard_row_idx"] = idx
        item[f"{tag}_source_file"] = item.pop("_source_file", "")
        item[f"{tag}_source_row_idx"] = item.pop("_source_row_idx", None)
        item[f"{tag}_source_shard"] = item.pop("_source_shard", None)
        item["_shard_row_idx"] = idx
        out.append(item)
    return out


def round_robin_cap(rows: list[dict], cap: int, labels: list[str]) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[label(row)].append(row)
    selected: list[dict] = []
    while len(selected) < cap:
        progressed = False
        for lab in labels:
            bucket = by_label[lab]
            if bucket:
                selected.append(bucket.pop(0))
                progressed = True
                if len(selected) >= cap:
                    break
        if not progressed:
            break
    return selected


def make_trial_plan(
    *,
    design_id: str,
    run_id: str,
    shard_id: int,
    shard_rows: list[dict],
    shard_hash: str,
    workers: list[dict],
    seed: int,
    labels: list[str],
) -> list[dict]:
    now = int(time.time())
    plan: list[dict] = []
    seq = 0
    for worker in workers:
        agent = worker["agent"]
        channel = worker["channels"][0]
        task = worker["tasks"][0]
        cap = int(worker.get("max_attacks", len(shard_rows)))
        selected = round_robin_cap(list(shard_rows), min(cap, len(shard_rows)), labels)
        for row in selected:
            family = row["family_sha256"]
            plan.append(
                {
                    "schema_version": 1,
                    "design_id": design_id,
                    "planned_trial_id": f"{design_id}:{seq:09d}",
                    "run_id": run_id,
                    "assignment_order": seq,
                    "job_tag": f"atk:{agent}:s{shard_id}:{channel}+t:{task}",
                    "agent_id": agent,
                    "shard": shard_id,
                    "shard_file": f"shards/shard_{shard_id}.jsonl",
                    "shard_file_sha256": shard_hash,
                    "shard_row_idx": int(row["_shard_row_idx"]),
                    "channel": channel,
                    "task": task,
                    "is_control": False,
                    "matched_pair": False,
                    "multi_turn": False,
                    "split": "all",
                    "attack_id": row["id"],
                    "primary_unit_id": f"family:{family}",
                    "family_sha256": family,
                    "text_sha256": row["text_sha256"],
                    "text_sha256_normalized": row["text_sha256_normalized"],
                    "attack_label": row["label"],
                    "language": row["language"],
                    "source": row["source"],
                    "source_version": row["source_version"],
                    "quality_tier": row["quality_tier"],
                    "origin": row["origin"],
                    "cell_id": (f"agent={agent}|channel={channel}|task={task}|label={row['label']}|control=false"),
                    "block_id": f"family:{family}",
                    "stratum_id": f"label={row['label']}",
                    "repeat_idx": 0,
                    "n_repeats_planned": 1,
                    "randomization_seed": seed,
                    "planned_at_unix": now,
                    "sampling_weight": 1.0,
                }
            )
            seq += 1
    random.Random(seed).shuffle(plan)
    for idx, row in enumerate(plan):
        row["assignment_order"] = idx
        row["wave_id"] = f"wave_{idx // max(1, len(workers)):04d}"
    return plan


def write_fleet(
    *,
    design_id: str,
    run_id: str,
    shard_id: int,
    shard_rows: list[dict],
    spec_path: Path,
    design_dir: Path,
    workers: list[dict],
    seed: int,
    labels: list[str],
    comment: str,
    target: str,
    estimate: str,
) -> dict:
    shard_path = SHARDS / f"shard_{shard_id}.jsonl"
    write_jsonl(shard_path, shard_rows)
    shard_hash = file_sha256(shard_path)
    spec = {
        "_comment": comment,
        "_target": target,
        "_data": f"shards/shard_{shard_id}.jsonl; excludes prior plans, confirmed attacks, and observed result rows.",
        "_design_id": design_id,
        "_trial_plan": f"results/designs/{design_id}/trial_plan.jsonl",
        "_walltime_estimate": estimate,
        "max_concurrency": 6,
        "workers": workers,
    }
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    plan = make_trial_plan(
        design_id=design_id,
        run_id=run_id,
        shard_id=shard_id,
        shard_rows=shard_rows,
        shard_hash=shard_hash,
        workers=workers,
        seed=seed,
        labels=labels,
    )
    plan_path = design_dir / "trial_plan.jsonl"
    write_jsonl(plan_path, plan)
    summary_path = design_dir / "trial_plan_summary.json"
    summary = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "git_head": git_head(),
        "design_id": design_id,
        "run_id": run_id,
        "seed": seed,
        "shard": str(shard_path.relative_to(ROOT)),
        "shard_sha256": shard_hash,
        "spec_path": str(spec_path.relative_to(ROOT)),
        "spec_sha256": file_sha256(spec_path),
        "trial_plan": str(plan_path.relative_to(ROOT)),
        "trial_plan_sha256": file_sha256(plan_path),
        "selected_primary_units": len(shard_rows),
        "planned_trials": len(plan),
        "by_agent": dict(Counter(r["agent_id"] for r in plan)),
        "by_channel": dict(Counter(r["channel"] for r in plan)),
        "by_label": dict(Counter(r["attack_label"] for r in plan)),
        "selected_by_label": dict(Counter(row["label"] for row in shard_rows)),
        "selected_by_quality_tier": dict(Counter(row["quality_tier"] for row in shard_rows)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    excluded = collect_excluded()
    candidates = load_candidates(excluded)
    used: set[str] = set()

    free_rows_raw = select_balanced(candidates, labels=LABELS, per_label=12, seed=2026050606, used=used)
    used.update(primary_id(r) for r in free_rows_raw)
    free_rows = prepare_shard(free_rows_raw, 9606, "free_reliable")
    free_workers = [
        {
            "_lane": "copilot gpt-5-mini file",
            "agent": "copilot_cli",
            "shards": [9606],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 72,
            "n_repeats": 1,
        },
        {
            "_lane": "copilot gpt-5-mini tool",
            "agent": "copilot_cli",
            "shards": [9606],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 72,
            "n_repeats": 1,
        },
        {
            "_lane": "nous kimi file",
            "agent": "hermes_nous_kimi_k2_6",
            "shards": [9606],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 72,
            "n_repeats": 1,
        },
        {
            "_lane": "nous kimi tool",
            "agent": "hermes_nous_kimi_k2_6",
            "shards": [9606],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 72,
            "n_repeats": 1,
        },
        {
            "_lane": "nous qwen coder file",
            "agent": "hermes_nous_qwen3_coder_plus",
            "shards": [9606],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 64,
            "n_repeats": 1,
        },
        {
            "_lane": "nous qwen coder tool",
            "agent": "hermes_nous_qwen3_coder_plus",
            "shards": [9606],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 64,
            "n_repeats": 1,
        },
        {
            "_lane": "nous arcee file",
            "agent": "hermes_nous_arcee_trinity_thinking",
            "shards": [9606],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 64,
            "n_repeats": 1,
        },
        {
            "_lane": "nous step flash file",
            "agent": "hermes_nous_step_flash",
            "shards": [9606],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 64,
            "n_repeats": 1,
        },
        {
            "_lane": "openrouter free gpt-oss file",
            "agent": "hermes_or_gpt_oss_120b_free",
            "shards": [9606],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 48,
            "n_repeats": 1,
        },
        {
            "_lane": "openrouter free gpt-oss tool",
            "agent": "hermes_or_gpt_oss_120b_free",
            "shards": [9606],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 48,
            "n_repeats": 1,
        },
    ]
    free_summary = write_fleet(
        design_id="D-free-reliable-discovery-20260506",
        run_id="free_reliable_discovery_20260506",
        shard_id=9606,
        shard_rows=free_rows,
        spec_path=ROOT / "scripts" / "fleet_free_reliable_discovery_20260506.json",
        design_dir=DESIGNS / "D-free-reliable-discovery-20260506",
        workers=free_workers,
        seed=2026050606,
        labels=LABELS,
        comment="Free/reliable discovery fleet using Copilot CLI GPT-5-mini, Nous Portal models, and guarded OR GPT OSS free. Bad OR-free lanes are intentionally omitted.",
        target="Fresh uncovered data not present in prior design plans or observed run rows.",
        estimate="~2-5h; launch after confirmation run finishes.",
    )

    targeted_rows_raw = select_balanced(
        candidates,
        labels=TARGET_LABELS,
        per_label=24,
        seed=2026050607,
        used=used,
    )
    targeted_rows = prepare_shard(targeted_rows_raw, 9607, "targeted_labels")
    targeted_workers = [
        {
            "_lane": "codex file",
            "agent": "codex_cli",
            "shards": [9607],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 96,
            "n_repeats": 1,
        },
        {
            "_lane": "codex tool",
            "agent": "codex_cli",
            "shards": [9607],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 96,
            "n_repeats": 1,
        },
        {
            "_lane": "haiku file limited",
            "agent": "claude_cli_haiku",
            "shards": [9607],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 48,
            "n_repeats": 1,
        },
        {
            "_lane": "minimax m2.7 file",
            "agent": "hermes_minimax_m2_7",
            "shards": [9607],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 96,
            "n_repeats": 1,
        },
        {
            "_lane": "minimax m2.7 tool",
            "agent": "hermes_minimax_m2_7",
            "shards": [9607],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 96,
            "n_repeats": 1,
        },
        {
            "_lane": "kimi file",
            "agent": "hermes_nous_kimi_k2_6",
            "shards": [9607],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 96,
            "n_repeats": 1,
        },
        {
            "_lane": "kimi tool",
            "agent": "hermes_nous_kimi_k2_6",
            "shards": [9607],
            "channels": ["tool_output"],
            "tasks": ["triage_log"],
            "max_attacks": 96,
            "n_repeats": 1,
        },
        {
            "_lane": "qwen file",
            "agent": "hermes_nous_qwen3_coder_plus",
            "shards": [9607],
            "channels": ["file_content"],
            "tasks": ["readme_summarize"],
            "max_attacks": 64,
            "n_repeats": 1,
        },
    ]
    targeted_summary = write_fleet(
        design_id="D-targeted-labels-fresh-20260506",
        run_id="targeted_labels_fresh_20260506",
        shard_id=9607,
        shard_rows=targeted_rows,
        spec_path=ROOT / "scripts" / "fleet_targeted_labels_fresh_20260506.json",
        design_dir=DESIGNS / "D-targeted-labels-fresh-20260506",
        workers=targeted_workers,
        seed=2026050607,
        labels=TARGET_LABELS,
        comment="Targeted fresh-data sweep over the labels with the best recent yield: semantic manipulation, cognitive-state, content injection, and behavioral control.",
        target="Fresh uncovered data for high-yield labels only.",
        estimate="~2-5h; run after confirmation/free fleets depending on quota.",
    )

    manifest = {
        "schema_version": 1,
        "generated_at_unix": int(time.time()),
        "excluded_primary_units": len(excluded),
        "fresh_candidates": len(candidates),
        "free_reliable": free_summary,
        "targeted_labels": targeted_summary,
    }
    manifest_path = ROOT / "results" / "designs" / "next_fleets_20260506_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
