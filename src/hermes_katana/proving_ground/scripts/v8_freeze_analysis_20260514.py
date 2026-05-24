#!/usr/bin/env python3
"""Freeze v8 fleet artifacts and build cleaned cross-validation analysis.

This is intentionally source-preserving: raw result JSONLs are copied into a
timestamped snapshot first, then all derived artifacts are produced from that
snapshot. Invalid infrastructure rows are excluded from valid/effective
analysis but retained with reasons for retry triage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
AGENT_RUNS = RESULTS / "agent_shard_runs"
SNAPSHOT_ROOT = RESULTS / "snapshots"
HERMES_KATANA_RESULTS = ROOT.parent / "hermes-katana" / "results"

RUNS = [
    {
        "run_id": "v8_main_20260511",
        "cohort": "minimax_main",
        "design_id": "D-v8-untested-synth-minimax-2x2-resume-20260512",
    },
    {
        "run_id": "v8_crossval_new_effective_20260513",
        "cohort": "codex_spark_wave1_initial",
        "design_id": "D-crossval-new-effective-20260513",
    },
    {
        "run_id": "v8_crossval_codex_new_effective_20260513",
        "cohort": "codex_spark_wave1",
        "design_id": "D-crossval-codex-new-effective-20260513",
    },
    {
        "run_id": "v8_crossval_spark2_new_effective_20260514",
        "cohort": "spark2",
        "design_id": "D-crossval-spark2-new-effective-20260514",
    },
    {
        "run_id": "v8_crossval_router2_new_effective_20260513",
        "cohort": "router2",
        "design_id": "D-crossval-router2-new-effective-20260513",
    },
    {
        "run_id": "v8_crossval_free3_new_effective_20260514",
        "cohort": "free3",
        "design_id": "D-crossval-free3-new-effective-20260514",
    },
]
RUN_IDS = [r["run_id"] for r in RUNS]
RUN_BY_ID = {r["run_id"]: r for r in RUNS}
DESIGN_IDS = [r["design_id"] for r in RUNS if r.get("design_id")]

OPENAI_BATCH_DIRS = [
    "openai_batch_v8_probe_20260512_all",
]


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(errors="ignore", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                yield {
                    "_parse_error": str(exc),
                    "_source_file": str(path),
                    "_source_line_no": line_no,
                    "_raw_line_sha256": hashlib.sha256(line.encode()).hexdigest(),
                }
                continue
            yield row


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def invalid_reason(row: dict) -> str | None:
    if row.get("_parse_error"):
        return "parse_error"
    if row.get("row_valid") is False:
        return str(row.get("invalid_reason") or "row_valid_false")
    if row.get("invalid_run") is True:
        return str(row.get("invalid_reason") or "invalid_run")
    if row.get("baseline_valid") is False:
        return str(row.get("baseline_invalid_reason") or "baseline_invalid")
    if row.get("attack_run_valid") is False:
        return str(row.get("attack_run_invalid_reason") or "attack_run_invalid")
    return None


def key_tuple(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("attack_id") or ""),
        str(row.get("channel") or ""),
        str(row.get("task") or row.get("task_name") or "secrets_audit"),
    )


def key_string(key: tuple[str, str, str]) -> str:
    return "|".join(key)


def cell_tuple(row: dict) -> tuple[str, str, str, str, str, int]:
    return (
        str(row.get("run_id") or ""),
        str(row.get("agent_id") or ""),
        str(row.get("attack_id") or ""),
        str(row.get("channel") or ""),
        str(row.get("task") or row.get("task_name") or "secrets_audit"),
        int(row.get("repeat_idx") or 0),
    )


def trial_key(row: dict) -> str:
    planned = row.get("planned_trial_id")
    if planned:
        return str(planned)
    return "|".join(map(str, cell_tuple(row)))


def row_line_identity(row: dict) -> str:
    src = row.get("_source_file") or ""
    line = row.get("_source_line_no") or ""
    return f"{src}:{line}"


def model_family(agent: str, cohort: str | None = None) -> str:
    if agent.startswith("hermes_minimax"):
        return "minimax"
    if agent.startswith("codex_cli"):
        return "codex_spark"
    if cohort == "spark2":
        return "codex_spark"
    if cohort == "router2":
        return "router2"
    if cohort == "free3":
        return "free3"
    if "arcee" in agent:
        return "free3"
    if agent.startswith("hermes_or_"):
        return "openrouter_free"
    return "other"


def compact_worker(row: dict, prefix: str = "worker") -> dict:
    fields = [
        "run_id",
        "agent_id",
        "shard",
        "channel",
        "task",
        "attack_id",
        "attack_label",
        "family_sha256",
        "text_sha256_normalized",
        "attack_text_sha256",
        "repeat_idx",
        "effective",
        "canary_leaked",
        "collapsed",
        "severity",
        "severity_top_signal",
        "row_valid",
        "invalid_run",
        "baseline_valid",
        "attack_run_valid",
    ]
    out = {f"{prefix}_{k}": row.get(k) for k in fields if k in row}
    out[f"{prefix}_cohort"] = row.get("_cohort")
    out[f"{prefix}_source_file"] = row.get("_source_file")
    out[f"{prefix}_source_line_no"] = row.get("_source_line_no")
    return out


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def ps_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["ps", "-eo", "pid,ppid,pgid,stat,etime,cmd"],
            text=True,
            errors="ignore",
        )
    except Exception as exc:
        return f"ps failed: {exc}\n"


def create_snapshot(args: argparse.Namespace) -> int:
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = args.snapshot_root / f"v8_freeze_{timestamp}"
    if snap.exists() and not args.overwrite:
        raise SystemExit(f"snapshot already exists: {snap}")
    snap.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": str(ROOT),
        "run_ids": RUN_IDS,
        "design_ids": DESIGN_IDS,
        "paths": {},
        "copied": {
            "agent_shard_run_files": [],
            "fleet_run_dirs": [],
            "design_dirs": [],
            "queue_files": [],
            "spec_files": [],
            "backup_dirs": [],
        },
    }

    agent_dst = snap / "agent_shard_runs"
    for run_id in RUN_IDS:
        files = sorted(AGENT_RUNS.glob(f"*run_{run_id}*"))
        for src in files:
            if src.is_file():
                dst = agent_dst / src.name
                copy_file(src, dst)
                manifest["copied"]["agent_shard_run_files"].append(str(dst.relative_to(snap)))

    for run_id in RUN_IDS:
        src = RESULTS / "fleet_runs" / run_id
        dst = snap / "fleet_runs" / run_id
        if src.exists():
            copy_tree(src, dst)
            manifest["copied"]["fleet_run_dirs"].append(str(dst.relative_to(snap)))

    backup_root = RESULTS / "agent_shard_runs_backups"
    if backup_root.exists():
        backup_dst_root = snap / "agent_shard_runs_backups"
        for src in sorted(backup_root.glob("v8_main_20260511*")):
            if not src.is_dir():
                continue
            dst = backup_dst_root / src.name
            copy_tree(src, dst)
            manifest["copied"]["backup_dirs"].append(str(dst.relative_to(snap)))

    for design_id in DESIGN_IDS:
        src = RESULTS / "designs" / design_id
        dst = snap / "designs" / design_id
        if src.exists():
            copy_tree(src, dst)
            manifest["copied"]["design_dirs"].append(str(dst.relative_to(snap)))
        q = RESULTS / "queues" / f"{design_id}.source_pairs.jsonl"
        if q.exists():
            dst_q = snap / "queues" / q.name
            copy_file(q, dst_q)
            manifest["copied"]["queue_files"].append(str(dst_q.relative_to(snap)))

    for spec in sorted((ROOT / "scripts").glob("fleet*crossval*effective*202605*.json")):
        dst = snap / "scripts" / spec.name
        copy_file(spec, dst)
        manifest["copied"]["spec_files"].append(str(dst.relative_to(snap)))
    main_spec = ROOT / "scripts" / "fleet_v8_untested_synth_minimax_2x2_resume.json"
    if main_spec.exists():
        dst = snap / "scripts" / main_spec.name
        copy_file(main_spec, dst)
        manifest["copied"]["spec_files"].append(str(dst.relative_to(snap)))

    (snap / "process_snapshot.txt").write_text(ps_snapshot(), encoding="utf-8")
    manifest["paths"] = {
        "snapshot": str(snap),
        "agent_shard_runs": str(agent_dst),
        "process_snapshot": str(snap / "process_snapshot.txt"),
    }
    write_json(snap / "manifest.json", manifest)
    print(json.dumps({"snapshot": str(snap), "manifest": str(snap / "manifest.json")}, indent=2))
    return 0


def load_status_totals(snapshot: Path) -> dict:
    totals = {
        "by_run": defaultdict(lambda: Counter()),
        "by_agent": defaultdict(lambda: Counter()),
        "by_channel": defaultdict(lambda: Counter()),
        "by_shard": defaultdict(lambda: Counter()),
    }
    for path in sorted((snapshot / "agent_shard_runs").glob("*.status.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = data.get("run_id")
        if run_id not in RUN_BY_ID:
            continue
        total = int(data.get("total") or 0)
        done = int(data.get("done") or 0)
        effective = int(data.get("effective") or 0)
        invalid = int(data.get("invalid_runs") or 0)
        canary = int(data.get("canary_leaks") or 0)
        dims = [
            totals["by_run"][run_id],
            totals["by_agent"][str(data.get("agent_id") or "unknown")],
            totals["by_channel"][str(data.get("channel") or "unknown")],
            totals["by_shard"][str(data.get("shard") or "unknown")],
        ]
        for bucket in dims:
            bucket["planned_status_total"] += total
            bucket["done_status"] += done
            bucket["effective_status"] += effective
            bucket["invalid_status"] += invalid
            bucket["canary_status"] += canary
    return {k: {kk: dict(vv) for kk, vv in val.items()} for k, val in totals.items()}


def load_trial_plans(snapshot: Path) -> tuple[list[dict], dict[str, dict]]:
    plans: list[dict] = []
    by_id: dict[str, dict] = {}
    for design_id in DESIGN_IDS:
        path = snapshot / "designs" / design_id / "trial_plan.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            row["_design_id"] = design_id
            plans.append(row)
            if row.get("planned_trial_id"):
                by_id[str(row["planned_trial_id"])] = row
    return plans, by_id


def load_rows(snapshot: Path) -> tuple[list[dict], list[dict], Counter, Counter]:
    valid: list[dict] = []
    invalid: list[dict] = []
    source_counts = Counter()
    invalid_reasons = Counter()
    for path in sorted((snapshot / "agent_shard_runs").glob("*.jsonl")):
        for line_no, row in enumerate(read_jsonl(path), 1):
            if not isinstance(row, dict):
                continue
            if "_source_file" not in row:
                row["_source_file"] = str(path)
            row["_source_line_no"] = row.get("_source_line_no") or line_no
            run_id = str(row.get("run_id") or "")
            if run_id not in RUN_BY_ID:
                continue
            row["_cohort"] = RUN_BY_ID[run_id]["cohort"]
            row["_model_family"] = model_family(str(row.get("agent_id") or ""), row["_cohort"])
            source_counts[run_id] += 1
            reason = invalid_reason(row)
            if reason:
                row["_invalid_reason"] = reason
                invalid.append(row)
                invalid_reasons[reason] += 1
            else:
                valid.append(row)
    return valid, invalid, source_counts, invalid_reasons


def invalid_dedupe_key(row: dict) -> tuple:
    return (
        row.get("run_id"),
        row.get("agent_id"),
        row.get("shard"),
        row.get("channel"),
        row.get("task") or row.get("task_name") or "secrets_audit",
        row.get("attack_id"),
        int(row.get("repeat_idx") or 0),
        row.get("_invalid_reason") or invalid_reason(row) or "unknown",
    )


def load_backed_up_invalid_rows(snapshot: Path, current_invalid: list[dict]) -> tuple[list[dict], Counter]:
    """Load invalid rows that were pruned from live files but retained in backups."""
    seen = {invalid_dedupe_key(row) for row in current_invalid}
    backup_invalid: list[dict] = []
    reasons = Counter()
    backup_root = snapshot / "agent_shard_runs_backups"
    if not backup_root.exists():
        return backup_invalid, reasons
    for path in sorted(backup_root.glob("v8_main_20260511*/*.jsonl")):
        for line_no, row in enumerate(read_jsonl(path), 1):
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "")
            if run_id not in RUN_BY_ID:
                continue
            reason = invalid_reason(row)
            if not reason:
                continue
            row["_source_file"] = str(path)
            row["_source_line_no"] = line_no
            row["_source_kind"] = "backup_pruned_invalid"
            row["_cohort"] = RUN_BY_ID[run_id]["cohort"]
            row["_model_family"] = model_family(str(row.get("agent_id") or ""), row["_cohort"])
            row["_invalid_reason"] = reason
            key = invalid_dedupe_key(row)
            if key in seen:
                continue
            seen.add(key)
            backup_invalid.append(row)
            reasons[reason] += 1
    return backup_invalid, reasons


def counter_table(rows: Iterable[dict], keys: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for key in keys:
        c: dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            val = str(row.get(key) or "unknown")
            c[val]["valid"] += 1
            if row.get("effective"):
                c[val]["effective"] += 1
            if row.get("canary_leaked"):
                c[val]["canary"] += 1
        out[key] = {
            val: {
                **dict(counts),
                "effective_rate": round(counts["effective"] / counts["valid"], 4) if counts["valid"] else 0.0,
            }
            for val, counts in sorted(c.items())
        }
    return out


def group_confirmations(valid: list[dict], invalid: list[dict], plans: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str], dict] = {}

    def ensure(key: tuple[str, str, str]) -> dict:
        if key not in groups:
            groups[key] = {
                "attack_id": key[0],
                "channel": key[1],
                "task": key[2],
                "group_key": key_string(key),
                "labels": set(),
                "families": set(),
                "planned_trials": 0,
                "valid_rows": 0,
                "invalid_rows": 0,
                "effective_rows": 0,
                "canary_rows": 0,
                "valid_agents": set(),
                "effective_agents": set(),
                "valid_runs": set(),
                "effective_runs": set(),
                "valid_model_families": set(),
                "effective_model_families": set(),
                "valid_cohorts": set(),
                "effective_cohorts": set(),
                "invalid_reasons": Counter(),
            }
        return groups[key]

    for plan in plans:
        rec = ensure(key_tuple(plan))
        rec["planned_trials"] += 1
        if plan.get("attack_label"):
            rec["labels"].add(str(plan["attack_label"]))
        if plan.get("family_sha256"):
            rec["families"].add(str(plan["family_sha256"]))

    for row in invalid:
        rec = ensure(key_tuple(row))
        rec["invalid_rows"] += 1
        rec["invalid_reasons"][row.get("_invalid_reason") or "unknown"] += 1
        if row.get("attack_label"):
            rec["labels"].add(str(row["attack_label"]))
        if row.get("family_sha256"):
            rec["families"].add(str(row["family_sha256"]))

    for row in valid:
        rec = ensure(key_tuple(row))
        agent = str(row.get("agent_id") or "unknown")
        run_id = str(row.get("run_id") or "unknown")
        cohort = str(row.get("_cohort") or "unknown")
        family = str(row.get("_model_family") or model_family(agent, cohort))
        rec["valid_rows"] += 1
        rec["valid_agents"].add(agent)
        rec["valid_runs"].add(run_id)
        rec["valid_cohorts"].add(cohort)
        rec["valid_model_families"].add(family)
        if row.get("attack_label"):
            rec["labels"].add(str(row["attack_label"]))
        if row.get("family_sha256"):
            rec["families"].add(str(row["family_sha256"]))
        if row.get("effective"):
            rec["effective_rows"] += 1
            rec["effective_agents"].add(agent)
            rec["effective_runs"].add(run_id)
            rec["effective_cohorts"].add(cohort)
            rec["effective_model_families"].add(family)
        if row.get("canary_leaked"):
            rec["canary_rows"] += 1

    out = []
    for rec in groups.values():
        eff_families = set(rec["effective_model_families"])
        valid_families = set(rec["valid_model_families"])
        minimax_effective = "minimax" in eff_families
        codex_effective = "codex_spark" in eff_families
        router_effective = bool(eff_families & {"router2", "free3", "openrouter_free"})
        arcee_effective = any("arcee" in a for a in rec["effective_agents"])
        cross_valid = bool(valid_families - {"minimax", "other"})
        cross_effective = bool(eff_families - {"minimax", "other"})
        if minimax_effective and codex_effective and router_effective:
            agreement = "confirmed_codex_and_router"
        elif minimax_effective and codex_effective:
            agreement = "confirmed_codex_spark"
        elif minimax_effective and router_effective:
            agreement = "confirmed_router_free"
        elif minimax_effective and cross_valid and not cross_effective:
            agreement = "not_reproduced"
        elif minimax_effective:
            agreement = "minimax_only"
        elif cross_effective:
            agreement = "cross_only_effective"
        else:
            agreement = "not_effective"

        out.append(
            {
                "group_key": rec["group_key"],
                "attack_id": rec["attack_id"],
                "channel": rec["channel"],
                "task": rec["task"],
                "labels": sorted(rec["labels"]),
                "families": sorted(rec["families"]),
                "planned_trials": rec["planned_trials"],
                "valid_rows": rec["valid_rows"],
                "invalid_rows": rec["invalid_rows"],
                "effective_rows": rec["effective_rows"],
                "canary_rows": rec["canary_rows"],
                "valid_agents": sorted(rec["valid_agents"]),
                "effective_agents": sorted(rec["effective_agents"]),
                "valid_runs": sorted(rec["valid_runs"]),
                "effective_runs": sorted(rec["effective_runs"]),
                "valid_model_families": sorted(rec["valid_model_families"]),
                "effective_model_families": sorted(rec["effective_model_families"]),
                "valid_cohorts": sorted(rec["valid_cohorts"]),
                "effective_cohorts": sorted(rec["effective_cohorts"]),
                "minimax_effective": minimax_effective,
                "codex_spark_effective": codex_effective,
                "router_or_free_effective": router_effective,
                "arcee_effective": arcee_effective,
                "cross_valid_rows": sum(1 for fam in rec["valid_model_families"] if fam not in {"minimax", "other"}),
                "cross_effective": cross_effective,
                "agreement_class": agreement,
                "invalid_reasons": dict(rec["invalid_reasons"]),
            }
        )
    out.sort(
        key=lambda r: (
            r["agreement_class"],
            -r["effective_rows"],
            r["attack_id"],
            r["channel"],
            r["task"],
        )
    )
    return out


def load_openai_predictions(batch_root: Path) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    for batch_name in OPENAI_BATCH_DIRS:
        path = batch_root / batch_name / "predictions.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            for key in (row.get("id"), row.get("text_sha256")):
                if key and key not in predictions:
                    predictions[str(key)] = row
    return predictions


def join_batch_refusals(valid: list[dict], batch_root: Path) -> tuple[list[dict], list[dict], Counter]:
    predictions = load_openai_predictions(batch_root)
    joined = []
    verdicts = Counter()
    unique: dict[str, dict] = {}
    for row in valid:
        if not row.get("effective"):
            continue
        candidates = [
            row.get("attack_id"),
            row.get("attack_text_sha256"),
            row.get("text_sha256_normalized"),
            row.get("family_sha256"),
        ]
        pred = None
        for cand in candidates:
            if cand and str(cand) in predictions:
                pred = predictions[str(cand)]
                break
        if not pred:
            continue
        verdict = str(pred.get("verdict") or "unknown")
        verdicts[verdict] += 1
        if verdict != "refuse":
            continue
        item = {
            "openai_id": pred.get("id"),
            "openai_verdict": pred.get("verdict"),
            "openai_confidence": pred.get("confidence"),
            "openai_label": pred.get("label"),
            "openai_reason": pred.get("reason"),
            **compact_worker(row),
        }
        joined.append(item)
        unique.setdefault(str(row.get("attack_id")), item)
    return joined, list(unique.values()), verdicts


def retry_candidates(
    valid: list[dict],
    invalid: list[dict],
    plans: list[dict],
    confirmations: list[dict],
) -> list[dict]:
    valid_trials = {trial_key(r) for r in valid}
    invalid_by_trial: dict[str, list[dict]] = defaultdict(list)
    for row in invalid:
        invalid_by_trial[trial_key(row)].append(row)

    conf_by_key = {row["group_key"]: row for row in confirmations}
    candidates = []
    for plan in plans:
        run_id = str(plan.get("run_id") or "")
        if run_id == "v8_main_20260511":
            continue
        tkey = trial_key(plan)
        if tkey in valid_trials:
            continue
        key = key_string(key_tuple(plan))
        conf = conf_by_key.get(key, {})
        if not conf.get("minimax_effective"):
            continue
        invalid_rows = invalid_by_trial.get(tkey, [])
        status = "invalid" if invalid_rows else "missing"
        reasons = Counter(r.get("_invalid_reason") or "unknown" for r in invalid_rows)
        agent = str(plan.get("agent_id") or "")
        family = model_family(agent, RUN_BY_ID.get(run_id, {}).get("cohort"))
        priority = 0
        if conf.get("cross_effective"):
            priority += 50
        if conf.get("effective_rows", 0) >= 2:
            priority += 20
        if family == "codex_spark":
            priority += 15
        if family in {"router2", "free3", "openrouter_free"}:
            priority += 10
        if "arcee" in agent:
            priority += 10
        if status == "invalid":
            priority += 5
        if priority < 25:
            continue
        candidates.append(
            {
                "priority": priority,
                "status": status,
                "invalid_reasons": dict(reasons),
                "planned_trial_id": plan.get("planned_trial_id"),
                "run_id": run_id,
                "design_id": plan.get("design_id") or plan.get("_design_id"),
                "agent_id": agent,
                "model_family": family,
                "attack_id": plan.get("attack_id"),
                "attack_label": plan.get("attack_label"),
                "channel": plan.get("channel"),
                "task": plan.get("task") or "secrets_audit",
                "group_key": key,
                "current_agreement_class": conf.get("agreement_class"),
                "current_effective_agents": conf.get("effective_agents", []),
                "current_effective_model_families": conf.get("effective_model_families", []),
                "retry_value": (
                    "could strengthen cross-family agreement"
                    if conf.get("cross_effective")
                    else "could test strong MiniMax-only finding"
                ),
            }
        )
    candidates.sort(key=lambda r: (-r["priority"], r["status"], r["run_id"], r["agent_id"]))
    return candidates


def rate_table(rows: list[dict], key: str, min_valid: int = 1, limit: int | None = None) -> list[dict]:
    c: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        val = str(row.get(key) or "unknown")
        c[val]["valid"] += 1
        if row.get("effective"):
            c[val]["effective"] += 1
        if row.get("canary_leaked"):
            c[val]["canary"] += 1
    out = []
    for val, counts in c.items():
        if counts["valid"] < min_valid:
            continue
        out.append(
            {
                key: val,
                "valid": counts["valid"],
                "effective": counts["effective"],
                "canary": counts["canary"],
                "effective_rate": counts["effective"] / counts["valid"] if counts["valid"] else 0.0,
            }
        )
    out.sort(key=lambda r: (-r["effective_rate"], -r["valid"], r[key]))
    return out[:limit] if limit else out


def channel_sensitivity(rows: list[dict]) -> list[dict]:
    by_label_channel: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        key = (str(row.get("attack_label") or "unknown"), str(row.get("channel") or "unknown"))
        by_label_channel[key]["valid"] += 1
        if row.get("effective"):
            by_label_channel[key]["effective"] += 1
    labels = sorted({label for label, _ in by_label_channel})
    out = []
    for label in labels:
        fc = by_label_channel[(label, "file_content")]
        to = by_label_channel[(label, "tool_output")]
        if not fc["valid"] or not to["valid"]:
            continue
        fc_rate = fc["effective"] / fc["valid"]
        to_rate = to["effective"] / to["valid"]
        out.append(
            {
                "attack_label": label,
                "file_content_valid": fc["valid"],
                "file_content_effective_rate": fc_rate,
                "tool_output_valid": to["valid"],
                "tool_output_effective_rate": to_rate,
                "delta_tool_minus_file": to_rate - fc_rate,
            }
        )
    out.sort(key=lambda r: abs(r["delta_tool_minus_file"]), reverse=True)
    return out


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines) + "\n"


def pct(n: int | float, d: int | float) -> str:
    return f"{(100 * n / d):.1f}%" if d else "0.0%"


def build_report(summary: dict, confirmations: list[dict], retry_rows: list[dict]) -> str:
    valid = summary["counts"]["valid_rows"]
    invalid = summary["counts"]["invalid_rows"]
    effective = summary["counts"]["effective_rows"]
    canary = summary["counts"]["canary_rows"]
    confirm_counts = Counter(r["agreement_class"] for r in confirmations)

    lines = [
        "# V8 Freeze Analysis Report",
        "",
        f"Snapshot: `{summary['snapshot']}`",
        "",
        "## Executive Summary",
        "",
        f"- Source worker rows in current frozen JSONLs: {valid + invalid:,}.",
        f"- Valid rows retained for analysis: {valid:,}; current invalid infrastructure rows excluded: {invalid:,}.",
        f"- Previously pruned invalid rows retained from backups: {summary['counts'].get('backed_up_invalid_rows', 0):,}.",
        f"- Effective rows among valid rows: {effective:,} ({pct(effective, valid)}); canary rows: {canary:,}.",
        f"- Cross-model confirmation groups keyed by `attack_id + channel + task`: {len(confirmations):,}.",
        f"- Retry candidates are limited to {len(retry_rows):,} high-value missing/invalid planned cells.",
        "",
        "## Fleet Totals",
        "",
    ]

    run_rows = []
    for run_id, vals in summary["by_run"].items():
        run_rows.append(
            [
                f"`{run_id}`",
                vals.get("planned_status_total", 0),
                vals.get("valid", 0),
                vals.get("invalid", 0),
                vals.get("effective", 0),
                vals.get("canary", 0),
                pct(vals.get("effective", 0), vals.get("valid", 0)),
            ]
        )
    lines.append(markdown_table(["Run", "Planned/status", "Valid", "Invalid", "Effective", "Canary", "ASR"], run_rows))

    agent_rows = []
    for row in summary["rankings"]["agents"][:16]:
        agent_rows.append(
            [
                f"`{row['agent_id']}`",
                row["valid"],
                row["effective"],
                row["canary"],
                pct(row["effective"], row["valid"]),
            ]
        )
    lines += ["", "### By Agent", "", markdown_table(["Agent", "Valid", "Effective", "Canary", "ASR"], agent_rows)]

    chan_rows = []
    for row in summary["rankings"]["channels"]:
        chan_rows.append(
            [f"`{row['channel']}`", row["valid"], row["effective"], row["canary"], pct(row["effective"], row["valid"])]
        )
    lines += ["", "### By Channel", "", markdown_table(["Channel", "Valid", "Effective", "Canary", "ASR"], chan_rows)]

    shard_rows = []
    shard_items = sorted(
        summary["by_shard"].items(),
        key=lambda item: (-item[1].get("effective", 0), -item[1].get("valid", 0), item[0]),
    )[:16]
    for shard, vals in shard_items:
        shard_rows.append(
            [
                f"`{shard}`",
                vals.get("planned_status_total", 0),
                vals.get("valid", 0),
                vals.get("invalid", 0),
                vals.get("effective", 0),
                vals.get("canary", 0),
                pct(vals.get("effective", 0), vals.get("valid", 0)),
            ]
        )
    lines += [
        "",
        "### By Shard",
        "",
        markdown_table(["Shard", "Planned/status", "Valid", "Invalid", "Effective", "Canary", "ASR"], shard_rows),
    ]

    invalid_reason_rows = []
    current_invalid = summary.get("invalid_reasons", {})
    backup_invalid = summary.get("backup_invalid_reasons", {})
    for reason in sorted(set(current_invalid) | set(backup_invalid)):
        invalid_reason_rows.append([f"`{reason}`", current_invalid.get(reason, 0), backup_invalid.get(reason, 0)])
    lines += [
        "",
        "### Invalid Reasons",
        "",
        markdown_table(["Reason", "Current frozen rows", "Pruned backup rows"], invalid_reason_rows),
    ]

    lines += ["", "## Cross-Validation Agreement", ""]
    agree_rows = [[f"`{k}`", v] for k, v in sorted(confirm_counts.items())]
    lines.append(markdown_table(["Agreement class", "Groups"], agree_rows))
    lines += [
        "",
        f"- MiniMax-only effective groups: {confirm_counts.get('minimax_only', 0):,}.",
        f"- Confirmed by Codex/Spark: {confirm_counts.get('confirmed_codex_spark', 0) + confirm_counts.get('confirmed_codex_and_router', 0):,}.",
        f"- Confirmed by router/free agents: {confirm_counts.get('confirmed_router_free', 0) + confirm_counts.get('confirmed_codex_and_router', 0):,}.",
        f"- Not reproduced after cross-family valid attempts: {confirm_counts.get('not_reproduced', 0):,}.",
    ]

    confirmed_examples = [
        r
        for r in confirmations
        if r["agreement_class"] in {"confirmed_codex_spark", "confirmed_router_free", "confirmed_codex_and_router"}
    ][:12]
    ex_rows = [
        [
            f"`{r['attack_id']}`",
            f"`{r['channel']}`",
            ",".join(r["labels"][:2]),
            ",".join(r["effective_model_families"]),
            r["effective_rows"],
        ]
        for r in confirmed_examples
    ]
    lines += [
        "",
        "### Confirmed Examples",
        "",
        markdown_table(["Attack", "Channel", "Label", "Effective families", "Rows"], ex_rows),
    ]

    lines += ["", "## Attack-Family Ranking", ""]
    label_rows = []
    for row in summary["rankings"]["labels"]:
        label_rows.append(
            [
                f"`{row['attack_label']}`",
                row["valid"],
                row["effective"],
                row["canary"],
                pct(row["effective"], row["valid"]),
            ]
        )
    lines.append(markdown_table(["Label", "Valid", "Effective", "Canary", "ASR"], label_rows))

    fam_rows = []
    for row in summary["rankings"]["family_sha256"][:12]:
        fam_rows.append(
            [f"`{row['family_sha256'][:16]}`", row["valid"], row["effective"], pct(row["effective"], row["valid"])]
        )
    lines += [
        "",
        "### Strongest Repeated Families",
        "",
        markdown_table(["Family SHA prefix", "Valid", "Effective", "ASR"], fam_rows),
    ]

    sens_rows = []
    for row in summary["channel_sensitivity"][:10]:
        sens_rows.append(
            [
                f"`{row['attack_label']}`",
                pct(row["file_content_effective_rate"], 1),
                pct(row["tool_output_effective_rate"], 1),
                f"{row['delta_tool_minus_file']:+.3f}",
            ]
        )
    lines += [
        "",
        "### Channel Sensitivity",
        "",
        markdown_table(["Label", "File ASR", "Tool ASR", "Tool minus file"], sens_rows),
    ]

    lines += [
        "",
        "## Data Quality Guidance",
        "",
        "- Use `worker_valid_rows_all.jsonl` and `worker_effective_rows_all.jsonl` for analysis.",
        "- Discard rows in `worker_invalid_rows_current.jsonl` and `worker_invalid_rows_with_backups.jsonl` from ASR and agreement summaries; they are retained only for audit and retry planning.",
        "- `worker_invalid_rows_all.jsonl` is the invalid set present in the frozen live JSONLs; `worker_invalid_rows_with_backups.jsonl` also includes invalid rows pruned before this freeze.",
        "- Treat `batch_refused_but_agent_effective.jsonl` as evidence of a static-vs-agentic gap, not as a success predictor.",
        "- Free-router data is useful but noisy. Arcee effective rows are higher-confidence than rows from failed or rate-limited OpenRouter cells.",
        "- Small targeted slices with very high ASR should not be compared directly to broad MiniMax lanes.",
        "",
        "## Retry Guidance",
        "",
        f"High-value retry candidates: {len(retry_rows):,}. The list is limited to missing/invalid planned cells tied to strong MiniMax evidence or existing cross-family findings.",
    ]
    retry_preview = [
        [
            r["priority"],
            r["status"],
            f"`{r['agent_id']}`",
            f"`{r['attack_id']}`",
            f"`{r['channel']}`",
            r["retry_value"],
        ]
        for r in retry_rows[:15]
    ]
    lines.append(markdown_table(["Priority", "Status", "Agent", "Attack", "Channel", "Value"], retry_preview))

    lines += [
        "",
        "## Recommended Next Wave",
        "",
        "Do not launch another broad OpenAI batch or free-router wave from this snapshot. A focused retry wave is justified only for the listed high-priority missing/invalid cells, especially Codex/Spark or Arcee cells that would confirm or challenge existing MiniMax findings.",
        "",
        "## Verification",
        "",
    ]
    for item in summary["verification"]:
        status = "PASS" if item["ok"] else "FAIL"
        lines.append(f"- {status}: {item['name']} - {item['detail']}")
    lines.append("")
    return "\n".join(lines)


def derive(args: argparse.Namespace) -> int:
    snapshot = args.snapshot
    if snapshot is None:
        candidates = sorted(args.snapshot_root.glob("v8_freeze_*"))
        if not candidates:
            raise SystemExit("no snapshot found")
        snapshot = candidates[-1]
    derived = snapshot / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    plans, _plans_by_id = load_trial_plans(snapshot)
    valid, invalid, source_counts, invalid_reasons = load_rows(snapshot)
    backup_invalid, backup_invalid_reasons = load_backed_up_invalid_rows(snapshot, invalid)
    invalid_with_backups = invalid + backup_invalid
    effective_rows = [row for row in valid if row.get("effective")]
    canary_rows = [row for row in valid if row.get("canary_leaked")]
    status_totals = load_status_totals(snapshot)
    confirmations = group_confirmations(valid, invalid, plans)
    batch_join, batch_unique, joined_verdicts = join_batch_refusals(valid, args.batch_root)
    retries = retry_candidates(valid, invalid, plans, confirmations)

    counts = {
        "valid_rows_all": write_jsonl(derived / "worker_valid_rows_all.jsonl", valid),
        "invalid_rows_all": write_jsonl(derived / "worker_invalid_rows_all.jsonl", invalid),
        "invalid_rows_current": write_jsonl(derived / "worker_invalid_rows_current.jsonl", invalid),
        "invalid_rows_with_backups": write_jsonl(
            derived / "worker_invalid_rows_with_backups.jsonl", invalid_with_backups
        ),
        "effective_rows_all": write_jsonl(derived / "worker_effective_rows_all.jsonl", effective_rows),
        "cross_model_confirmations": write_jsonl(derived / "cross_model_confirmations.jsonl", confirmations),
        "batch_refused_but_agent_effective": write_jsonl(
            derived / "batch_refused_but_agent_effective.jsonl", batch_join
        ),
        "batch_refused_but_agent_effective_unique": write_jsonl(
            derived / "batch_refused_but_agent_effective_unique.jsonl", batch_unique
        ),
        "retry_candidates": write_jsonl(derived / "retry_candidates.jsonl", retries),
    }

    for run_id in RUN_IDS:
        run_valid = [r for r in valid if r.get("run_id") == run_id]
        run_invalid = [r for r in invalid if r.get("run_id") == run_id]
        safe = run_id.replace("-", "_")
        counts[f"valid_{safe}"] = write_jsonl(derived / f"worker_valid_rows_{safe}.jsonl", run_valid)
        counts[f"invalid_{safe}"] = write_jsonl(derived / f"worker_invalid_rows_{safe}.jsonl", run_invalid)

    spot_effective = effective_rows[:10]
    spot_invalid = invalid[:10]
    counts["spotcheck_effective_sample"] = write_jsonl(derived / "spotcheck_effective_sample.jsonl", spot_effective)
    counts["spotcheck_invalid_sample"] = write_jsonl(derived / "spotcheck_invalid_sample.jsonl", spot_invalid)

    by_run: dict[str, Counter] = defaultdict(Counter)
    by_agent: dict[str, Counter] = defaultdict(Counter)
    by_channel: dict[str, Counter] = defaultdict(Counter)
    by_shard: dict[str, Counter] = defaultdict(Counter)
    for row in valid:
        dims = [
            by_run[str(row.get("run_id") or "unknown")],
            by_agent[str(row.get("agent_id") or "unknown")],
            by_channel[str(row.get("channel") or "unknown")],
            by_shard[str(row.get("shard") or "unknown")],
        ]
        for bucket in dims:
            bucket["valid"] += 1
            if row.get("effective"):
                bucket["effective"] += 1
            if row.get("canary_leaked"):
                bucket["canary"] += 1
    for row in invalid:
        for bucket in [
            by_run[str(row.get("run_id") or "unknown")],
            by_agent[str(row.get("agent_id") or "unknown")],
            by_channel[str(row.get("channel") or "unknown")],
            by_shard[str(row.get("shard") or "unknown")],
        ]:
            bucket["invalid"] += 1

    # Merge status-level planned/done totals into run/agent/channel/shard summaries.
    for name, target in [
        ("by_run", by_run),
        ("by_agent", by_agent),
        ("by_channel", by_channel),
        ("by_shard", by_shard),
    ]:
        for key, vals in status_totals.get(name, {}).items():
            target[key].update(vals)

    rankings = {
        "agents": rate_table(valid, "agent_id", limit=50),
        "channels": rate_table(valid, "channel"),
        "labels": rate_table(valid, "attack_label", min_valid=10),
        "family_sha256": rate_table(valid, "family_sha256", min_valid=3, limit=50),
    }

    verification = []
    parsed_total = sum(source_counts.values())
    verification.append(
        {
            "name": "valid_invalid_reconcile",
            "ok": parsed_total == len(valid) + len(invalid),
            "detail": f"parsed={parsed_total} valid+invalid={len(valid) + len(invalid)}",
        }
    )
    verification.append(
        {
            "name": "invalid_excluded_from_effective",
            "ok": not any(invalid_reason(row) for row in effective_rows),
            "detail": f"effective_rows_checked={len(effective_rows)}",
        }
    )
    verification.append(
        {
            "name": "cross_model_key_fields_present",
            "ok": all(r["attack_id"] and r["channel"] and r["task"] for r in confirmations),
            "detail": "keys are attack_id + channel + task",
        }
    )
    verification.append(
        {
            "name": "spotcheck_samples_written",
            "ok": len(spot_effective) == min(10, len(effective_rows)) and len(spot_invalid) == min(10, len(invalid)),
            "detail": "spotcheck_effective_sample.jsonl and spotcheck_invalid_sample.jsonl",
        }
    )
    ps_text = ps_snapshot()
    forbidden = [
        line
        for line in ps_text.splitlines()
        if ("batch_run.py" in line or "batch_watcher.py" in line or "openai_batch" in line)
        and "v8_freeze_analysis_20260514.py" not in line
    ]
    verification.append(
        {
            "name": "no_openai_batch_runner_started",
            "ok": not forbidden,
            "detail": f"matching_processes={len(forbidden)}",
        }
    )

    summary = {
        "snapshot": str(snapshot),
        "derived_dir": str(derived),
        "run_ids": RUN_IDS,
        "counts": {
            "source_rows_by_run": dict(source_counts),
            "valid_rows": len(valid),
            "invalid_rows": len(invalid),
            "backed_up_invalid_rows": len(backup_invalid),
            "invalid_rows_with_backups": len(invalid_with_backups),
            "effective_rows": len(effective_rows),
            "canary_rows": len(canary_rows),
            "planned_crossval_trials": len(plans),
            "cross_model_confirmation_groups": len(confirmations),
            "batch_refused_effective_rows": len(batch_join),
            "batch_refused_effective_unique_attacks": len(batch_unique),
            "retry_candidates": len(retries),
            **counts,
        },
        "invalid_reasons": dict(invalid_reasons),
        "backup_invalid_reasons": dict(backup_invalid_reasons),
        "joined_effective_openai_verdicts": dict(joined_verdicts),
        "by_run": {k: dict(v) for k, v in sorted(by_run.items())},
        "by_agent": {k: dict(v) for k, v in sorted(by_agent.items())},
        "by_channel": {k: dict(v) for k, v in sorted(by_channel.items())},
        "by_shard": {k: dict(v) for k, v in sorted(by_shard.items())},
        "rankings": rankings,
        "channel_sensitivity": channel_sensitivity(valid),
        "agreement_counts": dict(Counter(r["agreement_class"] for r in confirmations)),
        "verification": verification,
    }
    write_json(derived / "summary.json", summary)
    report = build_report(summary, confirmations, retries)
    (derived / "report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {"derived": str(derived), "summary": str(derived / "summary.json"), "report": str(derived / "report.md")},
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("snapshot", help="copy run artifacts into a timestamped snapshot")
    p_snap.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    p_snap.add_argument("--timestamp", default=None)
    p_snap.add_argument("--overwrite", action="store_true")
    p_snap.set_defaults(func=create_snapshot)

    p_derive = sub.add_parser("derive", help="build cleaned JSONLs and report from a snapshot")
    p_derive.add_argument("--snapshot-root", type=Path, default=SNAPSHOT_ROOT)
    p_derive.add_argument("--snapshot", type=Path, default=None)
    p_derive.add_argument("--batch-root", type=Path, default=HERMES_KATANA_RESULTS)
    p_derive.set_defaults(func=derive)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
