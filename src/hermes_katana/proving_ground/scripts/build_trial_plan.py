#!/usr/bin/env python3
"""Build a deterministic planned-trial manifest for a fleet spec.

The manifest makes denominators explicit before collection. It is intentionally
lightweight and backwards-compatible: the existing runner can continue to use
legacy shard/max_attacks execution, while reports/quarantine can compare
observed rows against this plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from hermes_katana.proving_ground.paths import pg_root
from hermes_katana.proving_ground.scripts.fleet import _expand, _load_spec


SCHEMA_VERSION = 1
ROOT = pg_root()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return None


def shard_path(shard: int) -> Path:
    return ROOT / "shards" / f"shard_{shard:04d}.jsonl"


def load_shard_rows(shard: int, *, split: str) -> tuple[Path, list[dict]]:
    path = shard_path(shard)
    if not path.exists():
        alt = ROOT / "shards" / f"shard_{shard:03d}.jsonl"
        if alt.exists():
            path = alt
    rows: list[dict] = []
    with path.open(errors="ignore", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_shard_row_idx"] = idx
            if split == "all" or str(row.get("split", "all")) == split:
                rows.append(row)
    return path, rows


def primary_unit(row: dict) -> str:
    return str(
        row.get("family_sha256")
        or row.get("text_sha256_normalized")
        or row.get("text_sha256")
        or row.get("id")
        or sha256_text(str(row.get("text", "")))
    )


def attack_id(row: dict) -> str:
    return str(row.get("id") or row.get("attack_id") or f"atk_{primary_unit(row)[:16]}")


def stratum_id(row: dict, fields: Iterable[str]) -> str:
    parts = []
    for field in fields:
        value = row.get(field)
        if value is None and field == "attack_label":
            value = row.get("label")
        if value is None and field == "language":
            value = row.get("lang") or "unknown"
        if value is None:
            value = "unknown"
        parts.append(f"{field}={value}")
    return "|".join(parts)


def balanced_take(rows: list[dict], *, n: int, seed: int, strata: list[str]) -> list[dict]:
    if n <= 0:
        return []
    if not strata:
        rng = random.Random(seed)
        out = list(rows)
        rng.shuffle(out)
        return out[:n]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[stratum_id(row, strata)].append(row)
    rng = random.Random(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    selected: list[dict] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < n and keys:
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop(0))
        keys = [k for k in keys if groups[k]]
        cursor += 1
    return selected


def make_trial_plan(spec: dict, *, design_id: str, run_id: str | None, seed: int, strata: list[str]) -> list[dict]:
    jobs = _expand(spec, run_id=run_id or "PLANNED")
    trials: list[dict] = []
    seq = 0
    for job_idx, job in enumerate(jobs):
        path, rows = load_shard_rows(job.shard, split=job.split)
        if not rows:
            raise ValueError(f"job {job.tag()} has no shard rows after split={job.split}")
        selected = balanced_take(rows, n=job.max_attacks, seed=seed + job_idx, strata=strata)
        if len(selected) < job.max_attacks:
            raise ValueError(
                f"job {job.tag()} requested max_attacks={job.max_attacks} but only {len(selected)} rows are available"
            )
        shard_hash = file_sha256(path)
        for row in selected:
            pu = primary_unit(row)
            label = row.get("label") or row.get("attack_label") or "unknown"
            task = row.get("task") or "code_review"
            for repeat_idx in range(job.n_repeats):
                planned_trial_id = f"{design_id}:{seq:09d}"
                cell_id = (
                    f"agent={job.agent}|channel={job.channel}|task={task}|"
                    f"label={label}|control={str(job.control).lower()}"
                )
                trials.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "design_id": design_id,
                        "planned_trial_id": planned_trial_id,
                        "run_id": run_id,
                        "assignment_order": seq,
                        "job_tag": job.tag(),
                        "agent_id": job.agent,
                        "shard": job.shard,
                        "shard_file": str(path.relative_to(ROOT)),
                        "shard_file_sha256": shard_hash,
                        "shard_row_idx": row.get("_shard_row_idx"),
                        "channel": job.channel,
                        "task": task,
                        "is_control": bool(job.control),
                        "matched_pair": bool(job.matched_pair),
                        "multi_turn": bool(job.multi_turn),
                        "split": job.split,
                        "attack_id": attack_id(row),
                        "primary_unit_id": f"family:{pu}",
                        "family_sha256": row.get("family_sha256") or pu,
                        "text_sha256": row.get("text_sha256") or sha256_text(str(row.get("text", ""))),
                        "text_sha256_normalized": row.get("text_sha256_normalized") or pu,
                        "attack_label": label,
                        "language": row.get("language") or row.get("lang") or "unknown",
                        "source": row.get("source") or "unknown",
                        "source_version": row.get("source_version") or "unknown",
                        "quality_tier": row.get("quality_tier") or "unknown",
                        "origin": row.get("origin") or "unknown",
                        "cell_id": cell_id,
                        "block_id": f"family:{pu}",
                        "stratum_id": stratum_id(row, strata),
                        "repeat_idx": repeat_idx,
                        "n_repeats_planned": job.n_repeats,
                        "randomization_seed": seed,
                        "planned_at_unix": int(time.time()),
                        "sampling_weight": 1.0,
                    }
                )
                seq += 1
    random.Random(seed).shuffle(trials)
    for idx, row in enumerate(trials):
        row["assignment_order"] = idx
        row["wave_id"] = f"wave_{idx // max(1, int(spec.get('max_concurrency', 1))):04d}"
    return trials


def summarize(trials: list[dict], *, design_id: str, spec_path: Path, seed: int, strata: list[str]) -> dict:
    by_cell = Counter(t["cell_id"] for t in trials)
    by_stratum = Counter(t["stratum_id"] for t in trials)
    by_agent = Counter(t["agent_id"] for t in trials)
    by_channel = Counter(t["channel"] for t in trials)
    return {
        "schema_version": SCHEMA_VERSION,
        "design_id": design_id,
        "spec_path": str(spec_path),
        "spec_sha256": file_sha256(spec_path),
        "git_head": git_head(),
        "seed": seed,
        "strata": strata,
        "planned_trials": len(trials),
        "planned_attack_trials": sum(1 for t in trials if not t["is_control"]),
        "planned_control_trials": sum(1 for t in trials if t["is_control"]),
        "unique_primary_units": len({t["primary_unit_id"] for t in trials}),
        "by_cell": dict(sorted(by_cell.items())),
        "by_stratum": dict(sorted(by_stratum.items())),
        "by_agent": dict(sorted(by_agent.items())),
        "by_channel": dict(sorted(by_channel.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--design-id", required=True)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seed", type=int, default=20260503)
    ap.add_argument("--strata", nargs="*", default=["label", "language", "source"])
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    spec_path = args.spec if args.spec.is_absolute() else ROOT / args.spec
    spec = _load_spec(spec_path)
    out_dir = args.out_dir or ROOT / "results" / "designs" / args.design_id
    out_dir.mkdir(parents=True, exist_ok=True)
    trials = make_trial_plan(spec, design_id=args.design_id, run_id=args.run_id, seed=args.seed, strata=args.strata)
    write_jsonl(out_dir / "trial_plan.jsonl", trials)
    summary = summarize(trials, design_id=args.design_id, spec_path=spec_path, seed=args.seed, strata=args.strata)
    (out_dir / "trial_plan_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "design_id": args.design_id,
        "created_at_unix": int(time.time()),
        "trial_plan": "trial_plan.jsonl",
        "trial_plan_summary": "trial_plan_summary.json",
        "trial_plan_sha256": file_sha256(out_dir / "trial_plan.jsonl"),
        "summary_sha256": file_sha256(out_dir / "trial_plan_summary.json"),
    }
    (out_dir / "design_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
