#!/usr/bin/env python3
"""Build Scabbard-compatible 128-dim zvec centroid artifacts.

This is a maintenance tool for rebuilding ``attack_centroids_128d.npz`` from
audited JSONL training data. It deliberately writes reports and manifests next
to the artifact so future refreshes are reproducible instead of being a silent
``np.savez`` one-off.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_katana.scabbard.feature_extractor import CentroidDetector  # noqa: E402

RUNTIME_CATEGORIES = tuple(CentroidDetector.CATEGORIES)
DEFAULT_ARTIFACT_NAME = "attack_centroids_128d.npz"
DEFAULT_LABEL_POLICY = "v3_scabbard"

LABEL_POLICIES: dict[str, dict[str, Any]] = {
    "v3_scabbard": {
        "description": (
            "Fit the full V3 Scabbard centroid taxonomy, including encoding_evasion and persona_jailbreak "
            "as first-class centroid slots."
        ),
        "map": {},
        "exclude": set(),
        "fail_unknown_attack_labels": True,
    },
    "strict": {
        "description": "Use only labels that already match runtime centroid slots; fail on any other attack label.",
        "map": {},
        "exclude": set(),
        "fail_unknown_attack_labels": True,
    },
    "report_only": {
        "description": "Do not fit non-runtime labels; report any unknown attack labels without failing.",
        "map": {},
        "exclude": set(),
        "fail_unknown_attack_labels": False,
    },
}


@dataclass(frozen=True)
class DatasetRow:
    row_id: str
    text: str
    label: str
    is_attack: bool
    split: str
    source: str
    origin: str
    file: str
    line_no: int


@dataclass(frozen=True)
class PolicyDecision:
    fit_label: str | None
    reason: str
    original_label: str


@dataclass
class PreparedRows:
    included: dict[str, list[DatasetRow]]
    included_count: int
    mapped_counts: Counter[str]
    excluded_counts: Counter[str]
    original_label_counts: Counter[str]
    source_counts: Counter[str]
    origin_counts: Counter[str]
    split_counts: Counter[str]
    row_count: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def split_csv(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("split list cannot be empty")
    return items


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args: str) -> str | None:
    try:
        out = subprocess.check_output(("git", *args), cwd=ROOT, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    return out.strip() or None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def row_text(raw: dict[str, Any], *, path: Path, line_no: int) -> str:
    for key in ("text", "prompt", "content"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"{path}:{line_no}: row has no non-empty text/prompt/content field")


def read_jsonl(path: Path, *, default_split: str) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            label = str(raw.get("label") or raw.get("category") or "").strip()
            if not label:
                raise ValueError(f"{path}:{line_no}: row has no label/category field")
            row_id = str(raw.get("id") or raw.get("attack_id") or f"{path.name}:{line_no}")
            rows.append(
                DatasetRow(
                    row_id=row_id,
                    text=row_text(raw, path=path, line_no=line_no),
                    label=label,
                    is_attack=parse_bool(raw.get("is_attack", label != "clean")),
                    split=str(raw.get("split") or default_split),
                    source=str(raw.get("source") or ""),
                    origin=str(raw.get("origin") or ""),
                    file=str(path),
                    line_no=line_no,
                )
            )
    return rows


def split_path(data_dir: Path, split: str) -> Path:
    direct = data_dir / f"{split}.jsonl"
    nested = data_dir / "splits" / f"{split}.jsonl"
    if nested.is_file():
        return nested
    return direct


def load_split_rows(data_dir: Path, splits: Iterable[str]) -> tuple[list[DatasetRow], list[Path]]:
    rows: list[DatasetRow] = []
    paths: list[Path] = []
    for split in splits:
        path = split_path(data_dir, split)
        if not path.is_file():
            raise FileNotFoundError(f"missing split file for {split!r}: {path}")
        rows.extend(read_jsonl(path, default_split=split))
        paths.append(path)
    return rows, paths


def apply_label_policy(row: DatasetRow, policy_name: str) -> PolicyDecision:
    if policy_name not in LABEL_POLICIES:
        raise ValueError(f"unknown label policy: {policy_name}")
    if not row.is_attack or row.label == "clean":
        return PolicyDecision(None, "clean_or_control", row.label)

    policy = LABEL_POLICIES[policy_name]
    label_map = policy["map"]
    excluded = policy["exclude"]

    if row.label in RUNTIME_CATEGORIES:
        return PolicyDecision(row.label, "direct", row.label)
    if row.label in label_map:
        return PolicyDecision(str(label_map[row.label]), "mapped", row.label)
    if row.label in excluded:
        return PolicyDecision(None, "excluded_by_policy", row.label)
    if policy["fail_unknown_attack_labels"]:
        raise ValueError(
            f"{row.file}:{row.line_no}: attack label {row.label!r} is not covered by policy {policy_name!r}"
        )
    return PolicyDecision(None, "unknown_attack_label", row.label)


def prepare_rows(
    rows: Iterable[DatasetRow],
    *,
    policy_name: str,
    max_rows_per_class: int = 0,
    seed: int = 1337,
) -> PreparedRows:
    included: dict[str, list[DatasetRow]] = {cat: [] for cat in RUNTIME_CATEGORIES}
    mapped_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    original_label_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    row_count = 0

    for row in rows:
        row_count += 1
        original_label_counts[row.label] += 1
        source_counts[row.source or "<missing>"] += 1
        origin_counts[row.origin or "<missing>"] += 1
        split_counts[row.split] += 1
        decision = apply_label_policy(row, policy_name)
        if decision.fit_label is None:
            excluded_counts[f"{decision.reason}:{decision.original_label}"] += 1
            continue
        included[decision.fit_label].append(row)
        if decision.reason == "mapped":
            mapped_counts[f"{decision.original_label}->{decision.fit_label}"] += 1

    if max_rows_per_class > 0:
        rng = random.Random(seed)
        for label, label_rows in included.items():
            if len(label_rows) > max_rows_per_class:
                sampled = list(label_rows)
                rng.shuffle(sampled)
                included[label] = sampled[:max_rows_per_class]
                excluded_counts[f"max_rows_per_class:{label}"] += len(label_rows) - max_rows_per_class

    included_count = sum(len(label_rows) for label_rows in included.values())
    return PreparedRows(
        included=included,
        included_count=included_count,
        mapped_counts=mapped_counts,
        excluded_counts=excluded_counts,
        original_label_counts=original_label_counts,
        source_counts=source_counts,
        origin_counts=origin_counts,
        split_counts=split_counts,
        row_count=row_count,
    )


def counter_json(counter: Counter[str], *, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(k): int(v) for k, v in items}


def prepared_summary(prepared: PreparedRows) -> dict[str, Any]:
    return {
        "rows_seen": prepared.row_count,
        "rows_used_for_centroids": prepared.included_count,
        "runtime_categories": list(RUNTIME_CATEGORIES),
        "included_by_category": {cat: len(prepared.included[cat]) for cat in RUNTIME_CATEGORIES},
        "mapped_counts": counter_json(prepared.mapped_counts),
        "excluded_counts": counter_json(prepared.excluded_counts),
        "original_label_counts": counter_json(prepared.original_label_counts),
        "split_counts": counter_json(prepared.split_counts),
        "source_counts_top20": counter_json(prepared.source_counts, limit=20),
        "origin_counts": counter_json(prepared.origin_counts),
    }


def label_policy_summary(policy_name: str) -> dict[str, Any]:
    policy = LABEL_POLICIES[policy_name]
    return {
        "name": policy_name,
        "description": policy["description"],
        "map": dict(policy["map"]),
        "exclude": sorted(policy["exclude"]),
        "fail_unknown_attack_labels": bool(policy["fail_unknown_attack_labels"]),
    }


def assert_min_rows(prepared: PreparedRows, min_per_class: int) -> None:
    short = {cat: len(prepared.included[cat]) for cat in RUNTIME_CATEGORIES if len(prepared.included[cat]) < min_per_class}
    if short:
        raise ValueError(f"insufficient rows for centroid categories: {short}; min_per_class={min_per_class}")


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def normalize_rows(matrix: Any) -> Any:
    import numpy as np

    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D embedding matrix, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    return arr / norms


def normalize_vector(vector: Any) -> Any:
    import numpy as np

    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("cannot normalize empty or non-finite centroid")
    return arr / norm


def build_centroids(prepared: PreparedRows, embedder: Any, *, batch_size: int) -> dict[str, Any]:
    import numpy as np

    centroids: dict[str, Any] = {}
    for cat in RUNTIME_CATEGORIES:
        rows = prepared.included[cat]
        texts = [row.text for row in rows]
        total: Any | None = None
        count = 0
        for batch in batched(texts, batch_size):
            embs = normalize_rows(embedder.encode_batch(batch))
            if embs.shape[1] != 128:
                raise ValueError(f"zvec embedder returned {embs.shape[1]} dims, expected 128")
            total = embs.sum(axis=0) if total is None else total + embs.sum(axis=0)
            count += embs.shape[0]
        if total is None or count == 0:
            raise ValueError(f"no rows available for centroid category {cat!r}")
        centroid = normalize_vector(total / np.float32(count))
        if centroid.shape != (128,):
            raise ValueError(f"centroid {cat!r} has shape {centroid.shape}, expected (128,)")
        centroids[cat] = centroid.astype(np.float32)
    return centroids


def validate_centroids(centroids: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    stats: dict[str, Any] = {}
    missing = [cat for cat in RUNTIME_CATEGORIES if cat not in centroids]
    if missing:
        raise ValueError(f"artifact is missing centroid keys: {missing}")
    for cat in RUNTIME_CATEGORIES:
        arr = np.asarray(centroids[cat], dtype=np.float32)
        if arr.shape != (128,):
            raise ValueError(f"centroid {cat!r} has shape {arr.shape}, expected (128,)")
        if not np.isfinite(arr).all():
            raise ValueError(f"centroid {cat!r} contains non-finite values")
        norm = float(np.linalg.norm(arr))
        stats[cat] = {"shape": list(arr.shape), "norm": norm}
        if not 0.98 <= norm <= 1.02:
            raise ValueError(f"centroid {cat!r} has unexpected norm {norm:.6f}")
    return stats


def save_centroids(path: Path, centroids: dict[str, Any], *, overwrite: bool) -> None:
    import numpy as np

    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{cat: centroids[cat] for cat in RUNTIME_CATEGORIES})


def load_centroids_npz(path: Path) -> dict[str, Any]:
    import numpy as np

    data = np.load(path)
    return {cat: np.asarray(data[cat], dtype=np.float32) for cat in data.files}


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower))


def score_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "p99": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def score_dataset(
    rows: list[DatasetRow],
    *,
    detector: CentroidDetector,
    embedder: Any,
    batch_size: int,
    label_policy: str,
) -> dict[str, Any]:
    import numpy as np

    scores_all: list[float] = []
    scores_attack: list[float] = []
    scores_clean: list[float] = []
    scores_in_policy_attack: list[float] = []
    max_scores_by_original_label: dict[str, list[float]] = defaultdict(list)
    max_scores_by_fit_label: dict[str, list[float]] = defaultdict(list)
    own_slot_scores_by_fit_label: dict[str, list[float]] = defaultdict(list)
    top1_total = 0
    top1_correct = 0
    policy_counts: Counter[str] = Counter()
    category_index = {cat: i for i, cat in enumerate(RUNTIME_CATEGORIES)}

    for batch_rows in (rows[start : start + batch_size] for start in range(0, len(rows), batch_size)):
        embs = normalize_rows(embedder.encode_batch([row.text for row in batch_rows]))
        for row, emb in zip(batch_rows, embs):
            sims = detector.compute_distances(emb)
            max_score = float(np.max(sims)) if len(sims) else 0.0
            scores_all.append(max_score)
            max_scores_by_original_label[row.label].append(max_score)
            if row.is_attack:
                scores_attack.append(max_score)
            else:
                scores_clean.append(max_score)

            decision = apply_label_policy(row, label_policy)
            policy_counts[decision.reason if decision.fit_label is None else "in_policy_attack"] += 1
            if decision.fit_label is None:
                continue
            scores_in_policy_attack.append(max_score)
            max_scores_by_fit_label[decision.fit_label].append(max_score)
            own_slot_scores_by_fit_label[decision.fit_label].append(float(sims[category_index[decision.fit_label]]))
            top1_total += 1
            top_cat = RUNTIME_CATEGORIES[int(np.argmax(sims))]
            if top_cat == decision.fit_label:
                top1_correct += 1

    threshold_candidates: dict[str, Any] = {}
    if scores_in_policy_attack and scores_clean:
        for clean_percentile in (0.90, 0.95, 0.99):
            threshold = percentile(scores_clean, clean_percentile)
            if threshold is None:
                continue
            recall = sum(score >= threshold for score in scores_in_policy_attack) / len(scores_in_policy_attack)
            threshold_candidates[f"clean_p{int(clean_percentile * 100):02d}"] = {
                "threshold": threshold,
                "in_policy_attack_recall": recall,
                "clean_percentile": clean_percentile,
            }

    return {
        "rows": len(rows),
        "score_all": score_stats(scores_all),
        "score_attack": score_stats(scores_attack),
        "score_in_policy_attack": score_stats(scores_in_policy_attack),
        "score_clean": score_stats(scores_clean),
        "max_score_by_original_label": {
            label: score_stats(values) for label, values in sorted(max_scores_by_original_label.items())
        },
        "max_score_by_fit_label": {label: score_stats(values) for label, values in sorted(max_scores_by_fit_label.items())},
        "own_slot_score_by_fit_label": {
            label: score_stats(values) for label, values in sorted(own_slot_scores_by_fit_label.items())
        },
        "threshold_candidates": threshold_candidates,
        "top1_accuracy_in_policy": (top1_correct / top1_total) if top1_total else None,
        "top1_correct": top1_correct,
        "top1_total": top1_total,
        "policy_counts": counter_json(policy_counts),
    }


def compare_centroids(path: Path, compare_path: Path) -> dict[str, Any]:
    import numpy as np

    current = load_centroids_npz(path)
    baseline = load_centroids_npz(compare_path)
    out: dict[str, Any] = {"baseline": str(compare_path), "categories": {}}
    for cat in RUNTIME_CATEGORIES:
        if cat not in current or cat not in baseline:
            out["categories"][cat] = {"missing": True}
            continue
        a = normalize_vector(current[cat])
        b = normalize_vector(baseline[cat])
        out["categories"][cat] = {
            "cosine": float(np.dot(a, b)),
            "l2_delta": float(np.linalg.norm(a - b)),
            "current_norm": float(np.linalg.norm(current[cat])),
            "baseline_norm": float(np.linalg.norm(baseline[cat])),
        }
    return out


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preview_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Hermes Katana zvec Centroid Rebuild",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Label policy: `{report['label_policy']['name']}`",
        f"- Rows used: `{report['fit_summary']['rows_used_for_centroids']}`",
        "",
        "## Included Rows",
        "",
    ]
    for cat, count in report["fit_summary"]["included_by_category"].items():
        lines.append(f"- `{cat}`: {count}")
    lines.extend(["", "## Exclusions", ""])
    exclusions = report["fit_summary"].get("excluded_counts") or {}
    if exclusions:
        for reason, count in exclusions.items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None")
    if "evaluation" in report:
        lines.extend(["", "## Evaluation", ""])
        for name, result in report["evaluation"].items():
            p50 = result["score_all"]["p50"]
            p95 = result["score_all"]["p95"]
            lines.append(f"- `{name}`: rows={result['rows']} p50={p50} p95={p95}")
    lines.append("")
    return "\n".join(lines)


def source_manifest(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        manifest[str(path)] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return manifest


def resolve_zvec_paths(args: argparse.Namespace) -> dict[str, str | None]:
    zvec_base = args.zvec_base
    backbone = args.backbone
    projector = args.projector
    tokenizer = args.tokenizer
    if zvec_base is not None:
        backbone = backbone or zvec_base / "backbone_fp32"
        projector = projector or zvec_base / "projector_fp32.pt"
        tokenizer = tokenizer or zvec_base / "tokenizer"
    return {
        "zvec_base": str(zvec_base) if zvec_base else None,
        "backbone": str(backbone) if backbone else None,
        "projector": str(projector) if projector else None,
        "tokenizer": str(tokenizer) if tokenizer else None,
    }


def make_embedder(args: argparse.Namespace) -> Any:
    os.environ["HERMES_KATANA_DEVICE"] = args.device
    from hermes_katana.scabbard.embedder import ZvecEmbedder

    paths = resolve_zvec_paths(args)
    return ZvecEmbedder(
        model_path=paths["backbone"],
        projector_path=paths["projector"],
        tokenizer_path=paths["tokenizer"],
        device=args.device,
    )


def manifest_payload(
    *,
    args: argparse.Namespace,
    artifact: Path | None,
    fit_paths: list[Path],
    report_path: Path | None,
) -> dict[str, Any]:
    files = list(fit_paths)
    metadata = args.data / "metadata.json" if args.data else None
    manifest = args.data / "MANIFEST.sha256" if args.data else None
    if metadata and metadata.is_file():
        files.append(metadata)
    if manifest and manifest.is_file():
        files.append(manifest)
    if artifact and artifact.is_file():
        files.append(artifact)
    if report_path and report_path.is_file():
        files.append(report_path)
    return {
        "generated_at": utc_now(),
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "source_files": source_manifest(files),
        "zvec_paths": resolve_zvec_paths(args),
    }


def evaluate_requested(
    *,
    args: argparse.Namespace,
    detector: CentroidDetector,
    embedder: Any,
) -> dict[str, Any]:
    if args.data is None:
        return {}
    results: dict[str, Any] = {}
    eval_splits = split_csv(args.eval_splits, default=("train", "val", "test"))
    for split in eval_splits:
        path = split_path(args.data, split)
        if path.is_file():
            rows = read_jsonl(path, default_split=split)
            results[f"split:{split}"] = score_dataset(
                rows,
                detector=detector,
                embedder=embedder,
                batch_size=args.batch_size,
                label_policy=args.label_policy,
            )
    if args.eval_gold:
        path = args.data / "eval" / "gold_confirmed.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"--eval-gold requested but missing: {path}")
        results["gold_confirmed"] = score_dataset(
            read_jsonl(path, default_split="gold"),
            detector=detector,
            embedder=embedder,
            batch_size=args.batch_size,
            label_policy=args.label_policy,
        )
    if args.eval_hard_negatives:
        path = args.data / "hard_negatives.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"--eval-hard-negatives requested but missing: {path}")
        results["hard_negatives"] = score_dataset(
            read_jsonl(path, default_split="hard_negative"),
            detector=detector,
            embedder=embedder,
            batch_size=args.batch_size,
            label_policy=args.label_policy,
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebuild Scabbard-compatible 128-dim zvec centroids from audited JSONL data."
    )
    parser.add_argument("--data", type=Path, help="Dataset root containing splits/, eval/, metadata.json.")
    parser.add_argument("--out-dir", type=Path, help="Directory for artifact, report, manifest, and preview outputs.")
    parser.add_argument("--artifact", type=Path, help="Artifact to verify, or build destination if --out-dir is omitted.")
    parser.add_argument("--compare", type=Path, help="Existing centroid .npz to compare against during verification.")
    parser.add_argument("--fit-splits", help="Comma-separated split names to fit centroids from. Defaults to train.")
    parser.add_argument("--splits", help="Backward-compatible alias for --fit-splits.")
    parser.add_argument("--eval-splits", default="train,val,test", help="Comma-separated split names to evaluate.")
    parser.add_argument("--label-policy", choices=sorted(LABEL_POLICIES), default=DEFAULT_LABEL_POLICY)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", help="Torch device for zvec embedding. Defaults to cpu.")
    parser.add_argument("--zvec-base", type=Path, help="Directory containing backbone_fp32/, projector_fp32.pt, tokenizer/.")
    parser.add_argument("--backbone", type=Path, help="Override zvec backbone directory.")
    parser.add_argument("--projector", type=Path, help="Override zvec projector_fp32.pt.")
    parser.add_argument("--tokenizer", type=Path, help="Override zvec tokenizer directory.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect data and label policy without embedding or writing npz.")
    parser.add_argument("--verify", action="store_true", help="Load artifact through CentroidDetector and optionally evaluate.")
    parser.add_argument("--eval-gold", action="store_true", help="Evaluate eval/gold_confirmed.jsonl during --verify.")
    parser.add_argument("--eval-hard-negatives", action="store_true", help="Evaluate hard_negatives.jsonl during --verify.")
    parser.add_argument("--write-manifest", action="store_true", help="Write artifact manifest JSON.")
    parser.add_argument("--write-report", action="store_true", help="Write build or verify report JSON.")
    parser.add_argument("--write-preview", action="store_true", help="Write a short Markdown preview.")
    parser.add_argument("--min-per-class", type=int, default=25)
    parser.add_argument("--max-rows-per-class", type=int, default=0, help="Optional deterministic cap for debugging.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing artifact/report outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.min_per_class <= 0:
        parser.error("--min-per-class must be positive")
    if args.splits and args.fit_splits:
        parser.error("--splits and --fit-splits cannot both be set")

    fit_splits = split_csv(args.splits or args.fit_splits, default=("train",))
    # Default mode is build. ``--verify`` is intentionally verify-only so a
    # verification run cannot accidentally overwrite an existing artifact.
    build_requested = not args.dry_run and not args.verify

    if args.data is None and (args.dry_run or build_requested or args.eval_gold or args.eval_hard_negatives):
        parser.error("--data is required for dry-run, build, or data-backed verification")
    if build_requested and args.out_dir is None and args.artifact is None:
        parser.error("build requires --out-dir or --artifact")

    out_dir = args.out_dir
    artifact = args.artifact
    if build_requested and artifact is None:
        artifact = out_dir / DEFAULT_ARTIFACT_NAME if out_dir is not None else None
    if args.verify and artifact is None:
        artifact = out_dir / DEFAULT_ARTIFACT_NAME if out_dir is not None else None
    if args.verify and artifact is None:
        parser.error("--verify requires --artifact or --out-dir")

    fit_rows: list[DatasetRow] = []
    fit_paths: list[Path] = []
    prepared: PreparedRows | None = None
    if args.data is not None:
        fit_rows, fit_paths = load_split_rows(args.data, fit_splits)
        prepared = prepare_rows(
            fit_rows,
            policy_name=args.label_policy,
            max_rows_per_class=args.max_rows_per_class,
            seed=args.seed,
        )

    report: dict[str, Any] = {
        "generated_at": utc_now(),
        "mode": "dry_run" if args.dry_run else "build" if build_requested else "verify",
        "label_policy": label_policy_summary(args.label_policy),
        "fit_splits": list(fit_splits),
        "zvec_paths": resolve_zvec_paths(args),
        "artifact": str(artifact) if artifact else None,
    }
    if prepared is not None:
        report["fit_summary"] = prepared_summary(prepared)

    if args.dry_run:
        if prepared is None:
            raise AssertionError("prepared rows missing for dry-run")
        assert_min_rows(prepared, args.min_per_class)
        print(json.dumps(report["fit_summary"], indent=2, sort_keys=True))
        if args.write_report:
            if out_dir is None:
                parser.error("--write-report with --dry-run requires --out-dir")
            write_json(out_dir / "attack_centroids_128d.dry_run.json", report)
        return 0

    start = time.monotonic()
    report_path: Path | None = None

    if build_requested:
        if prepared is None:
            raise AssertionError("prepared rows missing for build")
        if artifact is None:
            parser.error("build requires an artifact path")
        assert_min_rows(prepared, args.min_per_class)
        embedder = make_embedder(args)
        centroids = build_centroids(prepared, embedder, batch_size=args.batch_size)
        report["centroid_health"] = validate_centroids(centroids)
        save_centroids(artifact, centroids, overwrite=args.overwrite)
        report["artifact_sha256"] = sha256_file(artifact)
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
        print(f"wrote {artifact}")
    else:
        embedder = None

    if args.verify:
        if artifact is None:
            parser.error("--verify requires an artifact path")
        loaded = load_centroids_npz(artifact)
        report["centroid_health"] = validate_centroids(loaded)
        detector = CentroidDetector.load(str(artifact))
        if len(detector.centroids) != len(RUNTIME_CATEGORIES):
            raise ValueError(f"CentroidDetector loaded {len(detector.centroids)} categories, expected {len(RUNTIME_CATEGORIES)}")
        if embedder is None and (args.data is not None or args.eval_gold or args.eval_hard_negatives):
            embedder = make_embedder(args)
        if embedder is not None and args.data is not None:
            report["evaluation"] = evaluate_requested(args=args, detector=detector, embedder=embedder)
        if args.compare is not None:
            drift = compare_centroids(artifact, args.compare)
            report["drift"] = drift
            if out_dir is not None:
                write_json(out_dir / "attack_centroids_128d.drift.json", drift)
        report["artifact_sha256"] = sha256_file(artifact)
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
        print(f"verified {artifact}")

    if out_dir is not None and args.write_report:
        suffix = "report" if build_requested else "verify"
        report_path = out_dir / f"attack_centroids_128d.{suffix}.json"
        write_json(report_path, report)
    if out_dir is not None and args.write_manifest:
        manifest = manifest_payload(args=args, artifact=artifact, fit_paths=fit_paths, report_path=report_path)
        write_json(out_dir / "attack_centroids_128d.manifest.json", manifest)
    if out_dir is not None and args.write_preview:
        (out_dir / "attack_centroids_128d.preview.md").write_text(preview_markdown(report), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
