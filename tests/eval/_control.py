"""Shared execution controls for the evaluation harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

EVAL_EXECUTION_ENV = "HERMES_KATANA_RUN_EVALS"
EVAL_MAX_RECORDS_ENV = "HERMES_KATANA_EVAL_MAX_RECORDS"
EVAL_CORPUS_ENV = "HERMES_KATANA_EVAL_CORPUS"
FULL_SEMANTIC_BACKENDS = frozenset({"contrastive", "zvec_quantized"})


def eval_execution_enabled() -> bool:
    """Return whether the heavy eval suite is explicitly enabled."""
    return os.getenv(EVAL_EXECUTION_ENV) == "1"


def eval_execution_skip_reason() -> str:
    """Human-readable guidance for intentionally running evals."""
    return f"Eval suite is disabled by default. Set {EVAL_EXECUTION_ENV}=1 to run it intentionally."


def configured_max_records() -> int | None:
    """Return the optional eval corpus cap from the environment."""
    raw = os.getenv(EVAL_MAX_RECORDS_ENV)
    if raw is None:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def default_eval_corpus_path(project_root: Path) -> Path:
    """Return the preferred eval corpus path for this repo.

    Prefer the strict held-out artifact once it exists; otherwise fall back to
    the legacy normalized-clean corpus.
    """
    strict_path = project_root / "research" / "wild-attacks-2026-04-05" / "normalized-strict-heldout.jsonl"
    if strict_path.exists():
        return strict_path
    return project_root / "research" / "wild-attacks-2026-04-05" / "normalized-clean.jsonl"


def configured_eval_corpus_path(project_root: Path) -> Path:
    """Return the configured eval corpus path, honoring env override."""
    override = os.getenv(EVAL_CORPUS_ENV)
    if override:
        return Path(override)
    return default_eval_corpus_path(project_root)


def load_jsonl_records(path: Path, *, max_records: int | None = None) -> list[dict]:
    """Load JSONL records from disk with an optional cap."""
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_records is not None and len(records) >= max_records:
                break
    return records


def current_semantic_backend_name() -> str | None:
    """Return the active semantic backend name when available."""
    try:
        from hermes_katana.scanner.semantic_recall import semantic_backend_status
    except Exception:
        return None

    try:
        backend = semantic_backend_status().get("backend")
    except Exception:
        return None
    return backend if isinstance(backend, str) else None


def eval_false_positive_ceiling() -> float:
    """Return the FP ceiling appropriate for the active semantic backend."""
    backend = current_semantic_backend_name()
    return 0.05 if backend in FULL_SEMANTIC_BACKENDS else 0.15


def baseline_label_scanner_reference(
    baseline: dict[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return per-scanner baseline data for a label or a skip reason.

    Older baselines only store overall per-scanner totals, which are not
    comparable to injection-only eval slices. This helper centralizes the
    compatibility checks so both pytest and run_eval.py compare like-for-like.
    """

    per_label = baseline.get("per_label")
    if not isinstance(per_label, dict):
        return None, "Baseline lacks per-label results; refresh it with run_eval.py --update-baseline."

    label_entry = per_label.get(label)
    if not isinstance(label_entry, dict):
        return None, f"Baseline has no {label!r} label entry; refresh it with run_eval.py --update-baseline."

    per_scanner = label_entry.get("per_scanner")
    if not isinstance(per_scanner, dict):
        return None, (
            f"Baseline lacks per-scanner data for label {label!r}; refresh it with run_eval.py --update-baseline."
        )

    saved_backend = baseline.get("semantic_backend", {}).get("backend")
    current_backend = current_semantic_backend_name()
    if isinstance(saved_backend, str) and isinstance(current_backend, str) and saved_backend != current_backend:
        return None, (
            f"Baseline semantic backend {saved_backend!r} does not match "
            f"current {current_backend!r}; refresh baseline in this runtime."
        )

    return per_scanner, None
