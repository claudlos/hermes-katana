"""Tests for evaluation harness execution controls."""

from __future__ import annotations

import json

from tests.eval import _control as control
from tests.eval._control import (
    EVAL_CORPUS_ENV,
    EVAL_EXECUTION_ENV,
    EVAL_MAX_RECORDS_ENV,
    baseline_label_scanner_reference,
    configured_max_records,
    configured_eval_corpus_path,
    default_eval_corpus_path,
    eval_false_positive_ceiling,
    eval_execution_enabled,
    eval_execution_skip_reason,
    load_jsonl_records,
)


def test_eval_execution_enabled(monkeypatch):
    monkeypatch.delenv(EVAL_EXECUTION_ENV, raising=False)
    assert eval_execution_enabled() is False

    monkeypatch.setenv(EVAL_EXECUTION_ENV, "1")
    assert eval_execution_enabled() is True


def test_eval_execution_skip_reason_mentions_env():
    assert EVAL_EXECUTION_ENV in eval_execution_skip_reason()


def test_configured_max_records_handles_invalid_values(monkeypatch):
    monkeypatch.delenv(EVAL_MAX_RECORDS_ENV, raising=False)
    assert configured_max_records() is None

    monkeypatch.setenv(EVAL_MAX_RECORDS_ENV, "25")
    assert configured_max_records() == 25

    monkeypatch.setenv(EVAL_MAX_RECORDS_ENV, "0")
    assert configured_max_records() is None

    monkeypatch.setenv(EVAL_MAX_RECORDS_ENV, "not-a-number")
    assert configured_max_records() is None


def test_default_eval_corpus_prefers_strict(tmp_path):
    root = tmp_path
    corpus_dir = root / "research" / "wild-attacks-2026-04-05"
    corpus_dir.mkdir(parents=True)
    legacy = corpus_dir / "normalized-clean.jsonl"
    strict = corpus_dir / "normalized-strict-heldout.jsonl"
    legacy.write_text("", encoding="utf-8")

    assert default_eval_corpus_path(root) == legacy

    strict.write_text("", encoding="utf-8")
    assert default_eval_corpus_path(root) == strict


def test_configured_eval_corpus_path_honors_override(monkeypatch, tmp_path):
    override = tmp_path / "custom.jsonl"
    monkeypatch.setenv(EVAL_CORPUS_ENV, str(override))

    assert configured_eval_corpus_path(tmp_path) == override


def test_load_jsonl_records_honors_max_records(tmp_path):
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        "\n".join(json.dumps({"id": idx, "attack_text": f"attack-{idx}"}) for idx in range(4)) + "\n",
        encoding="utf-8",
    )

    records = load_jsonl_records(path, max_records=2)

    assert [record["id"] for record in records] == [0, 1]


def test_eval_false_positive_ceiling_tracks_backend(monkeypatch):
    monkeypatch.setattr(control, "current_semantic_backend_name", lambda: "contrastive")
    assert eval_false_positive_ceiling() == 0.05

    monkeypatch.setattr(control, "current_semantic_backend_name", lambda: "minilm_fallback")
    assert eval_false_positive_ceiling() == 0.15


def test_baseline_label_scanner_reference_requires_per_scanner_data():
    baseline = {"per_label": {"injection": {"total": 10, "caught": 9, "coverage": 0.9}}}

    reference, reason = baseline_label_scanner_reference(baseline, label="injection")

    assert reference is None
    assert "per-scanner data" in reason


def test_baseline_label_scanner_reference_detects_backend_mismatch(monkeypatch):
    monkeypatch.setattr(control, "current_semantic_backend_name", lambda: "minilm_fallback")
    baseline = {
        "semantic_backend": {"backend": "contrastive"},
        "per_label": {"injection": {"per_scanner": {"injection": {"deny": 3}}}},
    }

    reference, reason = baseline_label_scanner_reference(baseline, label="injection")

    assert reference is None
    assert "does not match current" in reason


def test_baseline_label_scanner_reference_returns_matching_slice(monkeypatch):
    monkeypatch.setattr(control, "current_semantic_backend_name", lambda: "minilm_fallback")
    baseline = {
        "semantic_backend": {"backend": "minilm_fallback"},
        "per_label": {"injection": {"per_scanner": {"injection": {"deny": 3}}}},
    }

    reference, reason = baseline_label_scanner_reference(baseline, label="injection")

    assert reference == {"injection": {"deny": 3}}
    assert reason is None
