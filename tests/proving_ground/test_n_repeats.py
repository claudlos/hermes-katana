"""Tests for G2 (Q-E): within-cell repeats — _load_done_keys + resume logic."""

from __future__ import annotations

import json
from pathlib import Path


from run_agent_shard import (
    _load_done_keys,
    _load_done_attack_ids,
    OUTPUT_SCHEMA_VERSION,
)


def _write_rows(p: Path, rows: list[dict]) -> None:
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_done_keys_no_file(tmp_path):
    out = _load_done_keys(tmp_path / "nonexistent.jsonl")
    assert out == set()


def test_load_done_keys_pre_g2_rows_default_to_repeat_zero(tmp_path):
    """Pre-G2 rows lack repeat_idx; resume must treat them as (id, 0)."""
    p = tmp_path / "out.jsonl"
    _write_rows(
        p,
        [
            {"attack_id": "atk_a", "effective": True},
            {"attack_id": "atk_b", "effective": False},
        ],
    )
    keys = _load_done_keys(p)
    assert keys == {("atk_a", 0), ("atk_b", 0)}


def test_load_done_keys_g2_rows_distinguish_repeats(tmp_path):
    """G2 rows carry repeat_idx; resume must distinguish (id, 0) from (id, 1)."""
    p = tmp_path / "out.jsonl"
    _write_rows(
        p,
        [
            {"attack_id": "atk_a", "repeat_idx": 0, "effective": True},
            {"attack_id": "atk_a", "repeat_idx": 1, "effective": False},
            {"attack_id": "atk_b", "repeat_idx": 0, "effective": True},
        ],
    )
    keys = _load_done_keys(p)
    assert keys == {("atk_a", 0), ("atk_a", 1), ("atk_b", 0)}


def test_load_done_keys_filters_by_run_id_when_requested(tmp_path):
    """A new campaign must not resume/skip from rows written by an older run_id."""
    p = tmp_path / "out.jsonl"
    _write_rows(
        p,
        [
            {"run_id": "old", "attack_id": "atk_a", "repeat_idx": 0},
            {"run_id": "new", "attack_id": "atk_a", "repeat_idx": 1},
            {"run_id": "new", "attack_id": "atk_b", "repeat_idx": 0},
            {"attack_id": "legacy", "repeat_idx": 0},
        ],
    )
    assert _load_done_keys(p, run_id="new") == {("atk_a", 1), ("atk_b", 0)}


def test_load_done_attack_ids_legacy_returns_just_ids(tmp_path):
    """Legacy callers that ask for attack_ids only get the de-duped id set."""
    p = tmp_path / "out.jsonl"
    _write_rows(
        p,
        [
            {"attack_id": "atk_a", "repeat_idx": 0},
            {"attack_id": "atk_a", "repeat_idx": 1},
            {"attack_id": "atk_b", "repeat_idx": 0},
        ],
    )
    ids = _load_done_attack_ids(p)
    assert ids == {"atk_a", "atk_b"}


def test_load_done_keys_skips_malformed_lines(tmp_path):
    p = tmp_path / "out.jsonl"
    p.write_text('{"attack_id":"atk_a","repeat_idx":0}\nthis is not json\n{"attack_id":"atk_b","repeat_idx":2}\n')
    keys = _load_done_keys(p)
    assert keys == {("atk_a", 0), ("atk_b", 2)}


def test_schema_version_bumped_for_g2():
    """Sentinel: when G2 is rolled back this assertion guards the rollback."""
    assert OUTPUT_SCHEMA_VERSION >= 5, "schema version should be >=5 once G2 (repeat_idx) is in"
