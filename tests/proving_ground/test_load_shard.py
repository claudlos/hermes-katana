"""Tests for run_agent_shard._load_shard --split filter (G1)."""

from __future__ import annotations

import pytest

from hermes_katana.proving_ground.run_agent_shard import _load_shard


@pytest.fixture()
def shard_workspace(tmp_path, monkeypatch):
    shards = tmp_path / "shards"
    control = shards / "control"
    control.mkdir(parents=True)
    (shards / "shard_100.jsonl").write_text(
        "\n".join(
            [
                '{"id":"train-1","split":"train","prompt":"train row one"}',
                '{"id":"train-2","split":"train","prompt":"train row two"}',
                '{"id":"val-1","split":"val","prompt":"val row"}',
                '{"id":"test-1","split":"test","prompt":"test row"}',
            ]
        )
        + "\n"
    )
    (shards / "shard_001.jsonl").write_text(
        "\n".join(
            [
                '{"id":"old-train","split":"train","prompt":"legacy train row"}',
                '{"id":"old-val","split":"val","prompt":"legacy val row"}',
            ]
        )
        + "\n"
    )
    (control / "shard_ctrl_001.jsonl").write_text(
        "\n".join(
            [
                '{"id":"ctrl-1","prompt":"benign control one"}',
                '{"id":"ctrl-2","prompt":"benign control two"}',
            ]
        )
        + "\n"
    )
    monkeypatch.chdir(tmp_path)
    return {
        "shard100_counts": {"train": 2, "val": 1, "test": 1},
        "control_rows": 2,
    }


def test_default_all_returns_all_rows(shard_workspace):
    rows = _load_shard(100, split="all")
    assert len(rows) == sum(shard_workspace["shard100_counts"].values())


def test_split_train_filters_to_train_rows_only(shard_workspace):
    rows = _load_shard(100, split="train")
    expected = shard_workspace["shard100_counts"].get("train", 0)
    assert len(rows) == expected
    assert all(r.get("split") == "train" for r in rows), "filter must not leak rows of a different split"


def test_split_val_filters_to_val_rows_only(shard_workspace):
    rows = _load_shard(100, split="val")
    expected = shard_workspace["shard100_counts"].get("val", 0)
    assert len(rows) == expected
    assert all(r.get("split") == "val" for r in rows)


def test_split_test_returns_zero_when_no_test_rows(shard_workspace):
    """Shard 1 has no test rows by current convention; the filter should
    silently return [] rather than raise. The runner's outer wrapper is
    responsible for noticing 'no work' and exiting early with a hint."""
    rows = _load_shard(1, split="test")
    assert rows == []


def test_control_mode_forces_split_all(shard_workspace):
    """Control shards are not split-labelled. Passing --split=train
    against --control must NOT silently filter them to zero. The runner
    overrides split to 'all' inside _load_shard when control=True."""
    rows_train = _load_shard(1, control=True, split="train")
    rows_all = _load_shard(1, control=True, split="all")
    assert len(rows_train) == len(rows_all) == shard_workspace["control_rows"]
