"""Tests for run_agent_shard._load_shard --split filter (G1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_katana.proving_ground.run_agent_shard import _load_shard


REPO_ROOT = Path(__file__).resolve().parents[1]


def _shard_exists(shard_id: int, control: bool = False) -> bool:
    if control:
        return (REPO_ROOT / "shards" / "control" / f"shard_ctrl_{shard_id:03d}.jsonl").exists()
    return (REPO_ROOT / "shards" / f"shard_{shard_id:03d}.jsonl").exists()


def _split_counts(shard_id: int) -> dict[str, int]:
    """Read shard once, return raw split-value counts. Used to derive
    expected values for the assertions below without hardcoding numbers
    that drift as the corpus evolves."""
    counts: dict[str, int] = {}
    with (REPO_ROOT / "shards" / f"shard_{shard_id:03d}.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            s = r.get("split", "<missing>")
            counts[s] = counts.get(s, 0) + 1
    return counts


@pytest.fixture(scope="module")
def shard100_counts():
    if not _shard_exists(100):
        pytest.skip("shard 100 not present in this checkout")
    return _split_counts(100)


def test_default_all_returns_all_rows(shard100_counts):
    rows = _load_shard(100, split="all")
    assert len(rows) == sum(shard100_counts.values())


def test_split_train_filters_to_train_rows_only(shard100_counts):
    rows = _load_shard(100, split="train")
    expected = shard100_counts.get("train", 0)
    assert len(rows) == expected
    assert all(r.get("split") == "train" for r in rows), "filter must not leak rows of a different split"


def test_split_val_filters_to_val_rows_only(shard100_counts):
    rows = _load_shard(100, split="val")
    expected = shard100_counts.get("val", 0)
    assert len(rows) == expected
    assert all(r.get("split") == "val" for r in rows)


def test_split_test_returns_zero_when_no_test_rows():
    """Shard 1 has no test rows by current convention; the filter should
    silently return [] rather than raise. The runner's outer wrapper is
    responsible for noticing 'no work' and exiting early with a hint."""
    if not _shard_exists(1):
        pytest.skip("shard 1 not present")
    rows = _load_shard(1, split="test")
    assert rows == []


def test_control_mode_forces_split_all():
    """Control shards are not split-labelled. Passing --split=train
    against --control must NOT silently filter them to zero. The runner
    overrides split to 'all' inside _load_shard when control=True."""
    if not _shard_exists(1, control=True):
        pytest.skip("control shard 1 not present")
    rows_train = _load_shard(1, control=True, split="train")
    rows_all = _load_shard(1, control=True, split="all")
    assert len(rows_train) == len(rows_all) > 0
