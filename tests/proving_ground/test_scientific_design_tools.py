from __future__ import annotations

import json

import pytest

from hermes_katana.proving_ground.research.statistics import (
    cluster_bootstrap_mean_ci,
    paired_cluster_bootstrap_ci,
    sample_size_paired_delta,
    sample_size_single_proportion,
)
from scripts import build_trial_plan as btp
from scripts.quarantine_invalid_rows import denominator_summary, quarantine_rows


def test_power_planning_uses_independent_family_units():
    assert sample_size_single_proportion(half_width=0.05, conf=0.95, p=0.5) == 385
    assert sample_size_paired_delta(delta=0.10, discordance=0.30, power=0.80, alpha=0.05) == 228


def test_cluster_bootstrap_collapses_repeats_before_ci():
    point, lo, hi = cluster_bootstrap_mean_ci(
        {
            "family:a": [1, 1, 0],  # family rate 2/3
            "family:b": [0, 0],  # family rate 0
            "family:c": [1],  # family rate 1
        },
        iters=200,
        seed=7,
    )
    assert point == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_paired_cluster_bootstrap_uses_only_common_families():
    point, lo, hi, n = paired_cluster_bootstrap_ci(
        {"a": [1, 1], "b": [0], "extra_a": [1]},
        {"a": [0, 1], "b": [0], "extra_b": [0]},
        iters=200,
        seed=4,
    )
    assert n == 2
    assert point == pytest.approx(((1.0 - 0.5) + (0.0 - 0.0)) / 2)
    assert lo <= point <= hi


def test_quarantine_splits_invalid_duplicate_orphan_and_missing_rows():
    plan = {
        "p1": {"planned_trial_id": "p1", "cell_id": "c1"},
        "p2": {"planned_trial_id": "p2", "cell_id": "c1"},
    }
    observed = [
        {
            "planned_trial_id": "p1",
            "run_id": "r",
            "row_valid": True,
            "effective": True,
            "is_control": False,
            "cell_id": "c1",
        },
        {
            "planned_trial_id": "p1",
            "run_id": "r",
            "row_valid": True,
            "effective": False,
            "is_control": False,
            "cell_id": "c1",
        },
        {
            "planned_trial_id": "p3",
            "run_id": "r",
            "row_valid": True,
            "effective": True,
            "is_control": False,
            "cell_id": "c1",
        },
        {"run_id": "r", "row_valid": False, "invalid_reason": "timeout", "effective": False, "is_control": False},
    ]
    buckets = quarantine_rows(plan_rows=plan, observed_rows=observed, run_id="r")
    summary = denominator_summary(buckets, planned_n=len(plan))
    assert summary["valid_attack_rows"] == 1
    assert summary["duplicate_trial_rows"] == 1
    assert summary["orphan_observed_rows"] == 1
    assert summary["invalid_infrastructure_rows"] == 1
    assert summary["missing_planned_trials"] == 1


def test_quarantine_can_explicitly_exclude_known_bad_rows():
    observed = [
        {
            "run_id": "bad-run",
            "agent_id": "hermes_openai_codex",
            "channel": "file_content",
            "row_valid": True,
            "effective": True,
            "is_control": False,
            "cell_id": "c1",
        },
        {
            "run_id": "good-run",
            "agent_id": "codex_cli",
            "channel": "file_content",
            "row_valid": True,
            "effective": True,
            "is_control": False,
            "cell_id": "c1",
        },
    ]
    buckets = quarantine_rows(
        plan_rows={},
        observed_rows=observed,
        exclude_rules=[
            {
                "reason": "cwd_contamination",
                "run_id": "bad-run",
                "agent_id": "hermes_openai_codex",
            }
        ],
    )
    summary = denominator_summary(buckets, planned_n=0)
    assert summary["valid_attack_rows"] == 1
    assert summary["excluded_rows"] == 1
    assert buckets["excluded_rows"][0]["quarantine_reason"] == "cwd_contamination"


def test_build_trial_plan_balances_and_repeats_with_stable_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(btp, "ROOT", tmp_path)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    rows = [
        {"id": "a1", "text": "attack one", "label": "x", "source": "s1", "split": "test", "family_sha256": "fa"},
        {"id": "a2", "text": "attack two", "label": "y", "source": "s1", "split": "test", "family_sha256": "fb"},
        {"id": "a3", "text": "attack three", "label": "x", "source": "s2", "split": "test", "family_sha256": "fc"},
        {"id": "a4", "text": "attack four", "label": "y", "source": "s2", "split": "train", "family_sha256": "fd"},
    ]
    with (shard_dir / "shard_001.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    spec = {
        "max_concurrency": 2,
        "workers": [
            {
                "agent": "agent_a",
                "shards": [1],
                "channels": ["file_content"],
                "max_attacks": 2,
                "n_repeats": 2,
                "split": "test",
            }
        ],
    }
    trials = btp.make_trial_plan(spec, design_id="D-test", run_id="run-test", seed=123, strata=["label", "source"])
    assert len(trials) == 4
    assert {t["planned_trial_id"] for t in trials} == {
        "D-test:000000000",
        "D-test:000000001",
        "D-test:000000002",
        "D-test:000000003",
    }
    assert {t["repeat_idx"] for t in trials} == {0, 1}
    assert all(t["run_id"] == "run-test" for t in trials)
    assert all(t["primary_unit_id"].startswith("family:") for t in trials)
    assert all(t["stratum_id"] for t in trials)
    assert all(t["split"] == "test" for t in trials)
    assert {t["assignment_order"] for t in trials} == {0, 1, 2, 3}
