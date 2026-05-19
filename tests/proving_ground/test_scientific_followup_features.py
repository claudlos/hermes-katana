from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_agent_shard as ras
from scripts import analyze_asr_scientific as aas
from scripts import build_corpus
from scripts import fleet
from scripts import rescore_semantic
from scripts import synth_to_shards


def test_trial_plan_slice_filters_and_resumes_by_planned_trial_id(tmp_path):
    plan = tmp_path / "trial_plan.jsonl"
    rows = [
        {
            "design_id": "D",
            "planned_trial_id": "p1",
            "run_id": "r1",
            "agent_id": "agent_a",
            "shard": 7,
            "channel": "file_content",
            "is_control": False,
            "split": "test",
            "attack_id": "atk1",
            "repeat_idx": 0,
            "n_repeats_planned": 2,
            "cell_id": "cell/a",
            "primary_unit_id": "family:f1",
            "block_id": "family:f1",
            "stratum_id": "label=x",
        },
        {
            "design_id": "D",
            "planned_trial_id": "p2",
            "run_id": "r1",
            "agent_id": "agent_a",
            "shard": 7,
            "channel": "file_content",
            "is_control": False,
            "split": "test",
            "attack_id": "atk2",
            "repeat_idx": 1,
            "n_repeats_planned": 2,
        },
        {
            "design_id": "D",
            "planned_trial_id": "p3",
            "run_id": "other",
            "agent_id": "agent_a",
            "shard": 7,
            "channel": "file_content",
            "is_control": False,
            "split": "test",
            "attack_id": "atk3",
            "repeat_idx": 0,
        },
    ]
    plan.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done = tmp_path / "done.jsonl"
    done.write_text(json.dumps({"run_id": "r1", "planned_trial_id": "p1"}) + "\n")

    attacks = [
        {"id": "atk1", "text": "one", "label": "x"},
        {"id": "atk2", "text": "two", "label": "y"},
    ]
    pending, selected, done_ids = ras._pending_from_trial_plan(
        plan,
        attacks,
        done,
        run_id="r1",
        shard_id=7,
        agent_id="agent_a",
        channel="file_content",
        control=False,
        split="test",
    )
    assert done_ids == {"p1"}
    assert [p[2]["planned_trial_id"] for p in pending] == ["p2"]
    assert selected == 2
    assert pending[0][0]["id"] == "atk2"
    assert pending[0][1] == 1


def test_fleet_job_threads_trial_plan_and_updates_run_meta(tmp_path):
    job = fleet.Job(
        agent="agent_a",
        shard=1,
        channel="file_content",
        max_attacks=5,
        run_id="runx",
        trial_plan=Path("results/designs/D/trial_plan.jsonl"),
    )
    cmd = job.cmd()
    assert "--trial-plan" in cmd
    assert "results/designs/D/trial_plan.jsonl" in cmd

    dirs = fleet.RunDirs("meta-test")
    monkey_base = tmp_path / "fleet_runs"
    original = fleet.FLEET_RUNS
    fleet.FLEET_RUNS = monkey_base
    try:
        dirs = fleet.RunDirs("meta-test")
        dirs.ensure()
        dirs.meta.write_text(json.dumps({"run_id": "meta-test", "started_at": 1}))
        fleet._update_run_meta(
            dirs,
            finished_at=2,
            finished_at_iso="done",
            done_jobs=3,
            failed_jobs=1,
            exit_code=1,
            interrupted=False,
            design_id="D",
            trial_plan="plan.jsonl",
        )
        meta = json.loads(dirs.meta.read_text())
        assert meta["done_jobs"] == 3
        assert meta["failed_jobs"] == 1
        assert meta["exit_code"] == 1
        assert meta["design_id"] == "D"
        assert meta["trial_plan"] == "plan.jsonl"
    finally:
        fleet.FLEET_RUNS = original


def test_scientific_analyzer_uses_family_clusters_and_paired_common_units():
    rows = [
        {"cell_id": "A", "primary_unit_id": "f1", "effective": True, "row_valid": True},
        {"cell_id": "A", "primary_unit_id": "f1", "effective": False, "row_valid": True},
        {"cell_id": "A", "primary_unit_id": "f2", "effective": True, "row_valid": True},
        {"cell_id": "B", "primary_unit_id": "f1", "effective": False, "row_valid": True},
        {"cell_id": "B", "primary_unit_id": "f2", "effective": True, "row_valid": True},
        {"cell_id": "B", "primary_unit_id": "f3", "effective": True, "row_valid": True},
        {"cell_id": "A", "primary_unit_id": "bad", "effective": True, "row_valid": False},
    ]
    result = aas.analyze_rows(rows, comparisons=[("A", "B")], bootstrap_iters=200, seed=3)
    assert result["cells"]["A"]["valid_families"] == 2
    assert result["cells"]["A"]["primary_asr"] == pytest.approx(0.75)
    assert result["comparisons"]["A::B"]["paired_families"] == 2
    assert result["comparisons"]["A::B"]["delta"] == pytest.approx(0.25)


def test_rescore_writes_reproducibility_metadata(tmp_path):
    in_file = tmp_path / "rows.jsonl"
    out_file = tmp_path / "rows.enriched.jsonl"
    in_file.write_text(json.dumps({"run_id": "r", "effective": False}) + "\n")
    out_file.write_text(json.dumps({"run_id": "r", "semantic_enriched": True, "effective": True}) + "\n")
    meta = rescore_semantic._build_rescore_metadata(
        input_path=in_file,
        output_path=out_file,
        rows_in=1,
        rows_out=1,
        effective_before=0,
        effective_after=1,
        run_id="r",
    )
    assert meta["input_sha256"]
    assert meta["output_sha256"]
    assert meta["scorer"]["name"] == "score_semantic"
    assert meta["reliability"]["judge_repeats"] == 1
    assert meta["effective_after"] == 1


def test_corpus_manifest_helpers_add_source_and_selection_checksums(tmp_path):
    src = tmp_path / "source.jsonl"
    rows = [
        {"id": "a", "text_sha256_normalized": "n1", "family_sha256": "f1", "label": "x"},
        {"id": "b", "text_sha256_normalized": "n2", "family_sha256": "f2", "label": "y"},
    ]
    src.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    entries = [{"file": "shard_001.jsonl", "sha256": "abc", "n_rows": 2}]
    meta = build_corpus._source_checksum_manifest(src, rows, entries, builder="test")
    assert meta["source_sha256"]
    assert meta["selected_rows_sha256"]
    assert meta["selected_row_count"] == 2
    assert meta["shards_sha256"]

    synth_meta = synth_to_shards._build_synth_manifest(
        input_paths=[src],
        written_paths=[],
        rows=[],
        first_shard=200,
        per_shard=177,
        by_run={},
        by_label={},
    )
    assert synth_meta["input_files"][0]["sha256"]
    assert synth_meta["selected_rows_sha256"]
