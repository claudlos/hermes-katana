"""Pipeline orchestrator — chains the canonical stages of a campaign.

Stages (executed in order):

    1. build-corpus     — scripts/build_corpus.py (attack / benign / multilingual)
    2. launch-fleet     — scripts/fleet.py launch (blocking; waits for exit)
    3. cross-reference  — scripts/cross_reference_confirm.py
    4. features         — scripts/features/* + scripts/export_channel_weights.py
    5. report           — scripts/report.py --run-id ...
    6. manifest         — scripts/build_manifest.py

Fresh-hit follow-up stages:

    postrun-followup     — promote completed scan hits into a confirmation queue
    launch-confirm-queue — run the generated confirmation queue fleet

Idempotency is provided by each individual stage:
- build_corpus overwrites shards/*.jsonl (safe if unchanged).
- run_agent_shard.py skips already-done attacks in the output JSONL.
- cross_reference_confirm.py fully overwrites confirmed/rejected/provisional.
- features scripts overwrite their outputs.

Typical invocations:

    # Full pipeline on an existing spec — auto-generates run_id
    python scripts/pipeline.py --spec scripts/fleet_v12.json

    # Analysis-only pass on a completed run
    python scripts/pipeline.py --run-id a5f3b2c1 --skip build-corpus launch-fleet

    # Fresh-hit funnel after a Haiku/Codex scan has completed
    python scripts/pipeline.py \
      --only postrun-followup launch-confirm-queue \
      --confirm-run-id confirm_queue_20260505_1911

    # Single stage
    python scripts/pipeline.py --run-id a5f3b2c1 --only report

    # Build corpus only (no fleet, no analysis)
    python scripts/pipeline.py --only build-corpus --corpus-mode attack

The orchestrator is intentionally thin — it's a record of the canonical
stage ordering, not a re-implementation. Each stage invokes the existing
script as a subprocess so it inherits that script's CLI, logs, and
idempotency guarantees. Stage failures halt the pipeline; re-run after
fixing and it'll resume from the failed stage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")

STAGES = [
    "build-corpus",
    "launch-fleet",
    "cross-reference",
    "features",
    "report",
    "manifest",
    "postrun-followup",
    "launch-confirm-queue",
]


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"  → exit={rc} elapsed={time.time() - t0:.1f}s", flush=True)
    return rc


def stage_build_corpus(args: argparse.Namespace) -> int:
    cmd = [PY, "scripts/build_corpus.py", args.corpus_mode]
    if args.corpus_mode == "attack" and args.corpus_num_shards:
        cmd += ["--num-shards", str(args.corpus_num_shards)]
    return _run(cmd)


def stage_launch_fleet(args: argparse.Namespace) -> int:
    cmd = [
        PY,
        "scripts/fleet.py",
        "launch",
        "--spec",
        args.spec,
        "--run-id",
        args.run_id,
    ]
    return _run(cmd)


def stage_cross_reference(args: argparse.Namespace) -> int:
    return _run([PY, "scripts/cross_reference_confirm.py"])


def stage_features(args: argparse.Namespace) -> int:
    out_dir = "results/scanner_feeds"
    commands = [
        [
            PY,
            "scripts/features/extract_trigger_ngrams.py",
            "--confirmed",
            "results/confirmed_attacks.jsonl",
            "--rejected",
            "results/rejected_attacks.jsonl",
            "--out-dir",
            out_dir,
            "--min-score",
            "0.01",
            "--min-df-confirmed",
            "5",
        ],
        [
            PY,
            "scripts/features/build_semantic_centroids.py",
            "--confirmed",
            "results/confirmed_attacks.jsonl",
            "--rejected",
            "results/rejected_attacks.jsonl",
            "--out-dir",
            out_dir,
        ],
        [
            PY,
            "scripts/features/cluster_cross_model_effects.py",
            "--confirmed",
            "results/confirmed_attacks.jsonl",
            "--rejected",
            "results/rejected_attacks.jsonl",
            "--out-dir",
            out_dir,
        ],
        [PY, "scripts/export_channel_weights.py", "--out-dir", out_dir],
    ]
    for cmd in commands:
        rc = _run(cmd)
        if rc != 0:
            return rc
    return 0


def stage_report(args: argparse.Namespace) -> int:
    if args.run_id == "":
        print("SKIP report: no run_id (set via --run-id or via launch-fleet)")
        return 0
    return _run([PY, "scripts/report.py", "--run-id", args.run_id])


def stage_manifest(args: argparse.Namespace) -> int:
    return _run([PY, "scripts/build_manifest.py"])


def stage_postrun_followup(args: argparse.Namespace) -> int:
    return _run(
        [
            PY,
            "scripts/postrun_followup_20260505.py",
            "--run-id",
            args.confirm_run_id,
            "--haiku-codex-run-id",
            args.haiku_codex_run_id,
            "--free-run-id",
            args.free_run_id,
        ]
    )


def stage_launch_confirm_queue(args: argparse.Namespace) -> int:
    cmd = [
        PY,
        "scripts/fleet.py",
        "launch",
        "--spec",
        args.confirm_spec,
        "--run-id",
        args.confirm_run_id,
        "--allow-no-prereg",
        "--trial-plan",
        args.confirm_trial_plan,
        "--design-id",
        args.confirm_design_id,
    ]
    return _run(cmd)


HANDLERS = {
    "build-corpus": stage_build_corpus,
    "launch-fleet": stage_launch_fleet,
    "cross-reference": stage_cross_reference,
    "features": stage_features,
    "report": stage_report,
    "manifest": stage_manifest,
    "postrun-followup": stage_postrun_followup,
    "launch-confirm-queue": stage_launch_confirm_queue,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--spec",
        default="scripts/fleet_v12.json",
        help="Fleet spec for launch-fleet stage",
    )
    p.add_argument("--run-id", default="", help="Campaign run_id (auto-gen by fleet if empty)")
    p.add_argument(
        "--confirm-run-id",
        default="confirm_queue_20260505_1911",
        help="run_id for launch-confirm-queue and postrun-followup output",
    )
    p.add_argument(
        "--haiku-codex-run-id",
        default="haiku_codex_confirm_20260505_1700",
        help="Completed Haiku/Codex scan run consumed by postrun-followup",
    )
    p.add_argument(
        "--free-run-id",
        default="free_fleet_uncovered_20260505_1512",
        help="Completed free-fleet run consumed by postrun-followup",
    )
    p.add_argument(
        "--confirm-spec",
        default="scripts/fleet_confirm_queue_20260505.json",
        help="Fleet spec generated by postrun-followup",
    )
    p.add_argument(
        "--confirm-trial-plan",
        default="results/designs/D-confirm-queue-20260505/trial_plan.jsonl",
        help="Trial plan generated by postrun-followup",
    )
    p.add_argument(
        "--confirm-design-id",
        default="D-confirm-queue-20260505",
        help="Design id generated by postrun-followup",
    )
    p.add_argument("--corpus-mode", default="attack", choices=["attack", "benign", "multilingual"])
    p.add_argument("--corpus-num-shards", type=int, default=None)
    p.add_argument("--only", nargs="+", choices=STAGES, default=None, help="Run only these stages")
    p.add_argument(
        "--skip",
        nargs="+",
        choices=STAGES,
        default=None,
        help="Skip these stages (e.g. --skip build-corpus launch-fleet for analysis-only)",
    )
    args = p.parse_args()

    # Generate a run_id now if not supplied and launch-fleet will run.
    if not args.run_id:
        import secrets

        args.run_id = secrets.token_hex(4)
        print(f"[pipeline] auto-generated run_id={args.run_id}")

    if args.only:
        stages = args.only
    else:
        stages = [s for s in STAGES if not (args.skip and s in args.skip)]

    print(f"[pipeline] run_id={args.run_id}  stages={stages}")

    for stage in stages:
        print(f"\n=== [pipeline] stage: {stage} ===")
        rc = HANDLERS[stage](args)
        if rc != 0:
            print(f"\n[pipeline] STAGE FAILED: {stage} (exit={rc})")
            print(f"[pipeline] resume with:  python scripts/pipeline.py --run-id {args.run_id} --only {stage}")
            return rc

    print(f"\n[pipeline] all stages complete for run_id={args.run_id}")
    print(f"[pipeline] report: results/reports/{args.run_id}/report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
