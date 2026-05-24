#!/usr/bin/env python3
"""Build a label-prioritized confirmation queue from v7's untested synth pool.

Background: training/data_v7/attacks.jsonl has 8,457 rows; 3,828 are
quality_tier=confirmed_* (n3+ effective on real models). The other 4,629
are synth (simula_dual_critic, synth_v5_critic_passed, synth_origin_balance_v2*)
that have never been put through the proving-ground harness. Per-label
confirmation imbalance is severe — persona_jailbreak 10% confirmed,
encoding_evasion 21% — exactly matching v14's per-family weakness.

This script:

  1. Reads hermes-katana/training/data_v7/attacks.jsonl.
  2. Filters to rows whose quality_tier is NOT confirmed_*.
  3. Sorts within label by quality_tier preference (higher-priority synth
     first: synth_v5_critic_passed > simula_dual_critic > origin_balance_v2*).
  4. Stacks the global queue so weak-label rows (persona, encoding) appear
     first; other labels follow proportionally.
  5. Writes the queue as JSONL shards at shards/shard_v8_<NNN>.jsonl. Each
     shard is ~150 rows so it fits comfortably in a single fleet worker
     pass.
  6. Writes a fleet manifest at scripts/fleet_v8_untested_synth_<ts>.json
     wiring the 10-agent panel (2x claude_cli_haiku, codex_cli, hermes-codex,
     2x mm2.5, 2x mm2.7, 2x qwen-on-nous) across file_content + tool_output
     channels.

Run:
    python scripts/build_v8_untested_synth_queue.py
    python scripts/build_v8_untested_synth_queue.py --max-per-label 200
    python scripts/build_v8_untested_synth_queue.py --rows-per-shard 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(
    os.environ.get(
        "HERMES_KATANA_ROOT",
        str(ROOT.parent / "hermes-katana"),
    )
)

ATTACKS = HERMES_ROOT / "training" / "data_v7" / "attacks.jsonl"

# Stacking order: weakest-confirmed labels first.
LABEL_PRIORITY = [
    "persona_jailbreak",  # 10.1% confirmed in v7
    "encoding_evasion",  # 20.9% confirmed
    "exfiltration_attempt",  # 35.1%
    "behavioral_control",  # 42.3%
    "semantic_manipulation",  # 52.5%
    "content_injection",  # 64.1%
    "cognitive_state_attack",  # 73.8%
    "jailbreak",  # 73.8%
]

# Within a label, prefer rows from the highest-effort synth pipelines first.
# synth_v5_critic_passed = went through v5 critic gate.
# simula_dual_critic = went through two-critic agreement.
# synth_origin_balance_v2_llm = LLM-rewrites of seeds for origin tiers.
# synth_origin_balance_v2 = hand-seeded origin-balanced rows.
QTIER_PREFERENCE = [
    "synth_v5_critic_passed",
    "simula_dual_critic",
    "synth_origin_balance_v2_llm",
    "synth_origin_balance_v2",
    "synth_origin_balance_v1",
]


def load_attacks():
    rows = []
    with ATTACKS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_untested(rows):
    """Keep only rows whose quality_tier is NOT confirmed_*."""
    return [r for r in rows if not str(r.get("quality_tier", "")).startswith("confirmed_")]


def stack_queue(untested, max_per_label):
    """Interleave labels by LABEL_PRIORITY, within label sorted by QTIER_PREFERENCE."""
    by_label = defaultdict(list)
    for r in untested:
        by_label[r.get("label", "?")].append(r)

    for lbl in by_label:
        by_label[lbl].sort(
            key=lambda r: (
                QTIER_PREFERENCE.index(r.get("quality_tier", ""))
                if r.get("quality_tier", "") in QTIER_PREFERENCE
                else len(QTIER_PREFERENCE),
                r.get("id", ""),
            )
        )
        if max_per_label is not None:
            by_label[lbl] = by_label[lbl][:max_per_label]

    # Interleave: each round-robin pulls one row from each label in priority order.
    queue = []
    pointers = {lbl: 0 for lbl in LABEL_PRIORITY}
    while True:
        progress = False
        for lbl in LABEL_PRIORITY:
            i = pointers[lbl]
            if i < len(by_label.get(lbl, [])):
                queue.append(by_label[lbl][i])
                pointers[lbl] = i + 1
                progress = True
        if not progress:
            break
    return queue, {lbl: len(by_label.get(lbl, [])) for lbl in LABEL_PRIORITY}


def shard_to_jsonl(rows, shard_id):
    """Convert v7 attack rows to shard format (preserve all fields plus shard ID stamp)."""
    out = []
    for r in rows:
        # Shard rows historically use 'id' and the same flat schema.
        out.append({**r, "shard": shard_id, "origin": r.get("origin", "user_input")})
    return out


def build_fleet_manifest(shards_used, output_path, run_id):
    """Wire the 11-worker panel.

    fleet.py rejects instances>1 (would collide on output files), so we
    expand multi-worker agents as separate entries either:
      * on different channels (no collision), or
      * on disjoint shard slices when both channels are taken.

    2x base agents (haiku, qwen) get one worker per channel.
    3x minimax agents (mm2.5, mm2.7) get worker per channel + an extra
    file_content worker on a shard-disjoint half (uses free-tier MM
    headroom we measured at ~60% utilization on the 2x layout).
    1x agent (hermes_openai_codex) gets a single file_content worker.

    codex_cli is temporarily excluded — Carlos's weekly quota at <5% on
    2026-05-11; re-add after reset, ideally switched to gpt-5.4-mini.
    """
    workers = []

    # 2x agents: one per channel, all shards.
    # claude_cli_haiku removed 2026-05-11 — Anthropic Max burns too fast +
    # buggy canary baseline (false-positive effectives).
    two_x_panel = [
        "hermes_nous_qwen3_coder_plus",  # 2x Qwen on Nous (free, reset)
    ]
    for agent in two_x_panel:
        for channel in ["file_content", "tool_output"]:
            workers.append(
                {
                    "agent": agent,
                    "shards": shards_used,
                    "channels": [channel],
                    "tasks": ["secrets_audit"],
                    "max_attacks": 9999,
                    "n_repeats": 1,
                }
            )

    # 3x MiniMax agents:
    #   * one tool_output worker over all shards
    #   * two file_content workers over disjoint shard halves
    #
    # This adds the one extra M2.5 and one extra M2.7 worker Carlos requested
    # without launching duplicate jobs against the same output file.
    mid = len(shards_used) // 2
    first_half = shards_used[:mid]
    second_half = shards_used[mid:]
    three_x_minimax = [
        "hermes_minimax_m2_5",  # 3x MM 2.5 (free, reset)
        "hermes_minimax_m2_7",  # 3x MM 2.7 (free, reset, shared pool)
    ]
    for agent in three_x_minimax:
        workers.append(
            {
                "agent": agent,
                "shards": shards_used,
                "channels": ["tool_output"],
                "tasks": ["secrets_audit"],
                "max_attacks": 9999,
                "n_repeats": 1,
            }
        )
        for shard_slice in [first_half, second_half]:
            workers.append(
                {
                    "agent": agent,
                    "shards": shard_slice,
                    "channels": ["file_content"],
                    "tasks": ["secrets_audit"],
                    "max_attacks": 9999,
                    "n_repeats": 1,
                }
            )

    # 1x agents: file_content only (primary lane).
    # codex_cli_gpt5_4_mini PERMANENTLY dropped: canary-detector false positives.
    # deepseek_v3_free + ling_2_6_1t_free dropped: OR 404 (model removed).
    # codex_cli + hermes_openai_codex RE-ADDED 2026-05-12 — ChatGPT plan reset +
    # fresh OPENAI_API_KEY in .env.
    # gemma + nemotron-nano dropped: OR free-tier saturates with >1 free OR worker.
    # GLM is the only OR-free that survives long enough; keep it.
    one_x_panel = [
        "codex_cli",  # GPT-5.5 via ChatGPT plan (reset)
        "hermes_openai_codex",  # codex via hermes-agent (new OPENAI_API_KEY)
        "hermes_or_glm_4_5_air_free",  # GLM/Zhipu family (free, OR)
    ]
    for agent in one_x_panel:
        workers.append(
            {
                "agent": agent,
                "shards": shards_used,
                "channels": ["file_content"],
                "tasks": ["secrets_audit"],
                "max_attacks": 9999,
                "n_repeats": 1,
            }
        )

    manifest = {
        "_comment": "v8 untested-synth confirmation push. Built 2026-05-11.",
        "_target": "Confirm as many untested synth rows from data_v7/attacks.jsonl as possible.",
        "_design_id": f"D-v8-untested-synth-{run_id}",
        # fleet.py uses global job concurrency, not persistent per-entry
        # worker slots. Cap this campaign at the six MiniMax lanes Carlos
        # requested; otherwise fast-completing/skipped non-MM jobs can let the
        # queue fill all active slots with MiniMax and overshoot the free-plan
        # 5-hour allowance.
        "max_concurrency": 6,
        "workers": workers,
    }
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Cap untested rows per label (default: no cap — drain the pool).",
    )
    ap.add_argument("--rows-per-shard", type=int, default=150)
    ap.add_argument(
        "--start-shard-id",
        type=int,
        default=9700,
        help="Shard IDs to allocate (we leave the 9600s for the prior 2026-05-06 work).",
    )
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id or f"v8_untested_synth_{time.strftime('%Y%m%d_%H%M%S')}"

    rows = load_attacks()
    print(f"[build] {ATTACKS}: {len(rows)} total attack rows")
    untested = filter_untested(rows)
    print(f"[build] untested synth: {len(untested)} rows")

    queue, per_label_counts = stack_queue(untested, args.max_per_label)
    print(f"[build] stacked queue size: {len(queue)}")
    print("[build] per-label slice taken:")
    for lbl in LABEL_PRIORITY:
        print(f"  {lbl:25s}: {per_label_counts.get(lbl, 0)}")

    # Slice into shards
    shards_dir = ROOT / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_files = []
    shard_ids = []
    for i in range(0, len(queue), args.rows_per_shard):
        chunk = queue[i : i + args.rows_per_shard]
        sid = args.start_shard_id + (i // args.rows_per_shard)
        path = shards_dir / f"shard_{sid:05d}.jsonl"
        rows_out = shard_to_jsonl(chunk, sid)
        with path.open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        shard_files.append(str(path.relative_to(ROOT)))
        shard_ids.append(sid)
    print(f"[build] wrote {len(shard_files)} shards to {shards_dir.relative_to(ROOT)}")

    # Write a queue manifest for traceability
    queue_path = ROOT / "results" / "queues" / f"queue_{run_id}.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("w", encoding="utf-8") as f:
        for r in queue:
            f.write(
                json.dumps(
                    {
                        "id": r.get("id"),
                        "label": r.get("label"),
                        "quality_tier": r.get("quality_tier"),
                        "source": r.get("source"),
                        "split": r.get("split"),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    print(f"[build] queue manifest -> {queue_path.relative_to(ROOT)}")

    # Fleet manifest
    fleet_path = ROOT / "scripts" / f"fleet_{run_id}.json"
    build_fleet_manifest(shard_ids, fleet_path, run_id)
    print(f"[build] fleet manifest -> {fleet_path.relative_to(ROOT)}")
    print(f"[build] run_id = {run_id}")
    print("[build] DONE")


if __name__ == "__main__":
    raise SystemExit(main())
