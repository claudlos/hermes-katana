"""Generate a per-campaign markdown report.

Reads `results/fleet_runs/<run_id>/run_meta.json` + streams the matching
`results/agent_shard_runs/*.jsonl` rows to produce a self-contained summary:
headline counts, top confirmed attacks, per-agent / per-channel / per-label
effectiveness, plus links to key artifacts.

Usage:
    python scripts/report.py --run-id a5f3b2c1          # write markdown
    python scripts/report.py --run-id a5f3b2c1 --stdout # also print to stdout

Output: `results/reports/<run_id>/report.md`
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLEET_RUNS = ROOT / "results" / "fleet_runs"
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
REPORTS = ROOT / "results" / "reports"


def _load_run_meta(run_id: str) -> dict | None:
    p = FLEET_RUNS / run_id / "run_meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _iter_run_rows(run_id: str):
    """Stream shard_run rows stamped with this run_id."""
    if not AGENT_RUNS.exists():
        return
    for p in sorted(AGENT_RUNS.glob("shard_*.jsonl")):
        if "_broken" in str(p):
            continue
        try:
            with p.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("run_id") == run_id:
                        yield row
        except FileNotFoundError:
            continue


def _trial_key(row: dict) -> tuple:
    return (
        row.get("run_id"),
        row.get("shard"),
        row.get("agent_id"),
        row.get("channel"),
        row.get("attack_id"),
        int(row.get("repeat_idx", 0) or 0),
        bool(row.get("is_control", False)),
    )


def _elapsed(meta: dict) -> str:
    started = meta.get("started_at")
    if not started:
        return "unknown"
    end = meta.get("finished_at", int(time.time()))
    delta = end - started
    h, rem = divmod(delta, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h {int(m):02d}m {int(s):02d}s"


def build_report(run_id: str) -> str:
    meta = _load_run_meta(run_id)
    if meta is None:
        return f"# campaign report — {run_id}\n\n**ERROR**: no run_meta.json found for run_id={run_id}\n"

    rows = 0
    invalid_rows = 0
    duplicate_rows = 0
    effective = 0
    canary = 0
    collapsed = 0
    refusal_spike = 0
    by_agent: Counter = Counter()
    by_channel: Counter = Counter()
    by_label: Counter = Counter()
    eff_by_agent: Counter = Counter()
    eff_by_channel: Counter = Counter()
    eff_by_label: Counter = Counter()
    top_attacks: Counter = Counter()
    attack_meta: dict[str, dict] = {}
    seen_trials: set[tuple] = set()

    for row in _iter_run_rows(run_id):
        key = _trial_key(row)
        if key in seen_trials:
            duplicate_rows += 1
            continue
        seen_trials.add(key)
        if row.get("invalid_run") or row.get("row_valid") is False:
            invalid_rows += 1
            continue
        rows += 1
        a = row.get("agent_id", "?")
        c = row.get("channel", "?")
        lbl = row.get("attack_label", "?")
        by_agent[a] += 1
        by_channel[c] += 1
        by_label[lbl] += 1
        if row.get("effective"):
            effective += 1
            eff_by_agent[a] += 1
            eff_by_channel[c] += 1
            eff_by_label[lbl] += 1
            aid = row.get("attack_id")
            if aid:
                top_attacks[aid] += 1
                attack_meta.setdefault(
                    aid,
                    {
                        "label": lbl,
                        "first_seen_agent": a,
                        "first_seen_channel": c,
                    },
                )
        if row.get("canary_leaked"):
            canary += 1
        if row.get("collapsed"):
            collapsed += 1
        if row.get("refusal_spike"):
            refusal_spike += 1

    spec = meta.get("spec", {})
    workers = spec.get("workers", [])
    agents_in_spec = sorted({w["agent"] for w in workers})

    lines: list[str] = []
    lines.append(f"# Campaign report — run_id `{run_id}`")
    lines.append("")
    lines.append(f"**Started**: {meta.get('started_at_iso', 'unknown')}  ")
    lines.append(f"**Elapsed**: {_elapsed(meta)}  ")
    lines.append(f"**Git HEAD**: `{meta.get('git_head', '?')}`  ")
    lines.append(f"**Spec**: `{meta.get('spec_path', '?')}`  ")
    lines.append(f"**Max concurrency**: {meta.get('max_concurrency')}  ")
    lines.append(f"**Jobs planned**: {meta.get('total_jobs')}  ")
    lines.append(f"**Agents**: {', '.join(agents_in_spec) or '?'}  ")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    lines.append(f"| Valid attack-session rows | {rows:,} |")
    lines.append(f"| Invalid infrastructure rows excluded | {invalid_rows:,} |")
    lines.append(f"| Duplicate trial rows excluded | {duplicate_rows:,} |")
    lines.append(f"| Effective | {effective:,}" + (f" ({effective / rows:.1%})" if rows else "") + " |")
    lines.append(f"| Canary leaked | {canary:,} |")
    lines.append(f"| Collapsed | {collapsed:,} |")
    lines.append(f"| Refusal spike | {refusal_spike:,} |")
    lines.append("")

    if by_agent:
        lines.append("## Effectiveness by agent")
        lines.append("")
        lines.append("| agent | rows | effective | rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for a, n in by_agent.most_common():
            e = eff_by_agent[a]
            lines.append(f"| `{a}` | {n:,} | {e:,} | {e / n:.1%} |")
        lines.append("")

    if by_channel:
        lines.append("## Effectiveness by channel")
        lines.append("")
        lines.append("| channel | rows | effective | rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for c, n in by_channel.most_common():
            e = eff_by_channel[c]
            lines.append(f"| `{c}` | {n:,} | {e:,} | {e / n:.1%} |")
        lines.append("")

    if by_label:
        lines.append("## Effectiveness by attack label")
        lines.append("")
        lines.append("| label | rows | effective | rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for lbl, n in by_label.most_common():
            e = eff_by_label[lbl]
            lines.append(f"| `{lbl}` | {n:,} | {e:,} | {e / n:.1%} |")
        lines.append("")

    if top_attacks:
        lines.append("## Top 20 effective attacks (by slot coverage)")
        lines.append("")
        lines.append("| attack_id | label | slots effective | first agent | first channel |")
        lines.append("| --- | --- | ---: | --- | --- |")
        for aid, n in top_attacks.most_common(20):
            m = attack_meta.get(aid, {})
            lines.append(
                f"| `{aid}` | {m.get('label', '?')} | {n} | "
                f"`{m.get('first_seen_agent', '?')}` | `{m.get('first_seen_channel', '?')}` |"
            )
        lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Supervisor log: `results/fleet_runs/{run_id}/supervisor.log`")
    lines.append(f"- Per-job logs: `results/fleet_runs/{run_id}/jobs/`")
    lines.append("- Raw rows: `results/agent_shard_runs/shard_*.jsonl` (filter by `run_id`)")
    lines.append(f"- Aggregate stats: `python scripts/query.py --run-id {run_id}`")
    lines.append("")
    lines.append("## Spec snapshot")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(spec, indent=2))
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument(
        "--stdout",
        action="store_true",
        help="also print to stdout after writing the file",
    )
    args = p.parse_args()

    out_dir = REPORTS / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.run_id)
    out_path = out_dir / "report.md"
    out_path.write_text(report)
    print(f"Wrote {out_path} ({len(report):,} bytes)")
    if args.stdout:
        print()
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
