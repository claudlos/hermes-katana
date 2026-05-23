"""ASR dashboard with Wilson 95% CIs.

Reads results/agent_shard_runs/*.jsonl and emits a markdown report of
attack-success rate per cell, each rate carrying its Wilson interval.

Cells covered:
  - per agent
  - per channel
  - per (agent, channel)
  - per attack_label
  - per (attack_label, agent)
  - if --corpus is given: per (label, technique) joined from corpus

Why CIs and not point estimates: AgentDojo / JailbreakBench leaderboards
report point estimates which silently mask sampling jitter. Wilson interval
is well-behaved near 0/1 and at small n; same primitive used elsewhere in
research/statistics.py.

Usage:
    python scripts/asr_dashboard.py
    python scripts/asr_dashboard.py --runs results/agent_shard_runs --out reports/asr.md
    python scripts/asr_dashboard.py --corpus synthdata/incoming/v5_synthdata_final_*.jsonl
    python scripts/asr_dashboard.py --json reports/asr.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from hermes_katana.proving_ground.research.statistics import wilson_ci, bootstrap_mean_ci  # noqa: E402


def load_runs(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    skipped = 0
    for pat in paths:
        for fp in glob.glob(pat, recursive=True):
            try:
                with open(fp) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            skipped += 1
            except OSError:
                continue
    if skipped:
        print(f"# warning: skipped {skipped} malformed lines", file=sys.stderr)
    return rows


def _trial_key(row: dict) -> tuple:
    """Identity of one planned trial row for duplicate detection."""
    return (
        row.get("run_id"),
        row.get("shard"),
        row.get("agent_id"),
        row.get("channel"),
        row.get("attack_id"),
        int(row.get("repeat_idx", 0) or 0),
        bool(row.get("is_control", False)),
    )


def validate_rows(
    rows: list[dict],
    *,
    run_id: str | None,
    allow_mixed_runs: bool,
    allow_duplicates: bool,
) -> tuple[list[dict], list[str]]:
    """Filter invalid rows and fail closed on mixed/duplicate campaigns."""
    notes: list[str] = []
    if run_id:
        before = len(rows)
        rows = [r for r in rows if r.get("run_id") == run_id]
        notes.append(f"Filtered to run_id={run_id}: {len(rows)}/{before} rows kept.")
    else:
        run_ids = sorted({r.get("run_id") for r in rows})
        if len(run_ids) > 1 and not allow_mixed_runs:
            raise SystemExit(
                "refusing to aggregate mixed run_ids without --run-id "
                "or --allow-mixed-runs; found " + ", ".join(map(str, run_ids[:12]))
            )

    invalid = [r for r in rows if r.get("invalid_run") or r.get("row_valid") is False]
    if invalid:
        notes.append(f"Excluded {len(invalid)} invalid infrastructure rows from ASR.")
        rows = [r for r in rows if not (r.get("invalid_run") or r.get("row_valid") is False)]

    seen: set[tuple] = set()
    dupes: list[tuple] = []
    for r in rows:
        key = _trial_key(r)
        if key in seen:
            dupes.append(key)
        seen.add(key)
    if dupes and not allow_duplicates:
        raise SystemExit(
            f"refusing to publish ASR with {len(dupes)} duplicate trial rows; "
            "deduplicate results or pass --allow-duplicates for forensic/debug output"
        )
    if dupes:
        notes.append(f"WARNING: kept {len(dupes)} duplicate trial rows (--allow-duplicates).")
    return rows, notes


def load_corpus(paths: list[str]) -> dict[str, dict]:
    """Map attack_id -> corpus row (for joining technique/source/etc.)."""
    by_id: dict[str, dict] = {}
    for pat in paths:
        for fp in glob.glob(pat):
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    aid = r.get("id") or r.get("attack_id")
                    if aid:
                        by_id[aid] = r
    return by_id


def fmt_pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def summarize(buckets: dict[tuple, dict]) -> list[tuple]:
    """Each bucket -> (key, k, n, rate, lo, hi). Sorted by rate desc, n desc."""
    out = []
    for key, agg in buckets.items():
        n = agg["n"]
        k = agg["k"]
        if n == 0:
            continue
        lo, hi = wilson_ci(k, n)
        out.append((key, k, n, k / n, lo, hi))
    out.sort(key=lambda r: (-r[3], -r[2]))
    return out


def render_table(rows: list[tuple], headers: list[str]) -> str:
    if not rows:
        return "_(no data)_\n"
    head = "| " + " | ".join(headers) + " |\n"
    sep = "|" + "|".join(["---"] * len(headers)) + "|\n"
    body = ""
    for key, k, n, rate, lo, hi in rows:
        if not isinstance(key, tuple):
            key = (key,)
        cells = list(key) + [
            str(n),
            str(k),
            fmt_pct(rate),
            f"{fmt_pct(lo)} – {fmt_pct(hi)}",
        ]
        body += "| " + " | ".join(str(c) for c in cells) + " |\n"
    return head + sep + body


def aggregate(rows: list[dict], key_fn) -> dict[tuple, dict]:
    buckets: dict[tuple, dict] = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        if r.get("matched_pair") is None:
            pass
        # exclude control / baseline rows from ASR
        if r.get("is_control"):
            continue
        if r.get("attack_label") == "baseline":
            continue
        key = key_fn(r)
        if key is None:
            continue
        buckets[key]["n"] += 1
        if r.get("effective"):
            buckets[key]["k"] += 1
    return buckets


def aggregate_repeats(rows: list[dict], key_fn) -> dict[tuple, dict[str, list[int]]]:
    """G2-aware: per cell, track per-attack (k, n). Used by collapse-repeats path.

    Returns {cell_key: {attack_id: [effective_count, total_count]}}.
    Cell-level rate is computed as mean of per-attack rates (k/n), then
    bootstrap CI is taken across attacks (NOT across rows). This is the
    correct repeated-measures aggregation when n_repeats > 1.
    """
    cells: dict[tuple, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.get("is_control"):
            continue
        if r.get("attack_label") == "baseline":
            continue
        key = key_fn(r)
        if key is None:
            continue
        aid = r.get("attack_id")
        if aid is None:
            continue
        cells[key][aid][1] += 1
        if r.get("effective"):
            cells[key][aid][0] += 1
    return cells


def summarize_collapsed(cells: dict[tuple, dict[str, list[int]]]) -> list[tuple]:
    """One row per cell: (key, n_attacks, total_trials, mean_attack_rate, lo, hi).
    Bootstrap CI across attack-level rates."""
    out = []
    for key, attacks in cells.items():
        if not attacks:
            continue
        per_attack_rates = [k / n for (k, n) in attacks.values() if n > 0]
        n_attacks = len(per_attack_rates)
        total_trials = sum(n for (_k, n) in attacks.values())
        mean, lo, hi = bootstrap_mean_ci(per_attack_rates)
        out.append((key, n_attacks, total_trials, mean, lo, hi))
    out.sort(key=lambda r: (-r[3], -r[2]))
    return out


def render_table_collapsed(rows: list[tuple], headers: list[str]) -> str:
    if not rows:
        return "_(no data)_\n"
    head = "| " + " | ".join(headers) + " |\n"
    sep = "|" + "|".join(["---"] * len(headers)) + "|\n"
    body = ""
    for key, n_attacks, total_trials, mean, lo, hi in rows:
        if not isinstance(key, tuple):
            key = (key,)
        cells = list(key) + [
            str(n_attacks),
            str(total_trials),
            fmt_pct(mean),
            f"{fmt_pct(lo)} – {fmt_pct(hi)}",
        ]
        body += "| " + " | ".join(str(c) for c in cells) + " |\n"
    return head + sep + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=["results/agent_shard_runs/*.jsonl"],
        help="glob(s) of agent_shard_runs JSONL",
    )
    ap.add_argument(
        "--corpus",
        nargs="*",
        default=[],
        help="optional corpus JSONL glob(s) for technique join",
    )
    ap.add_argument("--out", default=None, help="markdown output (default: stdout)")
    ap.add_argument("--json", default=None, help="JSON output of aggregated cells")
    ap.add_argument(
        "--collapse-repeats",
        action="store_true",
        help="When n_repeats > 1, aggregate per (cell, attack_id) "
        "first to per-attack rates, then bootstrap CI on the "
        "cell mean across attacks. Correctly accounts for "
        "repeated-measures clustering. Auto-enabled when "
        "any row has repeat_idx > 0.",
    )
    ap.add_argument("--run-id", default=None, help="only aggregate rows for this campaign")
    ap.add_argument(
        "--allow-mixed-runs",
        action="store_true",
        help="debug only: allow aggregating more than one run_id when --run-id is omitted",
    )
    ap.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="debug only: keep duplicate trial keys instead of failing closed",
    )
    args = ap.parse_args()

    rows = load_runs(args.runs)
    rows, validation_notes = validate_rows(
        rows,
        run_id=args.run_id,
        allow_mixed_runs=args.allow_mixed_runs,
        allow_duplicates=args.allow_duplicates,
    )
    if not rows:
        print("no rows loaded; check --runs glob", file=sys.stderr)
        sys.exit(1)
    corpus = load_corpus(args.corpus) if args.corpus else {}

    # G2 awareness: if any rows carry repeat_idx>0, raw row counting
    # overstates trial n (rows are repeated measures of the same attack,
    # not independent trials). Auto-enable --collapse-repeats; the raw
    # tables remain as a sanity-check sub-section.
    has_repeats = any(int(r.get("repeat_idx", 0)) > 0 for r in rows)
    n_with_repeats = sum(1 for r in rows if int(r.get("repeat_idx", 0)) > 0)
    collapse = args.collapse_repeats or has_repeats

    sections = []
    sections.append("# ASR dashboard\n\n")
    sections.append(f"Loaded **{len(rows)}** non-control attack rows from `{', '.join(args.runs)}`.\n")
    if validation_notes:
        sections.append("\n" + "\n".join(f"> {note}" for note in validation_notes) + "\n")
    if has_repeats:
        sections.append(
            f"\n> **G2 NOTE:** {n_with_repeats} of {len(rows)} rows carry "
            f"`repeat_idx > 0`. **Collapsed-repeats** tables (bootstrap CI on "
            f"the cell mean across per-attack rates) appear FIRST and are the "
            f"defensible numbers. Raw row-counted Wilson tables follow as a "
            f"sanity check — they over-state n by counting repeats as "
            f"independent trials.\n\n"
        )

    if collapse:
        # Collapsed-repeats sections (G2-aware). Bootstrap CI across attacks.
        sections.append("\n## [collapsed] Per agent\n\n")
        sections.append(
            render_table_collapsed(
                summarize_collapsed(aggregate_repeats(rows, lambda r: (r.get("agent_id", "?"),))),
                ["agent", "attacks", "trials", "mean ASR", "95% bootstrap CI"],
            )
        )

        sections.append("\n## [collapsed] Per channel\n\n")
        sections.append(
            render_table_collapsed(
                summarize_collapsed(aggregate_repeats(rows, lambda r: (r.get("channel", "?"),))),
                ["channel", "attacks", "trials", "mean ASR", "95% bootstrap CI"],
            )
        )

        sections.append("\n## [collapsed] Per (agent, channel)\n\n")
        sections.append(
            render_table_collapsed(
                summarize_collapsed(aggregate_repeats(rows, lambda r: (r.get("agent_id", "?"), r.get("channel", "?")))),
                [
                    "agent",
                    "channel",
                    "attacks",
                    "trials",
                    "mean ASR",
                    "95% bootstrap CI",
                ],
            )
        )

        sections.append("\n## [collapsed] Per attack label\n\n")
        sections.append(
            render_table_collapsed(
                summarize_collapsed(aggregate_repeats(rows, lambda r: (r.get("attack_label", "?"),))),
                ["label", "attacks", "trials", "mean ASR", "95% bootstrap CI"],
            )
        )

        sections.append(
            "\n---\n\n## Raw row-counted tables (sanity check; DO NOT publish these CIs when repeats > 1)\n\n"
        )
    if corpus:
        joined = sum(1 for r in rows if r.get("attack_id") in corpus)
        sections.append(f"Corpus join: **{joined}/{len(rows)}** rows matched a corpus entry.\n")
    sections.append("\nAll rates carry a Wilson 95% CI. Sorted by ASR descending, then by n.\n")

    # per agent
    sections.append("\n## Per agent\n\n")
    sections.append(
        render_table(
            summarize(aggregate(rows, lambda r: (r.get("agent_id", "?"),))),
            ["agent", "n", "k", "ASR", "95% CI"],
        )
    )

    # per channel
    sections.append("\n## Per channel\n\n")
    sections.append(
        render_table(
            summarize(aggregate(rows, lambda r: (r.get("channel", "?"),))),
            ["channel", "n", "k", "ASR", "95% CI"],
        )
    )

    # per agent × channel
    sections.append("\n## Per (agent, channel)\n\n")
    sections.append(
        render_table(
            summarize(aggregate(rows, lambda r: (r.get("agent_id", "?"), r.get("channel", "?")))),
            ["agent", "channel", "n", "k", "ASR", "95% CI"],
        )
    )

    # per attack_label
    sections.append("\n## Per attack label\n\n")
    sections.append(
        render_table(
            summarize(aggregate(rows, lambda r: (r.get("attack_label", "?"),))),
            ["label", "n", "k", "ASR", "95% CI"],
        )
    )

    # per (label, agent)
    sections.append("\n## Per (label, agent)\n\n")
    sections.append(
        render_table(
            summarize(aggregate(rows, lambda r: (r.get("attack_label", "?"), r.get("agent_id", "?")))),
            ["label", "agent", "n", "k", "ASR", "95% CI"],
        )
    )

    if corpus:
        # per (label, technique) — requires corpus join
        def key_lt(r):
            c = corpus.get(r.get("attack_id"))
            if not c:
                return None
            return (c.get("label", "?"), c.get("technique", "?"))

        sections.append("\n## Per (label, technique)  — corpus-joined\n\n")
        sections.append(
            render_table(
                summarize(aggregate(rows, key_lt)),
                ["label", "technique", "n", "k", "ASR", "95% CI"],
            )
        )

        # per (source, label)
        def key_sl(r):
            c = corpus.get(r.get("attack_id"))
            if not c:
                return None
            return (c.get("source", "?"), c.get("label", "?"))

        sections.append("\n## Per (source, label)  — corpus-joined\n\n")
        sections.append(
            render_table(
                summarize(aggregate(rows, key_sl)),
                ["source", "label", "n", "k", "ASR", "95% CI"],
            )
        )

    out = "".join(sections)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out)
        print(f"wrote {args.out} ({len(out)} chars)", file=sys.stderr)
    else:
        sys.stdout.write(out)

    if args.json:
        cells = {}
        for name, key_fn in [
            ("by_agent", lambda r: (r.get("agent_id", "?"),)),
            ("by_channel", lambda r: (r.get("channel", "?"),)),
            (
                "by_agent_channel",
                lambda r: (r.get("agent_id", "?"), r.get("channel", "?")),
            ),
            ("by_label", lambda r: (r.get("attack_label", "?"),)),
            (
                "by_label_agent",
                lambda r: (r.get("attack_label", "?"), r.get("agent_id", "?")),
            ),
        ]:
            cells[name] = [
                {
                    "key": list(k) if isinstance(k, tuple) else [k],
                    "k": ki,
                    "n": n,
                    "rate": rate,
                    "ci_low": lo,
                    "ci_high": hi,
                }
                for k, ki, n, rate, lo, hi in summarize(aggregate(rows, key_fn))
            ]
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(cells, indent=2))
        print(f"wrote {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
