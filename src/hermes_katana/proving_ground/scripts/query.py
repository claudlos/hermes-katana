"""Aggregate queries over results/agent_shard_runs/*.jsonl.

Answers:  how many (agent, shard, channel) combos are done? Which attacks
succeeded on which models? What's the effective rate this campaign?

Examples:

    # Fleet-level summary, all time
    python scripts/query.py

    # Just one campaign
    python scripts/query.py --run-id a5f3b2c1

    # Effective attacks for one agent × channel
    python scripts/query.py --agent claude_cli_haiku --channel code_comment --effective

    # Machine output for downstream scripts
    python scripts/query.py --run-id a5f3b2c1 --json

    # Per-label effectiveness breakdown
    python scripts/query.py --by label,channel

    # Per-(agent, channel, shard) completeness matrix (how many attacks run per slot)
    python scripts/query.py --coverage

Design: streams JSONL files sequentially, skip-filters early. For 120k rows
the full scan is ~3-5s on a reasonable disk — fast enough to stay
interactive, no index layer needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_RUNS = ROOT / "results" / "agent_shard_runs"
AGENT_RUNS_CONTROL = ROOT / "results" / "agent_shard_runs_control"
EXCLUSION_LIST = ROOT / "results" / "exclusion_list.json"


def _load_exclusion_keys() -> set[tuple]:
    """Load the parser-audit exclusion list keyed by (agent, shard, channel, attack_id)."""
    if not EXCLUSION_LIST.exists():
        return set()
    d = json.loads(EXCLUSION_LIST.read_text())
    out: set[tuple] = set()
    for r in d.get("rows", []):
        out.add((r.get("agent_id"), r.get("shard"), r.get("channel"), r.get("attack_id")))
    return out


def _matches(row: dict, args: argparse.Namespace, excl_keys: set[tuple]) -> bool:
    if args.run_id and row.get("run_id") != args.run_id:
        return False
    if args.agent and row.get("agent_id") != args.agent:
        return False
    if args.channel and row.get("channel") != args.channel:
        return False
    if args.shard is not None and row.get("shard") != args.shard:
        return False
    if args.label and row.get("attack_label") != args.label:
        return False
    if args.effective and not row.get("effective"):
        return False
    if args.attack_id and row.get("attack_id") != args.attack_id:
        return False
    if excl_keys:
        key = (
            row.get("agent_id"),
            row.get("shard"),
            row.get("channel"),
            row.get("attack_id"),
        )
        if key in excl_keys:
            return False
    return True


def _iter_rows(paths: list[Path], args: argparse.Namespace):
    excl_keys = _load_exclusion_keys() if getattr(args, "apply_exclusion", False) else set()
    if excl_keys:
        import sys as _sys

        print(
            f"[query] applying exclusion list: {len(excl_keys)} rows dropped",
            file=_sys.stderr,
        )
    for p in paths:
        try:
            with p.open() as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if _matches(row, args, excl_keys):
                        yield row
        except FileNotFoundError:
            continue


def _shard_paths(include_control: bool, control_only: bool) -> list[Path]:
    if control_only:
        roots = [AGENT_RUNS_CONTROL]
    else:
        roots = [AGENT_RUNS]
        if include_control:
            roots.append(AGENT_RUNS_CONTROL)
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(sorted(root.glob("shard_*.jsonl")))
    # Skip _broken_pre_fix and similar archived subdirs implicitly (glob doesn't recurse)
    # but filter obvious legacy backups.
    return [p for p in paths if "_broken" not in str(p)]


def cmd_summary(args: argparse.Namespace) -> int:
    rows_total = 0
    effective = 0
    canary = 0
    collapsed = 0
    refusal_spike = 0
    by_agent: Counter = Counter()
    by_channel: Counter = Counter()
    by_label: Counter = Counter()
    by_run: Counter = Counter()
    run_ids: set[str] = set()
    eff_by_agent: Counter = Counter()
    eff_by_channel: Counter = Counter()

    for row in _iter_rows(_shard_paths(include_control=False, control_only=args.control_only), args):
        rows_total += 1
        agent = row.get("agent_id", "?")
        channel = row.get("channel", "?")
        label = row.get("attack_label", "?")
        rid = row.get("run_id") or "(no-run-id)"
        by_agent[agent] += 1
        by_channel[channel] += 1
        by_label[label] += 1
        by_run[rid] += 1
        run_ids.add(rid)
        if row.get("effective"):
            effective += 1
            eff_by_agent[agent] += 1
            eff_by_channel[channel] += 1
        if row.get("canary_leaked"):
            canary += 1
        if row.get("collapsed"):
            collapsed += 1
        if row.get("refusal_spike"):
            refusal_spike += 1

    if args.json:
        out = {
            "n_rows": rows_total,
            "n_effective": effective,
            "n_canary_leaked": canary,
            "n_collapsed": collapsed,
            "n_refusal_spike": refusal_spike,
            "run_ids": sorted(run_ids),
            "by_agent": dict(by_agent),
            "by_channel": dict(by_channel),
            "by_label": dict(by_label),
            "by_run": dict(by_run),
            "eff_rate_by_agent": {a: round(eff_by_agent[a] / by_agent[a], 3) for a in by_agent if by_agent[a]},
            "eff_rate_by_channel": {
                c: round(eff_by_channel[c] / by_channel[c], 3) for c in by_channel if by_channel[c]
            },
        }
        print(json.dumps(out, indent=2))
        return 0

    filt_parts = []
    if args.run_id:
        filt_parts.append(f"run_id={args.run_id}")
    if args.agent:
        filt_parts.append(f"agent={args.agent}")
    if args.channel:
        filt_parts.append(f"channel={args.channel}")
    if args.shard is not None:
        filt_parts.append(f"shard={args.shard}")
    if args.label:
        filt_parts.append(f"label={args.label}")
    if args.effective:
        filt_parts.append("effective-only")
    if args.control_only:
        filt_parts.append("controls-only")
    filt = ", ".join(filt_parts) or "ALL"

    print(f"=== query — {filt} ===")
    print(f"rows: {rows_total:,}")
    if not rows_total:
        return 0
    print(f"effective: {effective:,} ({effective / rows_total:.1%})")
    print(f"  canary-leaked: {canary:,}  collapsed: {collapsed:,}  refusal-spike: {refusal_spike:,}")
    print(f"distinct run_ids: {len(run_ids)}")

    print("\nby agent                  rows   eff   eff%")
    for a, n in by_agent.most_common():
        e = eff_by_agent[a]
        print(f"  {a:<22} {n:>7,} {e:>5,} {e / n:>6.1%}")
    print("\nby channel             rows   eff   eff%")
    for c, n in by_channel.most_common():
        e = eff_by_channel[c]
        print(f"  {c:<18} {n:>7,} {e:>5,} {e / n:>6.1%}")
    print("\ntop labels")
    for lbl, n in by_label.most_common(12):
        print(f"  {lbl:<26} {n:>7,}")

    if len(run_ids) > 1 or (run_ids and next(iter(run_ids)) != "(no-run-id)"):
        print("\ntop run_ids")
        for rid, n in by_run.most_common(10):
            print(f"  {rid:<12} {n:>7,}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    """Completeness matrix: per (agent, shard, channel), how many attacks have rows?"""
    coverage: dict[tuple[str, int, str], int] = defaultdict(int)
    for row in _iter_rows(_shard_paths(include_control=False, control_only=False), args):
        k = (row.get("agent_id", "?"), row.get("shard", -1), row.get("channel", "?"))
        coverage[k] += 1
    if args.json:
        print(
            json.dumps(
                {f"{a}:{s}:{c}": n for (a, s, c), n in sorted(coverage.items())},
                indent=2,
            )
        )
        return 0
    print(f"=== coverage ({len(coverage):,} (agent,shard,channel) slots with ≥1 row) ===\n")
    # Per-agent totals
    by_agent: Counter = Counter()
    for (a, _, _), n in coverage.items():
        by_agent[a] += n
    print("total rows per agent:")
    for a, n in by_agent.most_common():
        slots = sum(1 for (ag, _, _) in coverage if ag == a)
        print(f"  {a:<26} {n:>8,} rows across {slots:>4} (shard×channel) slots")
    return 0


def cmd_list_effective(args: argparse.Namespace) -> int:
    """List attack_ids that were effective under the filters."""
    seen: set[str] = set()
    args.effective = True  # force filter
    for row in _iter_rows(_shard_paths(include_control=False, control_only=False), args):
        aid = row.get("attack_id")
        if aid and aid not in seen:
            seen.add(aid)
            if args.json:
                print(
                    json.dumps(
                        {
                            "attack_id": aid,
                            "label": row.get("attack_label"),
                            "agent_id": row.get("agent_id"),
                            "channel": row.get("channel"),
                            "shard": row.get("shard"),
                            "run_id": row.get("run_id"),
                        }
                    )
                )
            else:
                print(
                    f"{aid}  [{row.get('attack_label', '?'):<26}]  "
                    f"{row.get('agent_id', '?'):<24} {row.get('channel', '?'):<14} "
                    f"shard={row.get('shard')}"
                )
    if not args.json:
        print(f"\n{len(seen):,} distinct effective attack_ids")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # Filters — shared across modes.
    p.add_argument("--run-id", default=None)
    p.add_argument("--agent", default=None)
    p.add_argument(
        "--channel",
        default=None,
        choices=[None, "file_content", "code_comment", "data_row", "tool_output"],
    )
    p.add_argument("--shard", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--attack-id", default=None)
    p.add_argument("--effective", action="store_true", help="keep only rows where effective=True")
    p.add_argument(
        "--control-only",
        action="store_true",
        help="read agent_shard_runs_control/ instead",
    )
    p.add_argument(
        "--apply-exclusion",
        action="store_true",
        help="drop rows listed in results/exclusion_list.json "
        "(produced by scripts/audit_parsers.py --write-exclusion-list)",
    )
    p.add_argument("--json", action="store_true")
    # Modes.
    p.add_argument("--coverage", action="store_true", help="per-slot coverage matrix")
    p.add_argument(
        "--list-effective",
        action="store_true",
        help="list distinct effective attack_ids",
    )
    args = p.parse_args()

    if args.coverage:
        return cmd_coverage(args)
    if args.list_effective:
        return cmd_list_effective(args)
    return cmd_summary(args)


if __name__ == "__main__":
    sys.exit(main())
