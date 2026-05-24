"""Strip rows with silent API/auth errors from results/agent_shard_runs/.

An "API-error row" is one whose baseline.stdout_preview OR
attack_run.stdout_preview matches any known upstream-failure signature
(auth revoked, credit cap, rate limit, HTTP 4xx/5xx from provider).

These rows look structurally valid but the baseline and attack both
return the identical error string, so drift detection yields a false
negative (effective=False when we have no actual data). Keeping them
in the dataset biases paired analyses — e.g., hermes_nous_hermes4_70b
bare was 92% auth-error rows and showed +2.79pp *inversion* vs katana.

Usage:
    python scripts/strip_api_errors.py --dry-run   # report counts only
    python scripts/strip_api_errors.py             # rewrite files in place
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARD_RUNS = ROOT / "results" / "agent_shard_runs"

ERROR_PATTERNS = [
    re.compile(r"Refresh session has been revoked", re.I),
    re.compile(r"Key limit exceeded", re.I),
    re.compile(r"HTTP 40[13]", re.I),
    re.compile(r"HTTP 5\d\d", re.I),
    re.compile(r"API call failed after \d+ retries", re.I),
    re.compile(r"authentication (failed|error)", re.I),
    re.compile(r"insufficient.{0,20}(credit|quota|balance)", re.I),
    re.compile(r"rate.?limit.{0,40}exceeded", re.I),
    re.compile(r"unauthori[sz]ed", re.I),
]


def is_api_error(stdout: str, stderr: str = "") -> str | None:
    """Return matching pattern name if this output is an upstream API error."""
    if not stdout and not stderr:
        return None
    blob = (stdout or "") + "\n" + (stderr or "")
    # Short-circuit if output is unusually short AND looks error-shaped
    if len(stdout) < 300:
        for p in ERROR_PATTERNS:
            m = p.search(blob)
            if m:
                return m.group(0)[:60]
    else:
        # For longer outputs require the pattern to appear near the start —
        # some legitimate tool outputs mention these phrases in context.
        head = blob[:400]
        for p in ERROR_PATTERNS:
            m = p.search(head)
            if m:
                return m.group(0)[:60]
    return None


def row_has_api_error(row: dict) -> str | None:
    b = row.get("baseline") or {}
    a = row.get("attack_run") or {}
    bsig = is_api_error(b.get("stdout_preview", ""), b.get("stderr_preview", ""))
    if bsig:
        return f"baseline:{bsig}"
    asig = is_api_error(a.get("stdout_preview", ""), a.get("stderr_preview", ""))
    if asig:
        return f"attack:{asig}"
    return None


def process_file(path: Path, dry_run: bool) -> tuple[int, int, Counter]:
    """Return (kept, stripped, per_pattern_counts)."""
    kept_lines: list[str] = []
    stripped = 0
    pats: Counter = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                kept_lines.append(line)  # preserve non-JSON lines untouched
                continue
            sig = row_has_api_error(row)
            if sig:
                stripped += 1
                pats[sig] += 1
            else:
                kept_lines.append(line)
    kept = len(kept_lines)
    if not dry_run and stripped > 0:
        path.write_text("".join(kept_lines), encoding="utf-8")
    return kept, stripped, pats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--agent", default=None, help="filter filenames by agent substring")
    args = ap.parse_args()

    pattern = f"shard_*{args.agent}*.jsonl" if args.agent else "shard_*.jsonl"
    files = sorted(SHARD_RUNS.glob(pattern))
    print(f"scanning {len(files)} files (dry_run={args.dry_run})...")

    total_kept = 0
    total_stripped = 0
    all_pats: Counter = Counter()
    by_agent: dict[str, list[int]] = {}  # agent -> [kept, stripped]
    touched_files = 0
    for p in files:
        kept, stripped, pats = process_file(p, args.dry_run)
        total_kept += kept
        total_stripped += stripped
        all_pats.update(pats)
        if stripped:
            touched_files += 1
        # infer agent from filename: shard_NNN_<agent>[_channel].jsonl
        parts = p.stem.split("_", 2)
        # shard_041_claude_cli_haiku_katana_code_comment  →  claude_cli_haiku_katana_code_comment
        rest = parts[-1] if len(parts) == 3 else ""
        # strip trailing _channel (there are 4 canonical channels)
        for ch in ("_file_content", "_code_comment", "_tool_output", "_data_row"):
            if rest.endswith(ch):
                rest = rest[: -len(ch)]
                break
        agent = rest
        agg = by_agent.setdefault(agent, [0, 0])
        agg[0] += kept
        agg[1] += stripped

    print("\n=== totals ===")
    print(f"kept:     {total_kept:,}")
    print(
        f"stripped: {total_stripped:,}  ({100 * total_stripped / max(1, total_kept + total_stripped):.2f}% of all rows)"
    )
    print(f"files modified: {touched_files}")

    print("\n=== top error patterns ===")
    for sig, n in all_pats.most_common(12):
        print(f"  {n:>7,}  {sig}")

    print("\n=== worst-hit agents ===")
    rows = sorted(by_agent.items(), key=lambda kv: -kv[1][1])
    for agent, (k, s) in rows[:15]:
        if s == 0:
            continue
        pct = 100 * s / max(1, k + s)
        print(f"  {agent:<46}  stripped={s:>7,}  kept={k:>7,}  ({pct:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
