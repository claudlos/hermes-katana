"""B1 — empirical channel × attack-label effectiveness weights.

Reads results/agent_shard_runs/*.jsonl (agent-CLI observations) and computes
per-channel effectiveness baselines + per-(channel, label) multipliers.

Output: results/scanner_feeds/v2/channel_weights.json

hermes-katana uses per-file-type alert thresholds in its content_analyzer.
This file lets them multiply the default threshold by a channel-specific
factor — source code files (code_comment channel) get a ~2.4× tighter bar
than README files (file_content channel) based on what real agents do.

Why this matters: attacks hidden in code comments succeed on agents at
72% rate vs 30% for attacks in READMEs. A scanner that treats both
surfaces the same is blind to that gap.

Schema of the emitted JSON:
  {
    "version": 1,
    "source": "proving-ground agent-CLI battery",
    "n_observations": 531,
    "baseline_effective_rate_by_channel": {"file_content": 0.30, ...},
    "relative_channel_multiplier": {"file_content": 1.00, "code_comment": 2.4, ...},
    "per_label_channel_rates": {
        "semantic_manipulation": {"file_content": 0.35, "code_comment": 0.89, ...},
        ...
    },
    "recommended_scanner_threshold_multiplier": {
        "source_code":   0.42,   # lower threshold (alert sooner)
        "documentation": 1.00,   # baseline
        "log_file":      0.91,
        "data_file":     1.00,
    },
    "recommended_category_tighten_for_code": {
        "exfiltration_attempt": 4.0,
        "semantic_manipulation": 2.5,
        ...
    }
  }
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# Map our test channels onto hermes-katana's file-type categories.
# (We only populate source_code / documentation / log_file / data_file;
#  hermes-katana already knows how to identify those at ingestion.)
CHANNEL_TO_CATEGORY = {
    "code_comment": "source_code",
    "file_content": "documentation",
    "tool_output": "log_file",
    "data_row": "data_file",
}


def _load_rows(runs_dir: str) -> list[dict]:
    out: list[dict] = []
    for f in sorted(Path(runs_dir).glob("*.jsonl")):
        if "_broken_" in str(f):
            continue
        for line in f.open():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent-runs", default="results/agent_shard_runs")
    p.add_argument("--out-dir", default="results/scanner_feeds/v2")
    args = p.parse_args()

    rows = _load_rows(args.agent_runs)
    print(f"Loaded {len(rows)} agent-CLI rows from {args.agent_runs}")

    if not rows:
        print("No rows — nothing to compute.")
        return

    # --- Per-channel effectiveness ---
    ch_counts: dict[str, dict] = defaultdict(lambda: {"n": 0, "eff": 0, "canary": 0})
    for r in rows:
        ch = r.get("channel", "file_content")
        ch_counts[ch]["n"] += 1
        ch_counts[ch]["eff"] += int(bool(r.get("effective")))
        ch_counts[ch]["canary"] += int(bool(r.get("canary_leaked")))

    baseline_rate = {}
    for ch, d in ch_counts.items():
        baseline_rate[ch] = round(d["eff"] / d["n"], 4) if d["n"] else 0.0

    # File_content is the reference category (multiplier = 1.0). Others
    # scale relative to it — a channel that's 2.4× as effective is 2.4× as
    # risky in the scanner's eyes, so its threshold should be 1/2.4 as high.
    ref_rate = baseline_rate.get("file_content") or (max(baseline_rate.values()) if baseline_rate else 1.0)
    multipliers = {}
    for ch, r in baseline_rate.items():
        multipliers[ch] = round(r / ref_rate, 3) if ref_rate else 1.0

    # --- Per-(label, channel) effectiveness ---
    per_label: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"n": 0, "eff": 0}))
    for r in rows:
        ch = r.get("channel", "file_content")
        lbl = r.get("attack_label", "unknown")
        per_label[lbl][ch]["n"] += 1
        per_label[lbl][ch]["eff"] += int(bool(r.get("effective")))

    per_label_rates: dict[str, dict[str, float]] = {}
    for lbl, by_ch in per_label.items():
        per_label_rates[lbl] = {}
        for ch, d in by_ch.items():
            per_label_rates[lbl][ch] = round(d["eff"] / d["n"], 4) if d["n"] else 0.0

    # Recommended multiplier for how much to tighten the "source code"
    # scanner bar when the attack label is detectable up-front. Category
    # multiplier = code_rate / doc_rate, capped at 5× to avoid silly values
    # from tiny sample sizes.
    category_tighten_for_code = {}
    for lbl, rates in per_label_rates.items():
        doc = rates.get("file_content") or 0
        code = rates.get("code_comment") or 0
        if doc > 0:
            ratio = min(code / doc, 5.0)
            category_tighten_for_code[lbl] = round(ratio, 3)

    # Scanner threshold multipliers per file-type category. A channel 2.4×
    # more effective → threshold 1/2.4 × baseline → alert fires 2.4× sooner.
    scanner_threshold = {}
    for ch, mult in multipliers.items():
        category = CHANNEL_TO_CATEGORY.get(ch, "documentation")
        # Invert the multiplier: higher effectiveness → lower threshold.
        scanner_threshold[category] = round(1.0 / mult, 3) if mult else 1.0
    # Make sure documentation is the reference (1.0).
    if "documentation" in scanner_threshold:
        ref = scanner_threshold["documentation"]
        if ref:
            scanner_threshold = {k: round(v / ref, 3) for k, v in scanner_threshold.items()}

    payload = {
        "version": 1,
        "source": "proving-ground agent-CLI battery",
        "description": (
            "Empirical per-channel and per-(channel, label) effectiveness "
            "measurements from real agent runs. Use relative_channel_multiplier "
            "to adjust per-file-type alert thresholds, and "
            "per_label_channel_rates for per-attack-label sensitivity tuning."
        ),
        "n_observations": len(rows),
        "channel_counts": {k: v for k, v in ch_counts.items()},
        "baseline_effective_rate_by_channel": baseline_rate,
        "relative_channel_multiplier": multipliers,
        "recommended_scanner_threshold_multiplier": scanner_threshold,
        "per_label_channel_rates": per_label_rates,
        "recommended_category_tighten_for_code": category_tighten_for_code,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "channel_weights.json"
    out_file.write_text(json.dumps(payload, indent=2))

    print("\nBaseline effective rate by channel:")
    for ch, r in sorted(baseline_rate.items(), key=lambda kv: -kv[1]):
        print(f"  {ch:<18} {r * 100:5.1f}%  ({ch_counts[ch]['eff']}/{ch_counts[ch]['n']})")

    print("\nRelative channel multiplier (file_content = 1.00):")
    for ch, m in sorted(multipliers.items(), key=lambda kv: -kv[1]):
        print(f"  {ch:<18} {m}×")

    print("\nPer-label tighten for code vs docs (top 8):")
    ranked = sorted(category_tighten_for_code.items(), key=lambda kv: -kv[1])
    for lbl, t in ranked[:8]:
        print(f"  {lbl:<26} {t}×")

    print("\nRecommended scanner threshold multipliers (lower = alert sooner):")
    for cat, t in sorted(scanner_threshold.items(), key=lambda kv: kv[1]):
        print(f"  {cat:<18} {t}×")

    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
