"""Intern — the research agent's human CLI face.

Wraps `research.kernel.ResearchKernel` with a small command set:

    intern list-tools                    # which tools are available?
    intern list-hypotheses               # registered hypotheses + verdicts
    intern call <tool> [--args '{...}']  # invoke one tool, stream result
    intern status [--run-id R]           # kernel state + budget
    intern preregister --spec file.yaml  # register a hypothesis
    intern play <name>                   # run a hardcoded scientific workflow

The "play" mechanism is a named, idempotent sequence of tool calls +
claim submissions. It's how headline science gets run today without an
LLM planner; the planner slot is intentionally open for Phase 2.B.5.

Example:

    python -m hermes_katana.proving_ground.scripts.intern play quick-summary

    python -m hermes_katana.proving_ground.scripts.intern play resolve-harness-dominates-model \\
        --ccli-run-id mycam12 --hermes-run-id hrm-v7

Default `--run-id` is auto-generated per invocation; supply one to keep
events across invocations in the same research_runs/<id>/ bucket.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

from hermes_katana.proving_ground.research.kernel import ResearchKernel, KERNEL_ROOT  # noqa: E402
from hermes_katana.proving_ground.research.rigor import Claim, claim_from_proportion  # noqa: E402
from hermes_katana.proving_ground.research.events import Observation  # noqa: E402


# ---------------------------------------------------------------------------
# Plays
# ---------------------------------------------------------------------------


def play_quick_summary(k: ResearchKernel, args: argparse.Namespace) -> int:
    """Exploratory pass: list agents, query whole corpus, print effective rate."""
    r = k.call("list_agents", {}, rationale="sanity")
    n_agents = r.observation.data.get("count") if r.observation else "?"
    print(f"agents available: {n_agents}")

    r = k.call("query_corpus", {}, rationale="corpus-wide headline")
    if not r.observation:
        print("query_corpus failed")
        return 1
    data = r.observation.data
    total = data.get("n_rows", 0)
    eff = data.get("n_effective", 0)
    print(f"rows: {total:,}   effective: {eff:,} ({100 * eff / total:.1f}% overall)")

    # Emit an exploratory claim on the aggregate rate
    c = claim_from_proportion(
        hypothesis_id=None,
        primary_outcome="effective_rate_overall",
        k=eff,
        n=total,
    )
    res = k.submit_claim(c, [Observation(source="tool:query_corpus", data=data)])
    print(f"claim status: {res.kind}  CI=({c.ci[0] * 100:.2f}%, {c.ci[1] * 100:.2f}%)")

    # Effective rates by agent, top 5
    print("\ntop 5 agents by eff rate (min n=500):")
    rates: list[tuple[str, float, int]] = []
    for a, n in (data.get("by_agent") or {}).items():
        if n < 500:
            continue
        e = (data.get("eff_rate_by_agent") or {}).get(a, 0)
        rates.append((a, e, n))
    rates.sort(key=lambda t: -t[1])
    for a, e, n in rates[:5]:
        print(f"  {a:<26} {100 * e:>5.1f}% (n={n:,})")
    return 0


def play_resolve_harness_dominates(k: ResearchKernel, args: argparse.Namespace) -> int:
    """Paired comparison of two runs on the H-harness-dominates hypothesis.

    The hypothesis requires: same Claude Haiku model, CCLI harness vs. Hermes
    harness, McNemar test. This play implements the test given two run_ids
    (or two agent_ids sharing a corpus). It emits a Claim → Result pipeline
    that either resolves the hypothesis (if evidence is sufficient) or
    downgrades to an Observation.
    """
    hyp_id = "H-20260422-harness-dominates-model"
    from hermes_katana.proving_ground.research.registry import HypothesisRegistry

    reg = HypothesisRegistry()
    h = reg.load(hyp_id)
    print(f"[play resolve-harness-dominates] hypothesis: {h.title}")

    # Query each condition via the tool router
    a_agents = h.conditions["A"]["agent_ids"]
    b_agents = h.conditions["B"]["agent_ids"]

    def _query_for_agent(agent_id: str) -> dict:
        r = k.call("query_corpus", {"agent": agent_id}, rationale=f"rate for {agent_id}")
        return r.observation.data if r.observation else {}

    a_stats = {a: _query_for_agent(a) for a in a_agents}
    b_stats = {b: _query_for_agent(b) for b in b_agents}

    a_eff = sum(s.get("n_effective", 0) for s in a_stats.values())
    a_n = sum(s.get("n_rows", 0) for s in a_stats.values())
    b_eff = sum(s.get("n_effective", 0) for s in b_stats.values())
    b_n = sum(s.get("n_rows", 0) for s in b_stats.values())

    print(f"  A (claude_code_cli): {a_eff}/{a_n} = {100 * a_eff / max(a_n, 1):.1f}%")
    print(f"  B (hermes):          {b_eff}/{b_n} = {100 * b_eff / max(b_n, 1):.1f}%")

    # For a true paired McNemar we would need per-attack paired labels on
    # both harnesses. Absent pairing metadata, we fall back to a two-
    # proportion z-test's Wilson CI bracketing and flag as "preliminary —
    # paired test TBD when harness-ablation infrastructure lands."
    from hermes_katana.proving_ground.research.statistics import wilson_ci, cohens_h

    a_low, a_hi = wilson_ci(a_eff, a_n)
    b_low, b_hi = wilson_ci(b_eff, b_n)
    delta = (a_eff / max(a_n, 1)) - (b_eff / max(b_n, 1))
    h_effect = cohens_h(a_eff / max(a_n, 1), b_eff / max(b_n, 1))
    print(f"  CI(A): [{100 * a_low:.1f}%, {100 * a_hi:.1f}%]")
    print(f"  CI(B): [{100 * b_low:.1f}%, {100 * b_hi:.1f}%]")
    print(f"  delta = A - B = {100 * delta:+.1f}pp     Cohen's h = {h_effect:+.2f}")

    # Submit the claim; without paired metadata we can't compute McNemar here,
    # so the claim is marked with test_kind="two_prop_z_preliminary" and will
    # not pass a preregistration-grade rigor contract (missing paired test).
    # That's the point — the kernel will DOWNGRADE it until we have paired data.
    claim = Claim(
        hypothesis_id=hyp_id,
        primary_outcome="effective_rate_delta",
        value=delta,
        n_samples=min(a_n, b_n),
        ci=(
            delta - (a_hi - a_low) / 2 - (b_hi - b_low) / 2,
            delta + (a_hi - a_low) / 2 + (b_hi - b_low) / 2,
        ),  # rough bound
        baseline_run_id="aggregate-hermes",
        comparison_run_id="aggregate-claude-cli",
        test_kind="two_prop_z_preliminary",
        p_value=None,  # intentionally missing to force downgrade pre-pairing
        effect_size={"kind": "cohens_h", "value": h_effect},
        meta={
            "a_agents": a_agents,
            "b_agents": b_agents,
            "a_stats": a_stats,
            "b_stats": b_stats,
        },
    )
    supporting = [
        Observation(
            source="tool:query_corpus",
            data={"A": a_stats, "B": b_stats, "delta": delta},
        ),
    ]
    res = k.submit_claim(claim, supporting)
    print(f"  claim verdict: {res.kind}")
    if res.kind == "downgrade":
        print("  reasons:")
        for r in res.reasons:
            print(f"    - {r}")
        print("  → Paired test requires matched-pair harness infrastructure (Phase 2.D).")
        print("  → When that lands, re-run this play and the hypothesis can resolve.")
    return 0


def play_paired_harness_ablation(k: ResearchKernel, args: argparse.Namespace) -> int:
    """Invokes the harness-ablation module (McNemar paired analysis) and
    routes its Claim through the current kernel. This is the rigorous sibling
    of resolve-harness-dominates-model: paired per-attack, per-channel, with
    McNemar p-value + Wilson CIs + Cohen's h.

    Supports --harness-a / --harness-b to override the defaults
    (claude_code_cli vs hermes). The hypothesis is auto-resolved in the
    registry if p<.05 AND direction matches AND n>=min_n_per_condition;
    otherwise marked rejected or left preregistered.
    """
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "hermes_katana.proving_ground.scripts.harness_ablation",
        "--submit-to-kernel",
        "--run-id",
        k.run_id,
    ]
    if args.harness_a:
        cmd += ["--harness-a", args.harness_a]
    if args.harness_b:
        cmd += ["--harness-b", args.harness_b]
    if args.label_a:
        cmd += ["--label-a", args.label_a]
    if args.label_b:
        cmd += ["--label-b", args.label_b]
    return subprocess.call(cmd, cwd=str(ROOT))


PLAYS = {
    "quick-summary": play_quick_summary,
    "resolve-harness-dominates-model": play_resolve_harness_dominates,
    "paired-harness-ablation": play_paired_harness_ablation,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _new_run_id() -> str:
    return "intern-" + secrets.token_hex(4)


def _build_kernel(args: argparse.Namespace) -> ResearchKernel:
    run_id = args.run_id or _new_run_id()
    return ResearchKernel.build(
        run_id=run_id,
        auto_approve_human_gates=args.auto_approve,
    )


def cmd_list_tools(args: argparse.Namespace) -> int:
    k = _build_kernel(args)
    for t in k.router.list_tools():
        h = "HUMAN" if t["requires_human_approval"] else "auto"
        ro = "r/o" if t["read_only"] else "r/w"
        print(f"  [{h:<5} {ro:<3}] {t['name']:<20} {t['description']}")
    return 0


def cmd_list_hypotheses(args: argparse.Namespace) -> int:
    k = _build_kernel(args)
    r = k.call("list_hypotheses", {})
    for h in (r.observation.data or {}).get("hypotheses", []):
        status = h["status"].upper()
        if h.get("verdict"):
            status += f" ({h['verdict']})"
        print(f"  [{status:<25}] {h['id']}  {h['title'][:80]}")
    return 0


def cmd_call(args: argparse.Namespace) -> int:
    k = _build_kernel(args)
    tool_args = json.loads(args.args) if args.args else {}
    r = k.call(args.tool, tool_args, rationale=args.rationale or "")
    if r.gate_rejection:
        print(f"REJECTED: {r.gate_rejection.reason}")
        return 2
    if r.human_gate:
        print(f"HUMAN GATE: {r.human_gate.reason}")
        return 3
    if r.observation:
        print(json.dumps(r.observation.data, indent=2, default=str))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if args.run_id:
        k = _build_kernel(args)
        print(json.dumps(k.summary(), indent=2, default=str))
        return 0
    # No run_id: list all kernel runs
    if not KERNEL_ROOT.exists():
        print("(no research runs yet)")
        return 0
    print(f"{'run_id':<24}  events  hypotheses")
    for d in sorted(KERNEL_ROOT.iterdir()):
        ev_path = d / "events.jsonl"
        dag_path = d / "hypothesis_dag.json"
        n_events = 0
        if ev_path.exists():
            with ev_path.open(encoding="utf-8") as f:
                n_events = sum(1 for _ in f)
        n_hyp = 0
        if dag_path.exists():
            try:
                n_hyp = len(json.loads(dag_path.read_text(encoding="utf-8")).get("nodes", {}))
            except Exception:
                pass
        print(f"{d.name:<24}  {n_events:>6}  {n_hyp:>10}")
    return 0


def cmd_preregister(args: argparse.Namespace) -> int:
    k = _build_kernel(args)
    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    evt = k.propose_hypothesis(spec)
    print(f"preregistered: {evt.hypothesis_id}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    if args.name not in PLAYS:
        print(f"unknown play '{args.name}'. Available: {sorted(PLAYS)}")
        return 2
    k = _build_kernel(args)
    return PLAYS[args.name](k, args)


def main() -> int:
    p = argparse.ArgumentParser(prog="intern")
    p.add_argument("--run-id", default=None, help="kernel run_id; auto-generated if omitted")
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="bypass human gates (use ONLY when you trust the command)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-tools")
    sub.add_parser("list-hypotheses")

    c = sub.add_parser("call")
    c.add_argument("tool")
    c.add_argument("--args", default=None, help="JSON object")
    c.add_argument("--rationale", default=None)

    sub.add_parser("status")
    # --run-id is top-level

    pre = sub.add_parser("preregister")
    pre.add_argument("--spec", required=True)

    pl = sub.add_parser("play")
    pl.add_argument("name", help=f"one of: {sorted(PLAYS)}")
    pl.add_argument("--ccli-run-id", default=None)
    pl.add_argument("--hermes-run-id", default=None)
    pl.add_argument(
        "--harness-a",
        default=None,
        help="paired-harness-ablation: comma-sep agent_ids for harness A",
    )
    pl.add_argument(
        "--harness-b",
        default=None,
        help="paired-harness-ablation: comma-sep agent_ids for harness B",
    )
    pl.add_argument("--label-a", default=None)
    pl.add_argument("--label-b", default=None)

    args = p.parse_args()
    handler = {
        "list-tools": cmd_list_tools,
        "list-hypotheses": cmd_list_hypotheses,
        "call": cmd_call,
        "status": cmd_status,
        "preregister": cmd_preregister,
        "play": cmd_play,
    }[args.cmd]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
