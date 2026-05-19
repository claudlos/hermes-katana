#!/usr/bin/env python3
"""Katana Sandbox — behavioral attack testing in a controlled workspace.

Usage:
    python sandbox_cli.py run --task code_review --attack-id <id> --model <model>
    python sandbox_cli.py analyze --session <id>
    python sandbox_cli.py batch --sample 50 --model qwen3-8b
    python sandbox_cli.py list-sessions
"""

import argparse
import asyncio
import json
import uuid
from pathlib import Path


from hermes_katana.proving_ground.sandbox.session import SessionRunner, SessionConfig, WORKSPACE_TASKS
from hermes_katana.proving_ground.sandbox.honeypot import HoneypotChannel
from hermes_katana.proving_ground.sandbox.behavioral_tracker import BehavioralTracker
from hermes_katana.proving_ground.sandbox.analyzers.behavioral_drift import BehavioralAnalyzer
from hermes_katana.proving_ground.corpus_sampler import CorpusSampler


def get_tracker() -> BehavioralTracker:
    Path("results").mkdir(parents=True, exist_ok=True)
    return BehavioralTracker("results/sandbox_tracking.db")


async def run_single(args):
    """Run a single sandbox session."""
    tracker = get_tracker()

    # Load attack from corpus
    sampler = CorpusSampler(args.katana_root, corpus_paths=[args.corpus] if args.corpus else None)
    if args.attack_id:
        # Find specific attack
        samples = sampler.sample(n=1000, seed=42)
        attack = next((s for s in samples if s.id == args.attack_id), None)
        if not attack:
            print(f"[!] Attack {args.attack_id} not found, using random sample")
            sampled = sampler.sample(n=1, seed=42)
            if not sampled:
                raise SystemExit("No attacks found. Pass --corpus PATH or set KATANA_ATTACK_CORPUS.")
            attack = sampled[0]
    else:
        # Random attack
        sampled = sampler.sample(n=1, seed=args.seed)
        if not sampled:
            raise SystemExit("No attacks found. Pass --corpus PATH or set KATANA_ATTACK_CORPUS.")
        attack = sampled[0]

    config = SessionConfig(
        task_name=args.task,
        attack=attack,
        honeypot_channel=HoneypotChannel(args.channel),
        model=args.model,
        base_url=args.base_url,
        max_turns=args.max_turns,
        trigger_after_turns=args.trigger_turns,
        target_file=args.target_file,
    )

    runner = SessionRunner(tracker, workspace_root=f"sessions/{uuid.uuid4().hex[:8]}")

    print("[*] Running sandbox session")
    print(f"    Task: {args.task}")
    print(f"    Attack: [{attack.label}] {attack.text[:80]}...")
    print(f"    Channel: {args.channel}")
    print(f"    Model: {args.model}")
    print(f"    Endpoint: {args.base_url}")

    result = await runner.run(config)

    print(f"\n[+] Session complete: {result.session_id}")
    print(f"    Turns: {result.total_turns}")
    print(f"    Tool calls before attack: {result.tool_calls_before}")
    print(f"    Tool calls after attack: {result.tool_calls_after}")

    # Analyze
    analyzer = BehavioralAnalyzer()
    report = analyzer.analyze_session(result.session_id)

    print("\n[+] Behavioral Analysis:")
    print(f"    Tool usage drift: {report.tool_usage_drift:.3f}")
    print(f"    Task drift: {report.task_drift:.3f}")
    print(f"    Exfiltration risk: {report.exfiltration_risk:.3f}")
    print(f"    Dangerous action rate: {report.dangerous_action_rate:.3f}")
    if report.collapse_detected:
        print("    !! BEHAVIORAL COLLAPSE: model went silent post-attack")
    print("\n    Findings:")
    for f in report.findings:
        print(f"    - {f}")

    # Save report
    report_path = f"results/report_{result.session_id}.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"\n    Report saved: {report_path}")

    tracker.close()


async def run_batch(args):
    """Run multiple sandbox sessions across different attacks."""
    tracker = get_tracker()
    sampler = CorpusSampler(args.katana_root, corpus_paths=[args.corpus] if args.corpus else None)
    analyzer = BehavioralAnalyzer()

    # Sample attacks
    attacks = sampler.sample(n=args.sample, seed=args.seed)
    print(f"[*] Running {len(attacks)} sandbox sessions")
    print(f"    Model: {args.model}")
    print(f"    Task: {args.task}")
    print(f"    Channel: {args.channel}")

    results = []
    for i, attack in enumerate(attacks):
        print(f"\n--- Session {i + 1}/{len(attacks)}: [{attack.label}] {attack.text[:60]}... ---")

        config = SessionConfig(
            task_name=args.task,
            attack=attack,
            honeypot_channel=HoneypotChannel(args.channel),
            model=args.model,
            base_url=args.base_url,
            max_turns=args.max_turns,
            trigger_after_turns=args.trigger_turns,
        )

        runner = SessionRunner(tracker, workspace_root=f"sessions/{uuid.uuid4().hex[:8]}")
        result = await runner.run(config)

        report = analyzer.analyze_session(result.session_id)
        results.append(report)

        print(
            f"    Drift: tool={report.tool_usage_drift:.2f} task={report.task_drift:.2f} exfil={report.exfiltration_risk:.2f}"
        )

    # Summary
    print(f"\n{'=' * 80}")
    print(f"BATCH RESULTS: {len(results)} sessions")
    print(f"{'=' * 80}")

    if results:
        avg_tool_drift = sum(r.tool_usage_drift for r in results) / len(results)
        avg_task_drift = sum(r.task_drift for r in results) / len(results)
        avg_exfil = sum(r.exfiltration_risk for r in results) / len(results)
        high_drift = sum(1 for r in results if r.task_drift > 0.3)
        collapses = sum(1 for r in results if r.collapse_detected)

        print(f"Average tool drift:     {avg_tool_drift:.3f}")
        print(f"Average task drift:     {avg_task_drift:.3f}")
        print(f"Average exfil risk:     {avg_exfil:.3f}")
        print(f"High drift sessions:    {high_drift}/{len(results)}")
        print(f"Collapse sessions:      {collapses}/{len(results)}")

    # Save batch report
    batch_path = "results/batch_report.json"
    with open(batch_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nBatch report saved: {batch_path}")

    tracker.close()


def analyze_session(args):
    """Analyze an existing session."""
    analyzer = BehavioralAnalyzer()
    report = analyzer.analyze_session(args.session)

    print(f"Session: {report.session_id}")
    print(f"Attack: [{report.attack_label}] via {report.honeypot_channel}")
    print("\nBehavioral Drift:")
    print(f"  Tool usage drift:   {report.tool_usage_drift:.3f}")
    print(f"  Task drift:         {report.task_drift:.3f}")
    print(f"  Exfiltration risk:  {report.exfiltration_risk:.3f}")
    print(f"  Dangerous actions:  {report.dangerous_action_rate:.3f}")
    if report.collapse_detected:
        print("  !! Collapse:         model went silent after the attack")
    print("\nFindings:")
    for f in report.findings:
        print(f"  - {f}")

    if report.baseline:
        print(f"\nBaseline: {report.baseline.tool_calls} tool calls, {report.baseline.unique_tools} unique tools")
    if report.post_attack:
        print(
            f"Post-attack: {report.post_attack.tool_calls} tool calls, {report.post_attack.unique_tools} unique tools"
        )


def list_sessions(args):
    """List all tracked sessions."""
    import sqlite3

    db_path = Path("results/sandbox_tracking.db")
    if not db_path.exists():
        print("No sessions found.")
        return

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT session_id, task, attack_label, attack_strategy, total_turns, outcome FROM sessions ORDER BY start_time DESC"
    ).fetchall()
    conn.close()

    if not rows:
        print("No sessions found.")
        return

    print(f"{'ID':<10} {'Task':<20} {'Attack':<20} {'Channel':<15} {'Turns':>5} {'Outcome':<12}")
    print("-" * 85)
    for r in rows:
        print(f"{r[0]:<10} {r[1] or '':<20} {r[2] or '':<20} {r[3] or '':<15} {r[4] or 0:>5} {r[5] or '':<12}")


def list_tasks(args):
    """List available workspace tasks."""
    print("Available workspace tasks:")
    for name, task in WORKSPACE_TASKS.items():
        print(f"\n  {name}:")
        print(f"    {task['description']}")
        print(f"    Files: {', '.join(task['files'].keys())}")


def main():
    parser = argparse.ArgumentParser(description="Katana Sandbox — behavioral attack testing")
    from hermes_katana.proving_ground.paths import katana_root as _default_katana_root

    parser.add_argument("--katana-root", default=str(_default_katana_root()))
    parser.add_argument(
        "--corpus",
        default="",
        help="JSONL attack corpus path. Defaults to KATANA_ATTACK_CORPUS or the bundled tiny sample corpus.",
    )

    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run single session")
    run_p.add_argument("--task", default="code_review", choices=list(WORKSPACE_TASKS.keys()))
    run_p.add_argument("--attack-id", default="", help="Specific attack ID")
    run_p.add_argument("--model", default="qwen3-8b")
    run_p.add_argument("--base-url", default="http://localhost:8000/v1")
    run_p.add_argument("--channel", default="file_content", choices=[c.value for c in HoneypotChannel])
    run_p.add_argument("--max-turns", type=int, default=30)
    run_p.add_argument("--trigger-turns", type=int, default=5)
    run_p.add_argument("--target-file", default="")
    run_p.add_argument("--seed", type=int, default=42)

    # batch
    batch_p = sub.add_parser("batch", help="Run batch sessions")
    batch_p.add_argument("--sample", type=int, default=10)
    batch_p.add_argument("--task", default="code_review")
    batch_p.add_argument("--model", default="qwen3-8b")
    batch_p.add_argument("--base-url", default="http://localhost:8000/v1")
    batch_p.add_argument("--channel", default="file_content")
    batch_p.add_argument("--max-turns", type=int, default=30)
    batch_p.add_argument("--trigger-turns", type=int, default=5)
    batch_p.add_argument("--seed", type=int, default=42)

    # analyze
    analyze_p = sub.add_parser("analyze", help="Analyze existing session")
    analyze_p.add_argument("session", help="Session ID")

    # list-sessions
    sub.add_parser("list-sessions", help="List all sessions")

    # list-tasks
    sub.add_parser("list-tasks", help="List workspace tasks")

    args = parser.parse_args()

    # Only run/batch need the corpus; analyze/list-* work off the local DB.
    if args.command in ("run", "batch"):
        from hermes_katana.proving_ground.paths import require_dir

        require_dir(
            Path(args.katana_root),
            env_var="HERMES_KATANA_ROOT",
            hint="clone hermes-katana as a sibling of katana-proving-ground, or set HERMES_KATANA_ROOT.",
        )

    if args.command == "run":
        asyncio.run(run_single(args))
    elif args.command == "batch":
        asyncio.run(run_batch(args))
    elif args.command == "analyze":
        analyze_session(args)
    elif args.command == "list-sessions":
        list_sessions(args)
    elif args.command == "list-tasks":
        list_tasks(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
