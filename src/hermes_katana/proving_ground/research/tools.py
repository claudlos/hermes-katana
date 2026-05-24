"""Tool router for the research agent.

Every tool is declared with:
  - name, description, JSON schema of params
  - handler function (pure: takes params → returns observation payload)
  - requires_human_approval (bool): gated actions (launch fleet, stop run)
  - budget_estimator (optional): predicted cost BEFORE execution

The router's execute() enforces the lint gate:
  1. schema validity (every required field present, types match)
  2. budget headroom (estimator vs. remaining)
  3. doom-loop check (if detector has fired, only read-only tools allowed)
  4. human approval (pauses with HumanGate event if required)

This first version wires up a MINIMAL useful set:
  - Query / discovery (read-only, always allowed)
  - Campaign control (gated, humans approve non-trivial runs)
  - Registry ops (read-only + safe writes)

Attack synthesis tools land in Phase 3 (`research.tools_synth`).
Heavy analysis tools (detection bench, harness ablation) land as scripts
and are called via `run_script` rather than implemented here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from hermes_katana.proving_ground.research.budget import BudgetLedger
from hermes_katana.proving_ground.research.doom import DoomLoopDetector
from hermes_katana.proving_ground.research.events import Action, Observation, GateRejected, HumanGate


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


# ---------------------------------------------------------------------------
# Tool spec
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: dict  # JSON-schema-like
    handler: Callable[[dict], dict]  # (args) -> observation payload
    requires_human_approval: bool = False
    read_only: bool = True  # tools that only read (no state change)
    budget_estimator: Callable[[dict], dict] | None = None  # (args) -> {bucket: amount}
    schema_version: int = 1

    def validate_args(self, args: dict) -> list[str]:
        errors: list[str] = []
        required = self.params_schema.get("required", [])
        for k in required:
            if k not in args:
                errors.append(f"missing required param: {k}")
        props = self.params_schema.get("properties", {})
        for k, v in args.items():
            if k not in props:
                errors.append(f"unknown param: {k}")
                continue
            expected_type = props[k].get("type")
            if expected_type and not _type_ok(v, expected_type):
                errors.append(f"param {k}: expected {expected_type}, got {type(v).__name__}")
        return errors


def _type_ok(v: Any, json_type: str) -> bool:
    return {
        "string": isinstance(v, str),
        "integer": isinstance(v, int) and not isinstance(v, bool),
        "number": isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": isinstance(v, bool),
        "array": isinstance(v, list),
        "object": isinstance(v, dict),
    }.get(json_type, True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """What the kernel appends to its event stream after execute()."""

    action: Action
    observation: Observation | None = None
    gate_rejection: GateRejected | None = None
    human_gate: HumanGate | None = None


class ToolRouter:
    def __init__(
        self,
        *,
        budget: BudgetLedger | None = None,
        doom: DoomLoopDetector | None = None,
        auto_approve_human_gates: bool = False,
        run_id: str | None = None,
    ):
        self.tools: dict[str, ToolSpec] = {}
        self.budget = budget
        self.doom = doom
        self.auto_approve_human_gates = auto_approve_human_gates
        self.run_id = run_id

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.tools:
            raise ValueError(f"tool {spec.name} already registered")
        self.tools[spec.name] = spec

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "description": s.description,
                "params_schema": s.params_schema,
                "requires_human_approval": s.requires_human_approval,
                "read_only": s.read_only,
            }
            for s in self.tools.values()
        ]

    def execute(self, tool: str, args: dict, *, rationale: str = "") -> ExecutionResult:
        action = Action(
            tool=tool,
            args=args,
            rationale=rationale,
            status="issued",
            run_id=self.run_id,
        )

        # 1) Tool must exist
        spec = self.tools.get(tool)
        if not spec:
            gate = GateRejected(
                tool=tool,
                reason=f"unknown tool {tool}",
                args=args,
                run_id=self.run_id,
                parent_event_id=action.event_id,
            )
            action.status = "rejected"
            return ExecutionResult(action=action, gate_rejection=gate)

        # 2) Schema validity
        errors = spec.validate_args(args)
        if errors:
            gate = GateRejected(
                tool=tool,
                reason="; ".join(errors),
                args=args,
                run_id=self.run_id,
                parent_event_id=action.event_id,
            )
            action.status = "rejected"
            return ExecutionResult(action=action, gate_rejection=gate)

        # 3) Doom-loop: if detector fired, only read-only tools allowed
        if self.doom:
            self.doom.observe(tool, args)
            fires, details = self.doom.fires()
            if fires and not spec.read_only:
                gate = GateRejected(
                    tool=tool,
                    reason=f"doom-loop active (fingerprint={details['fingerprint']})",
                    args=args,
                    run_id=self.run_id,
                    parent_event_id=action.event_id,
                )
                action.status = "rejected"
                return ExecutionResult(action=action, gate_rejection=gate)

        # 4) Budget headroom
        if self.budget and spec.budget_estimator:
            est = spec.budget_estimator(args)
            for bucket, amount in est.items():
                state = self.budget.query(bucket)
                if amount > state["remaining"]:
                    gate = GateRejected(
                        tool=tool,
                        reason=f"budget: {bucket} need {amount} remaining {state['remaining']}",
                        args=args,
                        run_id=self.run_id,
                        parent_event_id=action.event_id,
                    )
                    action.status = "rejected"
                    return ExecutionResult(action=action, gate_rejection=gate)

        # 5) Human approval
        if spec.requires_human_approval and not self.auto_approve_human_gates:
            hg = HumanGate(
                action_id=action.event_id,
                reason=f"{tool} requires human approval",
                resolved_as=None,
                run_id=self.run_id,
                parent_event_id=action.event_id,
            )
            action.status = "issued"  # pending human
            return ExecutionResult(action=action, human_gate=hg)

        # Execute
        t0 = time.time()
        try:
            payload = spec.handler(args)
            action.status = "executed"
            obs = Observation(
                source=f"tool:{tool}",
                summary=f"{tool} ok ({time.time() - t0:.2f}s)",
                data=payload,
                run_id=self.run_id,
                parent_event_id=action.event_id,
            )
            if self.budget and spec.budget_estimator:
                for bucket, amount in spec.budget_estimator(args).items():
                    self.budget.charge(bucket, amount)
            return ExecutionResult(action=action, observation=obs)
        except Exception as e:
            action.status = "executed"
            obs = Observation(
                source=f"tool:{tool}",
                summary=f"{tool} FAILED: {e}",
                data={"error": str(e), "type": type(e).__name__},
                run_id=self.run_id,
                parent_event_id=action.event_id,
            )
            return ExecutionResult(action=action, observation=obs)


# ---------------------------------------------------------------------------
# Tool handlers — security research-specific
# ---------------------------------------------------------------------------


def _list_agents_handler(_args: dict) -> dict:

    from hermes_katana.proving_ground.sandbox.agent_cli_runner import AGENT_DRIVERS

    return {
        "count": len(AGENT_DRIVERS),
        "agents": [{"id": k, "description": v.description} for k, v in AGENT_DRIVERS.items()],
    }


def _list_shards_handler(_args: dict) -> dict:
    shards_dir = ROOT / "shards"
    if not shards_dir.exists():
        return {"count": 0, "shards": []}
    english = sorted(shards_dir.glob("shard_0??.jsonl"))
    multilingual = sorted(shards_dir.glob("shard_1??.jsonl"))
    control_dir = shards_dir / "control"
    control = sorted(control_dir.glob("shard_ctrl_???.jsonl")) if control_dir.exists() else []

    def _n_rows(p):
        with p.open(encoding="utf-8") as f:
            return sum(1 for _ in f)

    return {
        "n_english": len(english),
        "n_multilingual": len(multilingual),
        "n_control": len(control),
        "english": [{"path": str(p.relative_to(ROOT)), "n": _n_rows(p)} for p in english[:5]],
        "multilingual": [{"path": str(p.relative_to(ROOT)), "n": _n_rows(p)} for p in multilingual[:5]],
    }


def _script_module(name: str) -> list[str]:
    return [PY, "-m", f"hermes_katana.proving_ground.scripts.{name}"]


def _query_corpus_handler(args: dict) -> dict:
    """Invoke the query helper and parse its --json output."""
    cmd = [*_script_module("query"), "--json"]
    for k in ("run_id", "agent", "channel", "shard", "label", "attack_id"):
        if args.get(k) is not None:
            cmd += [f"--{k.replace('_', '-')}", str(args[k])]
        if k == "effective" and args.get("effective"):
            cmd.append("--effective")
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"query.py failed: {out.stderr[:400]}")
    return json.loads(out.stdout or "{}")


def _list_active_runs_handler(_args: dict) -> dict:
    """Return what fleets are active (reads results/fleet_runs/)."""
    import os

    runs_dir = ROOT / "results" / "fleet_runs"
    if not runs_dir.exists():
        return {"active": []}
    out = []
    for d in sorted(runs_dir.iterdir()):
        pid_file = d / "supervisor.pid"
        if not pid_file.exists():
            continue
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, 0)
            meta_file = d / "run_meta.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            out.append(
                {
                    "run_id": d.name,
                    "pid": pid,
                    "spec": meta.get("spec_path"),
                    "total_jobs": meta.get("total_jobs"),
                    "started_at_iso": meta.get("started_at_iso"),
                }
            )
        except (ProcessLookupError, ValueError, FileNotFoundError):
            continue
    return {"active": out}


def _list_hypotheses_handler(_args: dict) -> dict:
    from hermes_katana.proving_ground.research.registry import HypothesisRegistry

    reg = HypothesisRegistry()
    return {
        "hypotheses": [
            {
                "id": h.id,
                "title": h.title,
                "status": h.status,
                "verdict": (h.resolution or {}).get("verdict") if h.resolution else None,
            }
            for h in reg.list_all()
        ]
    }


def _generate_report_handler(args: dict) -> dict:
    run_id = args["run_id"]
    out = subprocess.run(
        [*_script_module("report"), "--run-id", run_id],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        encoding="utf-8",
    )
    if out.returncode != 0:
        raise RuntimeError(f"report.py failed: {out.stderr[:400]}")
    return {
        "run_id": run_id,
        "report_path": f"results/reports/{run_id}/report.md",
        "stdout_tail": out.stdout[-500:],
    }


def _launch_fleet_handler(args: dict) -> dict:
    spec_path = args["spec"]
    run_id = args.get("run_id")
    cmd = [*_script_module("fleet"), "launch", "--spec", spec_path]
    if run_id:
        cmd += ["--run-id", run_id]
    # Blocking would tie up the agent for hours; spawn detached.
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"pid": proc.pid, "spec": spec_path, "run_id": run_id, "spawned": True}


def _stop_fleet_handler(args: dict) -> dict:
    run_id = args["run_id"]
    cmd = [*_script_module("fleet"), "stop", "--run-id", run_id]
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=30, encoding="utf-8")
    return {"run_id": run_id, "stdout": out.stdout, "returncode": out.returncode}


def _harness_matrix_handler(args: dict) -> dict:
    cmd = _script_module("harness_matrix")
    if args.get("apply_exclusion"):
        cmd.append("--apply-exclusion")
    if args.get("min_n"):
        cmd += ["--min-n", str(args["min_n"])]
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=300, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"harness_matrix failed: {out.stderr[-400:]}")
    report_path = ROOT / "results" / "harness_matrix.json"
    return {
        "report_path": str(report_path.relative_to(ROOT)),
        "result": json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None,
        "stdout_tail": out.stdout[-500:],
    }


def _factorial_decompose_handler(args: dict) -> dict:
    cmd = _script_module("factorial_decompose")
    if args.get("apply_exclusion"):
        cmd.append("--apply-exclusion")
    sub = args.get("subsample") or 80000
    cmd += ["--subsample", str(sub)]
    if args.get("interactions"):
        cmd.append("--interactions")
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"factorial_decompose failed: {out.stderr[-400:]}")
    report_path = ROOT / "results" / "factorial.json"
    return {
        "report_path": str(report_path.relative_to(ROOT)),
        "result": json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None,
        "stdout_tail": out.stdout[-800:],
    }


def _simulate_defense_handler(args: dict) -> dict:
    cmd = _script_module("simulate_katana_defense")
    if args.get("apply_exclusion"):
        cmd.append("--apply-exclusion")
    if args.get("threshold") is not None:
        cmd += ["--threshold", str(args["threshold"])]
    if args.get("max_rows"):
        cmd += ["--max-rows", str(args["max_rows"])]
    if args.get("sweep"):
        cmd.append("--sweep")
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"simulate_katana_defense failed: {out.stderr[-400:]}")
    out_name = "katana_defense_simulation_sweep.json" if args.get("sweep") else "katana_defense_simulation.json"
    report_path = ROOT / "results" / out_name
    return {
        "report_path": str(report_path.relative_to(ROOT)),
        "result": json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None,
        "stdout_tail": out.stdout[-800:],
    }


def _harness_ablation_handler(args: dict) -> dict:
    cmd = _script_module("harness_ablation")
    for k in ("harness_a", "harness_b", "label_a", "label_b"):
        if args.get(k):
            cmd += [f"--{k.replace('_', '-')}", args[k]]
    if args.get("submit_to_kernel"):
        cmd.append("--submit-to-kernel")
    if args.get("run_id"):
        cmd += ["--run-id", args["run_id"]]
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"harness_ablation failed: {out.stderr[-400:]}")
    report_path = ROOT / "results" / "harness_ablation.json"
    return {
        "report_path": str(report_path.relative_to(ROOT)),
        "result": json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None,
        "stdout_tail": out.stdout[-800:],
    }


def _detection_bench_handler(args: dict) -> dict:
    cmd = _script_module("detection_bench")
    if args.get("detectors"):
        cmd += ["--detectors", args["detectors"]]
    if args.get("apply_dedup"):
        cmd.append("--apply-dedup")
    if args.get("max_neg"):
        cmd += ["--max-neg", str(args["max_neg"])]
    if args.get("skip_channel_strat"):
        cmd.append("--skip-channel-strat")
    out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600, encoding="utf-8")
    if out.returncode != 0:
        raise RuntimeError(f"detection_bench failed: {out.stderr[-400:]}")
    report_path = ROOT / "results" / "detection_bench.json"
    return {
        "report_path": str(report_path.relative_to(ROOT)),
        "result": json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else None,
        "stdout_tail": out.stdout[-800:],
    }


# ---------------------------------------------------------------------------
# Default tool set
# ---------------------------------------------------------------------------


def default_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="list_agents",
            description="List available CLI-agent drivers the fleet can invoke.",
            params_schema={"type": "object", "properties": {}, "required": []},
            handler=_list_agents_handler,
            read_only=True,
        ),
        ToolSpec(
            name="list_shards",
            description="Inventory of attack/benign/multilingual shards.",
            params_schema={"type": "object", "properties": {}, "required": []},
            handler=_list_shards_handler,
            read_only=True,
        ),
        ToolSpec(
            name="query_corpus",
            description=(
                "Aggregate query over results/agent_shard_runs/*. "
                "Supports filters: run_id, agent, channel, shard, label, attack_id, effective."
            ),
            params_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "agent": {"type": "string"},
                    "channel": {"type": "string"},
                    "shard": {"type": "integer"},
                    "label": {"type": "string"},
                    "attack_id": {"type": "string"},
                    "effective": {"type": "boolean"},
                },
                "required": [],
            },
            handler=_query_corpus_handler,
            read_only=True,
        ),
        ToolSpec(
            name="list_active_runs",
            description="Return currently-active fleet supervisors (run_id + PID).",
            params_schema={"type": "object", "properties": {}, "required": []},
            handler=_list_active_runs_handler,
            read_only=True,
        ),
        ToolSpec(
            name="list_hypotheses",
            description="List all preregistered / resolved hypotheses.",
            params_schema={"type": "object", "properties": {}, "required": []},
            handler=_list_hypotheses_handler,
            read_only=True,
        ),
        ToolSpec(
            name="generate_report",
            description="Generate/refresh results/reports/<run_id>/report.md.",
            params_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
            },
            handler=_generate_report_handler,
            read_only=False,
        ),
        ToolSpec(
            name="launch_fleet",
            description="Spawn a detached proving-ground fleet supervisor with spec + run_id.",
            params_schema={
                "type": "object",
                "properties": {
                    "spec": {"type": "string"},
                    "run_id": {"type": "string"},
                },
                "required": ["spec"],
            },
            handler=_launch_fleet_handler,
            read_only=False,
            requires_human_approval=True,
            budget_estimator=lambda args: {"claude_max": 0.05},  # rough; real value depends on spec
        ),
        ToolSpec(
            name="stop_fleet",
            description="SIGINT a running fleet supervisor by run_id.",
            params_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
            },
            handler=_stop_fleet_handler,
            read_only=False,
            requires_human_approval=True,
        ),
        # Analysis tools: read-only but compute-heavy (~30s to 10min).
        ToolSpec(
            name="harness_matrix",
            description="Coverage matrix (model_family × harness_type) × Wilson-CI effective rate.",
            params_schema={
                "type": "object",
                "properties": {
                    "apply_exclusion": {"type": "boolean"},
                    "min_n": {"type": "integer"},
                },
                "required": [],
            },
            handler=_harness_matrix_handler,
            read_only=True,
        ),
        ToolSpec(
            name="factorial_decompose",
            description=(
                "Logistic regression (HC3 SEs) decomposing effective ~ "
                "model_family + harness_type + channel + model_size. "
                "Reports per-term odds ratio + CI + p-value."
            ),
            params_schema={
                "type": "object",
                "properties": {
                    "apply_exclusion": {"type": "boolean"},
                    "subsample": {"type": "integer"},
                    "interactions": {"type": "boolean"},
                },
                "required": [],
            },
            handler=_factorial_decompose_handler,
            read_only=True,
        ),
        ToolSpec(
            name="simulate_katana_defense",
            description=(
                "Counterfactual: if Katana sat between env and agent at "
                "score >= threshold, what would effective rate be? "
                "Returns base/defended rates + Cohen's h + per-cell breakdown."
            ),
            params_schema={
                "type": "object",
                "properties": {
                    "threshold": {"type": "number"},
                    "apply_exclusion": {"type": "boolean"},
                    "max_rows": {"type": "integer"},
                    "sweep": {"type": "boolean"},
                },
                "required": [],
            },
            handler=_simulate_defense_handler,
            read_only=True,
        ),
        ToolSpec(
            name="harness_ablation",
            description=(
                "Paired McNemar of same-attack × same-channel × harness_A vs B. "
                "Pass submit_to_kernel=true to route the verdict through "
                "the registry and auto-resolve matching preregistered hypotheses."
            ),
            params_schema={
                "type": "object",
                "properties": {
                    "harness_a": {"type": "string"},
                    "harness_b": {"type": "string"},
                    "label_a": {"type": "string"},
                    "label_b": {"type": "string"},
                    "submit_to_kernel": {"type": "boolean"},
                    "run_id": {"type": "string"},
                },
                "required": [],
            },
            handler=_harness_ablation_handler,
            read_only=True,
        ),
        ToolSpec(
            name="detection_bench",
            description=(
                "Channel-stratified detection benchmark. Runs selected "
                "detectors over confirmed + benign control, reports AUC / F1 / "
                "recall @ {1,5}% FPR with CIs."
            ),
            params_schema={
                "type": "object",
                "properties": {
                    "detectors": {"type": "string"},
                    "apply_dedup": {"type": "boolean"},
                    "max_neg": {"type": "integer"},
                    "skip_channel_strat": {"type": "boolean"},
                },
                "required": [],
            },
            handler=_detection_bench_handler,
            read_only=True,
        ),
    ]


def build_default_router(
    *,
    budget: BudgetLedger | None = None,
    doom: DoomLoopDetector | None = None,
    run_id: str | None = None,
    auto_approve_human_gates: bool = False,
) -> ToolRouter:
    router = ToolRouter(
        budget=budget,
        doom=doom,
        run_id=run_id,
        auto_approve_human_gates=auto_approve_human_gates,
    )
    for spec in default_tools():
        router.register(spec)
    return router
