"""Hypothesis preregistration — the confirmatory/exploratory boundary.

Every headline scientific claim should be preregistered *before* data is
collected or analyzed for it. Otherwise, the researcher (or the agent) is
running an ever-growing multiple-comparisons test against any pattern that
looks interesting — a classic garden-of-forking-paths problem.

Preregistration here is:
  - Lightweight — one YAML per hypothesis, committed to git.
  - Binding — resolving a hypothesis writes its verdict into the same file.
  - Distinguishes confirmatory vs. exploratory — the `status` field records
    whether a hypothesis was registered before data or after.

Schema (per hypothesis YAML):

    id: H-20260422-harness-dominates-model
    title: "Harness architecture dominates model alignment for injection resistance"
    statement: |
      Holding the underlying model constant (Claude Haiku 4.5), the same
      injection corpus produces lower effective-rate under Claude Code CLI
      than under Hermes-agent harness.
    predicted_direction: less
    primary_outcome: effective_rate_delta      # rate(ccli) - rate(hermes)
    statistical_test: mcnemar                   # paired binary outcomes
    significance_level: 0.05
    min_n_per_condition: 200
    conditions:
      A: {harness: claude_cli, model: claude_cli_haiku}
      B: {harness: hermes,     model: hermes_claude_haiku}
    registered_at: 2026-04-22T17:00:00Z
    registered_by: carlos
    status: preregistered                       # preregistered | resolved | abandoned
    resolution: null
    # When resolved, resolution is:
    #   resolved_at: ...
    #   resolved_run_id: ...
    #   p_value: 0.003
    #   effect_size: {kind: cohens_h, value: -0.58}
    #   verdict: supported | rejected | inconclusive
    #   notes: one-paragraph interpretation

Usage (library):

    from hermes_katana.proving_ground.research.registry import HypothesisRegistry
    reg = HypothesisRegistry()
    h = reg.register(spec_dict)                 # writes YAML file
    reg.resolve(h.id, run_id="a5f3b2c1",
                p_value=0.003, effect_size={"kind":"cohens_h","value":-0.58},
                verdict="supported", notes="Claude Code CCLI reduced...")

Usage (CLI):

    python -m research.registry list
    python -m research.registry show H-20260422-harness-dominates-model
    python -m research.registry register --spec hyp.yaml
    python -m research.registry resolve H-... --run-id X --p 0.003 \
        --effect-kind cohens_h --effect-value -0.58 --verdict supported
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml  # pyyaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "research" / "hypotheses"

VALID_DIRECTIONS = {"greater", "less", "two-sided", "equivalence"}
VALID_TESTS = {
    "mcnemar",
    "wilson_ci",
    "paired_bootstrap",
    "chi2",
    "proportion_diff",
    "bootstrap",
    "equivalence_tost",
    "permutation",
}
VALID_STATUS = {"preregistered", "resolved", "abandoned"}
VALID_VERDICTS = {"supported", "rejected", "inconclusive"}


@dataclass
class Hypothesis:
    id: str
    title: str
    statement: str
    predicted_direction: str
    primary_outcome: str
    statistical_test: str
    significance_level: float
    min_n_per_condition: int
    conditions: dict
    registered_at: str
    registered_by: str
    status: str = "preregistered"
    resolution: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class HypothesisRegistry:
    def __init__(self, root: Path = DEFAULT_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ I/O
    def _path(self, hyp_id: str) -> Path:
        return self.root / f"{hyp_id}.yaml"

    def load(self, hyp_id: str) -> Hypothesis:
        p = self._path(hyp_id)
        if not p.exists():
            raise KeyError(f"no hypothesis registered with id={hyp_id}")
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return Hypothesis(**d)

    def save(self, h: Hypothesis) -> None:
        p = self._path(h.id)
        # Atomic write: tmp + rename
        tmp = p.with_suffix(".yaml.tmp")
        tmp.write_text(
            yaml.safe_dump(h.to_dict(), sort_keys=False, default_flow_style=False, width=120), encoding="utf-8"
        )
        tmp.replace(p)

    def list_all(self) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        for p in sorted(self.root.glob("*.yaml")):
            try:
                out.append(Hypothesis(**(yaml.safe_load(p.read_text(encoding="utf-8")) or {})))
            except Exception as e:
                print(f"WARN: failed to load {p.name}: {e}", file=sys.stderr)
        return out

    # ----------------------------------------------------------- registration
    def register(self, spec: dict, *, allow_overwrite: bool = False) -> Hypothesis:
        self._validate(spec)
        hyp_id = spec["id"]
        p = self._path(hyp_id)
        if p.exists() and not allow_overwrite:
            raise FileExistsError(
                f"hypothesis {hyp_id} already registered at {p}. "
                "Use --allow-overwrite to replace (AUDIT RISK — avoid post-hoc changes)."
            )
        h = Hypothesis(
            id=hyp_id,
            title=spec["title"],
            statement=spec["statement"],
            predicted_direction=spec["predicted_direction"],
            primary_outcome=spec["primary_outcome"],
            statistical_test=spec["statistical_test"],
            significance_level=float(spec.get("significance_level", 0.05)),
            min_n_per_condition=int(spec.get("min_n_per_condition", 30)),
            conditions=spec["conditions"],
            registered_at=spec.get(
                "registered_at",
                datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            ),
            registered_by=spec.get("registered_by", "unknown"),
            status="preregistered",
            resolution=None,
        )
        self.save(h)
        return h

    def resolve(
        self,
        hyp_id: str,
        *,
        run_id: str,
        p_value: float,
        effect_size: dict,
        verdict: str,
        notes: str = "",
    ) -> Hypothesis:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {VALID_VERDICTS}")
        h = self.load(hyp_id)
        if h.status == "resolved":
            raise ValueError(
                f"hypothesis {hyp_id} already resolved "
                f"(verdict={h.resolution.get('verdict') if h.resolution else '?'})."
            )
        h.status = "resolved"
        h.resolution = {
            "resolved_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "resolved_run_id": run_id,
            "p_value": float(p_value),
            "effect_size": effect_size,
            "verdict": verdict,
            "notes": notes,
        }
        self.save(h)
        return h

    def abandon(self, hyp_id: str, reason: str) -> Hypothesis:
        h = self.load(hyp_id)
        h.status = "abandoned"
        h.resolution = {
            "resolved_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "notes": f"ABANDONED: {reason}",
        }
        self.save(h)
        return h

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _validate(spec: dict) -> None:
        required = {
            "id",
            "title",
            "statement",
            "predicted_direction",
            "primary_outcome",
            "statistical_test",
            "conditions",
        }
        missing = required - spec.keys()
        if missing:
            raise ValueError(f"missing keys: {sorted(missing)}")
        if spec["predicted_direction"] not in VALID_DIRECTIONS:
            raise ValueError(
                f"predicted_direction must be one of {VALID_DIRECTIONS}, got {spec['predicted_direction']!r}"
            )
        if spec["statistical_test"] not in VALID_TESTS:
            raise ValueError(f"statistical_test must be one of {VALID_TESTS}, got {spec['statistical_test']!r}")
        if not isinstance(spec["conditions"], dict) or len(spec["conditions"]) < 2:
            raise ValueError("conditions must be a dict with at least 2 named conditions")
        # Stable-id hygiene check: recommend H-<date>-<slug> form.
        if not spec["id"].startswith("H-"):
            raise ValueError("id should start with 'H-' (e.g. H-20260422-harness-dominates-model)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_hyp(h: Hypothesis, verbose: bool = False) -> None:
    status = h.status.upper()
    if h.resolution and h.status == "resolved":
        status += f" ({h.resolution['verdict']})"
    line = f"[{status:<25}] {h.id}  {h.title[:80]}"
    print(line)
    if verbose:
        print(yaml.safe_dump(h.to_dict(), sort_keys=False, default_flow_style=False))


def main() -> int:
    p = argparse.ArgumentParser(prog="research.registry")
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="list all hypotheses with status")
    lp.add_argument("--status", choices=sorted(VALID_STATUS | {"all"}), default="all")

    sp = sub.add_parser("show", help="show full YAML for one hypothesis")
    sp.add_argument("id")

    rp = sub.add_parser("register", help="register a hypothesis from a YAML spec file")
    rp.add_argument("--spec", required=True, help="path to YAML hypothesis spec")
    rp.add_argument("--allow-overwrite", action="store_true")

    resp = sub.add_parser("resolve", help="record resolution of a preregistered hypothesis")
    resp.add_argument("id")
    resp.add_argument("--run-id", required=True)
    resp.add_argument("--p", type=float, required=True, dest="p_value")
    resp.add_argument("--effect-kind", required=True, help="e.g. cohens_h, delta_p, auc_delta")
    resp.add_argument("--effect-value", type=float, required=True)
    resp.add_argument("--verdict", choices=sorted(VALID_VERDICTS), required=True)
    resp.add_argument("--notes", default="")

    ap = sub.add_parser("abandon", help="mark hypothesis as abandoned")
    ap.add_argument("id")
    ap.add_argument("--reason", required=True)

    args = p.parse_args()
    reg = HypothesisRegistry()

    if args.cmd == "list":
        hs = reg.list_all()
        if args.status != "all":
            hs = [h for h in hs if h.status == args.status]
        for h in hs:
            _print_hyp(h)
        print(f"\n{len(hs)} hypotheses")
        return 0

    if args.cmd == "show":
        h = reg.load(args.id)
        _print_hyp(h, verbose=True)
        return 0

    if args.cmd == "register":
        spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
        h = reg.register(spec, allow_overwrite=args.allow_overwrite)
        print(f"registered: {h.id}  at  {reg._path(h.id)}")
        return 0

    if args.cmd == "resolve":
        h = reg.resolve(
            args.id,
            run_id=args.run_id,
            p_value=args.p_value,
            effect_size={"kind": args.effect_kind, "value": args.effect_value},
            verdict=args.verdict,
            notes=args.notes,
        )
        print(f"resolved: {h.id}  verdict={h.resolution['verdict']}  p={args.p_value}")
        return 0

    if args.cmd == "abandon":
        h = reg.abandon(args.id, reason=args.reason)
        print(f"abandoned: {h.id}  ({args.reason})")
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
