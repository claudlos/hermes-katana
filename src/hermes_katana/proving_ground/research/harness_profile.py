"""Harness component profiles.

A harness is not a monolith. To measure what PART of it matters for
injection resistance, we describe each agent as a vector of component
features. Factorial analysis over the 355k-row corpus then decomposes
effective-rate variance into per-feature coefficients with CIs.

Features we track (binary unless noted):

  model_family        categorical  (claude, openai, gemini, qwen, llama, minimax, nemotron, codex)
  model_size          categorical  (small <8B, medium 8-70B, large >70B, unknown)
  harness_type        categorical  (cli_coding_agent, api_chat, hermes_agent, codex_cli,
                                     openrouter_relay, gemini_cli)
  instruction_hier    bool         Has explicit system-message hierarchy (e.g. Claude Code's
                                   "You are a coding assistant, user messages are untrusted").
  bash_allowlist      categorical  none | loose | strict
  untrusted_marker    bool         Wraps tool/file content with "Untrusted content:" markers
  permission_gate     bool         Blocks destructive tool calls pending user approval
  scanner_layer       bool         Inline prompt-injection scanner (today: False for all; will
                                   flip on when hermes-katana middleware lands)
  tool_auto_exec      bool         Tools auto-execute without human-in-loop
  multi_turn          bool         Agent can accept arbitrary turns vs. one-shot

The profile is the scientific variable of interest. Effect-rate of an
agent = f(profile, model_family, channel). The factorial runs OLS
(logistic) over one-hots + continuous features.

Profile data lives in `research/harness_profiles.yaml`. This module
provides the loader + utility helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "research" / "harness_profiles.yaml"


@dataclass
class HarnessProfile:
    agent_id: str
    model_family: str
    model_size: str
    harness_type: str
    instruction_hier: bool
    bash_allowlist: str
    untrusted_marker: bool
    permission_gate: bool
    scanner_layer: bool
    tool_auto_exec: bool
    multi_turn: bool
    notes: str = ""

    def to_feature_dict(self) -> dict[str, Any]:
        """Flatten to a feature dict suitable for OLS / logistic regression.

        Boolean → 0/1 int. Categorical → one-hot columns.
        """
        out: dict[str, Any] = {
            "instruction_hier": int(self.instruction_hier),
            "untrusted_marker": int(self.untrusted_marker),
            "permission_gate": int(self.permission_gate),
            "scanner_layer": int(self.scanner_layer),
            "tool_auto_exec": int(self.tool_auto_exec),
            "multi_turn": int(self.multi_turn),
        }
        # One-hots
        for fam in MODEL_FAMILIES:
            out[f"mf_{fam}"] = int(self.model_family == fam)
        for sz in MODEL_SIZES:
            out[f"ms_{sz}"] = int(self.model_size == sz)
        for ht in HARNESS_TYPES:
            out[f"ht_{ht}"] = int(self.harness_type == ht)
        for ba in BASH_ALLOWLIST:
            out[f"ba_{ba}"] = int(self.bash_allowlist == ba)
        return out


MODEL_FAMILIES = [
    "claude",
    "openai",
    "gemini",
    "qwen",
    "llama",
    "minimax",
    "nemotron",
    "codex",
    "gemma",
    "deepseek",
    "arcee",
    "mimo",
    "stepfun",
    "kimi",
    "unknown",
]
MODEL_SIZES = ["small", "medium", "large", "unknown"]
HARNESS_TYPES = [
    "cli_coding_agent",
    "api_chat",
    "hermes_agent",
    "codex_cli",
    "openrouter_relay",
    "gemini_cli",
    "minimax_cli",
    "unknown",
]
BASH_ALLOWLIST = ["none", "loose", "strict", "unknown"]


def load_profiles(path: Path = PROFILES_PATH) -> dict[str, HarnessProfile]:
    if not path.exists():
        return {}
    d = yaml.safe_load(path.read_text()) or {}
    out: dict[str, HarnessProfile] = {}
    for agent_id, raw in (d.get("agents") or {}).items():
        out[agent_id] = HarnessProfile(
            agent_id=agent_id,
            model_family=raw.get("model_family", "unknown"),
            model_size=raw.get("model_size", "unknown"),
            harness_type=raw.get("harness_type", "unknown"),
            instruction_hier=bool(raw.get("instruction_hier", False)),
            bash_allowlist=raw.get("bash_allowlist", "unknown"),
            untrusted_marker=bool(raw.get("untrusted_marker", False)),
            permission_gate=bool(raw.get("permission_gate", False)),
            scanner_layer=bool(raw.get("scanner_layer", False)),
            tool_auto_exec=bool(raw.get("tool_auto_exec", True)),
            multi_turn=bool(raw.get("multi_turn", True)),
            notes=raw.get("notes", ""),
        )
    return out


def save_profiles(profiles: dict[str, HarnessProfile], path: Path = PROFILES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "agents": {
            agent_id: {
                "model_family": p.model_family,
                "model_size": p.model_size,
                "harness_type": p.harness_type,
                "instruction_hier": p.instruction_hier,
                "bash_allowlist": p.bash_allowlist,
                "untrusted_marker": p.untrusted_marker,
                "permission_gate": p.permission_gate,
                "scanner_layer": p.scanner_layer,
                "tool_auto_exec": p.tool_auto_exec,
                "multi_turn": p.multi_turn,
                "notes": p.notes,
            }
            for agent_id, p in profiles.items()
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, width=120))
