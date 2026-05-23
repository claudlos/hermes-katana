#!/usr/bin/env python3
"""Regenerate or check built-in policy assets.

This keeps the canonical ``policies/*.yaml`` files mechanically formatted and
keeps the README preset behavior table synchronized with runtime decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
POLICIES_DIR = ROOT / "policies"
README = ROOT / "README.md"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_katana.policy.defaults import BUILTIN_POLICY_PRESETS  # noqa: E402
from hermes_katana.policy.engine import PolicyEngine  # noqa: E402

TABLE_START = "<!-- policy-table:start -->"
TABLE_END = "<!-- policy-table:end -->"


def _taint_context(field: str, *, level: int = 8) -> dict[str, Any]:
    return {
        "tainted_fields": {
            field: {
                "is_tainted": True,
                "source": "web_content",
                "labels": ["web_content"],
                "readers": [],
                "level": level,
            }
        }
    }


POLICY_TABLE_CASES = (
    ("Clean terminal", "terminal", {"command": "ls"}, {}),
    ("Tainted terminal", "terminal", {"command": "cat README.md"}, _taint_context("command")),
    ("Dangerous terminal", "terminal", {"command": "rm -rf /"}, _taint_context("command")),
    ("Clean unknown tool", "unknown_tool_xyz", {"arg": "hello"}, {}),
    ("Tainted read-only", "read_file", {"path": "README.md"}, _taint_context("path")),
)


def _display_action(action: str) -> str:
    return action.upper()


def render_policy_table() -> str:
    headings = ["Preset", *(case[0] for case in POLICY_TABLE_CASES)]
    lines = [
        TABLE_START,
        "| " + " | ".join(headings) + " |",
        "|--------|:---:|:---:|:---:|:---:|:---:|",
    ]

    for preset in BUILTIN_POLICY_PRESETS:
        engine = PolicyEngine.with_defaults(preset)
        decisions = [
            _display_action(engine.evaluate(tool, args, taint_context=taint_context).action.value)
            for _, tool, args, taint_context in POLICY_TABLE_CASES
        ]
        lines.append("| `" + preset + "` | " + " | ".join(decisions) + " |")

    lines.append(TABLE_END)
    return "\n".join(lines)


def _replace_marked_block(text: str, replacement: str) -> str:
    start = text.index(TABLE_START)
    end = text.index(TABLE_END, start) + len(TABLE_END)
    return text[:start] + replacement + text[end:]


def render_policy_yaml(preset: str) -> str:
    path = POLICIES_DIR / f"{preset}.yaml"
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} must contain a YAML mapping")

    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=False, width=100)
    return (
        f"# HermesKatana built-in policy preset: {preset}\n"
        "# Source of truth for PolicyEngine.with_defaults(); keep in sync via tests.\n\n"
        f"{body}"
    )


def generated_files() -> dict[Path, str]:
    files = {POLICIES_DIR / f"{preset}.yaml": render_policy_yaml(preset) for preset in BUILTIN_POLICY_PRESETS}
    readme_text = README.read_text(encoding="utf-8")
    files[README] = _replace_marked_block(readme_text, render_policy_table())
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Rewrite generated files in place.")
    parser.add_argument("--check", action="store_true", help="Fail if generated files are stale.")
    args = parser.parse_args()

    if not args.write and not args.check:
        args.check = True

    stale: list[Path] = []
    for path, expected in generated_files().items():
        current = path.read_text(encoding="utf-8")
        if current == expected:
            continue
        stale.append(path)
        if args.write:
            path.write_text(expected, encoding="utf-8", newline="\n")

    if stale and args.check:
        for path in stale:
            print(f"stale generated policy asset: {path.relative_to(ROOT)}", file=sys.stderr)
        print("Run: python scripts/generate_policy_assets.py --write", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
