"""Regression tests for built-in policy source-of-truth drift."""

from __future__ import annotations

from pathlib import Path

import yaml

from hermes_katana.policy.defaults import (
    BUILTIN_POLICY_PRESETS,
    BUILTIN_POLICY_SETS,
    builtin_policy_path,
    builtin_policy_sets,
)
from hermes_katana.policy.engine import PolicyEngine
from hermes_katana.policy.yaml_loader import load_policy_file

ROOT = Path(__file__).resolve().parents[2]


def _policy_fingerprint(policy) -> dict:
    return policy.model_dump(mode="json", exclude_none=True)


def _taint_context(field: str, *, level: int = 8) -> dict:
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


README_POLICY_CASES = (
    ("Clean terminal", "terminal", {"command": "ls"}, {}),
    ("Tainted terminal", "terminal", {"command": "cat README.md"}, _taint_context("command")),
    ("Dangerous terminal", "terminal", {"command": "rm -rf /"}, _taint_context("command")),
    ("Clean unknown tool", "unknown_tool_xyz", {"arg": "hello"}, {}),
    ("Tainted read-only", "read_file", {"path": "README.md"}, _taint_context("path")),
)


def _display_action(action: str) -> str:
    return action.upper()


def _readme_policy_table() -> dict[str, dict[str, str]]:
    lines = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = lines.index("<!-- policy-table:start -->")
    end = lines.index("<!-- policy-table:end -->", start)
    header = lines[start + 1]
    rows = []
    for line in lines[start + 3 : end]:
        rows.append(line)

    table: dict[str, dict[str, str]] = {}
    headings = [cell.strip() for cell in header.strip("|").split("|")]
    for row in rows:
        cells = [cell.strip().strip("`") for cell in row.strip("|").split("|")]
        preset = cells[0]
        table[preset] = dict(zip(headings[1:], cells[1:]))
    return table


def test_builtin_policy_constants_are_loaded_from_top_level_yaml():
    """The compatibility constants must be exact views of policies/*.yaml."""
    for preset in BUILTIN_POLICY_PRESETS:
        policy_path = ROOT / "policies" / f"{preset}.yaml"
        assert builtin_policy_path(preset) == policy_path

        with policy_path.open("r", encoding="utf-8") as fh:
            raw_yaml = yaml.safe_load(fh)

        assert builtin_policy_sets()[preset] == raw_yaml


def test_builtin_policy_compatibility_exports_are_read_only():
    policies = BUILTIN_POLICY_SETS["balanced"]["policies"]

    try:
        policies.append({"name": "mutated"})  # type: ignore[attr-defined]
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("BUILTIN_POLICY_SETS nested policy list is mutable")


def test_policy_engine_defaults_match_canonical_yaml_files():
    """with_defaults() must evaluate the same policies as load_policy_file()."""
    for preset in BUILTIN_POLICY_PRESETS:
        from_engine = {
            policy.name: _policy_fingerprint(policy) for policy in PolicyEngine.with_defaults(preset).list_policies()
        }
        from_yaml = {
            policy.name: _policy_fingerprint(policy)
            for policy in load_policy_file(ROOT / "policies" / f"{preset}.yaml").policies
        }

        assert from_engine == from_yaml


def test_readme_policy_preset_table_matches_runtime_defaults():
    table = _readme_policy_table()

    expected: dict[str, dict[str, str]] = {}
    for preset in BUILTIN_POLICY_PRESETS:
        engine = PolicyEngine.with_defaults(preset)
        expected[preset] = {
            label: _display_action(engine.evaluate(tool, args, taint_context=taint_context).action.value)
            for label, tool, args, taint_context in README_POLICY_CASES
        }

    assert table == expected
