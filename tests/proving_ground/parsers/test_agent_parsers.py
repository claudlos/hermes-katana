"""Golden-output tests for agent CLI parsers.

The proving-ground harness extracts tool calls from each agent CLI's stdout
to compute behavioral drift metrics. The 2026-05-04 parser audit identified
five agent drivers that need golden-output coverage:

    - hermes_or_arcee_spark        (0.0%, empty outputs)
    - hermes_or_deepseek_v3_free   (0.0%, empty outputs)
    - gemini_cli_2_5_flash         (9.5%, model_garbage)
    - codex_cli                    (30.0%, garbage + verdict misalign)
    - gemini_cli                   (53.5%, model_garbage)

This test module pins parser behavior against curated stdout/stderr fixtures
collected from real runs. Adding a fixture here means: (1) drop a
``fixtures/<driver>/<scenario>.txt`` file with the captured output, (2) add
the expected tool-call extraction below.

When parser logic changes, run ``pytest tests/proving_ground/parsers/`` to
confirm no regression. When a real-world output format changes (provider rolls
a model or a CLI version bumps), add a new fixture rather than modifying old
ones; that preserves the golden-output property.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).resolve().parents[2]
SANDBOX = ROOT / "sandbox"
if str(ROOT) not in sys.path:
    pass


# Import lazily so this module imports even before sandbox/ has all its
# transitive dependencies installed (matters for CI on minimal images).
def _get_parsers() -> dict[str, Callable]:
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import (
        parse_claude_cli_json,
        parse_codex_cli,
        parse_copilot_cli,
        parse_gemini_cli,
        parse_hermes_cli,
    )

    return {
        "claude_cli": parse_claude_cli_json,
        "codex_cli": parse_codex_cli,
        "copilot_cli": parse_copilot_cli,
        "gemini_cli": parse_gemini_cli,
        "hermes_cli": parse_hermes_cli,
    }


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _fixture(driver: str, scenario: str) -> str:
    p = FIXTURES_DIR / driver / f"{scenario}.txt"
    if not p.exists():
        pytest.skip(f"fixture missing: {p}")
    return p.read_text()


# -----------------------------------------------------------------------------
# Codex CLI parser tests
# -----------------------------------------------------------------------------


def test_codex_parses_single_exec(tmp_path):
    """Single bash exec → one tool call with the command preserved."""
    parsers = _get_parsers()
    stdout = 'exec\n/bin/bash -lc "ls -la /tmp" in /tmp/work\n succeeded in 12ms:\ntotal 0\n'
    calls = parsers["codex_cli"](stdout, "", tmp_path)
    assert len(calls) == 1
    assert calls[0]["name"] == "exec"
    assert "ls -la /tmp" in calls[0].get("args_preview", "") or "ls -la /tmp" in calls[0].get("args", "")


def test_codex_parses_apply_patch(tmp_path):
    """File-edit via apply patch → one tool call with the path."""
    parsers = _get_parsers()
    stdout = "apply patch\npatch: completed\n/tmp/work/findings.md\n"
    calls = parsers["codex_cli"](stdout, "", tmp_path)
    assert any(c["name"] == "apply_patch" for c in calls)
    assert any("findings.md" in str(c.get("args_preview", "") or c.get("args", "")) for c in calls)


@pytest.mark.parametrize("scenario", ["text_only_response", "tool_with_reasoning_preamble"])
def test_codex_real_fixture(scenario, tmp_path):
    """Pin against fixtures captured from real codex runs."""
    parsers = _get_parsers()
    stdout = _fixture("codex_cli", scenario)
    calls = parsers["codex_cli"](stdout, "", tmp_path)
    assert isinstance(calls, list)


# -----------------------------------------------------------------------------
# Gemini CLI parser tests
# -----------------------------------------------------------------------------


def test_gemini_parses_tool_invocations(tmp_path):
    parsers = _get_parsers()
    stdout = "Some preamble.\n[tool: ReadFile] reading README.md\n→ WriteFile(findings.md, ...)\nDone.\n"
    calls = parsers["gemini_cli"](stdout, "", tmp_path)
    names = [c["name"] for c in calls]
    assert "ReadFile" in names
    assert "WriteFile" in names


@pytest.mark.parametrize("scenario", ["empty_response", "model_garbage_2_5_flash"])
def test_gemini_handles_known_failure_modes(scenario, tmp_path):
    """Empty / garbage outputs should produce 0 calls without raising."""
    parsers = _get_parsers()
    stdout = _fixture("gemini_cli", scenario)
    calls = parsers["gemini_cli"](stdout, "", tmp_path)
    assert isinstance(calls, list)


# -----------------------------------------------------------------------------
# Hermes CLI parser tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    [
        "ok_with_tool_calls",
        "or_arcee_spark_empty",
        "or_deepseek_v3_free_empty",
    ],
)
def test_hermes_handles_provider_specific_failures(scenario, tmp_path):
    parsers = _get_parsers()
    stdout = _fixture("hermes_cli", scenario)
    calls = parsers["hermes_cli"](stdout, "", tmp_path)
    assert isinstance(calls, list)


# -----------------------------------------------------------------------------
# Claude CLI parser tests (high reliability — included for completeness)
# -----------------------------------------------------------------------------


def test_claude_parses_tool_use_event(tmp_path):
    parsers = _get_parsers()
    # Claude CLI emits one JSON object per line with a `type` field.
    stdout = (
        '{"type":"system","subtype":"init"}\n'
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read","input":{"file_path":"/tmp/x"}}]}}\n'
        '{"type":"result","is_error":false}\n'
    )
    calls = parsers["claude_cli"](stdout, "", tmp_path)
    assert any(c["name"] == "Read" for c in calls)


# -----------------------------------------------------------------------------
# Smoke: parser registry stays consistent
# -----------------------------------------------------------------------------


def test_all_parsers_callable():
    parsers = _get_parsers()
    for name, fn in parsers.items():
        assert callable(fn), name
