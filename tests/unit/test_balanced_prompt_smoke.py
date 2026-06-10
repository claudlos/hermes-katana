"""Smoke tests for benign prompt/tool cases that should not trip balanced mode."""

from __future__ import annotations

import pytest

from hermes_katana.policy.engine import PolicyEngine
from hermes_katana.policy.models import PolicyResult
from hermes_katana.scanner import ScanVerdict, scan_command, scan_input


BENIGN_INPUTS = [
    ("meeting", "Please schedule the meeting for Thursday at 10am and include the agenda."),
    ("docs", "Summarize the README and explain the setup steps."),
    ("training", "Write a short paragraph explaining safe tool-use boundaries in security training."),
    ("quoted_phrase", "In the fixture, the quoted phrase is example text rather than an action request."),
    ("policy_debug", "Check whether skill_view is allowed by the balanced policy and report the matched rule name."),
]

BENIGN_COMMANDS = [
    ("pwd", "pwd"),
    ("python_version", "python --version"),
    ("pytest_target", "pytest -q tests/test_policy_fixes.py"),
    ("git_status", "git status --short"),
]

KNOWN_CLEAN_TOOLS = [
    ("skill_view", {"name": "hermes-agent"}),
    ("skills_list", {}),
    ("write_file", {"path": "notes.txt", "content": "safe clean content"}),
    ("patch", {"path": "notes.txt", "old_string": "a", "new_string": "b"}),
    ("execute_code", {"code": "print('hello')"}),
    ("todo", {"todos": []}),
    ("process", {"action": "poll", "session_id": "abc"}),
    ("session_search", {"query": "katana"}),
    ("web_search", {"query": "Hermes Agent docs"}),
    ("web_extract", {"urls": ["https://example.com"]}),
]


@pytest.mark.parametrize(("label", "text"), BENIGN_INPUTS, ids=[case[0] for case in BENIGN_INPUTS])
def test_benign_prompt_smoke_does_not_block(label: str, text: str):
    result = scan_input(text, security_level="high")

    assert result.verdict != ScanVerdict.BLOCK, (label, result.summary)


@pytest.mark.parametrize(("label", "command"), BENIGN_COMMANDS, ids=[case[0] for case in BENIGN_COMMANDS])
def test_benign_command_smoke_does_not_block(label: str, command: str):
    result = scan_command(command)

    assert result.verdict != ScanVerdict.BLOCK, (label, result.summary)


@pytest.mark.parametrize(("tool_name", "args"), KNOWN_CLEAN_TOOLS, ids=[case[0] for case in KNOWN_CLEAN_TOOLS])
def test_balanced_known_clean_hermes_tools_allow(tool_name: str, args: dict):
    engine = PolicyEngine.with_defaults("balanced")
    result = engine.evaluate(tool_name, args, {})

    assert result.action == PolicyResult.ALLOW, (tool_name, result)
    assert result.matched_policy is not None
    assert result.matched_policy.name != "balanced_catchall_clean"


def test_balanced_unknown_clean_tool_still_escalates():
    engine = PolicyEngine.with_defaults("balanced")
    result = engine.evaluate("brand_new_tool", {"text": "hello"}, {})

    assert result.action == PolicyResult.ESCALATE
    assert result.matched_policy is not None
    assert result.matched_policy.name == "balanced_catchall_clean"
