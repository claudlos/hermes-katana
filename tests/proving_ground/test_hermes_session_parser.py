"""Tests for fix #1 (_hermes_tool_calls_from_session) and G3 provenance."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_katana.proving_ground.sandbox.agent_cli_runner import (
    _hermes_tool_calls_from_session,
    extract_hermes_provenance,
)


def _real_session_id_or_skip() -> str:
    """Pick the most recent .jsonl session file, return its id (no extension).
    Skip the test if the local hermes session dir is empty — these tests are
    integration-flavored and assume the developer has run hermes at least once."""
    sessions = sorted(Path.home().glob(".hermes/sessions/*.jsonl"))
    if not sessions:
        pytest.skip("no real hermes session files; run hermes-agent at least once")
    return sessions[-1].stem


def test_tool_calls_returns_none_for_unknown_session():
    """Sentinel: nonexistent ids must return None, not raise. Caller relies on
    the None signal to fall back to the regex parser."""
    out = _hermes_tool_calls_from_session("definitely_not_a_real_session_id_2026")
    assert out is None


def test_tool_calls_jsonl_format_is_read():
    """Fix #1: parser used to look for session_<id>.json which never exists
    on disk. Real format is <id>.jsonl. Reading a real session must return
    a list (possibly empty), never None."""
    sid = _real_session_id_or_skip()
    out = _hermes_tool_calls_from_session(sid)
    assert out is not None, "fix #1 regression — parser fell through to None"
    assert isinstance(out, list)
    # If the session has any assistant turns at all, expect at least one
    # tool call. (Hermes is tool-loop-driven; sessions with zero tool calls
    # are rare but possible — assert structurally instead of on count.)
    for c in out:
        assert "name" in c and "source" in c
        assert c["source"] == "hermes_session"


def test_extract_provenance_empty_input_returns_all_none():
    p = extract_hermes_provenance("", "")
    assert p == {"served_model": None, "served_platform": None, "session_id": None}


def test_extract_provenance_recovers_session_meta():
    """G3: with a real session_id present in stderr, provenance should
    surface served_model + served_platform from the session_meta first
    line of the .jsonl session file."""
    sid = _real_session_id_or_skip()
    fake_stderr = f"\nsession_id: {sid}\n"
    p = extract_hermes_provenance("", fake_stderr)
    assert p["session_id"] == sid
    # served_model + served_platform must come from the session_meta
    # record, not be silently null when the file exists.
    assert p["served_model"] is not None, "session_meta record should expose `model`; check session JSONL format"
    assert p["served_platform"] is not None
