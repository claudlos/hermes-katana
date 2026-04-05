"""Tests for hermes_katana.taint.registrar."""

from __future__ import annotations

import pytest

from hermes_katana.taint.labels import TaintLabel, TrustLevel
from hermes_katana.taint.registrar import (
    taint_delegated,
    taint_file_content,
    taint_llm_response,
    taint_mcp_description,
    taint_mcp_result,
    taint_memory,
    taint_tool_output,
    taint_user_input,
    taint_web_content,
)
from hermes_katana.taint.tracker import TaintTracker
from hermes_katana.taint.value import TaintedStr


@pytest.fixture(autouse=True)
def fresh_tracker():
    """Reset the global tracker before each test."""
    tracker = TaintTracker.get_instance()
    tracker.clear()
    yield tracker
    tracker.clear()


class TestTaintUserInput:
    def test_returns_tainted_str(self):
        result = taint_user_input("hello world")
        assert isinstance(result, TaintedStr)

    def test_user_label(self):
        result = taint_user_input("hello")
        assert TaintLabel.USER in result.labels

    def test_trusted(self):
        result = taint_user_input("hello")
        assert result.is_trusted()

    def test_custom_source_id(self):
        result = taint_user_input("hello", source_id="telegram")
        sources = list(result.sources)
        assert any(s.origin == "telegram" for s in sources)

    def test_metadata(self):
        result = taint_user_input("hello", source_id="cli", channel="main")
        sources = list(result.sources)
        assert any(s.metadata.get("channel") == "main" for s in sources)


class TestTaintToolOutput:
    def test_returns_tainted_str(self):
        result = taint_tool_output("file1.txt", "terminal")
        assert isinstance(result, TaintedStr)

    def test_tool_output_label(self):
        result = taint_tool_output("output", "read_file")
        assert TaintLabel.TOOL_OUTPUT in result.labels

    def test_conditional_trust(self):
        result = taint_tool_output("output", "read_file")
        sources = list(result.sources)
        assert all(s.trust_level == TrustLevel.CONDITIONAL for s in sources)

    def test_custom_trust(self):
        result = taint_tool_output("output", "read_file", trust=TrustLevel.TRUSTED)
        sources = list(result.sources)
        assert all(s.trust_level == TrustLevel.TRUSTED for s in sources)

    def test_origin_is_tool_name(self):
        result = taint_tool_output("output", "terminal")
        sources = list(result.sources)
        assert any(s.origin == "terminal" for s in sources)


class TestTaintWebContent:
    def test_untrusted(self):
        result = taint_web_content("page content", "https://evil.com")
        assert not result.is_trusted()

    def test_web_label(self):
        result = taint_web_content("page", "https://evil.com")
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_origin_is_url(self):
        result = taint_web_content("page", "https://example.com")
        sources = list(result.sources)
        assert any(s.origin == "https://example.com" for s in sources)


class TestTaintFileContent:
    def test_file_label(self):
        result = taint_file_content("data", "/etc/passwd")
        assert TaintLabel.FILE_CONTENT in result.labels

    def test_conditional_by_default(self):
        result = taint_file_content("data", "/tmp/safe.txt")
        sources = list(result.sources)
        assert all(s.trust_level == TrustLevel.CONDITIONAL for s in sources)


class TestTaintMcpResult:
    def test_mcp_label(self):
        result = taint_mcp_result("response", "my_server", "my_tool")
        assert TaintLabel.MCP_TOOL_RESULT in result.labels

    def test_untrusted(self):
        result = taint_mcp_result("response", "my_server")
        assert not result.is_trusted()

    def test_origin_includes_server(self):
        result = taint_mcp_result("response", "my_server", "my_tool")
        sources = list(result.sources)
        assert any("my_server" in s.origin for s in sources)


class TestTaintMcpDescription:
    def test_description_label(self):
        result = taint_mcp_description("A helpful tool", "server1", "tool1")
        assert TaintLabel.MCP_TOOL_DESCRIPTION in result.labels

    def test_untrusted(self):
        result = taint_mcp_description("description", "server1")
        assert not result.is_trusted()


class TestTaintLlmResponse:
    def test_agent_label(self):
        result = taint_llm_response("I can help with that", "claude-3")
        assert TaintLabel.AGENT in result.labels

    def test_conditional_trust(self):
        result = taint_llm_response("response", "gpt-4")
        sources = list(result.sources)
        assert all(s.trust_level == TrustLevel.CONDITIONAL for s in sources)


class TestTaintMemory:
    def test_memory_label(self):
        result = taint_memory("stored value", "user_prefs")
        assert TaintLabel.MEMORY in result.labels


class TestTaintDelegated:
    def test_delegated_label(self):
        result = taint_delegated("sub-agent output", "task-123")
        assert TaintLabel.AGENT_DELEGATED in result.labels

    def test_conditional_trust(self):
        result = taint_delegated("output")
        sources = list(result.sources)
        assert all(s.trust_level == TrustLevel.CONDITIONAL for s in sources)


class TestRegistrationTracking:
    """Verify that registrar functions actually register with the tracker."""

    def test_user_input_registered(self, fresh_tracker):
        taint_user_input("test")
        stats = fresh_tracker.stats
        assert stats.values_registered >= 1

    def test_multiple_registrations(self, fresh_tracker):
        taint_user_input("a")
        taint_web_content("b", "http://x.com")
        taint_tool_output("c", "terminal")
        stats = fresh_tracker.stats
        assert stats.values_registered >= 3
