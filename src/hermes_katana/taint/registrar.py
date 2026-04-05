"""Convenience functions for tainting data at agent entry points.

Each function wraps raw text in a :class:`TaintedStr` with the appropriate
:class:`Source` and registers it with the global :class:`TaintTracker`.
These are the primary API for Hermes plugin hooks that need to tag incoming
data with provenance metadata.

Usage::

    from hermes_katana.taint.registrar import taint_user_input, taint_tool_output

    tagged_input = taint_user_input("ls -la", source_id="cli")
    tagged_result = taint_tool_output("file1.txt\nfile2.txt", tool_name="terminal")
"""

from __future__ import annotations

import logging

from hermes_katana.taint.labels import Source, TrustLevel
from hermes_katana.taint.tracker import TaintTracker
from hermes_katana.taint.value import TaintedStr

logger = logging.getLogger(__name__)

__all__ = [
    "taint_user_input",
    "taint_tool_output",
    "taint_web_content",
    "taint_file_content",
    "taint_mcp_result",
    "taint_mcp_description",
    "taint_llm_response",
    "taint_memory",
    "taint_delegated",
]


def _get_tracker() -> TaintTracker:
    """Return the global tracker singleton."""
    return TaintTracker.get_instance()


def taint_user_input(
    text: str,
    source_id: str = "user_input",
    **metadata: str,
) -> TaintedStr:
    """Tag text as trusted user input.

    Args:
        text: Raw user input string.
        source_id: Identifier for the input channel (``cli``, ``telegram``, etc.).
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with USER label and TRUSTED trust level.
    """
    source = Source.user(origin=source_id, **metadata)
    tracker = _get_tracker()
    return tracker.register(text, source)


def taint_tool_output(
    result: str,
    tool_name: str,
    trust: TrustLevel = TrustLevel.CONDITIONAL,
    **metadata: str,
) -> TaintedStr:
    """Tag text as tool output.

    Args:
        result: The tool's return value as a string.
        tool_name: Name of the tool that produced this output.
        trust: Trust level (default CONDITIONAL).
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with TOOL_OUTPUT label.
    """
    source = Source.tool(tool_name=tool_name, trust=trust, **metadata)
    tracker = _get_tracker()
    return tracker.register(result, source)


def taint_web_content(
    content: str,
    url: str,
    **metadata: str,
) -> TaintedStr:
    """Tag text as untrusted web content.

    Args:
        content: Text fetched from the web.
        url: The source URL.
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with WEB_CONTENT label and UNTRUSTED trust level.
    """
    source = Source.web(url=url, **metadata)
    tracker = _get_tracker()
    return tracker.register(content, source)


def taint_file_content(
    content: str,
    path: str,
    trust: TrustLevel = TrustLevel.CONDITIONAL,
    **metadata: str,
) -> TaintedStr:
    """Tag text as file content.

    Args:
        content: Text read from a file.
        path: The file path.
        trust: Trust level (default CONDITIONAL).
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with FILE_CONTENT label.
    """
    source = Source.file(path=path, trust=trust, **metadata)
    tracker = _get_tracker()
    return tracker.register(content, source)


def taint_mcp_result(
    result: str,
    server_name: str,
    tool_name: str = "",
    **metadata: str,
) -> TaintedStr:
    """Tag text as an MCP tool result.

    Args:
        result: The MCP tool's return value.
        server_name: Name of the MCP server.
        tool_name: Name of the MCP tool (optional).
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with MCP_TOOL_RESULT label and UNTRUSTED trust level.
    """
    source = Source.mcp_tool_result(server=server_name, tool_name=tool_name, **metadata)
    tracker = _get_tracker()
    return tracker.register(result, source)


def taint_mcp_description(
    description: str,
    server_name: str,
    tool_name: str = "",
    **metadata: str,
) -> TaintedStr:
    """Tag text as an MCP tool description (highest-risk MCP label).

    Args:
        description: The tool description text.
        server_name: Name of the MCP server.
        tool_name: Name of the MCP tool.
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with MCP_TOOL_DESCRIPTION label and UNTRUSTED trust level.
    """
    source = Source.mcp_tool_description(server=server_name, tool_name=tool_name, **metadata)
    tracker = _get_tracker()
    return tracker.register(description, source)


def taint_llm_response(
    content: str,
    model: str = "unknown",
    **metadata: str,
) -> TaintedStr:
    """Tag text as LLM-generated content.

    Args:
        content: The LLM response text.
        model: The model identifier.
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with AGENT label and CONDITIONAL trust level.
    """
    source = Source.agent(model=model, **metadata)
    tracker = _get_tracker()
    return tracker.register(content, source)


def taint_memory(
    content: str,
    key: str,
    trust: TrustLevel = TrustLevel.CONDITIONAL,
    **metadata: str,
) -> TaintedStr:
    """Tag text as data from persistent memory.

    Args:
        content: The memory content.
        key: Memory key or identifier.
        trust: Trust level (default CONDITIONAL).
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with MEMORY label.
    """
    source = Source.memory(key=key, trust=trust, **metadata)
    tracker = _get_tracker()
    return tracker.register(content, source)


def taint_delegated(
    content: str,
    task_id: str = "",
    **metadata: str,
) -> TaintedStr:
    """Tag text as output from a delegated sub-agent.

    Args:
        content: The sub-agent's output.
        task_id: The delegation task ID.
        **metadata: Extra key-value pairs for the source record.

    Returns:
        A TaintedStr with AGENT_DELEGATED label and CONDITIONAL trust.
    """
    source = Source.agent_delegated(task_id=task_id, **metadata)
    tracker = _get_tracker()
    return tracker.register(content, source)
