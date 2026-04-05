"""Shared test fixtures for HermesKatana test suite."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Generator
from uuid import uuid4

import pytest


# ---------------------------------------------------------------------------
# Taint fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user_source():
    """Trusted user source."""
    from hermes_katana.taint.labels import Source

    return Source.user("test_user")


@pytest.fixture
def web_source():
    """Untrusted web source."""
    from hermes_katana.taint.labels import Source

    return Source.web("https://evil.example.com")


@pytest.fixture
def mcp_source():
    """Untrusted MCP source."""
    from hermes_katana.taint.labels import Source

    return Source.mcp("untrusted_mcp_server")


@pytest.fixture
def tool_source():
    """Conditional tool source."""
    from hermes_katana.taint.labels import Source

    return Source.tool("some_tool")


@pytest.fixture
def tracker():
    """Fresh scoped taint tracker."""
    from hermes_katana.taint.tracker import TaintTracker

    t = TaintTracker()
    yield t
    t.clear()


# ---------------------------------------------------------------------------
# Policy fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def balanced_engine():
    """PolicyEngine with balanced preset."""
    from hermes_katana.policy.engine import PolicyEngine

    return PolicyEngine.with_defaults("balanced")


@pytest.fixture
def paranoid_engine():
    """PolicyEngine with paranoid preset."""
    from hermes_katana.policy.engine import PolicyEngine

    return PolicyEngine.with_defaults("paranoid")


@pytest.fixture
def permissive_engine():
    """PolicyEngine with permissive preset."""
    from hermes_katana.policy.engine import PolicyEngine

    return PolicyEngine.with_defaults("permissive")


# ---------------------------------------------------------------------------
# Temp directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    """Writable temporary directory rooted inside the repo."""
    base_dir = Path.cwd() / ".pytest_tmp"
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def audit_path(tmp_dir: Path) -> Path:
    """Path for a temporary audit trail file."""
    return tmp_dir / "test_audit.jsonl"


@pytest.fixture
def vault_path(tmp_dir: Path) -> Path:
    """Path for a temporary vault file."""
    return tmp_dir / "test_vault.json"


# ---------------------------------------------------------------------------
# Taint context helpers
# ---------------------------------------------------------------------------


def make_taint_context(
    fields: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a taint_context dict for policy engine tests."""
    return {"tainted_fields": fields or {}}


def make_tainted_field(
    is_tainted: bool = True,
    source: str = "web_content",
    labels: list[str] | None = None,
    level: int = 5,
) -> dict[str, Any]:
    """Build a single tainted field entry."""
    return {
        "is_tainted": is_tainted,
        "source": source,
        "labels": labels or ["untrusted"],
        "readers": [],
        "level": level,
    }
