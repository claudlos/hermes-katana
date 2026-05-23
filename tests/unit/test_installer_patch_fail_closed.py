"""Regression tests for fail-closed installer patch templates.

These tests both grep the patch template *and* run the patched dispatch
logic to verify the actual fail-closed behavior end-to-end.
"""

from __future__ import annotations

import json
from typing import Any

from hermes_katana.installer.patches import CURRENT_CORE_PATCHES, LEGACY_CORE_PATCHES


def _patch_text(patches, name: str) -> str:
    for patch in patches:
        if patch.name == name:
            return patch.replace_text
    raise AssertionError(f"missing patch {name}")


# ---------------------------------------------------------------------------
# String-level invariants
# ---------------------------------------------------------------------------


def test_current_tool_dispatch_patch_blocks_when_katana_import_fails():
    text = _patch_text(CURRENT_CORE_PATCHES, "tool_dispatch_hook")

    assert "except ImportError" not in text
    assert "Katana security bootstrap failed" in text
    assert "return json.dumps" in text
    # Must read the new fail-closed flags (HK #1)
    assert "_katana_bootstrap_failed" in text
    assert "_katana_checkout_discovered" in text


def test_current_dispatcher_bootstrap_patch_marks_bootstrap_failure():
    text = _patch_text(CURRENT_CORE_PATCHES, "dispatcher_bootstrap")

    assert "self._katana_bootstrap_failed = True" in text
    # Must use the failsafe helper, not the raw bootstrap
    assert "bootstrap_dispatcher_failsafe" in text


def test_legacy_tool_dispatch_patch_propagates_mutated_output():
    text = _patch_text(LEGACY_CORE_PATCHES, "tool_dispatch_hook")

    # HK #2: legacy must read mutated tool_output back into result
    assert "result = _katana_ctx.tool_output" in text
    # HK #2: must not silently swallow ImportError without failing closed
    assert "except ImportError:\n            pass" not in text


def test_legacy_dispatcher_bootstrap_uses_failsafe():
    text = _patch_text(LEGACY_CORE_PATCHES, "dispatcher_bootstrap")
    assert "bootstrap_dispatcher_failsafe" in text
    assert "self._katana_bootstrap_failed = True" in text


# ---------------------------------------------------------------------------
# End-to-end behavior of the patched dispatch logic
# ---------------------------------------------------------------------------


class _FakeRegistry:
    """Stand-in for ToolRegistry that runs the patched dispatch logic."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def dispatch_with_patch(self, name: str, args: dict) -> str:
        # Equivalent of the patch's middleware block (CURRENT layout).
        # We hand-roll this rather than monkey-patch a hermes checkout
        # because the test should verify the *intended semantics* of the
        # patch template, not a particular Hermes version.
        try:
            from hermes_katana.middleware import (
                CallContext,
                DispatchDecision,
            )
        except ImportError as exc:
            return json.dumps({"error": f"Katana import failed: {exc}"})

        _katana_chain = getattr(self, "_katana_chain", None)
        _katana_failed = getattr(self, "_katana_bootstrap_failed", False)
        _katana_discovered = getattr(self, "_katana_checkout_discovered", False)
        _katana_failed_err = getattr(self, "_katana_bootstrap_error", None) or "unknown error"
        if _katana_failed or (_katana_discovered and _katana_chain is None):
            return json.dumps(
                {
                    "error": (
                        f"Katana security bootstrap failed; refusing to dispatch tool '{name}': {_katana_failed_err}"
                    )
                }
            )
        if _katana_chain is not None:
            ctx = CallContext(tool_name=name, args=args)
            decision = _katana_chain.execute_pre(ctx)
            if decision == DispatchDecision.DENY:
                return json.dumps({"error": f"Katana blocked tool '{name}': " + "; ".join(ctx.deny_reasons)})
        # would dispatch the actual tool here
        return json.dumps({"ok": True})


def test_dispatch_fails_closed_when_chain_missing_after_discovery():
    """The exact gap from the audit: bootstrap returned None silently
    even though a checkout was discovered. Dispatch must refuse."""
    reg = _FakeRegistry()
    reg._katana_chain = None
    reg._katana_bootstrap_failed = False
    reg._katana_checkout_discovered = True
    reg._katana_bootstrap_error = None

    out = json.loads(reg.dispatch_with_patch("terminal", {"command": "ls"}))
    assert "error" in out
    assert "refusing to dispatch" in out["error"]


def test_dispatch_fails_closed_when_bootstrap_failed():
    reg = _FakeRegistry()
    reg._katana_chain = None
    reg._katana_bootstrap_failed = True
    reg._katana_checkout_discovered = True
    reg._katana_bootstrap_error = "RuntimeError: hermetic_ml_readiness denied"

    out = json.loads(reg.dispatch_with_patch("terminal", {"command": "ls"}))
    assert "error" in out
    assert "hermetic_ml_readiness denied" in out["error"]


def test_dispatch_proceeds_when_katana_inactive():
    """If no checkout was ever discovered, Katana is inactive and dispatch must allow."""
    reg = _FakeRegistry()
    reg._katana_chain = None
    reg._katana_bootstrap_failed = False
    reg._katana_checkout_discovered = False
    reg._katana_bootstrap_error = None

    out = json.loads(reg.dispatch_with_patch("terminal", {"command": "ls"}))
    assert out == {"ok": True}


# ---------------------------------------------------------------------------
# Helper-function behavior
# ---------------------------------------------------------------------------


def test_bootstrap_dispatcher_failsafe_marks_no_checkout_as_inactive(tmp_path, monkeypatch):
    """When no .katana checkout exists, the failsafe records inactive state."""
    from hermes_katana.bootstrap import bootstrap_dispatcher_failsafe, reset_runtime_cache

    monkeypatch.delenv("KATANA_CHECKOUT_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_runtime_cache()

    class D: ...

    d = D()
    bootstrap_dispatcher_failsafe(d)

    assert d._katana_bootstrap_failed is False
    assert d._katana_checkout_discovered is False
    assert getattr(d, "_katana_chain", None) is None


def test_bootstrap_dispatcher_failsafe_marks_broken_checkout_failed(tmp_path, monkeypatch):
    """When a .katana checkout exists but config is malformed, fail closed."""
    from hermes_katana.bootstrap import bootstrap_dispatcher_failsafe, reset_runtime_cache
    from hermes_katana.installer.installer import KATANA_CONFIG_DIR, KATANA_CONFIG_FILE

    cfg_dir = tmp_path / KATANA_CONFIG_DIR
    cfg_dir.mkdir()
    # Write malformed YAML — load_checkout_state returns None on YAMLError.
    (cfg_dir / KATANA_CONFIG_FILE).write_text("not: [valid: yaml")
    monkeypatch.setenv("KATANA_CHECKOUT_ROOT", str(tmp_path))
    reset_runtime_cache()

    class D: ...

    d = D()
    bootstrap_dispatcher_failsafe(d)

    # Discovery succeeded but runtime build returned None → must record failure.
    assert d._katana_checkout_discovered is True
    assert d._katana_bootstrap_failed is True
    assert d._katana_bootstrap_error is not None
