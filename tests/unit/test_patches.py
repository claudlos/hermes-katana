"""Tests for the CURRENT_CORE_PATCHES patch definitions against the current-snapshot fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_katana.installer.patches import (
    CURRENT_CORE_PATCHES,
    Patch,
    PatchStatus,
    apply_patches,
    preview_apply_patches,
)

_CURRENT_SNAPSHOT = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "hermes_compat" / "hermes-current-snapshot"
)

_EXPECTED_PATCH_NAMES = [
    "tool_dispatch_hook",
    "dispatcher_bootstrap",
    "dispatcher_escalation_audit",
    "proxy_env_vars",
    "banner_integration",
    "docker_proxy_forwarding",
    "gateway_command_scanning",
]


class TestCurrentCorePatches:
    def test_has_exactly_seven_entries(self):
        assert len(CURRENT_CORE_PATCHES) == 7

    def test_patch_names_match_expected(self):
        names = [p.name for p in CURRENT_CORE_PATCHES]
        assert names == _EXPECTED_PATCH_NAMES

    def test_all_patches_have_sentinels(self):
        for patch in CURRENT_CORE_PATCHES:
            assert patch.sentinel, f"{patch.name}: sentinel is empty"
            assert "KATANA-PATCH" in patch.sentinel, f"{patch.name}: sentinel does not contain KATANA-PATCH marker"

    def test_each_sentinel_is_unique(self):
        sentinels = [p.sentinel for p in CURRENT_CORE_PATCHES]
        assert len(set(sentinels)) == len(sentinels), "Duplicate sentinels found"

    @pytest.mark.parametrize("patch", CURRENT_CORE_PATCHES, ids=[p.name for p in CURRENT_CORE_PATCHES])
    def test_target_file_exists_in_current_snapshot(self, patch):
        target = _CURRENT_SNAPSHOT / patch.target_file
        assert target.exists(), (
            f"Patch '{patch.name}' target '{patch.target_file}' does not exist in hermes-current-snapshot"
        )

    @pytest.mark.parametrize("patch", CURRENT_CORE_PATCHES, ids=[p.name for p in CURRENT_CORE_PATCHES])
    def test_search_text_found_in_current_snapshot(self, patch):
        target = _CURRENT_SNAPSHOT / patch.target_file
        content = target.read_text(encoding="utf-8")
        assert patch.search_text in content, (
            f"Patch '{patch.name}' search_text not found in hermes-current-snapshot/{patch.target_file}"
        )

    @pytest.mark.parametrize("patch", CURRENT_CORE_PATCHES, ids=[p.name for p in CURRENT_CORE_PATCHES])
    def test_sentinel_not_already_present_in_current_snapshot(self, patch):
        target = _CURRENT_SNAPSHOT / patch.target_file
        content = target.read_text(encoding="utf-8")
        assert patch.sentinel not in content, (
            f"Patch '{patch.name}' sentinel already present in "
            f"hermes-current-snapshot/{patch.target_file} (fixture was pre-patched?)"
        )

    def test_critical_patches_count(self):
        critical = [p for p in CURRENT_CORE_PATCHES if p.critical]
        assert len(critical) == 4, f"Expected 4 critical patches, got {len(critical)}: " + ", ".join(
            p.name for p in critical
        )

    def test_optional_patches_count(self):
        optional = [p for p in CURRENT_CORE_PATCHES if not p.critical]
        assert len(optional) == 3, f"Expected 3 optional patches, got {len(optional)}: " + ", ".join(
            p.name for p in optional
        )

    def test_tool_dispatch_hook_blocks_escalation_on_modern_hermes(self):
        """Modern Hermes has no source-patch approval callback at this point.
        ESCALATE must therefore block rather than silently proceed."""
        hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "tool_dispatch_hook")
        assert "_katana_escalate" not in hook.replace_text
        assert "DispatchDecision.ESCALATE" in hook.replace_text
        assert "pending approval" in hook.replace_text

    def test_modern_source_patches_discover_checkout_from_patched_files(self):
        """Hermes is often launched from a project cwd, not the Hermes checkout."""
        dispatch_hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "tool_dispatch_hook")
        bootstrap_hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "dispatcher_bootstrap")

        assert "discover_checkout_root(__file__)" in dispatch_hook.replace_text
        assert "bootstrap_dispatcher_failsafe(self, checkout_root=__file__)" in bootstrap_hook.replace_text

    def test_tool_dispatch_hook_deny_error_is_fstring(self):
        """Regression: the DENY error message must be an f-string so {name} is
        substituted with the actual tool name, not written literally."""
        hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "tool_dispatch_hook")
        # replace_text has already been through .format(), so braces are single here.
        # Check via regex: the opening quote must be preceded by 'f' (f-string),
        # not a bare double-quote (plain string that writes literal {name}).
        import re

        assert not re.search(r'(?<![fF])"Katana blocked tool', hook.replace_text), (
            'DENY error message must use f-string (f"...") so {name} is substituted, '
            "otherwise the tool name is written literally into the target file"
        )
        assert 'f"Katana blocked tool' in hook.replace_text, (
            "DENY error message must be an f-string: f\"Katana blocked tool '{name}'...\""
        )

    def test_dispatcher_escalation_audit_runs_post_dispatch_scan(self):
        """Result scanning now belongs at the model_tools transform boundary."""
        hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "dispatcher_escalation_audit")
        assert hook.target_file == "model_tools.py"
        assert "_katana_chain.execute_post(_katana_ctx)" in hook.replace_text
        assert "result = _katana_ctx.tool_output" in hook.replace_text

    def test_tool_dispatch_hook_resolves_escalation_via_shared_policy(self):
        """ESCALATE must route through the shared resolver so block/acp_prompt/
        auto_approve behave identically on the source-patch and native paths."""
        hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "tool_dispatch_hook")
        assert "from hermes_katana.escalation import resolve_escalation" in hook.replace_text
        assert "resolve_escalation(" in hook.replace_text
        assert "escalate_action" in hook.replace_text


class TestAmbiguousAnchorGuard:
    """A search anchor that matches more than one location must fail loudly,
    not silently patch the first match (critical for a security hook)."""

    def _make_patch(self) -> Patch:
        return Patch(
            name="ambiguous_test",
            description="test patch with a non-unique anchor",
            target_file="mod.py",
            search_text="# ANCHOR\n",
            replace_text="# KATANA\n# ANCHOR\n",
            sentinel="# KATANA",
            critical=True,
        )

    def test_apply_patches_errors_on_duplicate_anchor(self, tmp_path):
        (tmp_path / "mod.py").write_text("# ANCHOR\nx = 1\n# ANCHOR\ny = 2\n", encoding="utf-8")
        results = apply_patches(tmp_path, patches=[self._make_patch()])
        assert results[0].status == PatchStatus.ERROR
        assert "ambiguous" in results[0].message.lower()
        # File must be left untouched (not patched at the first match).
        assert "# KATANA" not in (tmp_path / "mod.py").read_text(encoding="utf-8")

    def test_preview_errors_on_duplicate_anchor(self, tmp_path):
        (tmp_path / "mod.py").write_text("# ANCHOR\nx = 1\n# ANCHOR\ny = 2\n", encoding="utf-8")
        results = preview_apply_patches(tmp_path, patches=[self._make_patch()])
        assert results[0].status == PatchStatus.ERROR
        assert "ambiguous" in results[0].message.lower()

    def test_unique_anchor_still_applies(self, tmp_path):
        (tmp_path / "mod.py").write_text("# ANCHOR\nx = 1\n", encoding="utf-8")
        results = apply_patches(tmp_path, patches=[self._make_patch()])
        assert results[0].status == PatchStatus.APPLIED
        assert "# KATANA" in (tmp_path / "mod.py").read_text(encoding="utf-8")


class TestRobustDispatchAnchor:
    def test_tool_dispatch_hook_anchor_is_multiline_and_unique(self):
        """The dispatch-hook anchor was widened from a single comment line to a
        multi-line snippet so a future Hermes refactor can't accidentally match
        it in two places."""
        hook = next(p for p in CURRENT_CORE_PATCHES if p.name == "tool_dispatch_hook")
        assert hook.search_text.count("\n") >= 3, "anchor should span multiple lines"
        assert hook.search_text.rstrip().endswith("try:")
        # And it must occur exactly once in the real snapshot.
        model_tools = (_CURRENT_SNAPSHOT / "model_tools.py").read_text(encoding="utf-8")
        assert model_tools.count(hook.search_text) == 1
