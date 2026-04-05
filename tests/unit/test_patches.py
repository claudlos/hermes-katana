"""Tests for the CURRENT_CORE_PATCHES patch definitions against the current-snapshot fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_katana.installer.patches import CURRENT_CORE_PATCHES

_CURRENT_SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "hermes_compat"
    / "hermes-current-snapshot"
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
            assert "KATANA-PATCH" in patch.sentinel, (
                f"{patch.name}: sentinel does not contain KATANA-PATCH marker"
            )

    def test_each_sentinel_is_unique(self):
        sentinels = [p.sentinel for p in CURRENT_CORE_PATCHES]
        assert len(set(sentinels)) == len(sentinels), "Duplicate sentinels found"

    @pytest.mark.parametrize("patch", CURRENT_CORE_PATCHES, ids=[p.name for p in CURRENT_CORE_PATCHES])
    def test_target_file_exists_in_current_snapshot(self, patch):
        target = _CURRENT_SNAPSHOT / patch.target_file
        assert target.exists(), (
            f"Patch '{patch.name}' target '{patch.target_file}' "
            f"does not exist in hermes-current-snapshot"
        )

    @pytest.mark.parametrize("patch", CURRENT_CORE_PATCHES, ids=[p.name for p in CURRENT_CORE_PATCHES])
    def test_search_text_found_in_current_snapshot(self, patch):
        target = _CURRENT_SNAPSHOT / patch.target_file
        content = target.read_text(encoding="utf-8")
        assert patch.search_text in content, (
            f"Patch '{patch.name}' search_text not found in "
            f"hermes-current-snapshot/{patch.target_file}"
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
        assert len(critical) == 4, (
            f"Expected 4 critical patches, got {len(critical)}: "
            + ", ".join(p.name for p in critical)
        )

    def test_optional_patches_count(self):
        optional = [p for p in CURRENT_CORE_PATCHES if not p.critical]
        assert len(optional) == 3, (
            f"Expected 3 optional patches, got {len(optional)}: "
            + ", ".join(p.name for p in optional)
        )
