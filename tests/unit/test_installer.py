"""Tests for HermesKatana installer detection and lifecycle behavior."""

from __future__ import annotations

import json
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import tomllib  # will fail on 3.10 without tomli
from pathlib import Path

import pytest

from hermes_katana.installer.installer import (
    KATANA_BACKUP_DIR,
    KATANA_CONFIG_DIR,
    KATANA_CONFIG_FILE,
    KatanaInstaller,
)
from hermes_katana.installer.patches import (
    CURRENT_CORE_PATCHES,
    LEGACY_CORE_PATCHES,
    _detect_hermes_layout,
)
from tests.hermes_compat import (
    HERMES_CURRENT_SNAPSHOT,
    HERMES_V010_CORE_SNAPSHOT,
    HERMES_V010_EXTENDED_SNAPSHOT,
    fixture_checkout,
    fixture_checkout_direct,
    supported_fixtures,
)


def _stub_ca_cert(self, target: Path) -> None:
    cert_dir = target / ".katana" / "certs"
    cert_dir.mkdir(parents=True, exist_ok=True)
    (cert_dir / "katana-ca.pem").write_text("stub-cert", encoding="utf-8")
    (cert_dir / "katana-ca-key.pem").write_text("stub-key", encoding="utf-8")


class TestDetectHermes:
    def test_rejects_generic_pyproject_without_patch_targets(self, tmp_dir):
        (tmp_dir / "pyproject.toml").write_text(
            '[project]\nname = "hermes"\n',
            encoding="utf-8",
        )

        assert KatanaInstaller().detect_hermes(tmp_dir) is False

    def test_accepts_patchable_checkout(self, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_CORE_SNAPSHOT, tmp_dir)
        assert KatanaInstaller().detect_hermes(checkout) is True

    def test_accepts_current_layout_checkout(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        assert KatanaInstaller().detect_hermes(checkout) is True


class TestDetectHermesLayout:
    def test_returns_legacy_for_v010_checkout(self, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        assert _detect_hermes_layout(checkout) == "legacy-v0.1.0"

    def test_returns_current_for_current_snapshot(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        assert _detect_hermes_layout(checkout) == "current"

    def test_raises_for_unsupported_layout(self, tmp_dir):
        # Empty directory — no markers at all
        empty = tmp_dir / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="Unsupported Hermes layout"):
            _detect_hermes_layout(empty)

    def test_raises_for_pyproject_only_checkout(self, tmp_dir):
        (tmp_dir / "pyproject.toml").write_text(
            '[project]\nname = "something"\nversion = "1.0"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unsupported Hermes layout"):
            _detect_hermes_layout(tmp_dir)


class TestInstallerLayoutAwareness:
    def test_current_layout_uses_current_core_patches(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        installer = KatanaInstaller()

        results = installer.install(checkout)

        patch_names = {r.name for r in results}
        expected_names = {p.name for p in CURRENT_CORE_PATCHES}
        assert patch_names == expected_names

    def test_current_layout_dry_run_shows_current_patch_names(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        installer = KatanaInstaller()

        results = installer.install(checkout, dry_run=True)

        planned_or_skipped = {r.name for r in results if r.status.value in ("planned", "skipped")}
        for patch in CURRENT_CORE_PATCHES:
            assert patch.name in planned_or_skipped

    def test_installer_refuses_unsupported_layout_with_clear_error(self, tmp_dir):
        unsupported = tmp_dir / "unsupported"
        unsupported.mkdir()
        (unsupported / "pyproject.toml").write_text(
            '[project]\nname = "something-else"\nversion = "1.0"\n',
            encoding="utf-8",
        )
        installer = KatanaInstaller()
        with pytest.raises(ValueError, match="Not a Hermes checkout") as exc_info:
            installer.install(unsupported)
        # Error message must mention BOTH layouts so users of current Hermes
        # checkouts aren't confused by legacy-only marker names.
        msg = str(exc_info.value)
        assert "current-layout markers" in msg, "install() error must list current-layout markers, not just legacy"
        assert "tools/registry.py" in msg
        assert "hermes_cli" in msg
        assert "legacy v0.1.0 markers" in msg

    def test_patches_for_warns_on_silent_fallback(self, tmp_dir, caplog):
        """_patches_for() should log a warning when layout detection fails
        instead of silently returning current-layout patches."""
        import logging

        unsupported = tmp_dir / "mystery"
        unsupported.mkdir()
        installer = KatanaInstaller()
        with caplog.at_level(logging.WARNING, logger="hermes_katana.installer.installer"):
            patches = installer._patches_for(unsupported)
        assert any("Could not detect Hermes layout" in r.message for r in caplog.records), (
            "silent fallback to 'current' layout must emit a warning"
        )
        # Falls back to current patches (safe default)
        from hermes_katana.installer.patches import CURRENT_CORE_PATCHES

        assert patches is CURRENT_CORE_PATCHES

    def test_current_markers_match_detect_hermes_layout(self):
        """_HERMES_CURRENT_MARKERS must be consistent with the paths checked
        by _detect_hermes_layout() in patches.py. If they disagree,
        detect_hermes() and _detect_hermes_layout() can return contradictory
        results for edge-case checkouts."""
        from hermes_katana.installer.installer import _HERMES_CURRENT_MARKERS

        # Both modules must agree on these anchor paths.
        assert "tools/registry.py" in _HERMES_CURRENT_MARKERS
        assert "hermes_cli" in _HERMES_CURRENT_MARKERS
        # The legacy-style "file inside dir" check must NOT be used — it
        # allowed detect_hermes() to say False while _detect_hermes_layout()
        # said "current" if the __init__.py was missing.
        assert "hermes_cli/__init__.py" not in _HERMES_CURRENT_MARKERS

    def test_current_layout_verify_after_install(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        installer = KatanaInstaller()

        installer.install(checkout)
        verify = installer.verify(checkout)

        assert verify.is_valid is True

    def test_current_layout_uninstall_reverts_patches(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        installer = KatanaInstaller()

        installer.install(checkout)
        results = installer.uninstall(checkout)

        reverted = {r.name for r in results if r.status.value == "reverted"}
        for patch in CURRENT_CORE_PATCHES:
            if patch.critical:
                assert patch.name in reverted


class TestInstallerLifecycle:
    def test_install_dry_run_previews_bootstrap_patch_without_writing(self, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        installer = KatanaInstaller()

        results = installer.install(checkout, dry_run=True)

        planned = {result.name for result in results if result.status.value == "planned"}
        assert "tool_dispatch_hook" in planned
        assert "dispatcher_bootstrap" in planned
        assert not (checkout / ".katana").exists()
        assert "[KATANA-PATCH]" not in (checkout / "hermes" / "tools" / "dispatch.py").read_text(encoding="utf-8")

    def test_install_backup_writes_manifest(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        installer = KatanaInstaller()
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)

        results = installer.install(checkout, backup=True)

        assert any(result.status.value == "applied" for result in results)
        manifest_path = installer.last_backup_manifest_path
        assert manifest_path is not None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["operation"] == "install"
        assert "hermes/tools/dispatch.py" in manifest["files"]

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows symlink creation requires admin or Developer Mode.",
    )
    def test_install_rejects_symlinked_katana_dir(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        outside = tmp_dir / "outside-katana"
        outside.mkdir()
        (checkout / KATANA_CONFIG_DIR).symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="Config directory path is a symlink"):
            KatanaInstaller().install(checkout)

        assert not (outside / KATANA_CONFIG_FILE).exists()

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows symlink creation requires admin or Developer Mode.",
    )
    def test_install_rejects_symlinked_katana_config_file(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        config_dir = checkout / KATANA_CONFIG_DIR
        config_dir.mkdir()
        outside_config = tmp_dir / "outside.yaml"
        outside_config.write_text("outside: true\n", encoding="utf-8")
        (config_dir / KATANA_CONFIG_FILE).symlink_to(outside_config)

        with pytest.raises(ValueError, match="Config file path is a symlink"):
            KatanaInstaller().install(checkout)

        assert outside_config.read_text(encoding="utf-8") == "outside: true\n"

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows symlink creation requires admin or Developer Mode.",
    )
    def test_install_backup_rejects_symlinked_katana_source(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        outside = tmp_dir / "outside-katana"
        outside.mkdir()
        (outside / KATANA_CONFIG_FILE).write_text("outside: true\n", encoding="utf-8")
        (checkout / KATANA_CONFIG_DIR).symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="Backup source path is a symlink"):
            KatanaInstaller().install(checkout, backup=True)

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows symlink creation requires admin or Developer Mode.",
    )
    def test_install_backup_rejects_symlinked_backup_dir(self, tmp_dir):
        checkout = fixture_checkout_direct(HERMES_CURRENT_SNAPSHOT, tmp_dir)
        outside = tmp_dir / "outside-backups"
        outside.mkdir()
        (checkout / KATANA_BACKUP_DIR).symlink_to(outside, target_is_directory=True)

        with pytest.raises(ValueError, match="Backup directory path escapes checkout"):
            KatanaInstaller().install(checkout, backup=True)

        assert not any(outside.iterdir())

    def test_uninstall_backup_survives_removed_checkout_state(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        installer = KatanaInstaller()
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)

        installer.install(checkout)
        results = installer.uninstall(checkout, backup=True)

        assert any(result.status.value == "reverted" for result in results)
        manifest_path = installer.last_backup_manifest_path
        assert manifest_path is not None
        assert manifest_path.exists()
        assert not (checkout / ".katana").exists()

    def test_restore_reverts_install_backup(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        installer = KatanaInstaller()
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)

        installer.install(checkout, backup=True)
        manifest_path = installer.last_backup_manifest_path
        assert manifest_path is not None

        actions = installer.restore(manifest_path)

        assert "Restore hermes/tools/dispatch.py" in actions
        assert not (checkout / ".katana").exists()
        assert not (checkout / ".katana-installed").exists()
        dispatch = (checkout / "hermes" / "tools" / "dispatch.py").read_text(encoding="utf-8")
        assert "[KATANA-PATCH]" not in dispatch

    def test_restore_recreates_uninstall_backup(self, monkeypatch, tmp_dir):
        checkout = fixture_checkout(HERMES_V010_EXTENDED_SNAPSHOT, tmp_dir)
        installer = KatanaInstaller()
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)

        installer.install(checkout)
        installer.uninstall(checkout, backup=True)
        manifest_path = installer.last_backup_manifest_path
        assert manifest_path is not None

        actions = installer.restore(manifest_path)

        assert "Restore .katana" in actions
        assert (checkout / ".katana").exists()
        assert (checkout / ".katana-installed").exists()
        verify = installer.verify(checkout)
        assert verify.is_valid is True

    def test_supported_checkout_snapshots_round_trip(self, monkeypatch, tmp_dir):
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)

        for fixture in supported_fixtures():
            checkout = fixture_checkout(fixture.id, tmp_dir / fixture.profile)
            installer = KatanaInstaller()

            results = installer.install(checkout)
            statuses = {result.name: result.status.value for result in results}

            assert statuses["tool_dispatch_hook"] == "applied"
            assert statuses["dispatcher_bootstrap"] == "applied"
            pyproject = tomllib.loads((checkout / "pyproject.toml").read_text(encoding="utf-8"))
            assert pyproject["project"]["version"] == fixture.hermes_version
            verify = installer.verify(checkout)
            assert verify.is_valid is True

            uninstall_results = installer.uninstall(checkout)
            reverted = {result.name: result.status.value for result in uninstall_results}
            assert reverted["dispatcher_bootstrap"] == "reverted"
            assert reverted["tool_dispatch_hook"] == "reverted"


@pytest.mark.parametrize(
    "fixture_id,layout",
    [
        (HERMES_V010_EXTENDED_SNAPSHOT, "legacy-v0.1.0"),
        (HERMES_CURRENT_SNAPSHOT, "current"),
    ],
)
class TestInstallerBothLayouts:
    """Core install/uninstall round-trip parametrized over both Hermes layouts."""

    def _checkout(self, fixture_id: str, tmp_dir: Path) -> Path:
        if fixture_id == HERMES_CURRENT_SNAPSHOT:
            return fixture_checkout_direct(fixture_id, tmp_dir)
        return fixture_checkout(fixture_id, tmp_dir)

    def test_detect_layout(self, tmp_dir, fixture_id, layout):
        checkout = self._checkout(fixture_id, tmp_dir)
        assert _detect_hermes_layout(checkout) == layout

    def test_install_applies_critical_patches(self, monkeypatch, tmp_dir, fixture_id, layout):
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        checkout = self._checkout(fixture_id, tmp_dir)
        installer = KatanaInstaller()

        results = installer.install(checkout)
        statuses = {r.name: r.status.value for r in results}

        expected_patches = LEGACY_CORE_PATCHES if layout == "legacy-v0.1.0" else CURRENT_CORE_PATCHES
        for patch in expected_patches:
            if patch.critical:
                assert statuses[patch.name] == "applied", (
                    f"{layout}: expected {patch.name} to be applied, got {statuses[patch.name]}"
                )

    def test_install_then_verify_is_valid(self, monkeypatch, tmp_dir, fixture_id, layout):
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        checkout = self._checkout(fixture_id, tmp_dir)
        installer = KatanaInstaller()

        installer.install(checkout)
        verify = installer.verify(checkout)

        assert verify.is_valid is True, f"{layout}: verify failed: {verify.issues}"

    def test_uninstall_reverts_critical_patches(self, monkeypatch, tmp_dir, fixture_id, layout):
        monkeypatch.setattr(KatanaInstaller, "_generate_ca_cert", _stub_ca_cert)
        checkout = self._checkout(fixture_id, tmp_dir)
        installer = KatanaInstaller()

        installer.install(checkout)
        results = installer.uninstall(checkout)
        reverted = {r.name: r.status.value for r in results}

        expected_patches = LEGACY_CORE_PATCHES if layout == "legacy-v0.1.0" else CURRENT_CORE_PATCHES
        for patch in expected_patches:
            if patch.critical:
                assert reverted[patch.name] == "reverted", (
                    f"{layout}: expected {patch.name} to be reverted, got {reverted[patch.name]}"
                )
