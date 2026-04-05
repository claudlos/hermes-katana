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

from hermes_katana.installer.installer import KatanaInstaller
from tests.hermes_compat import (
    HERMES_V010_CORE_SNAPSHOT,
    HERMES_V010_EXTENDED_SNAPSHOT,
    fixture_checkout,
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
