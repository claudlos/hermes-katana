"""Tests for compatibility snapshot maintenance tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_katana.installer.compat_snapshots import (
    REGISTRY_FILENAME,
    build_snapshot_record,
    compute_file_sha256,
    compute_tree_sha256,
    infer_hermes_version,
    load_snapshot_registry,
    main,
    refresh_snapshot_matrix,
    snapshot_id,
    snapshot_paths_for_profile,
    verify_source_provenance,
)
from hermes_katana.installer.patches import CORE_PATCHES


def _write_source_checkout(root: Path, version: str = "1.2.3") -> Path:
    source = root / "source"
    (source / "hermes" / "tools").mkdir(parents=True, exist_ok=True)
    (source / "hermes" / "ui").mkdir(parents=True, exist_ok=True)
    (source / "hermes" / "gateway").mkdir(parents=True, exist_ok=True)
    (source / "pyproject.toml").write_text(
        f'[project]\nname = "hermes"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    (source / "hermes" / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")

    for patch in CORE_PATCHES:
        target = source / patch.target_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {patch.name}\n", encoding="utf-8")

    return source


def _write_source_archive(root: Path, version: str = "1.2.3") -> Path:
    archive = root / f"hermes-{version}.tar.gz"
    archive.write_bytes(f"release:{version}".encode("utf-8"))
    return archive


class TestCompatSnapshots:
    def test_snapshot_paths_for_core_and_extended_profiles(self):
        core_paths = set(snapshot_paths_for_profile("core"))
        extended_paths = set(snapshot_paths_for_profile("extended"))
        optional_targets = {patch.target_file for patch in CORE_PATCHES if not patch.critical}

        assert "pyproject.toml" in core_paths
        assert "hermes/__init__.py" in core_paths
        assert optional_targets.isdisjoint(core_paths)
        assert optional_targets.issubset(extended_paths)

    def test_refresh_snapshot_matrix_writes_profiles_and_registry(self, tmp_dir):
        source = _write_source_checkout(tmp_dir)
        fixtures_root = tmp_dir / "fixtures"
        tree_sha256 = compute_tree_sha256(source)

        records = refresh_snapshot_matrix(
            source,
            fixtures_root=fixtures_root,
            source_ref="v1.2.3",
            source_tree_sha256=tree_sha256,
        )

        assert [record.id for record in records] == [
            "hermes-v1.2.3-core-snapshot",
            "hermes-v1.2.3-extended-snapshot",
        ]
        assert (fixtures_root / "hermes-v1.2.3-core-snapshot" / "hermes" / "tools" / "dispatch.py").exists()
        assert not (fixtures_root / "hermes-v1.2.3-core-snapshot" / "hermes" / "ui" / "banner.py").exists()
        assert (fixtures_root / "hermes-v1.2.3-extended-snapshot" / "hermes" / "ui" / "banner.py").exists()

        registry = json.loads((fixtures_root / REGISTRY_FILENAME).read_text(encoding="utf-8"))
        assert registry["schema_version"] == 2
        assert registry["fixtures"][0]["source_ref"] == "v1.2.3"
        assert registry["fixtures"][0]["provenance"]["verification_mode"] == "tree_sha256"
        assert registry["fixtures"][0]["provenance"]["source_tree_sha256"] == tree_sha256

    def test_refresh_snapshot_matrix_replaces_existing_snapshot_when_requested(self, tmp_dir):
        source = _write_source_checkout(tmp_dir)
        fixtures_root = tmp_dir / "fixtures"
        tree_sha256 = compute_tree_sha256(source)

        refresh_snapshot_matrix(
            source,
            fixtures_root=fixtures_root,
            source_tree_sha256=tree_sha256,
        )
        banner = source / "hermes" / "ui" / "banner.py"
        banner.write_text("# refreshed\n", encoding="utf-8")
        updated_tree_sha256 = compute_tree_sha256(source)

        refresh_snapshot_matrix(
            source,
            fixtures_root=fixtures_root,
            source_tree_sha256=updated_tree_sha256,
            replace_existing=True,
        )

        copied = (
            fixtures_root
            / "hermes-v1.2.3-extended-snapshot"
            / "hermes"
            / "ui"
            / "banner.py"
        ).read_text(encoding="utf-8")
        assert copied == "# refreshed\n"

    def test_refresh_snapshot_matrix_main_supports_dry_run(self, tmp_dir, capsys):
        source = _write_source_checkout(tmp_dir, version="2.0.0")
        fixtures_root = tmp_dir / "fixtures"

        exit_code = main(
            [
                "--source",
                str(source),
                "--fixtures-root",
                str(fixtures_root),
                "--source-ref",
                "v2.0.0",
                "--dry-run",
            ]
        )

        output = capsys.readouterr().out
        assert exit_code == 0
        assert "Would refresh Hermes compatibility snapshots:" in output
        assert "hermes-v2.0.0-core-snapshot" in output
        assert "Source tree sha256:" in output
        assert "Source provenance not verified in preview." in output
        assert not (fixtures_root / REGISTRY_FILENAME).exists()

    def test_infer_hermes_version_and_registry_loader(self, tmp_dir):
        source = _write_source_checkout(tmp_dir, version="3.4.5")
        fixtures_root = tmp_dir / "fixtures"
        tree_sha256 = compute_tree_sha256(source)
        refresh_snapshot_matrix(
            source,
            fixtures_root=fixtures_root,
            source_tree_sha256=tree_sha256,
        )

        loaded = load_snapshot_registry(fixtures_root)

        assert infer_hermes_version(source) == "3.4.5"
        assert loaded[0] == build_snapshot_record(
            "3.4.5",
            "core",
            provenance={
                "verification_mode": "tree_sha256",
                "source_tree_sha256": tree_sha256,
            },
        )
        assert loaded[1].id == snapshot_id("3.4.5", "extended")

    def test_refresh_snapshot_matrix_requires_verified_provenance_for_writes(self, tmp_dir):
        source = _write_source_checkout(tmp_dir, version="4.0.0")

        with pytest.raises(ValueError, match="requires verified provenance"):
            refresh_snapshot_matrix(source, fixtures_root=tmp_dir / "fixtures")

    def test_verify_source_provenance_accepts_archive_checksum(self, tmp_dir):
        source = _write_source_checkout(tmp_dir, version="5.0.0")
        archive = _write_source_archive(tmp_dir, version="5.0.0")

        provenance = verify_source_provenance(
            source,
            source_archive=archive,
            archive_sha256=compute_file_sha256(archive),
        )

        assert provenance.verification_mode == "archive_sha256"
        assert provenance.source_archive == archive.name
        assert provenance.source_archive_sha256 == compute_file_sha256(archive)
        assert provenance.source_tree_sha256 == compute_tree_sha256(source)

    def test_verify_source_provenance_rejects_checksum_mismatch(self, tmp_dir):
        source = _write_source_checkout(tmp_dir, version="6.0.0")
        archive = _write_source_archive(tmp_dir, version="6.0.0")

        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_source_provenance(
                source,
                source_archive=archive,
                archive_sha256="0" * 64,
            )

    def test_refresh_snapshot_matrix_main_errors_without_verification(self, tmp_dir, capsys):
        source = _write_source_checkout(tmp_dir, version="7.0.0")

        with pytest.raises(SystemExit) as excinfo:
            main(["--source", str(source), "--fixtures-root", str(tmp_dir / "fixtures")])

        stderr = capsys.readouterr().err
        assert excinfo.value.code == 2
        assert "requires verified provenance" in stderr

    def test_repo_fixture_registry_entries_have_matching_tree_provenance(self):
        repo_fixtures_root = Path(__file__).resolve().parents[1] / "fixtures" / "hermes_compat"
        records = load_snapshot_registry(repo_fixtures_root)

        assert records
        for record in records:
            assert record.provenance["verification_mode"] == "tree_sha256"
            assert record.provenance["source_tree_sha256"] == compute_tree_sha256(
                repo_fixtures_root / record.directory
            )
