"""Tools for maintaining pinned Hermes compatibility snapshots."""

from __future__ import annotations

__all__ = [
    "CompatSnapshotRecord",
    "SourceProvenance",
    "snapshot_paths_for_profile",
    "infer_hermes_version",
    "snapshot_id",
    "load_snapshot_registry",
    "build_snapshot_record",
    "compute_file_sha256",
    "compute_tree_sha256",
    "verify_source_provenance",
    "refresh_snapshot_matrix",
    "main",
]


import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from hermes_katana.installer.installer import HERMES_MARKERS
from hermes_katana.installer.patches import CORE_PATCHES

REGISTRY_FILENAME = "fixtures.json"
SUPPORTED_PROFILES = ("core", "extended")
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURES_ROOT = REPO_ROOT / "tests" / "fixtures" / "hermes_compat"
PROFILE_ORDER = {name: index for index, name in enumerate(SUPPORTED_PROFILES)}
_VERSION_PATTERN = re.compile(r"^\s*version\s*=\s*['\"]([^'\"]+)['\"]\s*$", re.MULTILINE)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_IGNORED_TREE_DIRS = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class CompatSnapshotRecord:
    """Registry metadata for one supported Hermes snapshot."""

    id: str
    directory: str
    hermes_version: str
    profile: str
    support_tier: str
    source: str
    description: str
    expected: dict[str, str]
    source_ref: Optional[str] = None
    provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Serialize to a dictionary."""
        payload: dict[str, object] = {
            "id": self.id,
            "directory": self.directory,
            "hermes_version": self.hermes_version,
            "profile": self.profile,
            "support_tier": self.support_tier,
            "source": self.source,
            "description": self.description,
            "expected": dict(self.expected),
        }
        if self.source_ref:
            payload["source_ref"] = self.source_ref
        if self.provenance:
            payload["provenance"] = dict(self.provenance)
        return payload


@dataclass(frozen=True)
class SourceProvenance:
    """Verified provenance metadata for a Hermes source tree."""

    verification_mode: str
    source_tree_sha256: str
    source_archive: Optional[str] = None
    source_archive_sha256: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        """Serialize to a dictionary."""
        payload = {
            "verification_mode": self.verification_mode,
            "source_tree_sha256": self.source_tree_sha256,
        }
        if self.source_archive:
            payload["source_archive"] = self.source_archive
        if self.source_archive_sha256:
            payload["source_archive_sha256"] = self.source_archive_sha256
        return payload


def snapshot_paths_for_profile(profile: str) -> tuple[str, ...]:
    """Return the file set required for a snapshot profile."""
    selected = {"pyproject.toml", *HERMES_MARKERS}
    include_optional = profile == "extended"
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported snapshot profile: {profile}")

    for patch in CORE_PATCHES:
        if patch.critical or include_optional:
            selected.add(patch.target_file)

    return tuple(sorted(selected))


def infer_hermes_version(source_root: str | Path) -> str:
    """Infer the Hermes version from a checkout-local pyproject."""
    pyproject = Path(source_root).resolve() / "pyproject.toml"
    if not pyproject.exists():
        raise FileNotFoundError(f"Hermes source is missing pyproject.toml: {pyproject}")

    content = pyproject.read_text(encoding="utf-8")
    match = _VERSION_PATTERN.search(content)
    if not match:
        raise ValueError(f"Could not infer Hermes version from: {pyproject}")
    return match.group(1)


def snapshot_id(version: str, profile: str) -> str:
    """Build a canonical snapshot id from version and profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported snapshot profile: {profile}")
    return f"hermes-v{version}-{profile}-snapshot"


def load_snapshot_registry(fixtures_root: str | Path = DEFAULT_FIXTURES_ROOT) -> tuple[CompatSnapshotRecord, ...]:
    """Load the supported snapshot registry from disk."""
    registry_path = Path(fixtures_root).resolve() / REGISTRY_FILENAME
    if not registry_path.exists():
        return ()

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    records = []
    for raw in payload.get("fixtures", []):
        records.append(
            CompatSnapshotRecord(
                id=str(raw["id"]),
                directory=str(raw["directory"]),
                hermes_version=str(raw["hermes_version"]),
                profile=str(raw["profile"]),
                support_tier=str(raw["support_tier"]),
                source=str(raw["source"]),
                description=str(raw["description"]),
                expected=dict(raw["expected"]),
                source_ref=str(raw["source_ref"]) if raw.get("source_ref") else None,
                provenance=dict(raw.get("provenance", {})),
            )
        )
    return tuple(records)


def build_snapshot_record(
    version: str,
    profile: str,
    *,
    source_ref: str | None = None,
    provenance: dict[str, str] | None = None,
) -> CompatSnapshotRecord:
    """Create registry metadata for one supported snapshot."""
    snapshot_name = snapshot_id(version, profile)
    if profile == "core":
        description = (
            f"Pinned Hermes {version} core checkout layout with only required patch targets."
        )
        expected = {
            "critical_patches": "applied",
            "optional_patches": "skipped_cleanly",
        }
    else:
        description = (
            f"Pinned Hermes {version} extended checkout layout with optional UI, Docker, "
            "and gateway patch targets."
        )
        expected = {"all_patches": "applied"}

    return CompatSnapshotRecord(
        id=snapshot_name,
        directory=snapshot_name,
        hermes_version=version,
        profile=profile,
        support_tier="supported",
        source="generated_release_snapshot",
        description=description,
        expected=expected,
        source_ref=source_ref,
        provenance=dict(provenance or {}),
    )


def compute_file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    file_path = Path(path).resolve()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_tree_sha256(source_root: str | Path) -> str:
    """Compute a deterministic SHA-256 digest for a source tree."""
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Hermes source checkout does not exist: {root}")

    digest = hashlib.sha256()
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root)
        if any(part in _IGNORED_TREE_DIRS for part in relative.parts):
            continue
        if not candidate.is_file():
            continue

        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")

    return digest.hexdigest()


def verify_source_provenance(
    source_root: str | Path,
    *,
    source_archive: str | Path | None = None,
    archive_sha256: str | None = None,
    source_tree_sha256: str | None = None,
    require_verified: bool = True,
) -> SourceProvenance:
    """Verify and describe the provenance of a Hermes source checkout."""
    if source_archive is not None and archive_sha256 is None:
        raise ValueError("Archive provenance requires --archive-sha256.")
    if archive_sha256 is not None and source_archive is None:
        raise ValueError("Archive checksum verification requires --source-archive.")

    actual_tree_sha256 = compute_tree_sha256(source_root)
    verification_tags: list[str] = []
    archive_name: Optional[str] = None
    normalized_archive_sha256: Optional[str] = None

    if source_tree_sha256 is not None:
        expected_tree_sha256 = _normalize_sha256(source_tree_sha256, label="source tree checksum")
        if expected_tree_sha256 != actual_tree_sha256:
            raise ValueError(
                "Source tree checksum mismatch: "
                f"expected {expected_tree_sha256}, got {actual_tree_sha256}"
            )
        verification_tags.append("tree_sha256")

    if source_archive is not None:
        normalized_archive_sha256 = _normalize_sha256(archive_sha256, label="archive checksum")
        archive_path = Path(source_archive).resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Source archive does not exist: {archive_path}")
        actual_archive_sha256 = compute_file_sha256(archive_path)
        if normalized_archive_sha256 != actual_archive_sha256:
            raise ValueError(
                "Source archive checksum mismatch: "
                f"expected {normalized_archive_sha256}, got {actual_archive_sha256}"
            )
        archive_name = archive_path.name
        verification_tags.append("archive_sha256")

    if not verification_tags:
        if require_verified:
            raise ValueError(
                "Snapshot refresh requires verified provenance. Provide --source-archive "
                "with --archive-sha256, or provide --source-tree-sha256."
            )
        return SourceProvenance(
            verification_mode="unverified_preview",
            source_tree_sha256=actual_tree_sha256,
        )

    verification_mode = "+".join(sorted(verification_tags))
    return SourceProvenance(
        verification_mode=verification_mode,
        source_tree_sha256=actual_tree_sha256,
        source_archive=archive_name,
        source_archive_sha256=normalized_archive_sha256,
    )


def refresh_snapshot_matrix(
    source_root: str | Path,
    *,
    fixtures_root: str | Path = DEFAULT_FIXTURES_ROOT,
    profiles: Iterable[str] = SUPPORTED_PROFILES,
    version: str | None = None,
    source_ref: str | None = None,
    source_archive: str | Path | None = None,
    archive_sha256: str | None = None,
    source_tree_sha256: str | None = None,
    replace_existing: bool = False,
    dry_run: bool = False,
) -> tuple[CompatSnapshotRecord, ...]:
    """Refresh the supported snapshot matrix from a real Hermes checkout."""
    source_root = Path(source_root).resolve()
    fixtures_root = Path(fixtures_root).resolve()
    version = version or infer_hermes_version(source_root)
    selected_profiles = tuple(dict.fromkeys(profiles))
    provenance = verify_source_provenance(
        source_root,
        source_archive=source_archive,
        archive_sha256=archive_sha256,
        source_tree_sha256=source_tree_sha256,
        require_verified=not dry_run,
    )

    if not source_root.is_dir():
        raise FileNotFoundError(f"Hermes source checkout does not exist: {source_root}")

    records: list[CompatSnapshotRecord] = []
    for profile in selected_profiles:
        relative_paths = snapshot_paths_for_profile(profile)
        missing = [relative for relative in relative_paths if not (source_root / relative).exists()]
        if missing:
            raise FileNotFoundError(
                f"Hermes source checkout is missing paths for profile '{profile}': "
                + ", ".join(missing)
            )

        record = build_snapshot_record(
            version,
            profile,
            source_ref=source_ref,
            provenance=provenance.to_dict(),
        )
        snapshot_root = fixtures_root / record.directory
        if snapshot_root == source_root:
            raise ValueError("Source checkout cannot also be the destination snapshot directory")
        if snapshot_root.exists():
            if not replace_existing:
                raise FileExistsError(
                    f"Snapshot already exists: {snapshot_root}. Use replace_existing=True to overwrite it."
                )
            if not dry_run:
                shutil.rmtree(snapshot_root)

        if not dry_run:
            for relative in relative_paths:
                source = source_root / relative
                destination = snapshot_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

        records.append(record)

    if not dry_run:
        _write_snapshot_registry(fixtures_root, records)

    return tuple(records)


def _write_snapshot_registry(
    fixtures_root: Path,
    new_records: Iterable[CompatSnapshotRecord],
) -> None:
    """Merge and write the supported snapshot registry."""
    existing = list(load_snapshot_registry(fixtures_root))
    replacement_ids = {record.id for record in new_records}
    merged = [record for record in existing if record.id not in replacement_ids]
    merged.extend(new_records)
    merged.sort(key=lambda record: (record.hermes_version, PROFILE_ORDER[record.profile], record.id))

    payload = {
        "schema_version": 2,
        "fixtures": [record.to_dict() for record in merged],
    }
    registry_path = fixtures_root / REGISTRY_FILENAME
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the snapshot refresh workflow."""
    parser = argparse.ArgumentParser(
        description="Refresh pinned Hermes compatibility snapshots from a real Hermes checkout.",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the Hermes release checkout or extracted source tree.",
    )
    parser.add_argument(
        "--fixtures-root",
        default=str(DEFAULT_FIXTURES_ROOT),
        help="Directory that contains the pinned compatibility snapshots.",
    )
    parser.add_argument(
        "--profile",
        dest="profiles",
        action="append",
        choices=SUPPORTED_PROFILES,
        help="Snapshot profile to refresh. Defaults to both core and extended.",
    )
    parser.add_argument(
        "--version",
        help="Hermes version override. Defaults to the version inferred from pyproject.toml.",
    )
    parser.add_argument(
        "--source-ref",
        help="Optional release tag, commit, or archive label to store in the registry.",
    )
    parser.add_argument(
        "--source-archive",
        help="Path to the release archive used to produce --source.",
    )
    parser.add_argument(
        "--archive-sha256",
        help="Expected SHA-256 checksum for --source-archive.",
    )
    parser.add_argument(
        "--source-tree-sha256",
        help="Expected SHA-256 checksum for the extracted --source tree.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Overwrite existing snapshot directories for the selected version and profiles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the refresh without writing snapshot directories or the registry.",
    )
    args = parser.parse_args(argv)

    profiles = args.profiles or list(SUPPORTED_PROFILES)
    try:
        records = refresh_snapshot_matrix(
            args.source,
            fixtures_root=args.fixtures_root,
            profiles=profiles,
            version=args.version,
            source_ref=args.source_ref,
            source_archive=args.source_archive,
            archive_sha256=args.archive_sha256,
            source_tree_sha256=args.source_tree_sha256,
            replace_existing=args.replace_existing,
            dry_run=args.dry_run,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    action = "Would refresh" if args.dry_run else "Refreshed"
    print(f"{action} Hermes compatibility snapshots:")
    for record in records:
        print(f"- {record.id} ({record.profile}, Hermes {record.hermes_version})")
    provenance = records[0].provenance if records else {}
    if provenance:
        print(f"Source tree sha256: {provenance['source_tree_sha256']}")
        if provenance.get("source_archive"):
            print(
                "Source archive verified: "
                f"{provenance['source_archive']} ({provenance['source_archive_sha256']})"
            )
        elif provenance.get("verification_mode") == "unverified_preview":
            print(
                "Source provenance not verified in preview. Re-run with --source-archive "
                "and --archive-sha256, or with --source-tree-sha256."
            )
    if args.dry_run:
        print("Registry unchanged.")
    else:
        registry_path = Path(args.fixtures_root).resolve() / REGISTRY_FILENAME
        print(f"Updated registry: {registry_path}")
    return 0


def _normalize_sha256(value: str | None, *, label: str) -> str:
    """Normalize and validate a SHA-256 hex digest."""
    normalized = (value or "").strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid {label}: expected a 64-character hexadecimal SHA-256 digest.")
    return normalized
