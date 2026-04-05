"""Helpers for pinned Hermes compatibility snapshots."""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from hermes_katana.installer.compat_snapshots import CompatSnapshotRecord, load_snapshot_registry

_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "hermes_compat"
HERMES_V010_CORE_SNAPSHOT = "hermes-v0.1.0-core-snapshot"
HERMES_V010_EXTENDED_SNAPSHOT = "hermes-v0.1.0-extended-snapshot"


@lru_cache(maxsize=1)
def supported_fixtures() -> tuple[CompatSnapshotRecord, ...]:
    return load_snapshot_registry(_FIXTURE_ROOT)


def fixture_by_id(fixture_id: str) -> CompatSnapshotRecord:
    for fixture in supported_fixtures():
        if fixture.id == fixture_id:
            return fixture
    raise KeyError(f"Unknown Hermes compatibility fixture: {fixture_id}")


def fixture_checkout(fixture_id: str, tmp_dir: Path) -> Path:
    fixture = fixture_by_id(fixture_id)
    checkout = tmp_dir / fixture.directory
    shutil.copytree(_FIXTURE_ROOT / fixture.directory, checkout)
    return checkout
