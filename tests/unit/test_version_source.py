"""Regression tests for runtime version metadata drift."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import hermes_katana
from hermes_katana import hermes_plugin
from hermes_katana._version import ARTIFACT_REVISION, __version__
from hermes_katana.artifacts import DEFAULT_REVISION
from hermes_katana.cli import _support as cli_support
from hermes_katana.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_version_metadata_matches_pyproject():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    expected = pyproject["project"]["version"]
    assert __version__ == expected
    assert hermes_katana.__version__ == expected
    assert cli_main.VERSION == expected
    assert cli_support.VERSION == expected
    assert hermes_plugin.plugin_version == expected


def test_artifact_revision_uses_runtime_version():
    assert ARTIFACT_REVISION == f"v{__version__}"
    assert DEFAULT_REVISION == ARTIFACT_REVISION
