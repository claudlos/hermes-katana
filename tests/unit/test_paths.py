"""Tests for hermes_katana._paths and safe-home hardening.

Regression coverage: every path helper and every module that calls
Path.home() at import time must survive a cleared environment. On
Windows, Path.home() raises RuntimeError when USERPROFILE/HOMEDRIVE
are unset; without safe-home hardening, this crashes module imports
and breaks any test that wipes os.environ.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# _paths helper unit tests
# ---------------------------------------------------------------------------


class TestSafeHome:
    def test_returns_path_when_home_resolves(self):
        from hermes_katana._paths import safe_home

        result = safe_home()
        assert result is None or isinstance(result, Path)

    def test_returns_none_when_path_home_raises_runtime_error(self):
        from hermes_katana import _paths

        def raising_home():
            raise RuntimeError("Could not determine home directory.")

        with patch.object(_paths.Path, "home", staticmethod(raising_home)):
            assert _paths.safe_home() is None

    def test_returns_none_when_path_home_raises_key_error(self):
        from hermes_katana import _paths

        def raising_home():
            raise KeyError("HOME")

        with patch.object(_paths.Path, "home", staticmethod(raising_home)):
            assert _paths.safe_home() is None


class TestFallbackRoot:
    def test_returns_path_under_tempdir(self):
        from hermes_katana._paths import fallback_root

        root = fallback_root()
        assert isinstance(root, Path)
        # Must be under the system tempdir
        assert str(root).startswith(tempfile.gettempdir())
        # Must include the fallback subdir name
        assert root.name == "hermes-katana-fallback"

    def test_does_not_create_the_directory(self, tmp_path, monkeypatch):
        from hermes_katana import _paths

        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        root = _paths.fallback_root()
        assert not root.exists()


class TestHomeOrFallback:
    def test_returns_home_when_available(self):
        from hermes_katana._paths import home_or_fallback, safe_home

        home = safe_home()
        if home is None:
            pytest.skip("Home not resolvable in this environment")
        assert home_or_fallback() == home

    def test_returns_fallback_when_home_unavailable(self):
        from hermes_katana import _paths

        with patch.object(_paths, "safe_home", return_value=None):
            result = _paths.home_or_fallback()
            assert result == _paths.fallback_root()


# ---------------------------------------------------------------------------
# Import-time safety: modules must survive a crashing Path.home()
# ---------------------------------------------------------------------------


def _simulate_unresolvable_home(monkeypatch):
    """Monkeypatch Path.home to raise so we can test the fallback path."""

    def raising_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(raising_home))


_MODULES_UNDER_TEST = [
    "hermes_katana.config",
    "hermes_katana.audit.trail",
    "hermes_katana.vault.store",
    "hermes_katana.vault.expiry",
    "hermes_katana.vault.access_log",
    "hermes_katana.vault.migrate",
]


@pytest.fixture
def reimport_safely():
    """Reimport a module, then restore the original on teardown.

    This is critical: other tests hold references to these modules
    (e.g. `import hermes_katana.config as config_mod`), and if we leave
    a fresh module object in sys.modules, monkeypatches in those tests
    target the wrong object and fail.
    """
    saved: dict[str, object] = {}

    def _do_reimport(module_name: str):
        saved[module_name] = sys.modules.get(module_name)
        sys.modules.pop(module_name, None)
        return importlib.import_module(module_name)

    yield _do_reimport

    for name, original in saved.items():
        if original is not None:
            sys.modules[name] = original
        else:
            sys.modules.pop(name, None)


class TestImportSafety:
    @pytest.mark.parametrize("module_name", _MODULES_UNDER_TEST)
    def test_module_reimports_without_crashing_when_home_is_unresolvable(
        self, module_name, monkeypatch, reimport_safely
    ):
        """Reimporting any path-consuming module under a broken Path.home
        must not raise. Before the safe-home hardening, config.py crashed
        at import time and the others crashed on first helper call.
        """
        _simulate_unresolvable_home(monkeypatch)
        module = reimport_safely(module_name)
        assert module is not None


class TestPathHelperSafety:
    """Each default_*_path helper must return a usable Path even when
    Path.home() crashes - they must never raise."""

    def test_audit_default_audit_path(self, monkeypatch):
        _simulate_unresolvable_home(monkeypatch)
        from hermes_katana.audit.trail import default_audit_path

        result = default_audit_path()
        assert isinstance(result, Path)
        assert result.name == "audit.jsonl"

    def test_vault_default_vault_path(self, monkeypatch):
        _simulate_unresolvable_home(monkeypatch)
        from hermes_katana.vault.store import default_vault_path

        result = default_vault_path()
        assert isinstance(result, Path)
        assert result.name == "vault.json"

    def test_vault_expiry_default_path(self, monkeypatch):
        _simulate_unresolvable_home(monkeypatch)
        from hermes_katana.vault.expiry import _default_expiry_path

        result = _default_expiry_path()
        assert isinstance(result, Path)
        assert result.name == "vault_expiry.json"

    def test_vault_access_log_default_path(self, monkeypatch):
        _simulate_unresolvable_home(monkeypatch)
        from hermes_katana.vault.access_log import _default_access_log_path

        result = _default_access_log_path()
        assert isinstance(result, Path)
        assert result.name == "vault_access.jsonl"


class TestClearedEnvironImport:
    """End-to-end: import a module after clearing os.environ.

    This is the exact failure mode that started this PR. On Windows,
    clearing USERPROFILE/HOMEDRIVE/HOMEPATH makes Path.home() crash.
    """

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Only Windows derives home from env vars; POSIX uses /etc/passwd",
    )
    @pytest.mark.parametrize("module_name", _MODULES_UNDER_TEST)
    def test_reimport_under_cleared_environ(self, module_name, reimport_safely):
        # Save vars that Windows Path.home() consults
        keep = {k: os.environ.get(k) for k in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME")}
        try:
            for k in keep:
                os.environ.pop(k, None)
            # Must not raise. reimport_safely restores the original module
            # on teardown so other tests still see the expected object.
            reimport_safely(module_name)
        finally:
            for k, v in keep.items():
                if v is not None:
                    os.environ[k] = v
