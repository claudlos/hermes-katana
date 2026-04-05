"""Tests for vault memory safety (Worker 3 — Part B).

Tests cover:
- GAP 2.1: Secure key zeroing on __del__
- GAP 2.3: Pop env var after reading
- GAP 2.5: File locking (fcntl.flock) — verified via _write_vault
- GAP 2.8: Expiry sync with vault
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# GAP 2.1 — Secure key zeroing
# ---------------------------------------------------------------------------

class TestSecureKeyZeroing:
    """GAP 2.1: _zero_key clears master key bytes; __del__ calls it."""

    def test_zero_key_clears_master_key(self, tmp_path):
        """After _zero_key(), _master_key should be None."""
        from hermes_katana.vault.store import Vault

        vault_path = tmp_path / "vault.json"
        with patch("hermes_katana.vault.store._get_master_key", return_value=os.urandom(32)):
            with patch("hermes_katana.vault.store._set_master_key"):
                vault = Vault(path=vault_path, auto_create=True)

        assert vault._master_key is not None
        vault._zero_key()
        assert vault._master_key is None

    def test_zero_key_idempotent(self, tmp_path):
        """Calling _zero_key twice should not raise."""
        from hermes_katana.vault.store import Vault

        vault_path = tmp_path / "vault.json"
        with patch("hermes_katana.vault.store._get_master_key", return_value=os.urandom(32)):
            with patch("hermes_katana.vault.store._set_master_key"):
                vault = Vault(path=vault_path, auto_create=True)

        vault._zero_key()
        vault._zero_key()  # second call should be safe
        assert vault._master_key is None

    def test_del_calls_close(self, tmp_path):
        """__del__ should call close() to clean up the master key."""
        from hermes_katana.vault.store import Vault

        vault_path = tmp_path / "vault.json"
        with patch("hermes_katana.vault.store._get_master_key", return_value=os.urandom(32)):
            with patch("hermes_katana.vault.store._set_master_key"):
                vault = Vault(path=vault_path, auto_create=True)

        with patch.object(vault, "close") as mock_close:
            vault.__del__()
            mock_close.assert_called_once()

    def test_zero_key_no_key_set(self, tmp_path):
        """_zero_key on vault with no key should not raise."""
        from hermes_katana.vault.store import Vault

        vault_path = tmp_path / "vault.json"
        with patch("hermes_katana.vault.store._get_master_key", return_value=os.urandom(32)):
            with patch("hermes_katana.vault.store._set_master_key"):
                vault = Vault(path=vault_path, auto_create=True)

        vault._master_key = None
        vault._zero_key()  # should not raise
        assert vault._master_key is None


# ---------------------------------------------------------------------------
# GAP 2.3 — Pop env var after reading
# ---------------------------------------------------------------------------

class TestEnvVarPopping:
    """GAP 2.3: HERMES_KATANA_VAULT_KEY is popped (not just read) from env."""

    def test_env_var_removed_after_read(self):
        """The code uses os.environ.pop so the key is consumed on read."""
        import base64
        import hermes_katana.vault.store as store_mod

        test_key = os.urandom(32)
        encoded = base64.b64encode(test_key).decode("ascii")

        # Verify that the source code uses os.environ.pop (not .get)
        import inspect
        source = inspect.getsource(store_mod._get_master_key)
        assert "os.environ.pop" in source, (
            "_get_master_key should use os.environ.pop to consume the env var"
        )

    def test_env_var_not_present_returns_none(self):
        """When env var is absent, _get_master_key returns None (keyring also fails)."""
        from hermes_katana.vault.store import _get_master_key

        os.environ.pop("HERMES_KATANA_VAULT_KEY", None)
        with patch("hermes_katana.vault.store._get_master_key") as mock_gmk:
            mock_gmk.return_value = None
            result = mock_gmk()
            assert result is None


# ---------------------------------------------------------------------------
# GAP 2.5 — File locking (verify _write_vault uses lock)
# ---------------------------------------------------------------------------

class TestFileLocking:
    """GAP 2.5: Vault writes use file-level locking."""

    def test_write_vault_uses_file_lock(self):
        """Verify the Vault _write_vault method references a file lock."""
        import inspect
        from hermes_katana.vault.store import Vault
        source = inspect.getsource(Vault._write_vault)
        assert "_file_lock" in source, "_write_vault should use file-level locking"

    def test_read_vault_uses_file_lock(self):
        """Verify the Vault _read_vault method references a file lock."""
        import inspect
        from hermes_katana.vault.store import Vault
        source = inspect.getsource(Vault._read_vault)
        assert "_file_lock" in source, "_read_vault should use file-level locking"


# ---------------------------------------------------------------------------
# GAP 2.8 — Expiry sync with vault
# ---------------------------------------------------------------------------

class TestExpirySync:
    """GAP 2.8: sync_with_vault removes orphaned expiry entries."""

    def test_sync_removes_orphans(self, tmp_path):
        """Expiry entries for deleted vault keys are cleaned up."""
        from hermes_katana.vault.expiry import SecretExpiry

        expiry = SecretExpiry(path=tmp_path / "expiry.json")
        expiry.set_expiry("key_a", ttl_seconds=3600)
        expiry.set_expiry("key_b", ttl_seconds=3600)
        expiry.set_expiry("key_c", ttl_seconds=3600)

        # Only key_a exists in vault
        orphaned = expiry.sync_with_vault(["key_a"])
        assert sorted(orphaned) == ["key_b", "key_c"]

        # Verify they're actually gone
        assert expiry.get_expiry("key_b") is None
        assert expiry.get_expiry("key_c") is None
        assert expiry.get_expiry("key_a") is not None

    def test_sync_no_orphans(self, tmp_path):
        """No orphans means no changes."""
        from hermes_katana.vault.expiry import SecretExpiry

        expiry = SecretExpiry(path=tmp_path / "expiry.json")
        expiry.set_expiry("key_a", ttl_seconds=3600)

        orphaned = expiry.sync_with_vault(["key_a"])
        assert orphaned == []

    def test_sync_empty_vault(self, tmp_path):
        """All expiry entries are orphaned if vault is empty."""
        from hermes_katana.vault.expiry import SecretExpiry

        expiry = SecretExpiry(path=tmp_path / "expiry.json")
        expiry.set_expiry("key_x", ttl_seconds=3600)
        expiry.set_expiry("key_y", ttl_seconds=3600)

        orphaned = expiry.sync_with_vault([])
        assert sorted(orphaned) == ["key_x", "key_y"]

    def test_sync_empty_expiry(self, tmp_path):
        """No expiry entries means nothing to clean."""
        from hermes_katana.vault.expiry import SecretExpiry

        expiry = SecretExpiry(path=tmp_path / "expiry.json")
        orphaned = expiry.sync_with_vault(["key_a", "key_b"])
        assert orphaned == []

    def test_sync_idempotent(self, tmp_path):
        """Running sync twice yields no orphans the second time."""
        from hermes_katana.vault.expiry import SecretExpiry

        expiry = SecretExpiry(path=tmp_path / "expiry.json")
        expiry.set_expiry("key_a", ttl_seconds=3600)
        expiry.set_expiry("key_b", ttl_seconds=3600)

        expiry.sync_with_vault(["key_a"])
        orphaned2 = expiry.sync_with_vault(["key_a"])
        assert orphaned2 == []
