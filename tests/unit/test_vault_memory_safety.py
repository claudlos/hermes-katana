"""Tests for vault SecureBytes memory safety and env var cleanup."""

from __future__ import annotations
import os
import base64
import secrets
import pytest
from unittest.mock import patch

from hermes_katana.vault.store import SecureBytes, VaultError


class TestSecureBytes:
    def test_raw_returns_data(self):
        data = b"secret-key-material-32bytes!!!!!"
        sb = SecureBytes(data)
        assert sb.raw == data

    def test_close_zeros_buffer(self):
        data = b"secret-key-material-32bytes!!!!!"
        sb = SecureBytes(data)
        sb.close()
        assert all(b == 0 for b in sb._buf)

    def test_raw_after_close_raises(self):
        sb = SecureBytes(b"secret")
        sb.close()
        with pytest.raises(VaultError, match="closed"):
            _ = sb.raw

    def test_double_close_safe(self):
        sb = SecureBytes(b"secret")
        sb.close()
        sb.close()  # no raise

    def test_del_zeros_buffer(self):
        data = b"another-secret-key!!"
        sb = SecureBytes(data)
        buf_ref = sb._buf
        sb.__del__()
        assert all(b == 0 for b in buf_ref)

    def test_len(self):
        assert len(SecureBytes(b"12345")) == 5

    def test_bool_true_when_open(self):
        assert bool(SecureBytes(b"data")) is True

    def test_bool_false_when_closed(self):
        sb = SecureBytes(b"data")
        sb.close()
        assert bool(sb) is False

    def test_bool_false_when_empty(self):
        assert bool(SecureBytes(b"")) is False

    def test_large_buffer_zeros(self):
        data = secrets.token_bytes(4096)
        sb = SecureBytes(data)
        assert sb.raw == data
        sb.close()
        assert all(b == 0 for b in sb._buf)


class TestVaultEnvVarCleanup:
    def test_env_var_removed_after_read(self):
        """HERMES_KATANA_VAULT_KEY is consumed and removed from os.environ."""
        test_key = secrets.token_bytes(32)
        encoded = base64.b64encode(test_key).decode("ascii")

        with patch.dict(os.environ, {"HERMES_KATANA_VAULT_KEY": encoded}):
            with patch.dict("sys.modules", {"keyring": None}):
                from hermes_katana.vault.store import _get_master_key
                result = _get_master_key()
                assert "HERMES_KATANA_VAULT_KEY" not in os.environ
                assert result == test_key


class TestVaultCloseMethod:
    def test_vault_close_zeros_key(self):
        from hermes_katana.vault.store import Vault
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test_vault.json"
            test_key = secrets.token_bytes(32)
            encoded = base64.b64encode(test_key).decode("ascii")

            with patch.dict(os.environ, {"HERMES_KATANA_VAULT_KEY": encoded}):
                with patch.dict("sys.modules", {"keyring": None}):
                    vault = Vault(path=vault_path, auto_create=True)
                    assert vault._master_key is not None
                    buf_ref = vault._master_key._buf
                    vault.close()
                    assert vault._master_key is None
                    assert all(b == 0 for b in buf_ref)

    def test_vault_close_idempotent(self):
        from hermes_katana.vault.store import Vault
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir) / "test_vault.json"
            test_key = secrets.token_bytes(32)
            encoded = base64.b64encode(test_key).decode("ascii")

            with patch.dict(os.environ, {"HERMES_KATANA_VAULT_KEY": encoded}):
                with patch.dict("sys.modules", {"keyring": None}):
                    vault = Vault(path=vault_path, auto_create=True)
                    vault.close()
                    vault.close()  # no raise
