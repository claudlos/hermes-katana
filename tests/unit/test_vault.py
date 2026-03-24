"""Tests for HermesKatana vault (AES-256-GCM encrypted storage)."""

from __future__ import annotations

import base64
import os
import secrets
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_katana.vault.store import (
    Vault,
    VaultError,
    VaultKeyError,
    VaultLockedError,
    _decrypt_value,
    _encrypt_value,
)


# ======================================================================
# AES-256-GCM encryption primitives
# ======================================================================

class TestEncryptionPrimitives:
    def test_encrypt_decrypt_roundtrip(self):
        key = secrets.token_bytes(32)
        plaintext = "my super secret value"
        encrypted = _encrypt_value(plaintext, key)
        decrypted = _decrypt_value(encrypted, key)
        assert decrypted == plaintext

    def test_different_keys_different_ciphertext(self):
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        plaintext = "test value"
        ct1 = _encrypt_value(plaintext, key1)
        ct2 = _encrypt_value(plaintext, key2)
        assert ct1 != ct2

    def test_wrong_key_fails(self):
        key1 = secrets.token_bytes(32)
        key2 = secrets.token_bytes(32)
        encrypted = _encrypt_value("secret", key1)
        with pytest.raises(VaultError):
            _decrypt_value(encrypted, key2)

    def test_tampered_ciphertext_fails(self):
        key = secrets.token_bytes(32)
        encrypted = _encrypt_value("secret", key)
        # Tamper with the base64 data
        raw = base64.b64decode(encrypted)
        tampered = raw[:-1] + bytes([raw[-1] ^ 0xFF])
        tampered_b64 = base64.b64encode(tampered).decode()
        with pytest.raises(VaultError):
            _decrypt_value(tampered_b64, key)

    def test_empty_string_roundtrip(self):
        key = secrets.token_bytes(32)
        plaintext = ""
        encrypted = _encrypt_value(plaintext, key)
        decrypted = _decrypt_value(encrypted, key)
        assert decrypted == ""

    def test_unicode_roundtrip(self):
        key = secrets.token_bytes(32)
        plaintext = "unicode: \u00e9\u00e8\u00ea \u2603 \U0001f600"
        encrypted = _encrypt_value(plaintext, key)
        decrypted = _decrypt_value(encrypted, key)
        assert decrypted == plaintext


# ======================================================================
# Vault — high-level operations
# ======================================================================

class TestVault:
    @pytest.fixture
    def vault(self, vault_path):
        """Create a vault with a known test key via env var and mock keyring."""
        key = secrets.token_bytes(32)
        key_b64 = base64.b64encode(key).decode()
        with patch.dict(os.environ, {"HERMES_KATANA_VAULT_KEY": key_b64}), \
             patch("hermes_katana.vault.store._get_master_key", return_value=key), \
             patch("hermes_katana.vault.store._set_master_key"):
            v = Vault(path=vault_path, auto_create=True)
            yield v

    def test_set_and_get(self, vault):
        vault.set("API_KEY", "sk-test123")
        assert vault.get("API_KEY") == "sk-test123"

    def test_get_nonexistent_key(self, vault):
        with pytest.raises(VaultKeyError):
            vault.get("NONEXISTENT")

    def test_list_keys(self, vault):
        vault.set("KEY_A", "value_a")
        vault.set("KEY_B", "value_b")
        keys = vault.list_keys()
        assert "KEY_A" in keys
        assert "KEY_B" in keys
        assert keys == sorted(keys)  # Should be sorted

    def test_remove_key(self, vault):
        vault.set("TO_DELETE", "value")
        assert "TO_DELETE" in vault.list_keys()
        vault.remove("TO_DELETE")
        assert "TO_DELETE" not in vault.list_keys()

    def test_remove_nonexistent_key(self, vault):
        with pytest.raises(VaultKeyError):
            vault.remove("NONEXISTENT")

    def test_overwrite_key(self, vault):
        vault.set("KEY", "value1")
        vault.set("KEY", "value2")
        assert vault.get("KEY") == "value2"


# ======================================================================
# Circuit breaker (lock/unlock)
# ======================================================================

class TestVaultCircuitBreaker:
    @pytest.fixture
    def vault(self, vault_path):
        key = secrets.token_bytes(32)
        key_b64 = base64.b64encode(key).decode()
        with patch.dict(os.environ, {"HERMES_KATANA_VAULT_KEY": key_b64}), \
             patch("hermes_katana.vault.store._get_master_key", return_value=key), \
             patch("hermes_katana.vault.store._set_master_key"):
            v = Vault(path=vault_path, auto_create=True)
            yield v

    def test_lock_blocks_get(self, vault):
        vault.set("KEY", "value")
        vault.lock()
        assert vault.is_locked() is True
        with pytest.raises(VaultLockedError):
            vault.get("KEY")

    def test_lock_blocks_set(self, vault):
        vault.lock()
        with pytest.raises(VaultLockedError):
            vault.set("KEY", "value")

    def test_lock_blocks_list(self, vault):
        vault.lock()
        with pytest.raises(VaultLockedError):
            vault.list_keys()

    def test_unlock_restores_access(self, vault):
        vault.set("KEY", "value")
        vault.lock()
        assert vault.is_locked() is True
        vault.unlock()
        assert vault.is_locked() is False
        assert vault.get("KEY") == "value"

    def test_double_unlock_is_safe(self, vault):
        vault.lock()
        vault.unlock()
        vault.unlock()  # Should not raise
        assert vault.is_locked() is False

    def test_is_locked_initially_false(self, vault):
        assert vault.is_locked() is False
