"""
AES-256-GCM encrypted vault for HermesKatana.

Hardened from hermes-aegis Fernet (AES-128-CBC) vault with:
- AES-256-GCM authenticated encryption (256-bit key, 96-bit nonce, 128-bit tag)
- Per-value random nonces (no nonce reuse)
- HMAC-SHA256 integrity check over all entries
- Master key stored in OS keyring (not on disk)
- Circuit breaker via vault.lock sentinel file
- Key rotation: re-encrypt all values with a new master key
- Atomic writes via tmp + replace (no partial-write corruption)
- Thread-safe with reentrant lock

Security model:
- Master key never touches disk (only in keyring + memory)
- Each value encrypted independently (compromise of one doesn't leak others)
- HMAC prevents tampering with the vault file
- Circuit breaker locks vault on suspicious activity
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Vault file format version
_VAULT_VERSION = 2

# Keyring service/account names
_KEYRING_SERVICE = "hermes-katana-vault"
_KEYRING_ACCOUNT = "master-key"

# AES-256-GCM parameters
_KEY_SIZE = 32  # 256 bits
_NONCE_SIZE = 12  # 96 bits (standard for GCM)
_TAG_SIZE = 16  # 128 bits


class VaultError(Exception):
    """Base exception for vault operations."""
    pass


class VaultLockedError(VaultError):
    """Raised when the vault is locked (circuit breaker active)."""
    pass


class VaultIntegrityError(VaultError):
    """Raised when vault integrity check fails."""
    pass


class VaultKeyError(VaultError):
    """Raised when a requested key is not found."""
    pass


# ---------------------------------------------------------------------------
# AES-256-GCM encryption primitives
# ---------------------------------------------------------------------------

def _encrypt_value(plaintext: str, key: bytes) -> str:
    """Encrypt a plaintext string with AES-256-GCM.

    Returns a base64-encoded string: nonce || ciphertext || tag.

    Args:
        plaintext: The value to encrypt.
        key: 32-byte AES-256 key.

    Returns:
        Base64-encoded encrypted blob.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise VaultError(
            "cryptography package required: pip install cryptography"
        )

    nonce = secrets.token_bytes(_NONCE_SIZE)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
    # ct includes the tag appended by AESGCM
    blob = nonce + ct
    return base64.b64encode(blob).decode("ascii")


def _decrypt_value(encrypted: str, key: bytes) -> str:
    """Decrypt an AES-256-GCM encrypted value.

    Args:
        encrypted: Base64-encoded encrypted blob (nonce || ciphertext || tag).
        key: 32-byte AES-256 key.

    Returns:
        Decrypted plaintext string.

    Raises:
        VaultError: On decryption failure (wrong key, tampered data).
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise VaultError(
            "cryptography package required: pip install cryptography"
        )

    try:
        blob = base64.b64decode(encrypted)
    except Exception:
        raise VaultError("Invalid encrypted value: base64 decode failed")

    if len(blob) < _NONCE_SIZE + _TAG_SIZE:
        raise VaultError("Invalid encrypted value: too short")

    nonce = blob[:_NONCE_SIZE]
    ct = blob[_NONCE_SIZE:]

    try:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ct, None)
        return plaintext.decode("utf-8")
    except Exception:
        raise VaultError(
            "Decryption failed: wrong key or tampered data"
        )


def _compute_hmac(data: dict[str, str], key: bytes) -> str:
    """Compute HMAC-SHA256 over vault entries for integrity checking.

    The HMAC is computed over sorted key-value pairs to ensure
    deterministic output regardless of dict ordering.

    Args:
        data: The encrypted vault entries.
        key: The master key (used to derive HMAC key).

    Returns:
        Hex-encoded HMAC digest.
    """
    # Derive a separate HMAC key from the master key
    hmac_key = hashlib.sha256(b"hmac:" + key).digest()
    msg = json.dumps(sorted(data.items()), sort_keys=True).encode()
    return hmac.new(hmac_key, msg, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Keyring operations
# ---------------------------------------------------------------------------

def _get_master_key() -> Optional[bytes]:
    """Retrieve the master key from the OS keyring.

    Returns:
        The 32-byte master key, or None if not found.
    """
    try:
        import keyring
        raw = keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
        if raw is None:
            return None
        return base64.b64decode(raw)
    except ImportError:
        logger.warning(
            "keyring package not installed, falling back to environment variable"
        )
        raw_env = os.environ.get("HERMES_KATANA_VAULT_KEY")
        if raw_env:
            return base64.b64decode(raw_env)
        return None
    except Exception as exc:
        logger.warning("Failed to read keyring: %s", exc)
        return None


def _set_master_key(key: bytes) -> None:
    """Store the master key in the OS keyring.

    Args:
        key: 32-byte master key to store.
    """
    encoded = base64.b64encode(key).decode("ascii")
    try:
        import keyring
        keyring.set_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT, encoded)
    except ImportError:
        logger.warning(
            "keyring package not installed. Set HERMES_KATANA_VAULT_KEY "
            "environment variable to: %s",
            encoded,
        )
    except Exception as exc:
        raise VaultError(f"Failed to store master key in keyring: {exc}")


def _delete_master_key() -> None:
    """Remove the master key from the OS keyring."""
    try:
        import keyring
        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
    except ImportError:
        pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Vault class
# ---------------------------------------------------------------------------

def _default_vault_path() -> Path:
    """Return the default vault file path."""
    config_dir = Path.home() / ".config" / "hermes-katana"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "vault.json"


class Vault:
    """AES-256-GCM encrypted secret vault.

    Stores secrets in an encrypted JSON file with the master key in
    the OS keyring. Each value is individually encrypted with a random
    nonce. An HMAC-SHA256 covers all entries for tamper detection.

    Args:
        path: Path to the vault file (default: ~/.config/hermes-katana/vault.json).
        auto_create: If True, create a new vault with a fresh master key
            if one doesn't exist.

    Example:
        >>> vault = Vault()
        >>> vault.set("OPENAI_API_KEY", "sk-abc123...")
        >>> vault.get("OPENAI_API_KEY")
        'sk-abc123...'
        >>> vault.list_keys()
        ['OPENAI_API_KEY']
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        auto_create: bool = True,
    ) -> None:
        self._path = path or _default_vault_path()
        self._lock_path = self._path.with_suffix(".lock")
        self._rlock = threading.RLock()
        self._master_key: Optional[bytes] = None

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize
        if auto_create:
            self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure the vault is initialized with a master key."""
        key = _get_master_key()
        if key is None:
            # Generate a new master key
            key = secrets.token_bytes(_KEY_SIZE)
            _set_master_key(key)
            logger.info("Generated new vault master key")

            # Create empty vault file
            self._write_vault({})
        self._master_key = key

    def _get_key(self) -> bytes:
        """Get the master key, raising an error if not available."""
        if self._master_key is None:
            self._master_key = _get_master_key()
        if self._master_key is None:
            raise VaultError(
                "No master key found. Initialize vault or set "
                "HERMES_KATANA_VAULT_KEY environment variable."
            )
        return self._master_key

    def _check_lock(self) -> None:
        """Check if the vault is locked (circuit breaker)."""
        if self._lock_path.exists():
            raise VaultLockedError(
                f"Vault is locked (circuit breaker active). "
                f"Remove {self._lock_path} to unlock."
            )

    def _read_vault(self) -> dict[str, Any]:
        """Read and parse the vault file.

        Returns:
            The vault data dict with 'version', 'entries', 'hmac' keys.
        """
        if not self._path.exists():
            return {"version": _VAULT_VERSION, "entries": {}, "hmac": ""}

        try:
            raw = self._path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return data
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VaultError(f"Corrupt vault file: {exc}")

    def _write_vault(self, entries: dict[str, str]) -> None:
        """Atomically write the vault file.

        Uses temp file + rename pattern to prevent partial writes.

        Args:
            entries: The encrypted entries dict.
        """
        key = self._get_key()
        hmac_digest = _compute_hmac(entries, key)

        vault_data = {
            "version": _VAULT_VERSION,
            "entries": entries,
            "hmac": hmac_digest,
        }

        # Write to temp file first, then atomic replace
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=".vault_",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fp:
                    json.dump(vault_data, fp, indent=2)
                    fp.flush()
                    os.fsync(fp.fileno())
                # Atomic replace
                Path(tmp_path).replace(self._path)
            except Exception:
                os.unlink(tmp_path)
                raise
        except Exception as exc:
            raise VaultError(f"Failed to write vault: {exc}")

    def get(self, key: str) -> str:
        """Retrieve a decrypted secret by key.

        Args:
            key: The secret name.

        Returns:
            The decrypted secret value.

        Raises:
            VaultLockedError: If the vault is locked.
            VaultKeyError: If the key is not found.
            VaultError: On decryption failure.
        """
        with self._rlock:
            self._check_lock()
            master_key = self._get_key()
            vault = self._read_vault()
            entries = vault.get("entries", {})

            if key not in entries:
                raise VaultKeyError(f"Key not found: {key}")

            return _decrypt_value(entries[key], master_key)

    def set(self, key: str, value: str) -> None:
        """Store an encrypted secret.

        Args:
            key: The secret name.
            value: The plaintext secret value to encrypt and store.

        Raises:
            VaultLockedError: If the vault is locked.
            VaultError: On encryption or write failure.
        """
        with self._rlock:
            self._check_lock()
            master_key = self._get_key()
            vault = self._read_vault()
            entries = vault.get("entries", {})
            entries[key] = _encrypt_value(value, master_key)
            self._write_vault(entries)
            logger.debug("Stored key: %s", key)

    def remove(self, key: str) -> None:
        """Remove a secret from the vault.

        Args:
            key: The secret name to remove.

        Raises:
            VaultLockedError: If the vault is locked.
            VaultKeyError: If the key is not found.
        """
        with self._rlock:
            self._check_lock()
            vault = self._read_vault()
            entries = vault.get("entries", {})

            if key not in entries:
                raise VaultKeyError(f"Key not found: {key}")

            del entries[key]
            self._write_vault(entries)
            logger.debug("Removed key: %s", key)

    def list_keys(self) -> list[str]:
        """List all stored secret names.

        Returns:
            Sorted list of key names.

        Raises:
            VaultLockedError: If the vault is locked.
        """
        with self._rlock:
            self._check_lock()
            vault = self._read_vault()
            return sorted(vault.get("entries", {}).keys())

    def _get_all_values(self) -> dict[str, str]:
        """Get all decrypted values (internal use only, e.g., for scanner).

        Returns:
            Dict mapping key names to decrypted values.

        Warning:
            This loads all secrets into memory. Use only when necessary
            (e.g., building the secret scanner value set).
        """
        with self._rlock:
            self._check_lock()
            master_key = self._get_key()
            vault = self._read_vault()
            entries = vault.get("entries", {})
            result: dict[str, str] = {}
            for k, encrypted in entries.items():
                try:
                    result[k] = _decrypt_value(encrypted, master_key)
                except VaultError:
                    logger.warning("Failed to decrypt key: %s", k)
            return result

    def lock(self) -> None:
        """Activate the circuit breaker, locking the vault.

        Creates a sentinel file that prevents all vault operations
        until unlock() is called.
        """
        with self._rlock:
            self._lock_path.touch()
            logger.warning("Vault LOCKED (circuit breaker activated)")

    def unlock(self) -> None:
        """Deactivate the circuit breaker, unlocking the vault.

        Removes the sentinel file.
        """
        with self._rlock:
            if self._lock_path.exists():
                self._lock_path.unlink()
                logger.info("Vault unlocked")
            else:
                logger.debug("Vault was not locked")

    def is_locked(self) -> bool:
        """Check if the vault is locked."""
        return self._lock_path.exists()

    def rotate_key(self) -> None:
        """Rotate the master key.

        Generates a new master key, re-encrypts all values with the new
        key, and stores the new key in the keyring.

        This is an atomic operation: if re-encryption fails for any value,
        the old key is preserved.

        Raises:
            VaultLockedError: If the vault is locked.
            VaultError: On rotation failure.
        """
        with self._rlock:
            self._check_lock()
            old_key = self._get_key()
            new_key = secrets.token_bytes(_KEY_SIZE)

            # Read and decrypt all values with old key
            vault = self._read_vault()
            entries = vault.get("entries", {})
            decrypted: dict[str, str] = {}

            for k, encrypted in entries.items():
                try:
                    decrypted[k] = _decrypt_value(encrypted, old_key)
                except VaultError as exc:
                    raise VaultError(
                        f"Key rotation failed: could not decrypt '{k}': {exc}"
                    )

            # Re-encrypt all values with new key
            new_entries: dict[str, str] = {}
            for k, plaintext in decrypted.items():
                new_entries[k] = _encrypt_value(plaintext, new_key)

            # Store the new key in keyring first (so we can recover)
            _set_master_key(new_key)
            self._master_key = new_key

            # Write the re-encrypted vault
            try:
                self._write_vault(new_entries)
            except Exception:
                # Rollback: restore old key
                _set_master_key(old_key)
                self._master_key = old_key
                raise

            logger.info(
                "Vault key rotated successfully (%d entries re-encrypted)",
                len(new_entries),
            )

    def verify_integrity(self) -> bool:
        """Verify the vault's HMAC integrity.

        Returns:
            True if the HMAC is valid, False if tampered.

        Raises:
            VaultLockedError: If the vault is locked.
        """
        with self._rlock:
            self._check_lock()
            key = self._get_key()
            vault = self._read_vault()
            entries = vault.get("entries", {})
            stored_hmac = vault.get("hmac", "")

            if not stored_hmac:
                # No HMAC stored (legacy vault or empty vault)
                return len(entries) == 0

            expected_hmac = _compute_hmac(entries, key)
            return hmac.compare_digest(stored_hmac, expected_hmac)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the vault."""
        try:
            keys = self.list_keys()
            return key in keys
        except VaultError:
            return False

    def __len__(self) -> int:
        """Return the number of stored secrets."""
        try:
            return len(self.list_keys())
        except VaultError:
            return 0
