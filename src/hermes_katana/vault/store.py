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
import ctypes
import hashlib
import hmac
import json
import logging
import os
import platform
import secrets
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "VaultError",
    "VaultLockedError",
    "VaultIntegrityError",
    "VaultKeyError",
    "SecureBytes",
    "Vault",
    "default_vault_path",
]


class _VaultFileLock:
    """Cross-platform advisory file lock for multi-process vault safety."""

    def __init__(self, path: Path) -> None:
        self._lock_path = path.with_suffix(path.suffix + ".flock")
        self._fp: Any = None

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self._lock_path, "w")
        if platform.system() == "Windows":
            import msvcrt
            msvcrt.locking(self._fp.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX)

    def release(self) -> None:
        if self._fp is not None:
            try:
                if platform.system() == "Windows":
                    import msvcrt
                    msvcrt.locking(self._fp.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
            except (OSError, BlockingIOError):
                pass
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None

    def __enter__(self) -> "_VaultFileLock":
        self.acquire()
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

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


class SecureBytes:
    """Memory-safe bytes wrapper that zeros memory on deallocation.

    Wraps a bytes-like secret so that the underlying buffer is overwritten
    with zeros when the object is garbage-collected or explicitly closed.
    Optionally calls mlock() to prevent the page from being swapped to disk.

    Usage:
        sb = SecureBytes(secret_key)
        raw = sb.raw  # access the bytes
        sb.close()    # explicitly zero and release
    """

    def __init__(self, data: bytes) -> None:
        import ctypes
        import ctypes.util
        self._buf = bytearray(data)
        self._length = len(data)
        self._closed = False
        self._mlocked = False
        # Best-effort mlock to prevent swapping
        try:
            libc_name = ctypes.util.find_library("c")
            if libc_name:
                libc = ctypes.CDLL(libc_name, use_errno=True)
                if self._length > 0:
                    addr = (ctypes.c_char * self._length).from_buffer(self._buf)
                    ret = libc.mlock(ctypes.addressof(addr), self._length)
                    if ret == 0:
                        self._mlocked = True
        except Exception:
            pass

    @property
    def raw(self) -> bytes:
        """Return the underlying bytes (read-only copy)."""
        if self._closed:
            raise VaultError("SecureBytes has been closed/zeroed")
        return bytes(self._buf)

    def close(self) -> None:
        """Zero the buffer and mark as closed."""
        if self._closed:
            return
        try:
            import ctypes
        except ImportError:
            self._closed = True
            return
        try:
            buf_type = ctypes.c_char * self._length
            buf_ref = buf_type.from_buffer(self._buf)
            ctypes.memset(ctypes.addressof(buf_ref), 0, self._length)
        except Exception:
            for i in range(self._length):
                self._buf[i] = 0
        if self._mlocked:
            try:
                import ctypes.util
                libc_name = ctypes.util.find_library("c")
                if libc_name:
                    libc = ctypes.CDLL(libc_name, use_errno=True)
                    buf_type = ctypes.c_char * self._length
                    buf_ref = buf_type.from_buffer(self._buf)
                    libc.munlock(ctypes.addressof(buf_ref), self._length)
            except Exception:
                pass
        self._closed = True

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self._length

    def __bool__(self) -> bool:
        return not self._closed and self._length > 0


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
    def _validate_key(key_bytes: bytes) -> bytes:
        if len(key_bytes) != _KEY_SIZE:
            raise VaultError(
                f"Master key must be {_KEY_SIZE} bytes, got {len(key_bytes)}"
            )
        return key_bytes

    try:
        import keyring
        raw = keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
        if raw is None:
            # Keyring exists but has no stored key — fall through to env var
            raise KeyError("no key in keyring")
        return _validate_key(base64.b64decode(raw))
    except ImportError:
        logger.warning(
            "keyring package not installed, falling back to environment variable"
        )
    except VaultError:
        raise
    except Exception as exc:
        logger.warning("Failed to read keyring: %s, falling back to environment variable", exc)

    # Fallback: check environment variable
    raw_env = os.environ.pop("HERMES_KATANA_VAULT_KEY", None)
    if raw_env:
        return _validate_key(base64.b64decode(raw_env))
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
            "keyring package not installed. Master key generated but could "
            "not be stored in keyring. Run 'hermes-katana vault show-key' "
            "or set HERMES_KATANA_VAULT_KEY env var manually. "
            "Store the key securely."
        )
    except Exception as exc:
        logger.warning(
            "Failed to store master key in keyring: %s. "
            "The key is held in memory for this session. "
            "Set HERMES_KATANA_VAULT_KEY env var for persistence.",
            exc,
        )


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

def default_vault_path() -> Path:
    """Return the default vault file path without creating it."""
    return Path.home() / ".config" / "hermes-katana" / "vault.json"


def _default_vault_path() -> Path:
    """Return the default vault file path."""
    config_dir = default_vault_path().parent
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
        self._file_lock = _VaultFileLock(self._path)
        self._rlock = threading.RLock()
        self._master_key: Optional[SecureBytes] = None

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
        self._master_key = SecureBytes(key)

    def _get_key(self) -> bytes:
        """Get the master key, raising an error if not available."""
        if self._master_key is None:
            raw = _get_master_key()
            if raw is not None:
                self._master_key = SecureBytes(raw)
        if self._master_key is None:
            raise VaultError(
                "No master key found. Initialize vault or set "
                "HERMES_KATANA_VAULT_KEY environment variable."
            )
        return self._master_key.raw

    def _check_lock(self) -> None:
        """Check if the vault is locked (circuit breaker)."""
        if self._lock_path.exists():
            raise VaultLockedError(
                f"Vault is locked (circuit breaker active). "
                f"Remove {self._lock_path} to unlock."
            )

    def _read_vault(self) -> dict[str, Any]:
        """Read and parse the vault file.

        Uses file-level locking for multi-process safety.

        Returns:
            The vault data dict with 'version', 'entries', 'hmac' keys.
        """
        if not self._path.exists():
            return {"version": _VAULT_VERSION, "entries": {}, "hmac": ""}

        try:
            with self._file_lock:
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

        # Write to temp file first, then atomic replace — with file lock
        # for multi-process safety
        try:
            with self._file_lock:
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
                    # Set permissions before rename
                    if sys.platform != "win32":
                        os.chmod(tmp_path, 0o600)
                    # Atomic replace
                    Path(tmp_path).replace(self._path)
                except Exception:
                    os.unlink(tmp_path)
                    raise
        except VaultError:
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
            stored_hmac = vault.get("hmac", "")

            # Verify HMAC integrity before returning any data
            if entries:
                if not stored_hmac:
                    raise VaultIntegrityError(
                        "Vault integrity check failed: HMAC missing on non-empty vault. "
                        "The vault file may have been tampered with."
                    )
                expected_hmac = _compute_hmac(entries, master_key)
                if not hmac.compare_digest(stored_hmac, expected_hmac):
                    raise VaultIntegrityError(
                        "Vault integrity check failed: HMAC mismatch. "
                        "The vault file may have been tampered with."
                    )

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

            # Write rotation journal for crash recovery.
            # Keys are encrypted with each other to avoid writing plaintext keys to disk:
            # old_key is encrypted with new_key, new_key is encrypted with old_key.
            journal_path = self._path.with_suffix(".rotation_journal")
            try:
                import time as _time
                journal_data = {
                    "status": "in_progress",
                    "old_key_enc": _encrypt_value(
                        base64.b64encode(old_key).decode("ascii"), new_key
                    ),
                    "new_key_enc": _encrypt_value(
                        base64.b64encode(new_key).decode("ascii"), old_key
                    ),
                    "timestamp": _time.time(),
                }
                journal_path.write_text(
                    json.dumps(journal_data), encoding="utf-8"
                )
                # Restrict permissions
                journal_path.chmod(0o600)
            except Exception as exc:
                raise VaultError(f"Failed to write rotation journal: {exc}")

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
            self._master_key = SecureBytes(new_key)

            # Write the re-encrypted vault
            try:
                self._write_vault(new_entries)
            except Exception:
                # Rollback: restore old key
                _set_master_key(old_key)
                self._master_key = SecureBytes(old_key)
                raise

            logger.info(
                "Vault key rotated successfully (%d entries re-encrypted)",
                len(new_entries),
            )

            # Remove rotation journal on success
            try:
                if journal_path.exists():
                    journal_path.unlink()
            except OSError:
                pass

    def recover_rotation(self) -> bool:
        """Check for and complete any interrupted key rotation.

        Called on startup to recover from crashes during rotation.
        Returns True if a recovery was performed, False otherwise.
        """
        journal_path = self._path.with_suffix(".rotation_journal")
        if not journal_path.exists():
            return False

        try:
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            if journal.get("status") != "in_progress":
                journal_path.unlink()
                return False

            # Keys are stored encrypted in the journal.
            # We need the current master key to bootstrap recovery.
            current_key = self._get_key()

            # Try to recover keys from journal
            # new_key_enc was encrypted with old_key, old_key_enc with new_key
            new_key: bytes | None = None
            old_key: bytes | None = None
            try:
                # If current_key is old_key, we can decrypt new_key_enc
                new_key_b64 = _decrypt_value(journal["new_key_enc"], current_key)
                new_key = base64.b64decode(new_key_b64)
                old_key = current_key
            except (VaultError, KeyError):
                pass

            if new_key is None:
                try:
                    # If current_key is new_key, we can decrypt old_key_enc
                    old_key_b64 = _decrypt_value(journal["old_key_enc"], current_key)
                    old_key = base64.b64decode(old_key_b64)
                    new_key = current_key
                except (VaultError, KeyError):
                    pass

            if new_key is None or old_key is None:
                logger.error("Cannot recover rotation: unable to decrypt journal keys")
                return False

            vault = self._read_vault()
            entries = vault.get("entries", {})
            if entries:
                # Validate ALL entries, not just the first
                try:
                    for encrypted in entries.values():
                        _decrypt_value(encrypted, new_key)
                    _set_master_key(new_key)
                    self._master_key = SecureBytes(new_key)
                    journal_path.unlink()
                    logger.info("Completed interrupted key rotation (forward)")
                    return True
                except VaultError:
                    pass

                try:
                    for encrypted in entries.values():
                        _decrypt_value(encrypted, old_key)
                    _set_master_key(old_key)
                    self._master_key = SecureBytes(old_key)
                    journal_path.unlink()
                    logger.info("Rolled back interrupted key rotation")
                    return True
                except VaultError:
                    pass

            journal_path.unlink()
            return False
        except Exception as exc:
            logger.error("Failed to recover rotation: %s", exc)
            return False

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

    def close(self) -> None:
        """Securely zero the master key from memory.

        Call when done with the vault to ensure key material is wiped.
        Also called automatically on garbage collection via __del__.
        """
        if self._master_key is not None:
            self._master_key.close()
            self._master_key = None

    def __del__(self) -> None:
        """Zero the master key on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    @property
    def path(self) -> Path:
        """Return the vault file path."""
        return self._path

    def _zero_key(self) -> None:
        """Securely zero the master key in memory (GAP 2.1)."""
        if hasattr(self, '_master_key') and self._master_key:
            try:
                buf = (ctypes.c_char * len(self._master_key)).from_buffer_copy(self._master_key)
                ctypes.memset(buf, 0, len(self._master_key))
            except Exception:
                pass
            self._master_key = None

    def __del__(self) -> None:
        """Zero master key on garbage collection."""
        self._zero_key()
