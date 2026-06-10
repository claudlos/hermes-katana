"""
AES-256-GCM encrypted vault for HermesKatana.

Hardened from hermes-aegis Fernet (AES-128-CBC) vault with:
- AES-256-GCM authenticated encryption (256-bit key, 96-bit nonce, 128-bit tag)
- Per-value random nonces (no nonce reuse)
- HMAC-SHA256 integrity check over all entries
- Master key stored in OS keyring (not on disk)
- Circuit breaker via vault.lock scabbard file
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
import threading
import time
from pathlib import Path
from typing import Any, Optional

from hermes_katana._files import AdvisoryFileLock, atomic_write_text

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

# Vault file format version.
# v3 (audit hardening B6, 2026-06-09): HKDF-derived HMAC subkey, a write
# counter bound into the HMAC message, and per-entry AES-GCM AAD (the key
# name) on newly written values. v2 files verify via the legacy HMAC and
# upgrade in place on the next write; legacy (no-AAD) entry blobs remain
# decryptable until rewritten.
_VAULT_VERSION = 3

# Magic prefix inside the base64 blob marking an AAD-bound (v3) entry.
_AAD_MAGIC = b"HKV3"

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


def _process_is_running(pid: int) -> bool:
    """Return True when *pid* appears to be a live process."""
    if pid <= 0:
        return False

    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False

        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


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


def _encrypt_value(plaintext: str, key: bytes, *, aad: Optional[str] = None) -> str:
    """Encrypt a plaintext string with AES-256-GCM.

    Returns a base64-encoded string: nonce || ciphertext || tag.

    Args:
        plaintext: The value to encrypt.
        key: 32-byte AES-256 key.
        aad: Optional associated data bound into the GCM tag (the vault key
            name). Binding the name prevents an attacker who can edit the
            vault file from swapping ciphertexts between entries (audit
            hardening B6). When set, the blob carries the HKV3 magic prefix.

    Returns:
        Base64-encoded encrypted blob.
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise VaultError("cryptography package required: pip install cryptography")

    nonce = secrets.token_bytes(_NONCE_SIZE)
    aesgcm = AESGCM(key)
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad.encode("utf-8") if aad else None)
    # ct includes the tag appended by AESGCM
    blob = (_AAD_MAGIC + nonce + ct) if aad else (nonce + ct)
    return base64.b64encode(blob).decode("ascii")


def _decrypt_value(encrypted: str, key: bytes, *, aad: Optional[str] = None) -> str:
    """Decrypt an AES-256-GCM encrypted value.

    Args:
        encrypted: Base64-encoded encrypted blob. v3 blobs are
            ``HKV3 || nonce || ciphertext || tag`` and authenticate *aad*;
            legacy blobs are ``nonce || ciphertext || tag`` with no AAD and
            remain decryptable regardless of *aad* (pre-v3 compatibility).
        key: 32-byte AES-256 key.
        aad: Associated data the v3 blob was bound to (the vault key name).

    Returns:
        Decrypted plaintext string.

    Raises:
        VaultError: On decryption failure (wrong key, tampered data, or an
            entry moved to a different key name).
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise VaultError("cryptography package required: pip install cryptography")

    try:
        blob = base64.b64decode(encrypted)
    except Exception:
        raise VaultError("Invalid encrypted value: base64 decode failed")

    associated: Optional[bytes] = None
    if blob.startswith(_AAD_MAGIC):
        blob = blob[len(_AAD_MAGIC) :]
        associated = aad.encode("utf-8") if aad else b""

    if len(blob) < _NONCE_SIZE + _TAG_SIZE:
        raise VaultError("Invalid encrypted value: too short")

    nonce = blob[:_NONCE_SIZE]
    ct = blob[_NONCE_SIZE:]

    try:
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ct, associated)
        return plaintext.decode("utf-8")
    except Exception:
        raise VaultError("Decryption failed: wrong key, tampered data, or relocated entry")


def _derive_hmac_key(master_key: bytes) -> bytes:
    """Derive the v3 vault HMAC subkey from the master key via HKDF."""
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError:
        raise VaultError("cryptography package required: pip install cryptography")

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"hermes-katana-vault",
        info=b"vault-hmac:v3",
    ).derive(master_key)


def _compute_hmac(data: dict[str, str], key: bytes, *, counter: Optional[int] = None) -> str:
    """Compute HMAC-SHA256 over vault entries for integrity checking.

    The HMAC is computed over sorted key-value pairs to ensure
    deterministic output regardless of dict ordering.

    Args:
        data: The encrypted vault entries.
        key: The master key (used to derive HMAC key).
        counter: v3 write counter. When given, the subkey is HKDF-derived
            and the counter is bound into the authenticated message, so an
            attacker cannot tamper with the counter or transplant an HMAC
            between writes (audit hardening B6). When None, the legacy
            (v<=2) construction is used for verifying old vault files.

    Returns:
        Hex-encoded HMAC digest.
    """
    if counter is None:
        # Legacy (v<=2) construction — verification of existing files only.
        hmac_key = hashlib.sha256(b"hmac:" + key).digest()
        msg = json.dumps(sorted(data.items()), sort_keys=True).encode()
        return hmac.new(hmac_key, msg, hashlib.sha256).hexdigest()

    hmac_key = _derive_hmac_key(key)
    # The literal 3 (not _VAULT_VERSION) so v3 files stay verifiable after
    # any future format bump.
    msg = json.dumps(
        {"version": 3, "counter": counter, "entries": sorted(data.items())},
        sort_keys=True,
    ).encode()
    return hmac.new(hmac_key, msg, hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Keyring operations
# ---------------------------------------------------------------------------


# In-process cache for an env-provided master key (see _get_master_key).
_ENV_KEY_CACHE: Optional[bytes] = None


def _get_master_key() -> Optional[bytes]:
    """Retrieve the master key from the OS keyring.

    Returns:
        The 32-byte master key, or None if not found.
    """

    def _validate_key(key_bytes: bytes) -> bytes:
        if len(key_bytes) != _KEY_SIZE:
            raise VaultError(f"Master key must be {_KEY_SIZE} bytes, got {len(key_bytes)}")
        return key_bytes

    try:
        import keyring

        raw = keyring.get_password(_KEYRING_SERVICE, _KEYRING_ACCOUNT)
        if raw is None:
            # Keyring exists but has no stored key — fall through to env var
            raise KeyError("no key in keyring")
        return _validate_key(base64.b64decode(raw))
    except ImportError:
        logger.warning("keyring package not installed, falling back to environment variable")
    except VaultError:
        raise
    except Exception as exc:
        logger.warning("Failed to read keyring: %s, falling back to environment variable", exc)

    # Fallback: check environment variable. The var is popped so the key is
    # not left visible to child processes (GAP 2.3), but the value is cached
    # in-process — consuming it outright made every later call return None,
    # breaking rotation rollback and second readers (audit finding B5).
    global _ENV_KEY_CACHE
    raw_env = os.environ.pop("HERMES_KATANA_VAULT_KEY", None)
    if raw_env:
        key = _validate_key(base64.b64decode(raw_env))
        _ENV_KEY_CACHE = key
        return key
    if _ENV_KEY_CACHE is not None:
        return _ENV_KEY_CACHE
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
    global _ENV_KEY_CACHE
    _ENV_KEY_CACHE = None
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
    from hermes_katana._paths import home_or_fallback

    return home_or_fallback() / ".config" / "hermes-katana" / "vault.json"


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
        self._file_lock = AdvisoryFileLock(self._path)
        self._lock_state_guard = AdvisoryFileLock(self._lock_path, suffix=".guard")
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
                "No master key found. Initialize vault or set HERMES_KATANA_VAULT_KEY environment variable."
            )
        return self._master_key.raw

    def _zero_key(self) -> None:
        """Securely clear the in-memory master key."""
        if self._master_key is not None:
            self._master_key.close()
        self._master_key = None

    def _owner_pid(self) -> int:
        """Return this process ID for lock ownership metadata."""
        return os.getpid()

    def _owner_is_running(self, pid: int) -> bool:
        """Return True when a lock owner process appears to be live."""
        return _process_is_running(pid)

    def _read_lock_state_unlocked(self) -> dict[str, Any] | None:
        """Read the lock metadata while holding ``self._lock_state_guard``."""
        if not self._lock_path.exists():
            return None

        try:
            raw = self._lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            return {"path": str(self._lock_path), "status": "present"}

        if not raw:
            return {"path": str(self._lock_path), "status": "legacy"}

        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return {"path": str(self._lock_path), "status": "legacy"}

        if isinstance(state, dict):
            state.setdefault("path", str(self._lock_path))
            return state
        return {"path": str(self._lock_path), "status": "legacy"}

    def _current_lock_state(self) -> dict[str, Any] | None:
        """Return the active lock state, removing stale process-owned locks."""
        with self._lock_state_guard:
            state = self._read_lock_state_unlocked()
            if state is None:
                return None

            pid = state.get("pid")
            if isinstance(pid, int) and pid > 0 and not self._owner_is_running(pid):
                self._lock_path.unlink(missing_ok=True)
                logger.warning("Removed stale vault lock from dead PID %s", pid)
                return None

            return state

    def _check_lock(self) -> None:
        """Check if the vault is locked (circuit breaker)."""
        state = self._current_lock_state()
        if state is None:
            return

        owner = state.get("pid")
        reason = state.get("reason", "circuit_breaker")
        detail = f" pid={owner}" if isinstance(owner, int) else ""
        raise VaultLockedError(
            f"Vault is locked (circuit breaker active, reason={reason!r}{detail}). "
            f"Use Vault.unlock() to clear {self._lock_path}."
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
            if not isinstance(data, dict):
                raise VaultError("Corrupt vault file: root JSON value must be an object.")
            return data
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise VaultError(f"Corrupt vault file: {exc}")

    def _next_write_counter(self) -> int:
        """Return the successor of the current vault write counter."""
        try:
            current = self._read_vault().get("counter")
        except VaultError:
            current = None
        if not isinstance(current, int) or current < 0:
            current = 0
        return current + 1

    def _write_vault(self, entries: dict[str, str]) -> None:
        """Atomically write the vault file.

        Uses temp file + rename pattern to prevent partial writes. Every
        write bumps the authenticated counter, so external monitoring can
        detect a vault file rolled back to an earlier (validly MAC'd) state.

        Args:
            entries: The encrypted entries dict.
        """
        key = self._get_key()
        counter = self._next_write_counter()
        hmac_digest = _compute_hmac(entries, key, counter=counter)

        vault_data = {
            "version": _VAULT_VERSION,
            "counter": counter,
            "entries": entries,
            "hmac": hmac_digest,
        }

        # Write to temp file first, then atomic replace — with file lock
        # for multi-process safety
        try:
            with self._file_lock:
                atomic_write_text(self._path, json.dumps(vault_data, indent=2), mode=0o600)
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
            if not isinstance(entries, dict):
                raise VaultError("Corrupt vault file: entries must be a dictionary.")

            # Verify HMAC integrity before returning any data
            failure = self._integrity_failure(vault, master_key)
            if failure:
                raise VaultIntegrityError(f"Vault integrity check failed: {failure}")

            if key not in entries:
                raise VaultKeyError(f"Key not found: {key}")

            return _decrypt_value(entries[key], master_key, aad=key)

    @staticmethod
    def _integrity_failure(vault: dict[str, Any], master_key: bytes) -> Optional[str]:
        """Return a failure description for the vault HMAC, or None if intact.

        v3 files (counter present) verify with the HKDF subkey and the
        counter bound into the message; older files verify with the legacy
        construction and upgrade on their next write.
        """
        entries = vault.get("entries", {})
        stored_hmac = vault.get("hmac", "")
        if not entries:
            return None
        if not stored_hmac:
            return "HMAC missing on non-empty vault. The vault file may have been tampered with."

        version = vault.get("version")
        counter = vault.get("counter")
        if isinstance(version, int) and version >= 3 and isinstance(counter, int):
            expected = _compute_hmac(entries, master_key, counter=counter)
        else:
            expected = _compute_hmac(entries, master_key)
        if not hmac.compare_digest(stored_hmac, expected):
            return "HMAC mismatch. The vault file may have been tampered with."
        return None

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
            if not isinstance(entries, dict):
                raise VaultError("Corrupt vault file: entries must be a dictionary.")
            entries[key] = _encrypt_value(value, master_key, aad=key)
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
            if not isinstance(entries, dict):
                raise VaultError("Corrupt vault file: entries must be a dictionary.")

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
            entries = vault.get("entries", {})
            if not isinstance(entries, dict):
                raise VaultError("Corrupt vault file: entries must be a dictionary.")
            return sorted(entries.keys())

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
            if not isinstance(entries, dict):
                raise VaultError("Corrupt vault file: entries must be a dictionary.")
            result: dict[str, str] = {}
            for k, encrypted in entries.items():
                try:
                    result[k] = _decrypt_value(encrypted, master_key, aad=k)
                except VaultError:
                    logger.warning("Failed to decrypt key: %s", k)
            return result

    def lock(self) -> None:
        """Activate the circuit breaker, locking the vault.

        Creates a scabbard file that prevents all vault operations
        until unlock() is called.
        """
        with self._rlock:
            with self._lock_state_guard:
                state = self._read_lock_state_unlocked()
                owner = state.get("pid") if state else None
                current_pid = self._owner_pid()
                if (
                    state is not None
                    and isinstance(owner, int)
                    and owner != current_pid
                    and self._owner_is_running(owner)
                ):
                    logger.warning("Vault already locked by PID %s", owner)
                    return

                lock_state = {
                    "version": 1,
                    "reason": "circuit_breaker",
                    "pid": current_pid,
                    "host": platform.node(),
                    "locked_at": time.time(),
                }
                atomic_write_text(self._lock_path, json.dumps(lock_state, indent=2), mode=0o600)
            logger.warning("Vault LOCKED (circuit breaker activated)")

    def unlock(self) -> None:
        """Deactivate the circuit breaker, unlocking the vault.

        Removes the scabbard file.
        """
        with self._rlock:
            with self._lock_state_guard:
                state = self._read_lock_state_unlocked()
                if state is None:
                    logger.debug("Vault was not locked")
                    return

                owner = state.get("pid")
                if isinstance(owner, int) and owner != self._owner_pid() and self._owner_is_running(owner):
                    raise VaultLockedError(
                        f"Vault lock is owned by live PID {owner}; refusing to clear {self._lock_path}."
                    )

                self._lock_path.unlink(missing_ok=True)
                logger.info("Vault unlocked")

    def is_locked(self) -> bool:
        """Check if the vault is locked."""
        return self._current_lock_state() is not None

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
                        base64.b64encode(old_key).decode("ascii"), new_key, aad="rotation-journal:old_key"
                    ),
                    "new_key_enc": _encrypt_value(
                        base64.b64encode(new_key).decode("ascii"), old_key, aad="rotation-journal:new_key"
                    ),
                    "timestamp": _time.time(),
                }
                atomic_write_text(journal_path, json.dumps(journal_data), mode=0o600)
            except Exception as exc:
                raise VaultError(f"Failed to write rotation journal: {exc}")

            # Read and decrypt all values with old key
            vault = self._read_vault()
            entries = vault.get("entries", {})
            decrypted: dict[str, str] = {}

            for k, encrypted in entries.items():
                try:
                    decrypted[k] = _decrypt_value(encrypted, old_key, aad=k)
                except VaultError as exc:
                    raise VaultError(f"Key rotation failed: could not decrypt '{k}': {exc}")

            # Re-encrypt all values with new key
            new_entries: dict[str, str] = {}
            for k, plaintext in decrypted.items():
                new_entries[k] = _encrypt_value(plaintext, new_key, aad=k)

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
                new_key_b64 = _decrypt_value(journal["new_key_enc"], current_key, aad="rotation-journal:new_key")
                new_key = base64.b64decode(new_key_b64)
                old_key = current_key
            except (VaultError, KeyError):
                pass

            if new_key is None:
                try:
                    # If current_key is new_key, we can decrypt old_key_enc
                    old_key_b64 = _decrypt_value(journal["old_key_enc"], current_key, aad="rotation-journal:old_key")
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
                    for k, encrypted in entries.items():
                        _decrypt_value(encrypted, new_key, aad=k)
                    _set_master_key(new_key)
                    self._master_key = SecureBytes(new_key)
                    journal_path.unlink()
                    logger.info("Completed interrupted key rotation (forward)")
                    return True
                except VaultError:
                    pass

                try:
                    for k, encrypted in entries.items():
                        _decrypt_value(encrypted, old_key, aad=k)
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

            return self._integrity_failure(vault, key) is None

    @property
    def write_counter(self) -> int:
        """Authenticated vault write counter (0 for pre-v3/empty vaults).

        Each successful write increments it, and the value is bound into the
        vault HMAC. External monitoring can record the last seen value and
        flag a decrease, which indicates the vault file was rolled back to
        an earlier (validly MAC'd) state.
        """
        try:
            counter = self._read_vault().get("counter")
        except VaultError:
            return 0
        return counter if isinstance(counter, int) and counter >= 0 else 0

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
