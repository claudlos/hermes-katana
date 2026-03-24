"""
HermesKatana Vault Module - AES-256-GCM encrypted secret storage.

Provides hardened secret storage with:
- AES-256-GCM encryption (upgraded from Fernet/AES-128-CBC)
- Master key in OS keyring (platform-native credential storage)
- Individual value encryption with per-value nonces
- HMAC integrity verification over all entries
- Key rotation mechanism (re-encrypt all values with new key)
- Circuit breaker (vault.lock sentinel file)
- Atomic writes via temp file + rename
- Secret migration from env vars, config files, .env files

Usage:
    from hermes_katana.vault import Vault, VaultError

    vault = Vault()
    vault.set("OPENAI_API_KEY", "sk-...")
    key = vault.get("OPENAI_API_KEY")
    vault.rotate_key()
"""

from hermes_katana.vault.store import Vault, VaultError

__all__ = [
    "Vault",
    "VaultError",
]
