#!/usr/bin/env python3
"""Vault usage example — encrypted secret storage with integrity checks.

Demonstrates:
  - Creating a vault and storing secrets
  - Retrieving secrets
  - Listing keys
  - Key rotation
  - Integrity verification

Run:  python3 examples/vault_usage.py
"""

import base64 as _b64
import os
import secrets as _secrets
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hermes_katana.vault import Vault

# Use a temp directory so we don't pollute real vault
vault_dir = tempfile.mkdtemp(prefix="katana_vault_demo_")

# Set a demo master key (in production, use OS keyring or a secure key manager)
os.environ["HERMES_KATANA_VAULT_KEY"] = _b64.b64encode(_secrets.token_bytes(32)).decode()

vault = Vault(path=Path(vault_dir) / "demo.vault", auto_create=True)

# 1. Store secrets
print("=== Storing Secrets ===")
vault.set("api_key", "sk-ant-demo-1234567890abcdef")
vault.set("db_password", "super_secret_p@ssw0rd!")
vault.set("webhook_url", "https://hooks.example.com/abc123")
print(f"  Stored {len(vault.list_keys())} secrets: {vault.list_keys()}")

# 2. Retrieve a secret
print("\n=== Retrieving Secrets ===")
key = vault.get("api_key")
print(f"  api_key = {key[:10]}...{key[-4:]}")

# 3. Verify integrity (hash chain is intact)
print("\n=== Integrity Verification ===")
ok = vault.verify_integrity()
print(f"  Integrity check: {'PASS' if ok else 'FAIL'}")

# 4. Key rotation
print("\n=== Key Rotation ===")
vault.rotate_key()
print("  Master key rotated — all secrets re-encrypted")
val = vault.get("db_password")
print(f"  db_password still accessible: {val[:6]}...")

# Cleanup
vault.close()
shutil.rmtree(vault_dir, ignore_errors=True)
print("\n  (temp vault cleaned up)")
