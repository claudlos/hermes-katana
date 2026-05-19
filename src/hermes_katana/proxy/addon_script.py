"""
mitmproxy bootstrap script for HermesKatana.

This file is loaded by ``mitmdump -s`` and instantiates the real
``KatanaAddon`` using configuration passed through the environment.
"""

from __future__ import annotations

__all__ = ["addons"]

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from hermes_katana.proxy.addon import KatanaAddon
from hermes_katana.proxy.config import ProxyConfig

if TYPE_CHECKING:
    from hermes_katana.audit import AuditTrail
    from hermes_katana.vault import Vault

logger = logging.getLogger(__name__)


def _load_config() -> ProxyConfig:
    """Load proxy configuration from environment."""
    raw = os.environ.get("KATANA_PROXY_CONFIG_JSON")
    if raw:
        try:
            return ProxyConfig.model_validate_json(raw)
        except Exception:
            logger.warning("Invalid proxy config payload; falling back to defaults", exc_info=True)
    return ProxyConfig()


def _load_vault() -> Vault | None:
    """Load the vault backend for credential injection."""
    if os.environ.get("KATANA_PROXY_ENABLE_VAULT", "1") != "1":
        return None

    try:
        from hermes_katana.vault import Vault

        vault_path = os.environ.get("KATANA_PROXY_VAULT_PATH")
        if vault_path:
            return Vault(path=Path(vault_path), auto_create=False)
        return Vault(auto_create=False)
    except Exception:
        logger.debug("Proxy addon could not load vault backend", exc_info=True)
        return None


def _load_audit_trail() -> AuditTrail | None:
    """Load the audit trail backend for event logging."""
    if os.environ.get("KATANA_PROXY_ENABLE_AUDIT", "1") != "1":
        return None

    try:
        from hermes_katana.audit import AuditTrail

        audit_path = os.environ.get("KATANA_PROXY_AUDIT_PATH")
        return AuditTrail(path=Path(audit_path) if audit_path else None)
    except Exception:
        logger.debug("Proxy addon could not load audit trail", exc_info=True)
        return None


addons = [
    KatanaAddon(
        config=_load_config(),
        vault=_load_vault(),
        audit=_load_audit_trail(),
    )
]
