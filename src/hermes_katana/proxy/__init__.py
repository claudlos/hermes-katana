"""
HermesKatana Proxy Module - Hardened MITM proxy for LLM traffic.

Provides transparent interception of LLM API traffic with:
- Multi-layer request/response scanning (secrets, injection, content, unicode)
- Automatic API key injection from vault (12+ providers)
- Thread-safe rate limiting with anomaly escalation
- Response-body scanning for indirect prompt injection
- Domain allowlist enforcement

Usage:
    from hermes_katana.proxy import KatanaProxy, ProxyConfig

    config = ProxyConfig(port=8443)
    proxy = KatanaProxy(config)
    pid = proxy.start()
"""

from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.proxy.runner import KatanaProxy, default_pid_path

__all__ = [
    "KatanaProxy",
    "ProxyConfig",
    "default_pid_path",
]
