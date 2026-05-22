"""
Proxy configuration model for HermesKatana.

Defines all configurable parameters for the MITM proxy including
network settings, scan modes, and rate limiting.
"""

from __future__ import annotations

__all__ = [
    "ScanModes",
    "ProxyConfig",
]


from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# 1 MB default body scan limit
_DEFAULT_MAX_BODY_SCAN_SIZE = 1_048_576

ProxyMode = Literal["permissive", "strict", "max"]


class ScanModes(BaseModel):
    """Which scanning modes are active on the proxy."""

    secrets: bool = Field(
        default=True,
        description="Scan for secret/credential leakage in request and response bodies.",
    )
    injection: bool = Field(
        default=True,
        description="Scan for prompt injection payloads in requests and responses.",
    )
    content: bool = Field(
        default=True,
        description="Scan for content attacks (homograph URLs, ANSI, markdown, HTML).",
    )
    unicode: bool = Field(
        default=True,
        description="Scan for malicious Unicode (bidi overrides, zero-width, homoglyphs).",
    )

    model_config = {"frozen": False, "extra": "forbid"}


class ProxyConfig(BaseModel):
    """Configuration for the KatanaProxy MITM proxy.

    Attributes:
        port: Listening port for the proxy (default: 8443).
        host: Bind address (default: 127.0.0.1 - localhost only).
        ignore_hosts: Hostnames to pass through without scanning
            (e.g., sigstore/TUF update hosts).
        tls_verify: Whether to verify upstream TLS certificates.
        max_body_scan_size: Maximum body size (bytes) to scan. Bodies
            larger than this are passed through without scanning.
        scan_modes: Which scan modules are active.
        rate_limit_requests: Max requests per rate_limit_window.
        rate_limit_window: Time window for rate limiting (seconds).
        allowed_domains: If non-empty, only these domains are permitted.
            Empty list means all domains are allowed.
        inject_credentials: Whether to inject API keys from the vault.
        health_check_port: Port for the health check HTTP endpoint.
    """

    port: int = Field(
        default=8443,
        ge=1,
        le=65535,
        description="Proxy listening port.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Proxy bind address. Use 127.0.0.1 for local-only.",
    )
    ignore_hosts: list[str] = Field(
        default_factory=lambda: [
            # Sigstore / TUF update infrastructure
            "rekor.sigstore.dev",
            "fulcio.sigstore.dev",
            "tuf-repo-cdn.sigstore.dev",
            # OS update / package manager hosts
            "pypi.org",
            "files.pythonhosted.org",
        ],
        description="Hosts to pass through without scanning.",
    )
    tls_verify: bool = Field(
        default=True,
        description="Verify upstream TLS certificates.",
    )
    max_body_scan_size: int = Field(
        default=_DEFAULT_MAX_BODY_SCAN_SIZE,
        ge=0,
        description="Maximum body size in bytes to scan (0 = unlimited).",
    )
    scan_modes: ScanModes = Field(
        default_factory=ScanModes,
        description="Active scanning modes.",
    )
    scanner_security_level: Literal["low", "medium", "high"] = Field(
        default=None,  # type: ignore[assignment]
        description=(
            "Security level passed to scanner.scan_input for proxy request surfaces. "
            "When omitted, the level follows proxy mode: permissive=low, "
            "strict=medium, max=high."
        ),
    )
    rate_limit_requests: int = Field(
        default=50,
        ge=1,
        description="Maximum requests per rate_limit_window.",
    )
    rate_limit_window: float = Field(
        default=1.0,
        gt=0.0,
        description="Rate limit window in seconds.",
    )
    allowed_domains: list[str] = Field(
        default_factory=list,
        description="If non-empty, only allow traffic to these domains.",
    )
    inject_credentials: bool = Field(
        default=True,
        description="Inject API keys from vault for known LLM providers.",
    )
    health_check_port: Optional[int] = Field(
        default=None,
        ge=1,
        le=65535,
        description="Port for health check HTTP endpoint (None = disabled).",
    )
    max_request_body_size: int = Field(
        default=10_485_760,  # 10 MB
        ge=0,
        description="Maximum request body size in bytes (0 = unlimited).",
    )
    max_response_body_size: int = Field(
        default=52_428_800,  # 50 MB
        ge=0,
        description="Maximum response body size in bytes (0 = unlimited).",
    )
    graceful_shutdown_timeout: float = Field(
        default=10.0,
        ge=0.0,
        description="Seconds to wait for in-flight requests during shutdown.",
    )
    add_scanned_header: bool = Field(
        default=False,
        description="Inject X-Katana-Scanned header on responses (opt-in, disabled by default).",
    )
    mode: ProxyMode = Field(
        default="strict",
        description=(
            "Failure-handling mode. 'strict' and 'max' fail closed on scanner "
            "exceptions and oversized bodies; 'permissive' logs and allows traffic. "
            "Mirrors the policy engine's tri-mode model."
        ),
    )

    model_config = {"frozen": False, "extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def _default_scanner_security_level(cls, data: object) -> object:
        """Derive scanner strength from proxy fail-closed mode unless explicit."""
        if isinstance(data, dict) and data.get("scanner_security_level") is None:
            mode = data.get("mode", "strict")
            data = dict(data)
            data["scanner_security_level"] = {
                "permissive": "low",
                "strict": "medium",
                "max": "high",
            }.get(mode, "medium")
        return data

    @field_validator("host")
    @classmethod
    def _validate_host(cls, v: str) -> str:
        """Ensure host is a valid bind address."""
        v = v.strip()
        if not v:
            raise ValueError("host must not be empty")
        return v

    @field_validator("ignore_hosts")
    @classmethod
    def _validate_ignore_hosts(cls, v: list[str]) -> list[str]:
        """Normalize ignore_hosts to lowercase."""
        return [h.strip().lower() for h in v if h.strip()]

    @field_validator("allowed_domains")
    @classmethod
    def _validate_allowed_domains(cls, v: list[str]) -> list[str]:
        """Normalize allowed_domains to lowercase."""
        return [d.strip().lower() for d in v if d.strip()]
