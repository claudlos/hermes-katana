"""
Central configuration for HermesKatana.

Provides a single ``KatanaConfig`` Pydantic model that controls all aspects
of the security middleware: policy selection, scanning, proxy, vault, audit,
taint tracking, and more.

Configuration is loaded from ``~/.hermes-katana/config.yaml`` with fallback
to built-in defaults.  Environment variables prefixed with ``KATANA_`` can
override any setting (e.g. ``KATANA_LOG_LEVEL=DEBUG``).

Usage::

    from hermes_katana.config import load_config, KatanaConfig

    # Load from default location (or create defaults)
    config = load_config()

    # Access settings
    print(config.policy_preset)     # "balanced"
    print(config.proxy_port)        # 8443

    # Modify and save
    config.policy_preset = "paranoid"
    config.save()
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator

from hermes_katana._paths import home_or_fallback

__all__ = [
    "KatanaConfig",
    "load_config",
    "get_default_config",
    "config_path",
    "config_dir",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

# These are computed at import time so tests can monkeypatch them directly.
# On Windows, Path.home() can raise RuntimeError when USERPROFILE/HOMEDRIVE
# are unset (sandboxed envs, restricted service accounts). home_or_fallback()
# routes to a user-scoped tempdir in that case so module import cannot crash.
# In production the fallback branch should never fire — home is resolvable
# there and safe_home() emits a one-time warning if it isn't.
_CONFIG_DIR = home_or_fallback() / ".hermes-katana"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"
_VALID_PRESETS = {"paranoid", "balanced", "permissive"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


class KatanaConfig(BaseModel):
    """Central configuration for HermesKatana security middleware.

    All fields have sensible defaults.  Override via config file,
    environment variables (``KATANA_`` prefix), or programmatically.

    Attributes
    ----------
    policy_preset : str
        Built-in policy set to use.  One of: ``paranoid``, ``balanced``,
        ``permissive``.  Ignored when ``policy_path`` is set.
    policy_path : Optional[Path]
        Path to a custom policy YAML file.  When set, this takes priority
        over ``policy_preset``.
    scan_inputs : bool
        Enable input scanning (injection detection, secret leaks).
    scan_outputs : bool
        Enable output scanning (content analysis, taint propagation).
    scan_commands : bool
        Enable command scanning for terminal tool calls.
    proxy_enabled : bool
        Enable the mitmproxy-based network interception proxy.
    proxy_port : int
        Port for the interception proxy to listen on.
    vault_enabled : bool
        Enable the secret vault for credential management.
    audit_enabled : bool
        Enable structured audit logging for all policy decisions.
    audit_max_size_mb : int
        Maximum size of audit log files before rotation (in MB).
    taint_tracking : bool
        Enable taint tracking and information-flow control.
    strict_mode : bool
        When True, treat CONDITIONAL taint as UNTRUSTED (deny instead
        of escalate).  Recommended for high-security environments.
    domain_allowlist : list[str]
        List of domains the proxy should allow without interception.
        Useful for internal services and trusted APIs.
    log_level : str
        Logging level.  One of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """

    policy_preset: str = Field(
        default="balanced",
        description=("Built-in policy set. One of: paranoid, balanced, permissive. Ignored when policy_path is set."),
    )
    policy_path: Optional[Path] = Field(
        default=None,
        description=("Path to a custom policy YAML file. Takes priority over policy_preset."),
    )
    scan_inputs: bool = Field(
        default=True,
        description="Enable input scanning (injection detection, secret leaks).",
    )
    scan_outputs: bool = Field(
        default=True,
        description="Enable output scanning (content analysis, taint propagation).",
    )
    scan_commands: bool = Field(
        default=True,
        description="Enable command scanning for terminal tool calls.",
    )
    proxy_enabled: bool = Field(
        default=True,
        description="Enable the mitmproxy-based network interception proxy.",
    )
    proxy_port: int = Field(
        default=8443,
        ge=1,
        le=65535,
        description="Port for the interception proxy.",
    )
    vault_enabled: bool = Field(
        default=True,
        description="Enable the secret vault for credential management.",
    )
    audit_enabled: bool = Field(
        default=True,
        description="Enable structured audit logging for all policy decisions.",
    )
    audit_max_size_mb: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum audit log file size before rotation (MB).",
    )
    taint_tracking: bool = Field(
        default=True,
        description="Enable taint tracking and information-flow control.",
    )
    strict_mode: bool = Field(
        default=False,
        description=(
            "When True, treat CONDITIONAL taint as UNTRUSTED "
            "(deny instead of escalate). For high-security environments."
        ),
    )
    domain_allowlist: list[str] = Field(
        default_factory=list,
        description=("Domains the proxy should allow without interception. Example: ['api.internal.com', 'localhost']"),
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
    )

    model_config = {"extra": "ignore"}

    # -- Validators -----------------------------------------------------------

    @field_validator("policy_preset")
    @classmethod
    def _validate_preset(cls, v: str) -> str:
        """Ensure policy_preset is a known built-in set."""
        v = v.lower().strip()
        if v not in _VALID_PRESETS:
            raise ValueError(f"Invalid policy_preset '{v}'. Must be one of: {', '.join(sorted(_VALID_PRESETS))}")
        return v

    @field_validator("policy_path")
    @classmethod
    def _validate_policy_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate and sanitize the custom policy path.

        Rejects path traversal attempts (.. components) and ensures the
        resolved path doesn't escape expected directories.
        """
        if v is not None:
            raw = str(v)
            # Reject obvious path traversal patterns
            if ".." in raw.replace("\\", "/").split("/"):
                raise ValueError(
                    f"Path traversal detected in policy_path: '{raw}'. Relative '..' components are not allowed."
                )
            v = Path(v).expanduser().resolve()
            # Ensure it's a file path, not a directory
            if v.exists() and v.is_dir():
                raise ValueError(f"policy_path must be a file, not a directory: {v}")
            if not v.exists():
                logger.warning(
                    "Custom policy path does not exist: %s — will fall back to policy_preset",
                    v,
                )
        return v

    @field_validator("domain_allowlist")
    @classmethod
    def _validate_domain_allowlist(cls, v: list[str]) -> list[str]:
        """Validate domain allowlist entries.

        Rejects empty strings, entries with path components, and
        entries that look like full URLs rather than domain names.
        """
        validated = []
        for domain in v:
            domain = domain.strip().lower()
            if not domain:
                continue
            # Reject URLs masquerading as domains
            if "://" in domain:
                raise ValueError(
                    f"domain_allowlist entry looks like a URL, not a domain: '{domain}'. "
                    "Use bare domain names like 'api.example.com'."
                )
            # Reject path components
            if "/" in domain:
                raise ValueError(
                    f"domain_allowlist entry contains path component: '{domain}'. Use bare domain names only."
                )
            # Reject wildcard abuse (only leading *. is acceptable)
            if "*" in domain and not domain.startswith("*."):
                raise ValueError(
                    f"Invalid wildcard in domain_allowlist: '{domain}'. Only '*.example.com' patterns are allowed."
                )
            validated.append(domain)
        return validated

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a valid Python logging level."""
        v = v.upper().strip()
        if v not in _VALID_LOG_LEVELS:
            raise ValueError(f"Invalid log_level '{v}'. Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}")
        return v

    @field_validator("policy_path", mode="before")
    @classmethod
    def _validate_path_length(cls, v: Any) -> Any:
        """Reject excessively long paths."""
        if v is not None:
            raw = str(v)
            if len(raw) > 4096:
                raise ValueError(f"Path value too long ({len(raw)} chars, max 4096)")
        return v

    @field_validator("policy_preset", "log_level", mode="before")
    @classmethod
    def _validate_string_safety(cls, v: Any) -> Any:
        """Reject values with invalid or dangerous characters."""
        if isinstance(v, str):
            if len(v) > 256:
                raise ValueError(f"Config value too long ({len(v)} chars, max 256)")
            for ch in v:
                if ord(ch) < 32 and ch not in ("\n", "\t", "\r"):
                    raise ValueError(f"Config value contains invalid control character: U+{ord(ch):04X}")
        return v

    # -- Persistence ----------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the current configuration to a YAML file.

        Args:
            path: Destination file path.  Defaults to
                  ``~/.hermes-katana/config.yaml``.

        Returns:
            The path that was written to.
        """
        filepath = Path(path) if path else _CONFIG_FILE
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Serialize to dict, converting Path to string for YAML
        data = self.model_dump(mode="json", exclude_none=True)
        if data.get("policy_path") is not None:
            data["policy_path"] = str(data["policy_path"])

        # Write with header comment
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(
                "# HermesKatana Configuration\n"
                "# Generated by hermes-katana. Edit as needed.\n"
                "# Documentation: https://github.com/HermesKatana/hermes-katana\n"
                "\n"
            )
            yaml.safe_dump(
                data,
                fh,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                width=100,
            )

        logger.info("Configuration saved to %s", filepath)
        return filepath

    # -- Helpers --------------------------------------------------------------

    def effective_policy_path(self) -> Optional[Path]:
        """Return the policy file path to load, or None for built-in preset.

        If ``policy_path`` is set and the file exists, return it.
        Otherwise return None (caller should use ``policy_preset``).
        """
        if self.policy_path and self.policy_path.exists():
            return self.policy_path
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for display or export."""
        return self.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides with ``KATANA_`` prefix.

    Environment variables are mapped to config fields by lowering the
    suffix: ``KATANA_LOG_LEVEL`` → ``log_level``.

    Boolean fields accept ``true/false/1/0/yes/no``.
    List fields accept comma-separated values.
    """
    prefix = "KATANA_"
    _BOOL_TRUE = {"true", "1", "yes", "on"}
    _BOOL_FALSE = {"false", "0", "no", "off"}

    # Build a set of field names for lookup
    field_names = set(KatanaConfig.model_fields.keys())

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        field_name = key[len(prefix) :].lower()
        if field_name not in field_names:
            continue

        field_info = KatanaConfig.model_fields[field_name]
        annotation = field_info.annotation

        # Type coercion
        if annotation is bool or annotation == Optional[bool]:
            if value.lower() in _BOOL_TRUE:
                data[field_name] = True
            elif value.lower() in _BOOL_FALSE:
                data[field_name] = False
        elif annotation is int or annotation == Optional[int]:
            try:
                data[field_name] = int(value)
            except ValueError:
                logger.warning("Invalid integer for %s: %s — skipping", key, value)
                continue
        elif annotation == list[str]:
            data[field_name] = [s.strip() for s in value.split(",") if s.strip()]
        else:
            data[field_name] = value

        logger.debug("Config override from env: %s = %r", field_name, data[field_name])

    return data


def load_config(path: Optional[Path] = None) -> KatanaConfig:
    """Load configuration from a YAML file with fallback to defaults.

    Lookup order:
    1. Explicit ``path`` argument (if provided)
    2. ``~/.hermes-katana/config.yaml`` (if it exists)
    3. Built-in defaults

    Environment variables with ``KATANA_`` prefix override file values.

    Args:
        path: Optional explicit path to a config YAML file.

    Returns:
        A validated ``KatanaConfig`` instance.
    """
    data: dict[str, Any] = {}
    config_path = Path(path) if path else _CONFIG_FILE

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
            if isinstance(raw, dict):
                data = raw
                logger.info("Loaded configuration from %s", config_path)
            else:
                logger.warning(
                    "Config file %s is not a mapping — using defaults",
                    config_path,
                )
        except (yaml.YAMLError, OSError) as exc:
            logger.warning(
                "Failed to load config from %s: %s — using defaults",
                config_path,
                exc,
            )
    else:
        logger.debug(
            "Config file not found at %s — using built-in defaults",
            config_path,
        )

    # Apply environment variable overrides
    data = _apply_env_overrides(data)

    # Validate and return
    try:
        config = KatanaConfig.model_validate(data)
    except Exception as exc:
        logger.error("Configuration validation failed: %s — falling back to defaults", exc)
        config = KatanaConfig()

    return config


def get_default_config() -> KatanaConfig:
    """Return a ``KatanaConfig`` with all defaults (no file loading)."""
    return KatanaConfig()


def config_path() -> Path:
    """Return the default HermesKatana config file path."""
    return _CONFIG_FILE


def config_dir() -> Path:
    """Return the HermesKatana configuration directory path.

    Creates the directory if it doesn't exist.
    """
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return _CONFIG_DIR
