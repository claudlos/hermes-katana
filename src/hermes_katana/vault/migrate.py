"""
Secret discovery and migration for HermesKatana.

Scans multiple sources for secrets and migrates them into the vault:
1. Environment variables (highest priority)
2. Hermes config.yaml files
3. .env files (lowest priority)

After migration, source secrets can be securely deleted (overwritten
with zeros) to eliminate plaintext exposure on disk.

Priority order: env > hermes_config > dotenv
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from hermes_katana.vault.store import Vault

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known secret key patterns (env var names that are likely secrets)
# ---------------------------------------------------------------------------

_SECRET_KEY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r".*_API_KEY$", re.IGNORECASE),
    re.compile(r".*_API_TOKEN$", re.IGNORECASE),
    re.compile(r".*_SECRET_KEY$", re.IGNORECASE),
    re.compile(r".*_SECRET$", re.IGNORECASE),
    re.compile(r".*_PASSWORD$", re.IGNORECASE),
    re.compile(r".*_TOKEN$", re.IGNORECASE),
    re.compile(r".*_CREDENTIALS?$", re.IGNORECASE),
    re.compile(r".*_AUTH$", re.IGNORECASE),
    re.compile(r".*_ACCESS_KEY$", re.IGNORECASE),
    re.compile(r"^OPENAI_", re.IGNORECASE),
    re.compile(r"^ANTHROPIC_", re.IGNORECASE),
    re.compile(r"^GOOGLE_", re.IGNORECASE),
    re.compile(r"^GROQ_", re.IGNORECASE),
    re.compile(r"^TOGETHER_", re.IGNORECASE),
    re.compile(r"^OPENROUTER_", re.IGNORECASE),
    re.compile(r"^VERCEL_", re.IGNORECASE),
    re.compile(r"^DEEPSEEK_", re.IGNORECASE),
    re.compile(r"^MISTRAL_", re.IGNORECASE),
    re.compile(r"^COHERE_", re.IGNORECASE),
    re.compile(r"^REPLICATE_", re.IGNORECASE),
    re.compile(r"^HUGGINGFACE_", re.IGNORECASE),
    re.compile(r"^HF_", re.IGNORECASE),
    re.compile(r"^AWS_", re.IGNORECASE),
    re.compile(r"^AZURE_", re.IGNORECASE),
    re.compile(r"^GITHUB_TOKEN$", re.IGNORECASE),
    re.compile(r"^DATABASE_URL$", re.IGNORECASE),
    re.compile(r"^REDIS_URL$", re.IGNORECASE),
]

# Keys to always skip (not secrets)
_SKIP_KEYS: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "USER",
    "SHELL",
    "LANG",
    "TERM",
    "PWD",
    "OLDPWD",
    "HOSTNAME",
    "LOGNAME",
    "DISPLAY",
    "XDG_SESSION_TYPE",
    "XDG_RUNTIME_DIR",
    "XDG_DATA_DIRS",
    "XDG_CONFIG_DIRS",
    "EDITOR",
    "VISUAL",
    "PAGER",
    "SHLVL",
    "_",
    "TMPDIR",
    "TEMP",
    "TMP",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
})


def _is_secret_key(name: str) -> bool:
    """Check if an environment variable name looks like a secret."""
    if name in _SKIP_KEYS:
        return False
    return any(p.match(name) for p in _SECRET_KEY_PATTERNS)


@dataclass
class MigrationResult:
    """Result of a secret migration operation.

    Attributes:
        migrated: Number of secrets migrated to vault.
        skipped: Number of secrets skipped (already in vault or empty).
        deleted: Number of sources securely deleted.
        errors: List of error messages.
        sources: Mapping of migrated key -> source location.
    """

    migrated: int = 0
    skipped: int = 0
    deleted: int = 0
    errors: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source scanners
# ---------------------------------------------------------------------------

def _scan_env_vars() -> dict[str, str]:
    """Scan environment variables for secrets.

    Returns:
        Dict of {key_name: value} for detected secrets.
    """
    found: dict[str, str] = {}
    for key, value in os.environ.items():
        if _is_secret_key(key) and value.strip():
            found[key] = value
    return found


def _scan_hermes_config(config_path: Optional[Path] = None) -> dict[str, str]:
    """Scan a hermes config.yaml for secrets.

    Looks for keys like 'api_key', 'token', 'secret' in the config YAML.

    Args:
        config_path: Path to the config.yaml file. If None, searches
            common locations.

    Returns:
        Dict of {key_name: value} for detected secrets.
    """
    if config_path is None:
        # Search common locations
        candidates = [
            Path.home() / ".config" / "hermes" / "config.yaml",
            Path.home() / ".config" / "hermes" / "config.yml",
            Path.home() / ".hermes" / "config.yaml",
            Path.home() / ".hermes" / "config.yml",
            Path("config.yaml"),
            Path("config.yml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        return {}

    found: dict[str, str] = {}
    try:
        import yaml
        with open(config_path, "r") as fp:
            data = yaml.safe_load(fp)

        if isinstance(data, dict):
            _extract_secrets_from_dict(data, "", found)
    except ImportError:
        logger.debug("PyYAML not installed, skipping config.yaml scan")
    except Exception as exc:
        logger.debug("Error reading config: %s", exc)

    return found


def _extract_secrets_from_dict(
    data: dict,
    prefix: str,
    found: dict[str, str],
) -> None:
    """Recursively extract secret-looking values from a dict."""
    secret_keys = {
        "api_key", "api_token", "secret_key", "secret",
        "password", "token", "access_key", "credentials",
        "auth_key", "auth_token", "private_key",
    }

    for key, value in data.items():
        full_key = f"{prefix}{key}".upper().replace(".", "_").replace("-", "_")

        if isinstance(value, dict):
            _extract_secrets_from_dict(value, f"{full_key}_", found)
        elif isinstance(value, str) and value.strip():
            key_lower = key.lower()
            if key_lower in secret_keys or _is_secret_key(full_key):
                found[full_key] = value


def _scan_dotenv(dotenv_path: Optional[Path] = None) -> dict[str, str]:
    """Scan a .env file for secrets.

    Args:
        dotenv_path: Path to the .env file. If None, searches common locations.

    Returns:
        Dict of {key_name: value} for detected secrets.
    """
    if dotenv_path is None:
        candidates = [
            Path(".env"),
            Path.home() / ".env",
            Path(".env.local"),
            Path(".env.production"),
        ]
        for candidate in candidates:
            if candidate.exists():
                dotenv_path = candidate
                break

    if dotenv_path is None or not dotenv_path.exists():
        return {}

    found: dict[str, str] = {}
    try:
        with open(dotenv_path, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if _is_secret_key(key) and value:
                    found[key] = value
    except Exception as exc:
        logger.debug("Error reading .env: %s", exc)

    return found


# ---------------------------------------------------------------------------
# Secure delete
# ---------------------------------------------------------------------------

def _secure_delete_env_var(key: str) -> bool:
    """Remove a secret from environment variables.

    Args:
        key: The environment variable name to remove.

    Returns:
        True if the variable was removed.
    """
    if key in os.environ:
        os.environ.pop(key, None)
        return True
    return False


def _secure_delete_from_file(
    file_path: Path,
    key: str,
) -> bool:
    """Overwrite a secret value in a file with zeros.

    Reads the file, replaces the value for the given key with zeros
    of the same length, then writes back.

    Args:
        file_path: Path to the file containing the secret.
        key: The key name to zero out.

    Returns:
        True if the value was overwritten.
    """
    if not file_path.exists():
        return False

    try:
        content = file_path.read_text(encoding="utf-8")
        # Match key=value or key: value patterns
        patterns = [
            # .env style: KEY=value or KEY="value" or KEY='value'
            re.compile(
                rf"^({re.escape(key)}\s*=\s*)['\"]?(.+?)['\"]?\s*$",
                re.MULTILINE,
            ),
            # YAML style: key: value
            re.compile(
                rf"(\b{re.escape(key.lower())}\s*:\s*)['\"]?(.+?)['\"]?\s*$",
                re.MULTILINE | re.IGNORECASE,
            ),
        ]

        modified = False
        for pattern in patterns:
            def _zero_replace(m: re.Match) -> str:
                prefix = m.group(1)
                value = m.group(2)
                return prefix + "0" * len(value)

            new_content, count = pattern.subn(_zero_replace, content)
            if count > 0:
                content = new_content
                modified = True

        if modified:
            # Write with fsync to ensure overwrite reaches disk.
            # NOTE: On journaling filesystems (ext4, NTFS) or CoW
            # filesystems (btrfs, ZFS), the original data may persist
            # in the journal or as a previous snapshot. Full secure
            # deletion requires filesystem-specific tools or FDE.
            fd = os.open(str(file_path), os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(fd, content.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            return True
    except Exception as exc:
        logger.debug("Secure delete from %s failed: %s", file_path, exc)

    return False


# ---------------------------------------------------------------------------
# Migration main function
# ---------------------------------------------------------------------------

def discover_secrets(
    config_path: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
) -> dict[str, tuple[str, str]]:
    """Discover secrets from all sources with priority ordering.

    Priority: env > hermes_config > dotenv

    Args:
        config_path: Optional path to hermes config.yaml.
        dotenv_path: Optional path to .env file.

    Returns:
        Dict of {key_name: (value, source)} where source is
        'env', 'hermes_config', or 'dotenv'.
    """
    result: dict[str, tuple[str, str]] = {}

    # Lowest priority first, then overwrite with higher priority
    # .env (lowest)
    for key, value in _scan_dotenv(dotenv_path).items():
        result[key] = (value, "dotenv")

    # hermes config (medium)
    for key, value in _scan_hermes_config(config_path).items():
        result[key] = (value, "hermes_config")

    # Environment variables (highest)
    for key, value in _scan_env_vars().items():
        result[key] = (value, "env")

    return result


def migrate_secrets(
    vault: "Vault",
    config_path: Optional[Path] = None,
    dotenv_path: Optional[Path] = None,
    secure_delete: bool = True,
    dry_run: bool = False,
) -> MigrationResult:
    """Discover and migrate secrets from all sources into the vault.

    Scans environment variables, hermes config.yaml, and .env files
    for secrets, then stores them in the vault. Optionally securely
    deletes the originals.

    Args:
        vault: The vault to migrate secrets into.
        config_path: Optional path to hermes config.yaml.
        dotenv_path: Optional path to .env file.
        secure_delete: If True, overwrite originals with zeros after migration.
        dry_run: If True, report what would be migrated without doing it.

    Returns:
        MigrationResult with counts and details.
    """
    result = MigrationResult()
    discovered = discover_secrets(config_path, dotenv_path)

    for key, (value, source) in discovered.items():
        try:
            # Check if already in vault
            existing_keys = vault.list_keys()
            if key in existing_keys:
                result.skipped += 1
                continue

            if dry_run:
                result.migrated += 1
                result.sources[key] = source
                continue

            # Store in vault
            vault.set(key, value)
            result.migrated += 1
            result.sources[key] = source
            logger.info("Migrated secret '%s' from %s", key, source)

            # Secure delete from source
            if secure_delete:
                deleted = False
                if source == "env":
                    deleted = _secure_delete_env_var(key)
                elif source == "hermes_config" and config_path:
                    deleted = _secure_delete_from_file(config_path, key)
                elif source == "dotenv" and dotenv_path:
                    deleted = _secure_delete_from_file(dotenv_path, key)
                elif source == "dotenv":
                    # Try default .env locations
                    for p in [Path(".env"), Path(".env.local")]:
                        if p.exists() and _secure_delete_from_file(p, key):
                            deleted = True
                            break

                if deleted:
                    result.deleted += 1

        except Exception as exc:
            error_msg = f"Failed to migrate '{key}' from {source}: {exc}"
            result.errors.append(error_msg)
            logger.error(error_msg)

    return result
