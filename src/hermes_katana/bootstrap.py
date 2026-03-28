"""
Runtime bootstrap helpers for installed HermesKatana checkouts.

This module makes checkout-local installer state actionable at runtime.
It can discover an installed checkout, load ``.katana/katana.yaml``,
build the middleware chain from that state, export the matching
environment variables, and attach the runtime to a Hermes dispatcher.
"""

from __future__ import annotations

import inspect
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from hermes_katana.config import load_config
from hermes_katana.installer.installer import (
    KATANA_CONFIG_DIR,
    KATANA_CONFIG_FILE,
)
from hermes_katana.middleware.integration import create_default_chain
from hermes_katana.proxy import KatanaProxy, ProxyConfig
from hermes_katana.proxy.config import ScanModes

logger = logging.getLogger(__name__)

_MANAGED_ENV_KEYS = (
    "KATANA_ACTIVE",
    "KATANA_CHECKOUT_ROOT",
    "KATANA_CHECKOUT_CONFIG",
    "KATANA_POLICY_PRESET",
    "KATANA_POLICY_SOURCE",
    "KATANA_CA_CERT",
    "KATANA_PROXY_URL",
)
_TRUTHY = {"1", "true", "yes", "on"}
_RUNTIME_CACHE: dict[Path, "RuntimeBundle"] = {}


@dataclass
class CheckoutRuntimeState:
    """Normalized runtime state for one installed Hermes checkout."""

    checkout_root: Path
    config_path: Path
    policy_preset: str
    policy_dir: Optional[Path]
    policy_source: str
    scanner: dict[str, Any]
    taint_enabled: bool
    audit_enabled: bool
    audit_log_allow: bool
    audit_path: Optional[Path]
    proxy_enabled: bool
    proxy_config: Optional[ProxyConfig]
    ca_cert_path: Optional[Path]


@dataclass
class RuntimeBundle:
    """Concrete runtime objects derived from one checkout config."""

    state: CheckoutRuntimeState
    chain: Any
    proxy: Optional[KatanaProxy]
    audit_trail: Any
    vault: Any


def reset_runtime_cache() -> None:
    """Clear cached runtime bundles. Intended for tests."""
    _RUNTIME_CACHE.clear()


def discover_checkout_root(start: str | Path | None = None) -> Optional[Path]:
    """Discover an installed Hermes checkout by walking upward."""
    env_root = os.environ.get("KATANA_CHECKOUT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / KATANA_CONFIG_DIR / KATANA_CONFIG_FILE).exists():
            return root

    base = Path(start).expanduser().resolve() if start else Path.cwd().resolve()
    if base.is_file():
        base = base.parent

    for current in [base, *base.parents]:
        if (current / KATANA_CONFIG_DIR / KATANA_CONFIG_FILE).exists():
            return current

    return None


def load_checkout_state(checkout_root: str | Path | None = None) -> Optional[CheckoutRuntimeState]:
    """Load the installed checkout config and normalize it for runtime use."""
    root = (
        Path(checkout_root).expanduser().resolve()
        if checkout_root is not None
        else discover_checkout_root()
    )
    if root is None:
        return None

    config_path = root / KATANA_CONFIG_DIR / KATANA_CONFIG_FILE
    if not config_path.exists():
        return None

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        logger.warning("Could not load checkout config from %s", config_path, exc_info=True)
        return None

    if not isinstance(raw, dict):
        logger.warning("Checkout config at %s is not a mapping", config_path)
        return None

    global_config = load_config()

    policy_cfg = raw.get("policy", {}) if isinstance(raw.get("policy"), dict) else {}
    scan_cfg = raw.get("scanner", {}) if isinstance(raw.get("scanner"), dict) else {}
    taint_cfg = raw.get("taint", {}) if isinstance(raw.get("taint"), dict) else {}
    proxy_cfg = raw.get("proxy", {}) if isinstance(raw.get("proxy"), dict) else {}
    audit_cfg = raw.get("audit", {}) if isinstance(raw.get("audit"), dict) else {}

    policy_preset = str(policy_cfg.get("preset", global_config.policy_preset)).strip() or "balanced"
    policy_dir = _resolve_checkout_path(root, policy_cfg.get("custom_dir"))
    if policy_dir is not None and policy_dir.exists():
        policy_source = f"custom dir {policy_dir}"
    else:
        policy_dir = None
        policy_source = f"preset {policy_preset}"

    scanner = {
        "block_threshold": float(scan_cfg.get("block_threshold", 0.7)),
        "warn_threshold": float(scan_cfg.get("warn_threshold", 0.4)),
        "check_injection": bool(scan_cfg.get("check_injection", True)),
        "check_secrets": bool(scan_cfg.get("check_secrets", True)),
        "check_unicode": bool(scan_cfg.get("check_unicode", True)),
        "check_content": bool(scan_cfg.get("check_content", True)),
    }

    audit_enabled = bool(audit_cfg.get("enabled", global_config.audit_enabled))
    audit_dir = _resolve_checkout_path(root, audit_cfg.get("trail_dir")) or (
        root / KATANA_CONFIG_DIR / "audit"
    )
    audit_path = audit_dir / "audit.jsonl" if audit_enabled else None

    proxy_enabled = bool(proxy_cfg.get("enabled", global_config.proxy_enabled))
    proxy_config = None
    ca_cert_path = _resolve_checkout_path(root, proxy_cfg.get("ca_cert"))
    if proxy_enabled:
        proxy_config = ProxyConfig(
            host=str(proxy_cfg.get("listen_host", "127.0.0.1")),
            port=int(proxy_cfg.get("listen_port", global_config.proxy_port)),
            allowed_domains=list(global_config.domain_allowlist),
            inject_credentials=bool(global_config.vault_enabled),
            scan_modes=ScanModes(
                secrets=scanner["check_secrets"],
                injection=scanner["check_injection"],
                content=scanner["check_content"],
                unicode=scanner["check_unicode"],
            ),
        )

    return CheckoutRuntimeState(
        checkout_root=root,
        config_path=config_path,
        policy_preset=policy_preset,
        policy_dir=policy_dir,
        policy_source=policy_source,
        scanner=scanner,
        taint_enabled=bool(taint_cfg.get("enabled", global_config.taint_tracking)),
        audit_enabled=audit_enabled,
        audit_log_allow=bool(audit_cfg.get("log_allow", True)),
        audit_path=audit_path,
        proxy_enabled=proxy_enabled,
        proxy_config=proxy_config,
        ca_cert_path=ca_cert_path,
    )


def get_runtime_bundle(checkout_root: str | Path | None = None) -> Optional[RuntimeBundle]:
    """Return a cached runtime bundle for the installed checkout."""
    state = load_checkout_state(checkout_root)
    if state is None:
        return None

    cached = _RUNTIME_CACHE.get(state.checkout_root)
    if cached is not None:
        return cached

    runtime = _build_runtime_bundle(state)
    _RUNTIME_CACHE[state.checkout_root] = runtime
    return runtime


def compose_runtime_env(
    base_env: Optional[dict[str, str]] = None,
    *,
    checkout_root: str | Path | None = None,
    start_proxy: bool = False,
) -> dict[str, str]:
    """Compose a process environment from the installed checkout state."""
    env = dict(base_env) if base_env is not None else dict(os.environ)
    runtime = get_runtime_bundle(checkout_root)
    if runtime is None:
        return env

    for key in _MANAGED_ENV_KEYS:
        env.pop(key, None)

    env.update(_runtime_env_updates(runtime, start_proxy=start_proxy))
    return env


def prepare_runtime_environment(
    *,
    checkout_root: str | Path | None = None,
    start_proxy: bool = False,
) -> Optional[RuntimeBundle]:
    """Apply installed checkout state to the current process environment."""
    runtime = get_runtime_bundle(checkout_root)
    if runtime is None:
        return None

    updates = _runtime_env_updates(runtime, start_proxy=start_proxy)
    for key in _MANAGED_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(updates)
    return runtime


def ensure_dispatcher_bootstrap(
    dispatcher: Any,
    *,
    checkout_root: str | Path | None = None,
) -> Optional[RuntimeBundle]:
    """Attach a runtime chain and escalation handler to a Hermes dispatcher."""
    runtime = prepare_runtime_environment(checkout_root=checkout_root, start_proxy=False)
    if runtime is None:
        return None

    if getattr(dispatcher, "_katana_chain", None) is None:
        dispatcher._katana_chain = runtime.chain

    if getattr(dispatcher, "_katana_escalate", None) is None:
        dispatcher._katana_escalate = _build_default_escalator(dispatcher)

    dispatcher._katana_runtime = runtime
    return runtime


def _resolve_checkout_path(checkout_root: Path, raw_path: Any) -> Optional[Path]:
    """Resolve a path from checkout-local config."""
    if raw_path in (None, ""):
        return None

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = checkout_root / path
    return path.resolve()


def _build_runtime_bundle(state: CheckoutRuntimeState) -> RuntimeBundle:
    """Create concrete runtime objects for one checkout state."""
    from hermes_katana.audit import AuditTrail
    from hermes_katana.policy import PolicyEngine

    vault = _open_vault()
    audit_trail = AuditTrail(path=state.audit_path) if state.audit_path is not None else None

    if state.policy_dir is not None:
        policy_engine = PolicyEngine.from_directory(state.policy_dir)
    else:
        policy_engine = PolicyEngine.with_defaults(state.policy_preset)

    chain = create_default_chain(
        {
            "taint.enabled": state.taint_enabled,
            "scan.enabled": any(
                (
                    state.scanner["check_injection"],
                    state.scanner["check_secrets"],
                    state.scanner["check_unicode"],
                    state.scanner["check_content"],
                )
            ),
            "scan.block_threshold": state.scanner["block_threshold"],
            "scan.warn_threshold": state.scanner["warn_threshold"],
            "scan.check_injection": state.scanner["check_injection"],
            "scan.check_secrets": state.scanner["check_secrets"],
            "scan.check_unicode": state.scanner["check_unicode"],
            "scan.check_content": state.scanner["check_content"],
            "scan.vault_values": _collect_vault_values(vault),
            "policy.engine": policy_engine,
            "policy.preset": state.policy_preset,
            "audit.enabled": state.audit_enabled,
            "audit.log_allow": state.audit_log_allow,
            "audit.trail": audit_trail,
        }
    )

    proxy = None
    if state.proxy_enabled and state.proxy_config is not None:
        proxy = KatanaProxy(
            config=state.proxy_config,
            vault=vault,
            audit=audit_trail,
        )

    return RuntimeBundle(
        state=state,
        chain=chain,
        proxy=proxy,
        audit_trail=audit_trail,
        vault=vault,
    )


def _runtime_env_updates(runtime: RuntimeBundle, *, start_proxy: bool) -> dict[str, str]:
    """Build managed environment variables for one runtime bundle."""
    state = runtime.state
    updates = {
        "KATANA_ACTIVE": "1",
        "KATANA_CHECKOUT_ROOT": str(state.checkout_root),
        "KATANA_CHECKOUT_CONFIG": str(state.config_path),
        "KATANA_POLICY_PRESET": state.policy_preset,
        "KATANA_POLICY_SOURCE": state.policy_source,
    }

    if state.ca_cert_path is not None and state.ca_cert_path.exists():
        updates["KATANA_CA_CERT"] = str(state.ca_cert_path)

    if runtime.proxy is not None:
        status = runtime.proxy.status()
        if start_proxy and not status.get("running"):
            runtime.proxy.start()
            status = runtime.proxy.status()

        if status.get("running"):
            host = str(status.get("host", status["config"]["host"]))
            port = int(status.get("port", status["config"]["port"]))
            updates["KATANA_PROXY_URL"] = _build_proxy_url(host, port)

    return updates


def _open_vault():
    """Open the shared vault backend without creating it on demand."""
    try:
        from hermes_katana.vault import Vault

        return Vault(auto_create=False)
    except Exception:
        logger.debug("Runtime bootstrap could not open the vault", exc_info=True)
        return None


def _collect_vault_values(vault: Any) -> set[str]:
    """Collect current vault values for secret-leak detection."""
    if vault is None:
        return set()

    try:
        return {
            value
            for key in vault.list_keys()
            if (value := vault.get(key))
        }
    except Exception:
        logger.debug("Runtime bootstrap could not collect vault values", exc_info=True)
        return set()


def _build_default_escalator(dispatcher: Any):
    """Build the fallback async escalation handler for a dispatcher."""

    async def _katana_escalate(ctx: Any) -> bool:
        for name in (
            "request_approval",
            "request_escalation",
            "confirm_tool_use",
            "approve_tool_call",
            "approve",
        ):
            handler = getattr(dispatcher, name, None)
            if not callable(handler):
                continue

            result = handler(ctx)
            if inspect.isawaitable(result):
                result = await result
            return bool(result)

        auto_approve = os.environ.get("KATANA_AUTO_APPROVE_ESCALATIONS", "0").lower() in _TRUTHY
        if auto_approve:
            logger.warning(
                "KATANA_AUTO_APPROVE_ESCALATIONS is set — auto-approving "
                "escalation for tool '%s'. Disable this env var in production.",
                getattr(ctx, "tool_name", "unknown"),
            )
        return auto_approve

    return _katana_escalate


def _build_proxy_url(host: str, port: int) -> str:
    """Build the proxy URL string."""
    return f"http://{host}:{port}"
