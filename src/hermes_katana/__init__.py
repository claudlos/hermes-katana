"""
HermesKatana — defense-in-depth security toolkit for Hermes Agent.

CaMeL-inspired taint tracking, proxy-based secret guard, policy engine,
and red-team benchmarking for LLM agent tool use.

Usage::

    from hermes_katana import TaintTracker, PolicyEngine, Vault, scan_input

    # Taint tracking
    tracker = TaintTracker.get_instance()

    # Policy evaluation
    engine = PolicyEngine.with_defaults("balanced")

    # Encrypted secret storage
    vault = Vault()

    # Input scanning
    result = scan_input("user input text")
"""

from hermes_katana._version import __version__

# Scanner — multi-layer attack detection
from hermes_katana.scanner import ScanResult, ScanVerdict, scan_command, scan_input, scan_output

# Taint — CaMeL-inspired data-flow tracking
from hermes_katana.taint import (
    FlowAnalyzer,
    FlowDecision,
    FlowRule,
    Source,
    TaintedStr,
    TaintedValue,
    TaintLabel,
    TaintTracker,
    TrustLevel,
)

# Policy — declarative security rules
from hermes_katana.policy import (
    EvaluationResult,
    Policy,
    PolicyEngine,
    PolicyResult,
    PolicySet,
)

# Vault — encrypted secret storage
from hermes_katana.vault import Vault, VaultError, default_vault_path

# Audit — tamper-evident logging
from hermes_katana.audit import AuditEntry, AuditEventType, AuditTrail

# Middleware — tool-call interception chain
from hermes_katana.middleware import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
    MiddlewareChain,
)

# Proxy — MITM proxy for LLM traffic
from hermes_katana.proxy import KatanaProxy, ProxyConfig

# Config
from hermes_katana.config import KatanaConfig, load_config

__all__ = [
    # Meta
    "__version__",
    # Scanner
    "scan_input",
    "scan_output",
    "scan_command",
    "ScanResult",
    "ScanVerdict",
    # Taint
    "TaintTracker",
    "TaintedValue",
    "TaintedStr",
    "TaintLabel",
    "TrustLevel",
    "Source",
    "FlowDecision",
    "FlowRule",
    "FlowAnalyzer",
    # Policy
    "PolicyEngine",
    "PolicyResult",
    "EvaluationResult",
    "Policy",
    "PolicySet",
    # Vault
    "Vault",
    "VaultError",
    "default_vault_path",
    # Audit
    "AuditTrail",
    "AuditEntry",
    "AuditEventType",
    # Middleware
    "KatanaMiddleware",
    "MiddlewareChain",
    "CallContext",
    "DispatchDecision",
    # Proxy
    "KatanaProxy",
    "ProxyConfig",
    # Config
    "KatanaConfig",
    "load_config",
]
