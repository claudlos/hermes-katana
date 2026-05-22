"""
Policy data models for HermesKatana policy engine.

Defines the core types used to express declarative security policies:
- PolicyResult: the action to take (ALLOW, DENY, ESCALATE, LOG_ONLY)
- ConditionOperator: how to evaluate a condition against taint context
- Condition: a single predicate on a tool call's taint state
- Policy: a named rule binding tool patterns + conditions to an action
- PolicySet: a versioned collection of policies with metadata

All models use Pydantic v2 for validation, serialization, and schema generation.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "PolicyResult",
    "ConditionOperator",
    "Condition",
    "Policy",
    "PolicySet",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PolicyResult(str, enum.Enum):
    """Action the policy engine should take when a policy matches.

    ALLOW     – permit the tool call to proceed without modification.
    DENY      – block the tool call entirely; return an error to the caller.
    ESCALATE  – pause execution and request human approval before proceeding.
    LOG_ONLY  – allow the call but emit a structured audit-log entry.
    """

    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    LOG_ONLY = "log_only"


class ConditionOperator(str, enum.Enum):
    """Operators used inside policy conditions.

    CONTAINS_TAINT   – true when the specified field carries *any* taint label.
    SOURCE_IS        – true when the taint source matches the given value
                       (e.g. ``user_message``, ``web_content``, ``file``).
    READER_LACKS     – true when the taint reader set does NOT include the
                       given capability (e.g. ``trusted_executor``).
    MATCHES_PATTERN  – true when the field value matches a regex/glob pattern.
    ARGUMENT_MATCHES – true when a specific argument value matches a pattern
                       (inspects the raw argument, not taint metadata).
    TAINT_LEVEL_GTE  – true when the taint severity level is >= the given int.
    TAINT_LEVEL_LTE  – true when the taint severity level is <= the given int
                       (useful for "low taint = allow" rules).
    HAS_LABEL        – true when a specific taint label is present.
    """

    CONTAINS_TAINT = "contains_taint"
    SOURCE_IS = "source_is"
    READER_LACKS = "reader_lacks"
    MATCHES_PATTERN = "matches_pattern"
    ARGUMENT_MATCHES = "argument_matches"
    TAINT_LEVEL_GTE = "taint_level_gte"
    TAINT_LEVEL_LTE = "taint_level_lte"
    HAS_LABEL = "has_label"


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------


class Condition(BaseModel):
    """A single predicate evaluated against a tool call's taint context.

    Attributes:
        field:    The argument or metadata field to inspect (e.g. ``command``,
                  ``url``, ``*`` for any field).
        operator: How to compare ``field`` against ``value``.
        value:    The reference value for the comparison.  Interpretation
                  depends on ``operator``.
    """

    field: str = Field(
        ...,
        description="Argument name or metadata field to inspect. Use '*' for any field.",
    )
    operator: ConditionOperator = Field(
        ...,
        description="Comparison operator.",
    )
    value: Any = Field(
        default=None,
        description="Reference value. Meaning depends on operator.",
    )

    model_config = {"frozen": False, "extra": "forbid"}


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class Policy(BaseModel):
    """A single declarative security policy rule.

    Policies bind a *tool pattern* (glob) together with zero or more
    *conditions* to an *action* (PolicyResult).  When the policy engine
    evaluates a tool call it selects all matching policies, orders them by
    priority (highest first), and returns the action of the first policy
    whose conditions all evaluate to ``True``.

    Attributes:
        name:         Human-readable identifier (must be unique within a set).
        description:  Free-text explanation shown in audit logs and CLI output.
        tool_pattern: Glob pattern matched against the tool name
                      (e.g. ``terminal``, ``browser_*``, ``*``).
        conditions:   All conditions must be true for the policy to fire
                      (implicit AND).  An empty list means "always match".
        action:       What to do when the policy fires.
        priority:     Higher values are evaluated first.  Built-in defaults
                      use 0–100; user policies should use 100+.
        enabled:      Disabled policies are skipped during evaluation.
        tags:         Optional free-form tags for grouping / filtering.
    """

    name: str = Field(..., min_length=1, max_length=128)
    description: str = Field(default="")
    tool_pattern: str = Field(
        ...,
        min_length=1,
        description="Glob pattern matched against the tool name.",
    )
    conditions: list[Condition] = Field(default_factory=list)
    action: PolicyResult = Field(default=PolicyResult.DENY)
    priority: int = Field(default=50, ge=0, le=10000)
    enabled: bool = Field(default=True)
    tags: list[str] = Field(default_factory=list)

    model_config = {"frozen": False, "extra": "forbid"}

    @field_validator("tool_pattern")
    @classmethod
    def _validate_tool_pattern(cls, v: str) -> str:
        """Ensure the tool pattern is a reasonable glob."""
        # Reject empty or whitespace-only patterns
        stripped = v.strip()
        if not stripped:
            raise ValueError("tool_pattern must not be blank")
        return stripped


# ---------------------------------------------------------------------------
# PolicySet
# ---------------------------------------------------------------------------


class PolicySet(BaseModel):
    """A versioned, named collection of policies with metadata.

    PolicySets are the unit of distribution: they can be serialised to YAML,
    loaded from files, and extended (inherited) by other sets.

    Attributes:
        name:        Unique name (e.g. ``max``, ``balanced``).
        version:     SemVer-ish version string for change tracking.
        description: Human-readable description.
        extends:     Optional name of a parent PolicySet whose policies are
                     inherited (child policies override by name).
        author:      Who created / maintains this set.
        created_at:  Timestamp of creation.
        policies:    Ordered list of Policy objects.
        metadata:    Arbitrary key/value metadata.
    """

    name: str = Field(..., min_length=1, max_length=128)
    version: str = Field(default="1.0.0")
    description: str = Field(default="")
    extends: Optional[str] = Field(
        default=None,
        description="Name of a parent PolicySet to inherit from.",
    )
    author: str = Field(default="HermesKatana")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    policies: list[Policy] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False, "extra": "forbid"}

    # -- helpers -------------------------------------------------------------

    def get_policy(self, name: str) -> Optional[Policy]:
        """Return the first policy matching *name*, or ``None``."""
        for p in self.policies:
            if p.name == name:
                return p
        return None

    def enabled_policies(self) -> list[Policy]:
        """Return only enabled policies, sorted by descending priority."""
        return sorted(
            [p for p in self.policies if p.enabled],
            key=lambda p: p.priority,
            reverse=True,
        )

    def merge(self, other: "PolicySet") -> "PolicySet":
        """Merge *other* into this set.  Policies in *other* override same-name
        policies in *self*.  Returns a **new** PolicySet."""
        by_name: dict[str, Policy] = {p.name: p for p in self.policies}
        for p in other.policies:
            by_name[p.name] = p
        return PolicySet(
            name=other.name or self.name,
            version=other.version,
            description=other.description or self.description,
            extends=self.name,
            author=other.author or self.author,
            policies=list(by_name.values()),
            metadata={**self.metadata, **other.metadata},
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for YAML export."""
        return self.model_dump(mode="json", exclude_none=True)
