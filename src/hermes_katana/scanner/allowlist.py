"""False-positive suppression and allowlisting for scanner findings.

Production environments inevitably hit false positives: a database tool
legitimately uses SQL keywords, vault operations mention API key patterns,
terminal commands contain shell metacharacters.  This module provides a
structured way to suppress known FPs without weakening real detection.

Suppressions are:
- Scoped to specific tools (glob patterns)
- Pattern-based (regex match on the finding text)
- Time-bounded (optional expiry)
- Auditable (hit counts, creation dates)

Usage::

    from hermes_katana.scanner.allowlist import AllowlistManager

    mgr = AllowlistManager.with_defaults()
    finding = SomeFinding(matched_text="SELECT * FROM users", ...)
    if mgr.is_suppressed(finding, tool_name="database_query"):
        pass  # Don't alert on this
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suppression model
# ---------------------------------------------------------------------------


@dataclass
class Suppression:
    """A single false-positive suppression rule.

    Attributes:
        id: Unique identifier for this suppression.
        pattern: Regex to match against finding text (matched_text).
        tool_pattern: Glob to match against the tool name ('*' = all tools).
        category_pattern: Glob to match finding category (e.g., 'injection_*').
        reason: Human-readable reason for the suppression.
        created_at: Unix timestamp of creation.
        expires_at: Optional Unix timestamp of expiry (None = never).
        hit_count: Number of times this suppression has matched.
        enabled: Whether this suppression is active.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    pattern: str = ""
    tool_pattern: str = "*"
    category_pattern: str = "*"
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    hit_count: int = 0
    enabled: bool = True

    _compiled: Optional[re.Pattern] = field(default=None, repr=False, compare=False)

    @property
    def compiled_pattern(self) -> re.Pattern:
        """Lazily compile the regex pattern."""
        if self._compiled is None:
            try:
                self._compiled = re.compile(self.pattern, re.IGNORECASE)
            except re.error:
                logger.warning("Invalid suppression pattern: %r", self.pattern)
                self._compiled = re.compile(re.escape(self.pattern), re.IGNORECASE)
        return self._compiled

    @property
    def is_expired(self) -> bool:
        """Whether this suppression has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def is_active(self) -> bool:
        """Whether this suppression is enabled and not expired."""
        return self.enabled and not self.is_expired

    def matches_tool(self, tool_name: str) -> bool:
        """Check if this suppression applies to the given tool."""
        return fnmatch.fnmatch(tool_name, self.tool_pattern)

    def matches_category(self, category: str) -> bool:
        """Check if this suppression applies to the given finding category."""
        return fnmatch.fnmatch(category, self.category_pattern)

    def matches_text(self, text: str) -> bool:
        """Check if this suppression's pattern matches the finding text."""
        return bool(self.compiled_pattern.search(text))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for YAML/JSON export."""
        d: dict[str, Any] = {
            "id": self.id,
            "pattern": self.pattern,
            "tool_pattern": self.tool_pattern,
            "category_pattern": self.category_pattern,
            "reason": self.reason,
            "created_at": self.created_at,
            "hit_count": self.hit_count,
            "enabled": self.enabled,
        }
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Suppression:
        """Deserialize from dict."""
        return cls(
            id=d.get("id", uuid.uuid4().hex[:12]),
            pattern=d.get("pattern", ""),
            tool_pattern=d.get("tool_pattern", "*"),
            category_pattern=d.get("category_pattern", "*"),
            reason=d.get("reason", ""),
            created_at=d.get("created_at", time.time()),
            expires_at=d.get("expires_at"),
            hit_count=d.get("hit_count", 0),
            enabled=d.get("enabled", True),
        )


# ---------------------------------------------------------------------------
# Built-in suppressions for common false positives
# ---------------------------------------------------------------------------

_BUILTIN_SUPPRESSIONS: list[Suppression] = [
    # Vault operations legitimately handle API key patterns
    Suppression(
        id="builtin-vault-keys",
        pattern=r"(sk-|ghp_|AKIA|Bearer\s+)",
        tool_pattern="vault*",
        category_pattern="*secret*",
        reason="Vault tool legitimately handles secret patterns",
    ),
    # Database tools use SQL keywords
    Suppression(
        id="builtin-sql-in-db",
        pattern=r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b",
        tool_pattern="database*",
        category_pattern="*command*",
        reason="Database tools legitimately use SQL keywords",
    ),
    # Terminal tool: common safe patterns that look dangerous
    Suppression(
        id="builtin-terminal-grep",
        pattern=r"\bgrep\b.*\b(rm|kill|sudo)\b",
        tool_pattern="terminal",
        category_pattern="*command*",
        reason="Grepping for dangerous patterns is safe (reading, not executing)",
    ),
    # Read operations that mention sensitive paths
    Suppression(
        id="builtin-read-sensitive",
        pattern=r"/etc/(passwd|shadow|hosts)",
        tool_pattern="read_file",
        category_pattern="*command*",
        reason="Reading system files for inspection is a valid operation",
    ),
    # Code review and documentation mentioning security patterns
    Suppression(
        id="builtin-code-review",
        pattern=r"(ignore\s+previous|system\s+prompt|inject)",
        tool_pattern="read_file",
        category_pattern="*injection*",
        reason="Code/docs discussing security patterns are not injections",
    ),
    # Search tools looking for security patterns
    Suppression(
        id="builtin-search-security",
        pattern=r"(password|secret|key|token|credential)",
        tool_pattern="search_files",
        category_pattern="*secret*",
        reason="Searching for security-sensitive patterns is a valid audit operation",
    ),
    # --- New: SQL keywords in DB-adjacent tools (query builders, ORMs) ---
    Suppression(
        id="builtin-sql-orm",
        pattern=r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|JOIN|WHERE|GROUP\s+BY|ORDER\s+BY)\b",
        tool_pattern="*sql*",
        category_pattern="*",
        reason="SQL keywords are normal in SQL/ORM tools",
    ),
    Suppression(
        id="builtin-sql-migration",
        pattern=r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|TRUNCATE)\b",
        tool_pattern="*migrat*",
        category_pattern="*",
        reason="SQL keywords are normal in migration tools",
    ),
    # --- New: Shell commands in documentation context ---
    Suppression(
        id="builtin-shell-in-docs",
        pattern=r"(\$\s*(sudo|rm|chmod|chown|kill|curl|wget|apt|pip)|```(bash|sh|shell|console))",
        tool_pattern="*",
        category_pattern="*command*",
        reason="Shell commands in docs/code blocks are explanatory, not executable",
    ),
    # --- New: API key patterns in vault/config tools ---
    Suppression(
        id="builtin-keys-config",
        pattern=r"(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer)",
        tool_pattern="*config*",
        category_pattern="*secret*",
        reason="Config tools legitimately reference key/token field names",
    ),
    Suppression(
        id="builtin-keys-env",
        pattern=r"(API_KEY|SECRET_KEY|AUTH_TOKEN|BEARER|AWS_ACCESS)",
        tool_pattern="*env*",
        category_pattern="*secret*",
        reason="Environment variable tools handle credential names",
    ),
    # --- New: Password patterns in auth/credential management ---
    Suppression(
        id="builtin-password-auth",
        pattern=r"(password|passwd|credentials?|authenticate|login|bcrypt|argon2|hash)",
        tool_pattern="*auth*",
        category_pattern="*",
        reason="Auth tools legitimately handle password/credential patterns",
    ),
    Suppression(
        id="builtin-password-credential",
        pattern=r"(password|passwd|credentials?|secret|private[_-]?key)",
        tool_pattern="*credential*",
        category_pattern="*",
        reason="Credential management tools handle passwords by design",
    ),
    # --- New: Webhook URLs in integration/notification tools ---
    Suppression(
        id="builtin-webhook-integration",
        pattern=r"(webhook|callback|https?://[^\s]+(hook|notify|alert|event|slack|services))",
        tool_pattern="*integrat*",
        category_pattern="*",
        reason="Integration tools legitimately use webhook URLs",
    ),
    Suppression(
        id="builtin-webhook-notify",
        pattern=r"(webhook|callback|https?://[^\s]+/(hook|notify|slack|discord))",
        tool_pattern="*notif*",
        category_pattern="*",
        reason="Notification tools legitimately use webhook URLs",
    ),
]


# ---------------------------------------------------------------------------
# Allowlist manager
# ---------------------------------------------------------------------------


@dataclass
class AllowlistManager:
    """Manages false-positive suppressions for scanner findings.

    Holds an ordered list of suppressions and checks incoming findings
    against them.  Supports loading/saving from YAML files, and tracks
    hit counts for observability.

    Args:
        suppressions: Initial list of suppression rules.
        include_builtins: Whether to include built-in suppressions (default True).
    """

    suppressions: list[Suppression] = field(default_factory=list)
    include_builtins: bool = True

    def __post_init__(self):
        if self.include_builtins:
            # Prepend builtins (lower priority than user-defined)
            builtin_ids = {s.id for s in _BUILTIN_SUPPRESSIONS}
            existing_ids = {s.id for s in self.suppressions}
            for b in _BUILTIN_SUPPRESSIONS:
                if b.id not in existing_ids:
                    self.suppressions.append(b)

    @classmethod
    def with_defaults(cls) -> AllowlistManager:
        """Create a manager with only built-in suppressions."""
        return cls(suppressions=[], include_builtins=True)

    def is_suppressed(
        self,
        finding: Any,
        tool_name: str = "",
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Check if a finding should be suppressed.

        Args:
            finding: A scanner finding object. Must have ``matched_text``
                     attribute and optionally ``category``.
            tool_name: The tool that triggered the scan.
            context: Optional extra context dict. Recognized keys:

                - ``in_code_block`` (bool): Text is inside a fenced code block.
                - ``in_documentation`` (bool): Text is part of documentation.
                - ``full_text`` (str): The full surrounding text (used for
                  automatic documentation-mode detection).

        Returns:
            True if the finding matches an active suppression.
        """
        matched_text = getattr(finding, "matched_text", "")
        category = ""
        # Handle different finding types
        cat_attr = getattr(finding, "category", None)
        if cat_attr is not None:
            category = cat_attr.value if hasattr(cat_attr, "value") else str(cat_attr)

        ctx = context or {}

        # --- Documentation mode suppression ---
        if self._is_documentation_context(matched_text, ctx):
            logger.debug(
                "Finding suppressed by documentation mode: %s", matched_text[:80],
            )
            return True

        for sup in self.suppressions:
            if not sup.is_active:
                continue
            if not sup.matches_tool(tool_name):
                continue
            if category and not sup.matches_category(category):
                continue
            if sup.matches_text(matched_text):
                sup.hit_count += 1
                logger.debug(
                    "Finding suppressed by %s (%s): %s",
                    sup.id, sup.reason, matched_text[:80],
                )
                return True

        return False

    # ------------------------------------------------------------------
    # Documentation-mode detection
    # ------------------------------------------------------------------

    _CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    _DOC_LINE_RE = re.compile(r"^\s*(\$|#!?/|>>>|\.\.\.|//|/\*|\*)", re.MULTILINE)

    @staticmethod
    def _is_documentation_context(
        matched_text: str,
        ctx: dict[str, Any],
    ) -> bool:
        """Return True if the matched text appears to be in documentation.

        Checks explicit context flags first, then falls back to heuristic
        detection on the full surrounding text.
        """
        # Explicit flags from caller
        if ctx.get("in_code_block") or ctx.get("in_documentation"):
            return True

        full_text = ctx.get("full_text", "")
        if not full_text:
            return False

        # Check if matched_text falls inside a fenced code block
        for m in AllowlistManager._CODE_BLOCK_RE.finditer(full_text):
            if matched_text in m.group():
                return True

        # Check if the line containing matched_text starts with $ or #
        for line in full_text.splitlines():
            if matched_text in line and AllowlistManager._DOC_LINE_RE.match(line):
                return True

        return False

    def add_suppression(
        self,
        pattern: str,
        tool_pattern: str = "*",
        category_pattern: str = "*",
        reason: str = "",
        expires_in: Optional[float] = None,
    ) -> Suppression:
        """Add a new suppression rule.

        Args:
            pattern: Regex to match against finding text.
            tool_pattern: Glob for tool name (default '*').
            category_pattern: Glob for finding category.
            reason: Human-readable reason.
            expires_in: Seconds until expiry (None = never).

        Returns:
            The created Suppression object.
        """
        sup = Suppression(
            pattern=pattern,
            tool_pattern=tool_pattern,
            category_pattern=category_pattern,
            reason=reason,
            expires_at=time.time() + expires_in if expires_in else None,
        )
        self.suppressions.insert(0, sup)  # User rules first
        return sup

    def remove_suppression(self, suppression_id: str) -> bool:
        """Remove a suppression by ID.

        Args:
            suppression_id: The suppression's unique ID.

        Returns:
            True if the suppression was found and removed.
        """
        before = len(self.suppressions)
        self.suppressions = [s for s in self.suppressions if s.id != suppression_id]
        return len(self.suppressions) < before

    def load(self, path: str | Path) -> None:
        """Load suppressions from a YAML file.

        The YAML file should have a top-level ``suppressions`` key
        containing a list of suppression dicts.

        Args:
            path: Path to the YAML file.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Allowlist file not found: %s", path)
            return

        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            entries = raw.get("suppressions", [])
            if not isinstance(entries, list):
                return
            for entry in entries:
                if isinstance(entry, dict):
                    sup = Suppression.from_dict(entry)
                    self.suppressions.insert(0, sup)
            logger.info("Loaded %d suppressions from %s", len(entries), path)
        except Exception:
            logger.warning("Failed to load allowlist from %s", path, exc_info=True)

    def export(self, path: str | Path) -> None:
        """Export all suppressions to a YAML file.

        Args:
            path: Path to write the YAML file.
        """
        path = Path(path)
        data = {
            "version": "1.0",
            "suppressions": [s.to_dict() for s in self.suppressions],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.dump(data, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            logger.info("Exported %d suppressions to %s", len(self.suppressions), path)
        except Exception:
            logger.warning("Failed to export allowlist to %s", path, exc_info=True)

    def stats(self) -> dict[str, Any]:
        """Return suppression statistics."""
        active = [s for s in self.suppressions if s.is_active]
        expired = [s for s in self.suppressions if s.is_expired]
        total_hits = sum(s.hit_count for s in self.suppressions)
        return {
            "total_rules": len(self.suppressions),
            "active_rules": len(active),
            "expired_rules": len(expired),
            "total_hits": total_hits,
            "top_rules": sorted(
                [(s.id, s.hit_count, s.reason) for s in self.suppressions if s.hit_count > 0],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }
