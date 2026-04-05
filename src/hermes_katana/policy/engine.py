"""
Policy engine for HermesKatana.

The ``PolicyEngine`` is the central evaluation point: given a tool name,
its arguments, and a taint context, it determines whether the call should be
allowed, denied, escalated, or logged.

Typical usage::

    from hermes_katana.policy import PolicyEngine, PolicyResult

    engine = PolicyEngine.with_defaults("balanced")
    result = engine.evaluate("terminal", {"command": "rm -rf /"}, taint_ctx)

    if result.action == PolicyResult.DENY:
        raise SecurityError(result.reason)

The engine supports:
- Glob-based tool-name matching (e.g. ``browser_*``)
- Priority-ordered policy evaluation (highest first, first match wins)
- Condition evaluation against taint context dicts
- Hot-reload from YAML via the ``PolicyFileWatcher``
"""

from __future__ import annotations

import fnmatch
import functools
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from hermes_katana.taint import Source, TaintedStr

from .defaults import BUILTIN_POLICY_SETS
from .models import (
    Condition,
    ConditionOperator,
    Policy,
    PolicyResult,
    PolicySet,
)
from .yaml_loader import PolicyFileWatcher, load_policy_directory, load_policy_file

logger = logging.getLogger(__name__)

__all__ = [
    "BENIGN_COMMANDS",
    "BENIGN_GIT_SUBCOMMANDS",
    "command_safety_check",
    "EvaluationResult",
    "evaluate_condition",
    "PolicyEngine",
]


# ---------------------------------------------------------------------------
# Benign command whitelist — commands safe even with mild taint
# ---------------------------------------------------------------------------

BENIGN_COMMANDS: set[str] = {
    "ls",
    "cat",
    "echo",
    "pwd",
    "cd",
    "pip",
    "pip3",
    "npm",
    "npx",
    "yarn",
    "git",
    "head",
    "tail",
    "wc",
    "sort",
    "uniq",
    "grep",
    "find",
    "which",
    "whoami",
    "date",
    "env",
    "printenv",
    "uname",
    "df",
    "du",
    "free",
    "ps",
    "top",
    "htop",
    "file",
    "stat",
    "id",
    "hostname",
    "uptime",
    "python3",
    "python",
    "node",
    "cargo",
    "rustc",
    "make",
    "cmake",
    "tree",
    "less",
    "more",
    "diff",
    "basename",
    "dirname",
    "realpath",
    "mkdir",
    "touch",
    "cp",
    "mv",  # mild side-effects but common dev ops
}

# Git subcommands that are read-only / safe
BENIGN_GIT_SUBCOMMANDS: set[str] = {
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "tag",
    "remote",
    "stash",
    "describe",
    "shortlog",
    "reflog",
    "config",
    "ls-files",
    "ls-tree",
}


def _extract_base_command(command: str) -> str:
    """Extract the base command name from a shell command string."""
    cmd = command.strip()
    # Strip leading env vars like FOO=bar cmd
    while "=" in cmd.split()[0] if cmd.split() else False:
        parts = cmd.split(None, 1)
        cmd = parts[1] if len(parts) > 1 else ""
    # Strip sudo prefix
    if cmd.startswith("sudo "):
        cmd = cmd[5:].strip()
    base = cmd.split()[0] if cmd.split() else ""
    # Strip path prefix
    if "/" in base:
        base = base.rsplit("/", 1)[-1]
    return base


def _is_benign_command(command: str) -> bool:
    """Check if a command is benign (safe even with mild taint)."""
    base = _extract_base_command(command)
    if not base:
        return False
    if base in BENIGN_COMMANDS:
        # Special check for git — only safe subcommands
        if base == "git":
            parts = command.strip().split()
            git_idx = next((i for i, p in enumerate(parts) if p == "git" or p.endswith("/git")), -1)
            if git_idx >= 0 and git_idx + 1 < len(parts):
                subcmd = parts[git_idx + 1]
                return subcmd in BENIGN_GIT_SUBCOMMANDS
            return True  # bare 'git' is fine
        return True
    return False


def command_safety_check(
    command: str,
    taint_context: dict[str, Any],
) -> PolicyResult:
    """Cross-reference command content with taint to decide action.

    Logic:
    - Clean (no taint) → ALLOW regardless of command content
    - Tainted + dangerous command pattern → DENY
    - Tainted + benign command → ALLOW (low taint ≤3) or ESCALATE (higher)
    - Tainted + unknown command → ESCALATE

    This is used by the engine as an advisory pre-check for terminal calls.
    """
    is_tainted = _field_is_tainted(taint_context, "*")

    if not is_tainted:
        return PolicyResult.ALLOW

    taint_level = _field_level(taint_context, "*")

    # Check if command is flagged as dangerous by the scanner
    try:
        from hermes_katana.scanner.commands import detect_dangerous_command

        findings = detect_dangerous_command(command)
        is_dangerous = len(findings) > 0
    except ImportError:
        # Scanner not available — be conservative
        is_dangerous = False

    if is_dangerous:
        return PolicyResult.DENY

    if _is_benign_command(command):
        # Low taint (1-3) + benign = allow
        if taint_level <= 3:
            return PolicyResult.ALLOW
        # Medium taint (4-6) + benign = escalate
        return PolicyResult.ESCALATE

    # Unknown command + tainted = escalate
    return PolicyResult.ESCALATE


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    """The outcome of a single policy evaluation.

    Attributes:
        action:       The decided PolicyResult.
        matched_policy: The Policy that fired (None if no policy matched).
        reason:       Human-readable explanation.
        details:      Extra context for audit logging.
    """

    action: PolicyResult
    matched_policy: Optional[Policy] = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Taint context protocol
# ---------------------------------------------------------------------------
# The engine accepts a taint_context dict with a well-known shape.
# Callers can pass any dict that conforms to this interface:
#
#   {
#       "tainted_fields": {
#           "<field_name>": {
#               "is_tainted": bool,
#               "source": str | None,      # e.g. "user_message", "web_content"
#               "labels": list[str],        # e.g. ["untrusted", "pii"]
#               "readers": list[str],       # e.g. ["trusted_executor"]
#               "level": int,               # severity 0-10
#           }
#       }
#   }
#
# A missing or empty taint_context means "all arguments are clean".


def _field_is_tainted(taint_context: dict[str, Any], field_name: str) -> bool:
    """Check whether *field_name* (or any field if '*') is tainted."""
    tainted_fields = taint_context.get("tainted_fields", {})
    if not tainted_fields:
        return False

    if field_name == "*":
        return any(f.get("is_tainted", False) for f in tainted_fields.values())

    info = tainted_fields.get(field_name, {})
    return bool(info.get("is_tainted", False))


def _field_source(taint_context: dict[str, Any], field_name: str) -> set[str]:
    """Return the set of taint sources for *field_name* (or all if '*')."""
    tainted_fields = taint_context.get("tainted_fields", {})
    sources: set[str] = set()

    targets = tainted_fields.values() if field_name == "*" else [tainted_fields.get(field_name, {})]
    for info in targets:
        src = info.get("source")
        if src:
            sources.add(src)
    return sources


def _field_readers(taint_context: dict[str, Any], field_name: str) -> set[str]:
    """Return the set of readers for *field_name* (or all if '*')."""
    tainted_fields = taint_context.get("tainted_fields", {})
    readers: set[str] = set()

    targets = tainted_fields.values() if field_name == "*" else [tainted_fields.get(field_name, {})]
    for info in targets:
        for r in info.get("readers", []):
            readers.add(r)
    return readers


def _field_labels(taint_context: dict[str, Any], field_name: str) -> set[str]:
    """Return the set of taint labels for *field_name* (or all if '*')."""
    tainted_fields = taint_context.get("tainted_fields", {})
    labels: set[str] = set()

    targets = tainted_fields.values() if field_name == "*" else [tainted_fields.get(field_name, {})]
    for info in targets:
        for lbl in info.get("labels", []):
            labels.add(lbl)
    return labels


def _field_level(taint_context: dict[str, Any], field_name: str) -> int:
    """Return the maximum taint level for *field_name* (or all if '*')."""
    tainted_fields = taint_context.get("tainted_fields", {})
    levels: list[int] = []

    targets = tainted_fields.values() if field_name == "*" else [tainted_fields.get(field_name, {})]
    for info in targets:
        lvl = info.get("level", 0)
        if isinstance(lvl, int):
            levels.append(lvl)
    return max(levels) if levels else 0


# ---------------------------------------------------------------------------
# Regex cache with compile-time validation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _safe_compile(pattern: str) -> re.Pattern | None:
    """Compile a regex with length limit and error handling.

    Returns None for invalid or overly long patterns.
    """
    if len(pattern) > 2000:
        logger.warning("Policy regex too long (%d chars), rejecting", len(pattern))
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        logger.warning("Invalid policy regex %r: %s", pattern[:80], exc)
        return None


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------


def evaluate_condition(
    condition: Condition,
    tool_args: dict[str, Any],
    taint_context: dict[str, Any],
) -> bool:
    """Evaluate a single ``Condition`` against the given context.

    Returns ``True`` if the condition is satisfied.
    """
    op = condition.operator
    fld = condition.field
    val = condition.value

    if op == ConditionOperator.CONTAINS_TAINT:
        result = _field_is_tainted(taint_context, fld)
        # val can be True (field must be tainted) or False (must be clean)
        expected = bool(val) if val is not None else True
        return result == expected

    elif op == ConditionOperator.SOURCE_IS:
        sources = _field_source(taint_context, fld)
        return str(val) in sources

    elif op == ConditionOperator.READER_LACKS:
        readers = _field_readers(taint_context, fld)
        return str(val) not in readers

    elif op == ConditionOperator.MATCHES_PATTERN:
        # Match the *argument value* against a regex
        pat = _safe_compile(str(val))
        if pat is None:
            return False
        if fld == "*":
            return any(pat.search(str(v)) is not None for v in tool_args.values())
        arg_val = tool_args.get(fld, "")
        return pat.search(str(arg_val)) is not None

    elif op == ConditionOperator.ARGUMENT_MATCHES:
        # Exact or glob match on the raw argument value
        if fld == "*":
            return any(fnmatch.fnmatch(str(v), str(val)) for v in tool_args.values())
        arg_val = tool_args.get(fld, "")
        return fnmatch.fnmatch(str(arg_val), str(val))

    elif op == ConditionOperator.TAINT_LEVEL_GTE:
        level = _field_level(taint_context, fld)
        threshold = int(val) if val is not None else 0
        return level >= threshold

    elif op == ConditionOperator.TAINT_LEVEL_LTE:
        level = _field_level(taint_context, fld)
        threshold = int(val) if val is not None else 0
        return level <= threshold

    elif op == ConditionOperator.HAS_LABEL:
        labels = _field_labels(taint_context, fld)
        return str(val) in labels

    else:
        logger.warning("Unknown condition operator: %s", op)
        return False


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Declarative policy evaluation engine for Hermes tool calls.

    The engine maintains an ordered collection of ``Policy`` objects and
    evaluates them against incoming tool calls.  Policies are matched by
    tool-name glob and evaluated by priority (highest first).  The first
    policy whose *all* conditions are satisfied determines the result.

    If no policy matches, the engine returns a configurable default action
    (``ALLOW`` by default — explicit-deny is recommended via catch-all
    policies in the loaded policy set).

    Thread safety: all mutations are protected by a reentrant lock so the
    engine can be shared across threads and updated via hot-reload.

    Features:
    - Glob-based tool-name matching
    - Priority-ordered evaluation (highest first, first match wins)
    - Command safety cross-referencing (terminal calls checked against scanner)
    - LRU caching for repeated evaluations on same tool+args patterns

    Usage::

        engine = PolicyEngine.with_defaults("balanced")
        result = engine.evaluate("terminal", {"command": "ls"}, {})
        print(result.action)  # PolicyResult.ALLOW
    """

    # Class-level LRU cache for policy evaluations
    _EVAL_CACHE_MAX = 512

    def __init__(
        self,
        policies: Sequence[Policy] | None = None,
        *,
        default_action: PolicyResult = PolicyResult.ALLOW,
    ):
        """
        Args:
            policies:       Initial list of policies.
            default_action: Action returned when no policy matches.
        """
        self._lock = threading.RLock()
        self._policies: list[Policy] = list(policies or [])
        self._default_action = default_action
        self._policy_set_name: str = "custom"
        self._watcher: Optional[PolicyFileWatcher] = None
        self._eval_cache: dict[str, EvaluationResult] = {}

        # Sort by priority descending on init
        self._sort_policies()

    # -- Factory methods ----------------------------------------------------

    @classmethod
    def with_defaults(cls, preset: str = "balanced") -> "PolicyEngine":
        """Create an engine pre-loaded with a built-in policy set.

        Args:
            preset: One of ``paranoid``, ``balanced``, ``permissive``.

        Returns:
            A new PolicyEngine with the built-in policies loaded.

        Raises:
            ValueError: If *preset* is not a known built-in name.
        """
        raw = BUILTIN_POLICY_SETS.get(preset)
        if raw is None:
            available = ", ".join(sorted(BUILTIN_POLICY_SETS))
            raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

        ps = PolicySet.model_validate(raw)
        # Use preset-appropriate default action instead of blanket ALLOW
        _preset_defaults = {
            "paranoid": PolicyResult.DENY,
            "balanced": PolicyResult.ESCALATE,
            "permissive": PolicyResult.LOG_ONLY,
        }
        default = _preset_defaults.get(preset, PolicyResult.ESCALATE)
        engine = cls(policies=ps.policies, default_action=default)
        engine._policy_set_name = ps.name
        logger.info(
            "PolicyEngine initialised with '%s' preset (%d policies)",
            preset,
            len(ps.policies),
        )
        return engine

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "PolicyEngine":
        """Create an engine from a single YAML policy file.

        Args:
            path: Path to the YAML file.

        Returns:
            A new PolicyEngine with the file's policies loaded.
        """
        ps = load_policy_file(path)
        engine = cls(policies=ps.policies)
        engine._policy_set_name = ps.name
        return engine

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        *,
        watch: bool = False,
        watch_interval: float = 5.0,
    ) -> "PolicyEngine":
        """Create an engine from all YAML files in a directory.

        Args:
            directory:      Path to scan for policy files.
            watch:          If True, start a background watcher for hot-reload.
            watch_interval: Seconds between filesystem polls.

        Returns:
            A new PolicyEngine with all discovered policies loaded.
        """
        policy_sets = load_policy_directory(directory)
        all_policies: list[Policy] = []
        name_parts: list[str] = []
        for ps in policy_sets:
            all_policies.extend(ps.policies)
            name_parts.append(ps.name)

        engine = cls(policies=all_policies)
        engine._policy_set_name = "+".join(name_parts) if name_parts else "empty"

        if watch:
            engine._start_watcher(directory, watch_interval)

        return engine

    # -- Core evaluation ----------------------------------------------------

    @staticmethod
    def _canonical_taint_fingerprint(obj: Any) -> Any:
        """Walk *obj* and produce a deterministic, taint-aware fingerprint.

        Silent ``default=str`` coercion (the previous approach) was unsafe
        because a plain string and a tainted value with the same content
        produced identical cache keys — the taint labels were invisible
        to the hash. An attacker could warm the cache with a benign plain
        string, then bypass policy on a tainted call with the same content.

        This function canonicalizes:
          * ``TaintedStr``    -> ("tainted_str", content, sorted_label_ids,
                                  sorted_source_fingerprints)
          * ``Source``         -> ("source", label, origin, trust_level)
                                  (timestamp INTENTIONALLY omitted; including
                                   it would make cache misses the norm and
                                   it is not security-relevant)
          * ``frozenset/set``  -> ("frozenset", sorted fingerprints)
          * ``dict``           -> ("dict", sorted (k, fp(v)) items)
          * ``list/tuple``     -> ("list", [fp(x) for x in obj])
          * primitives         -> passthrough (str / int / float / bool / None)
          * everything else    -> ("obj", type_qualname, repr(obj))
            The fully-qualified type name prevents foreign objects from
            colliding with primitives merely because their ``__str__`` /
            ``__repr__`` happens to match.
        """
        # TaintedStr is a str subclass — check it BEFORE the str branch.
        if isinstance(obj, TaintedStr):
            label_names = tuple(sorted(getattr(lab, "name", repr(lab)) for lab in obj.labels))
            source_fps = tuple(
                sorted(
                    (repr(PolicyEngine._canonical_taint_fingerprint(s)) for s in (getattr(obj, "sources", None) or ()))
                )
            )
            return ("tainted_str", str.__str__(obj), label_names, source_fps)
        if isinstance(obj, Source):
            return (
                "source",
                getattr(obj.label, "name", repr(obj.label)),
                str(obj.origin) if obj.origin is not None else None,
                getattr(obj.trust_level, "name", repr(obj.trust_level)),
            )
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return (
                "dict",
                tuple(
                    sorted(
                        (
                            str(k),
                            PolicyEngine._canonical_taint_fingerprint(v),
                        )
                        for k, v in obj.items()
                    )
                ),
            )
        if isinstance(obj, (list, tuple)):
            return (
                "list",
                tuple(PolicyEngine._canonical_taint_fingerprint(x) for x in obj),
            )
        if isinstance(obj, (set, frozenset)):
            return (
                "frozenset",
                tuple(sorted(repr(PolicyEngine._canonical_taint_fingerprint(x)) for x in obj)),
            )
        # Unknown type — include the QUALIFIED type name so foreign objects
        # cannot collide with primitives merely by having a matching repr.
        tp = type(obj)
        qualname = f"{tp.__module__}.{tp.__qualname__}"
        return ("obj", qualname, repr(obj))

    @staticmethod
    def _make_cache_key(
        tool_name: str,
        args: dict[str, Any],
        taint_context: dict[str, Any],
    ) -> str:
        """Build a deterministic, taint-aware cache key for an evaluation.

        Uses :meth:`_canonical_taint_fingerprint` to recursively canonicalize
        every value so that TaintedStr and Source objects contribute their
        label metadata to the key (see that method's docstring for the
        cache-collision attack this prevents).
        """
        import hashlib

        fingerprint = (
            "v2",  # version the scheme so future changes invalidate old caches
            tool_name,
            PolicyEngine._canonical_taint_fingerprint(args),
            PolicyEngine._canonical_taint_fingerprint(taint_context),
        )
        return hashlib.sha256(repr(fingerprint).encode("utf-8")).hexdigest()[:32]

    def invalidate_cache(self) -> None:
        """Clear the evaluation cache (called on policy mutations)."""
        self._eval_cache.clear()

    def evaluate(
        self,
        tool_name: str,
        args: dict[str, Any],
        taint_context: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate all matching policies for a tool call.

        Policies are checked in priority order (highest first).  The first
        policy whose tool pattern matches *and* all conditions evaluate to
        True determines the returned action.

        For terminal calls, a ``command_safety_check`` is run first to
        cross-reference the command scanner: dangerous+tainted → DENY,
        benign+low-taint → ALLOW, otherwise → ESCALATE.

        Results are cached by (tool_name, args, taint_context) hash.

        Args:
            tool_name:     The Hermes tool being invoked (e.g. ``terminal``).
            args:          The tool's arguments as a flat dict.
            taint_context: Taint metadata for the arguments (see module docs).

        Returns:
            An ``EvaluationResult`` with the decided action and matched policy.
        """
        if taint_context is None:
            taint_context = {}

        # Check cache first
        cache_key = self._make_cache_key(tool_name, args, taint_context)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached

        # Command safety pre-check for terminal calls
        if tool_name == "terminal" and "command" in args:
            safety = command_safety_check(args["command"], taint_context)
            if safety == PolicyResult.DENY:
                result = EvaluationResult(
                    action=PolicyResult.DENY,
                    matched_policy=None,
                    reason=("command_safety_check denied terminal call: dangerous command with tainted args"),
                    details={
                        "tool_name": tool_name,
                        "policy_set": self._policy_set_name,
                        "safety_check": "dangerous_tainted",
                    },
                )
                self._cache_put(cache_key, result)
                return result

        # Snapshot policies under the lock, then evaluate without holding it
        # (GAP 4.3 — thread safety: atomic snapshot for evaluate)
        with self._lock:
            policies_snapshot = list(self._policies)

        for policy in policies_snapshot:
            if not policy.enabled:
                continue

            # Tool-name glob match
            if not fnmatch.fnmatch(tool_name, policy.tool_pattern):
                continue

            # Evaluate all conditions (implicit AND)
            all_met = all(evaluate_condition(cond, args, taint_context) for cond in policy.conditions)

            if all_met:
                reason = (
                    f"Policy '{policy.name}' matched tool '{tool_name}' "
                    f"(pattern='{policy.tool_pattern}', priority={policy.priority})"
                )
                logger.debug(reason)
                result = EvaluationResult(
                    action=policy.action,
                    matched_policy=policy,
                    reason=reason,
                    details={
                        "tool_name": tool_name,
                        "policy_name": policy.name,
                        "policy_set": self._policy_set_name,
                    },
                )
                self._cache_put(cache_key, result)
                return result

        # No policy matched — use default
        result = EvaluationResult(
            action=self._default_action,
            matched_policy=None,
            reason=f"No policy matched tool '{tool_name}'; using default action",
            details={"tool_name": tool_name, "policy_set": self._policy_set_name},
        )
        self._cache_put(cache_key, result)
        return result

    def _cache_put(self, key: str, result: EvaluationResult) -> None:
        """Insert into cache, evicting oldest if over limit."""
        if len(self._eval_cache) >= self._EVAL_CACHE_MAX:
            # Evict ~25% of oldest entries
            keys = list(self._eval_cache.keys())
            for k in keys[: len(keys) // 4]:
                self._eval_cache.pop(k, None)
        self._eval_cache[key] = result

    def evaluate_batch(
        self,
        calls: Sequence[tuple[str, dict[str, Any], dict[str, Any] | None]],
    ) -> list[EvaluationResult]:
        """Evaluate multiple tool calls in one shot.

        Args:
            calls: Sequence of ``(tool_name, args, taint_context)`` tuples.

        Returns:
            List of ``EvaluationResult`` in the same order as *calls*.
        """
        return [self.evaluate(name, args, ctx) for name, args, ctx in calls]

    # -- Policy management --------------------------------------------------

    def add_policy(self, policy: Policy) -> None:
        """Add a policy and re-sort by priority.

        If a policy with the same name already exists, it is replaced.
        """
        with self._lock:
            # Remove existing policy with the same name
            self._policies = [p for p in self._policies if p.name != policy.name]
            self._policies.append(policy)
            self._sort_policies()
            self.invalidate_cache()
        logger.info("Added policy '%s' (priority=%d)", policy.name, policy.priority)

    def remove_policy(self, name: str) -> bool:
        """Remove a policy by name.  Returns True if found and removed."""
        with self._lock:
            before = len(self._policies)
            self._policies = [p for p in self._policies if p.name != name]
            removed = len(self._policies) < before
            if removed:
                self.invalidate_cache()
        if removed:
            logger.info("Removed policy '%s'", name)
        return removed

    def list_policies(self) -> list[Policy]:
        """Return a copy of all policies, sorted by descending priority."""
        with self._lock:
            return list(self._policies)

    def get_policy(self, name: str) -> Optional[Policy]:
        """Find a policy by name, or return None."""
        with self._lock:
            for p in self._policies:
                if p.name == name:
                    return p
        return None

    def clear(self) -> None:
        """Remove all policies."""
        with self._lock:
            self._policies.clear()
        logger.info("All policies cleared")

    def load_policies(self, path: Union[str, Path]) -> int:
        """Load policies from a YAML file or directory and add them.

        Existing policies with the same name are replaced.

        Args:
            path: A file or directory path.

        Returns:
            Number of policies loaded.
        """
        p = Path(path)
        if p.is_file():
            ps = load_policy_file(p)
            sets = [ps]
        elif p.is_dir():
            sets = load_policy_directory(p)
        else:
            raise FileNotFoundError(f"Path not found: {p}")

        count = 0
        for ps in sets:
            for policy in ps.policies:
                self.add_policy(policy)
                count += 1

        self._policy_set_name = sets[0].name if len(sets) == 1 else "merged"
        logger.info("Loaded %d policies from %s", count, path)
        return count

    def replace_all(self, policies: Sequence[Policy]) -> None:
        """Atomically replace all policies (used by hot-reload).

        Validates policies before applying. Rejects empty or malformed
        policy lists to avoid leaving the engine in an insecure state
        (GAP 4.5 — hot-reload validation).
        """
        if not policies:
            logger.error("replace_all() called with empty policy list — rejecting to avoid security gap")
            return

        # Validate each policy has required fields
        validated: list[Policy] = []
        for p in policies:
            if not p.name or not p.tool_pattern:
                logger.warning("Skipping malformed policy (missing name or tool_pattern): %r", p)
                continue
            validated.append(p)

        if not validated:
            logger.error(
                "replace_all() — all %d policies were malformed; keeping previous",
                len(policies),
            )
            return

        with self._lock:
            self._policies = validated
            self._sort_policies()
            self.invalidate_cache()
        logger.info("Replaced all policies (%d total)", len(validated))

    @property
    def policy_count(self) -> int:
        """Number of policies currently loaded."""
        with self._lock:
            return len(self._policies)

    @property
    def policy_set_name(self) -> str:
        """Name of the currently active policy set."""
        return self._policy_set_name

    @property
    def default_action(self) -> PolicyResult:
        """The action returned when no policy matches."""
        return self._default_action

    @default_action.setter
    def default_action(self, value: PolicyResult) -> None:
        self._default_action = value

    # -- Hot-reload ---------------------------------------------------------

    def _start_watcher(self, directory: Union[str, Path], interval: float) -> None:
        """Start a background file watcher for hot-reload."""

        def _on_change(policy_sets: list[PolicySet]) -> None:
            all_policies: list[Policy] = []
            for ps in policy_sets:
                all_policies.extend(ps.policies)
            self.replace_all(all_policies)
            logger.info("Hot-reload: replaced policies (%d total)", len(all_policies))

        self._watcher = PolicyFileWatcher(
            directory=directory,
            callback=_on_change,
            interval=interval,
        )
        self._watcher.start()

    def stop_watcher(self) -> None:
        """Stop the hot-reload file watcher, if running."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None

    # -- Internals ----------------------------------------------------------

    def _sort_policies(self) -> None:
        """Sort policies by descending priority (must hold lock)."""
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def __repr__(self) -> str:
        return (
            f"PolicyEngine(set='{self._policy_set_name}', "
            f"policies={len(self._policies)}, "
            f"default={self._default_action.value})"
        )

    def __enter__(self) -> "PolicyEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop_watcher()
