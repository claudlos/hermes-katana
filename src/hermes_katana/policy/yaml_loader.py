"""
YAML policy file loader for HermesKatana.

Provides:
- ``load_policy_file(path)``    — load and validate a single YAML file
- ``load_policy_directory(dir)`` — load all ``*.yaml`` / ``*.yml`` files
- ``validate_policy_yaml(data)`` — schema validation with clear error messages
- ``export_policy_set(ps, path)`` — write a PolicySet to YAML
- ``PolicyFileWatcher``          — hot-reload watcher (filesystem polling)

Policy YAML schema
------------------

.. code-block:: yaml

   name: my-custom-policies
   version: "1.0.0"
   description: "My custom policy set"
   extends: balanced          # optional — inherit from a built-in set
   author: "Security Team"
   metadata:
     environment: production
   policies:
     - name: block_tainted_terminal
       description: "Block terminal with taint"
       tool_pattern: terminal
       conditions:
         - field: "*"
           operator: contains_taint
           value: true
       action: deny
       priority: 100
       enabled: true
       tags: [critical]
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml
from pydantic import ValidationError

from .defaults import BUILTIN_POLICY_SETS
from .models import PolicySet

logger = logging.getLogger(__name__)

__all__ = [
    "PolicyValidationError",
    "validate_policy_yaml",
    "load_policy_file",
    "load_policy_directory",
    "export_policy_set",
    "PolicyFileWatcher",
]

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_ACTIONS = {"allow", "deny", "escalate", "log_only"}
_VALID_OPERATORS = {
    "contains_taint",
    "source_is",
    "reader_lacks",
    "matches_pattern",
    "argument_matches",
    "taint_level_gte",
    "has_label",
}
_REQUIRED_POLICY_FIELDS = {"name", "tool_pattern"}


class PolicyValidationError(Exception):
    """Raised when a policy YAML file fails schema validation."""

    def __init__(self, message: str, errors: list[str] | None = None):
        self.errors = errors or []
        full = message
        if self.errors:
            full += "\n  • " + "\n  • ".join(self.errors)
        super().__init__(full)


def validate_policy_yaml(data: dict[str, Any]) -> list[str]:
    """Validate raw YAML data against the expected policy schema.

    Returns a list of human-readable error strings (empty = valid).
    """
    errors: list[str] = []

    if not isinstance(data, dict):
        errors.append("Root element must be a mapping (dict)")
        return errors

    # Top-level required fields
    if "name" not in data:
        errors.append("Missing required top-level field: 'name'")
    if "policies" not in data:
        errors.append("Missing required top-level field: 'policies'")
        return errors  # nothing else to check

    policies = data.get("policies", [])
    if not isinstance(policies, list):
        errors.append("'policies' must be a list")
        return errors

    for idx, policy in enumerate(policies):
        prefix = f"policies[{idx}]"
        if not isinstance(policy, dict):
            errors.append(f"{prefix}: must be a mapping, got {type(policy).__name__}")
            continue

        for req in _REQUIRED_POLICY_FIELDS:
            if req not in policy:
                errors.append(f"{prefix}: missing required field '{req}'")

        # Action validation
        action = policy.get("action")
        if action is not None and action not in _VALID_ACTIONS:
            errors.append(
                f"{prefix}: invalid action '{action}'. "
                f"Must be one of: {', '.join(sorted(_VALID_ACTIONS))}"
            )

        # Conditions validation
        conditions = policy.get("conditions", [])
        if not isinstance(conditions, list):
            errors.append(f"{prefix}: 'conditions' must be a list")
        else:
            for cidx, cond in enumerate(conditions):
                cprefix = f"{prefix}.conditions[{cidx}]"
                if not isinstance(cond, dict):
                    errors.append(f"{cprefix}: must be a mapping")
                    continue
                if "field" not in cond:
                    errors.append(f"{cprefix}: missing 'field'")
                if "operator" not in cond:
                    errors.append(f"{cprefix}: missing 'operator'")
                op = cond.get("operator")
                if op is not None and op not in _VALID_OPERATORS:
                    errors.append(
                        f"{cprefix}: invalid operator '{op}'. "
                        f"Must be one of: {', '.join(sorted(_VALID_OPERATORS))}"
                    )

        # Priority validation
        priority = policy.get("priority")
        if priority is not None:
            if not isinstance(priority, int) or priority < 0:
                errors.append(f"{prefix}: 'priority' must be a non-negative integer")

    return errors


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_policy_file(path: Union[str, Path]) -> PolicySet:
    """Load a single YAML policy file and return a validated ``PolicySet``.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        A validated ``PolicySet`` instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        PolicyValidationError: If schema validation fails.
        pydantic.ValidationError: If Pydantic model validation fails.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Policy file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise PolicyValidationError(f"Empty YAML file: {filepath}")

    # Schema validation
    errors = validate_policy_yaml(raw)
    if errors:
        raise PolicyValidationError(
            f"Validation failed for {filepath.name}",
            errors=errors,
        )

    # Handle inheritance
    extends = raw.get("extends")
    if extends:
        raw = _resolve_inheritance(raw, extends)

    try:
        return PolicySet.model_validate(raw)
    except ValidationError as exc:
        raise PolicyValidationError(
            f"Pydantic validation failed for {filepath.name}: {exc}"
        ) from exc


def load_policy_directory(
    directory: Union[str, Path],
    recursive: bool = False,
) -> list[PolicySet]:
    """Load all YAML policy files from a directory.

    Args:
        directory: Path to scan for ``*.yaml`` and ``*.yml`` files.
        recursive: If True, scan subdirectories as well.

    Returns:
        List of validated ``PolicySet`` instances.
    """
    dirpath = Path(directory)
    if not dirpath.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    glob_pattern = "**/*.yaml" if recursive else "*.yaml"
    yml_pattern = "**/*.yml" if recursive else "*.yml"

    files = sorted(set(dirpath.glob(glob_pattern)) | set(dirpath.glob(yml_pattern)))
    results: list[PolicySet] = []
    for fp in files:
        try:
            ps = load_policy_file(fp)
            results.append(ps)
            logger.info("Loaded policy set '%s' from %s (%d policies)", ps.name, fp.name, len(ps.policies))
        except (PolicyValidationError, ValidationError) as exc:
            logger.warning("Skipping invalid policy file %s: %s", fp.name, exc)

    return results


def _resolve_inheritance(data: dict[str, Any], parent_name: str) -> dict[str, Any]:
    """Merge *data* on top of the named parent policy set.

    The parent must be a built-in set name (paranoid, balanced, permissive).
    Raises PolicyValidationError if the parent is unknown to fail closed.
    """
    parent_raw = BUILTIN_POLICY_SETS.get(parent_name)
    if parent_raw is None:
        raise PolicyValidationError(
            f"Policy set extends unknown parent '{parent_name}'. "
            f"Available built-in sets: {', '.join(sorted(BUILTIN_POLICY_SETS))}"
        )

    # Build merged policy list: parent first, child overrides by name
    parent_policies = {p["name"]: p for p in parent_raw.get("policies", [])}
    for p in data.get("policies", []):
        parent_policies[p["name"]] = p

    merged = {**parent_raw, **data}
    merged["policies"] = list(parent_policies.values())
    # Keep the 'extends' field for provenance
    merged["extends"] = parent_name
    return merged


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_policy_set(
    policy_set: PolicySet,
    path: Union[str, Path],
    *,
    include_defaults: bool = True,
) -> Path:
    """Write a ``PolicySet`` to a YAML file.

    Args:
        policy_set: The PolicySet to export.
        path: Destination file path.
        include_defaults: If True, include fields even when they match defaults.

    Returns:
        The resolved Path that was written.
    """
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    data = policy_set.model_dump(
        mode="json",
        exclude_none=not include_defaults,
        exclude={"created_at"} if not include_defaults else set(),
    )

    # Convert datetime to ISO string for readability
    if "created_at" in data and data["created_at"] is not None:
        data["created_at"] = str(data["created_at"])

    with open(filepath, "w", encoding="utf-8") as fh:
        yaml.dump(
            data,
            fh,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )

    logger.info("Exported policy set '%s' to %s", policy_set.name, filepath)
    return filepath


# ---------------------------------------------------------------------------
# Hot-reload watcher
# ---------------------------------------------------------------------------


class PolicyFileWatcher:
    """Watches a directory for policy YAML changes and triggers a callback.

    Uses a simple polling approach (no OS-level inotify dependency) to
    remain portable across platforms.

    Usage::

        def on_change(policy_sets: list[PolicySet]) -> None:
            engine.replace_policies(policy_sets)

        watcher = PolicyFileWatcher("/etc/hermes/policies", on_change, interval=5.0)
        watcher.start()
        # ... later ...
        watcher.stop()
    """

    def __init__(
        self,
        directory: Union[str, Path],
        callback: Callable[[list[PolicySet]], None],
        interval: float = 5.0,
        recursive: bool = False,
    ):
        """
        Args:
            directory: Path to watch for YAML files.
            callback:  Called with newly-loaded PolicySets on change.
            interval:  Seconds between polls.
            recursive: Watch subdirectories as well.
        """
        self._directory = Path(directory)
        self._callback = callback
        self._interval = max(1.0, interval)
        self._recursive = recursive
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_mtimes: dict[str, float] = {}

    @property
    def is_running(self) -> bool:
        """Whether the watcher thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background polling thread."""
        if self.is_running:
            logger.warning("PolicyFileWatcher is already running")
            return

        self._stop_event.clear()
        # Capture initial state
        self._last_mtimes = self._snapshot_mtimes()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="PolicyFileWatcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "PolicyFileWatcher started: watching %s (interval=%.1fs)",
            self._directory,
            self._interval,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the watcher to stop and wait for the thread to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("PolicyFileWatcher stopped")

    def _poll_loop(self) -> None:
        """Main polling loop running on the background thread."""
        while not self._stop_event.is_set():
            try:
                current = self._snapshot_mtimes()
                if current != self._last_mtimes:
                    logger.info("Policy file change detected, reloading...")
                    self._last_mtimes = current
                    try:
                        policy_sets = load_policy_directory(
                            self._directory,
                            recursive=self._recursive,
                        )
                        # Count total YAML files to detect partial failures
                        glob_fn = self._directory.rglob if self._recursive else self._directory.glob
                        total_files = len(
                            list(glob_fn("*.yaml")) + list(glob_fn("*.yml"))
                        )
                        if not policy_sets:
                            logger.error(
                                "Policy reload produced 0 valid policy sets — "
                                "keeping previous policies to avoid security gap"
                            )
                        elif len(policy_sets) < total_files:
                            logger.error(
                                "Policy reload: only %d/%d files valid — "
                                "keeping previous policies to avoid partial coverage loss",
                                len(policy_sets),
                                total_files,
                            )
                        else:
                            self._callback(policy_sets)
                    except Exception:
                        logger.exception("Error reloading policies after file change")
            except Exception:
                logger.exception("Error in PolicyFileWatcher poll loop")

            self._stop_event.wait(timeout=self._interval)

    def _snapshot_mtimes(self) -> dict[str, float]:
        """Return a mapping of file paths to their modification times."""
        if not self._directory.is_dir():
            return {}

        mtimes: dict[str, float] = {}
        glob_fn = self._directory.rglob if self._recursive else self._directory.glob
        for ext in ("*.yaml", "*.yml"):
            for fp in glob_fn(ext):
                try:
                    mtimes[str(fp)] = os.path.getmtime(fp)
                except OSError:
                    pass  # file may have been deleted between glob and stat
        return mtimes
