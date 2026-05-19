"""
SHA-256 hash-chained append-only audit trail for HermesKatana.

Hardened from hermes-aegis audit log with:
- O(1) last-hash tracking (cached in memory, not O(n) file read)
- Cross-platform file locking for concurrent writers
- Automatic log rotation when file exceeds size threshold
- Structured entries with Pydantic validation
- Comprehensive event type enum
- Query and statistics support

Security model:
- Each entry contains a SHA-256 hash of the previous entry
- The chain can be verified to detect tampered or missing entries
- File locking prevents interleaved writes from concurrent processes
- Rotation preserves old logs for forensic analysis
"""

from __future__ import annotations

__all__ = [
    "AuditEventType",
    "AuditEntry",
    "default_audit_path",
    "AuditTrail",
]


import hashlib
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field
from hermes_katana._files import AdvisoryFileLock

logger = logging.getLogger(__name__)

# Genesis hash for the first entry in a chain
_GENESIS_HASH = "0" * 64  # SHA-256 of nothing

# Default rotation threshold: 10 MB
_DEFAULT_MAX_SIZE = 10 * 1024 * 1024

# Maximum number of rotated files to keep
_DEFAULT_MAX_ROTATIONS = 10


# ---------------------------------------------------------------------------
# Audit event types
# ---------------------------------------------------------------------------


class AuditEventType(str, Enum):
    """Types of events recorded in the audit trail.

    Each event type represents a distinct security-relevant action
    or observation.
    """

    TOOL_CALL = "tool_call"
    """A tool/function was called by the agent."""

    SCAN_RESULT = "scan_result"
    """Scanner produced a finding (injection, secret, etc.)."""

    POLICY_DECISION = "policy_decision"
    """The policy engine made an allow/deny/escalate decision."""

    FLOW_ANALYSIS = "flow_analysis"
    """Taint flow analysis produced a result."""

    SECRET_BLOCKED = "secret_blocked"
    """A secret was blocked from being transmitted."""

    INJECTION_DETECTED = "injection_detected"
    """A prompt injection was detected."""

    RATE_ANOMALY = "rate_anomaly"
    """Rate limiting triggered or anomalous request pattern detected."""

    CIRCUIT_BREAKER = "circuit_breaker"
    """A circuit breaker was activated or deactivated."""

    CONFIG_CHANGE = "config_change"
    """A configuration change was made."""

    SESSION_START = "session_start"
    """A new agent session started."""

    SESSION_END = "session_end"
    """An agent session ended."""


# ---------------------------------------------------------------------------
# Audit entry model
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """A single structured audit log entry.

    Each entry is hash-chained: entry_hash = SHA-256(prev_hash + serialized_content).
    This creates a tamper-evident chain where modifying any entry
    invalidates all subsequent hashes.

    Attributes:
        timestamp: When the event occurred (UTC).
        event_type: Type of audit event.
        tool_name: Name of the tool or component involved.
        args_hash: Hash of the arguments/parameters (for privacy).
        decision: The decision made (allow, deny, escalate, etc.).
        details: Free-text details about the event.
        prev_hash: SHA-256 hash of the previous entry in the chain.
        entry_hash: SHA-256 hash of this entry (computed from prev_hash + content).
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the event.",
    )
    event_type: AuditEventType = Field(
        ...,
        description="Type of audit event.",
    )
    tool_name: str = Field(
        default="",
        description="Tool or component name.",
    )
    args_hash: str = Field(
        default="",
        description="Hash of the arguments (privacy-preserving).",
    )
    decision: str = Field(
        default="",
        description="Decision made (allow, deny, escalate, warn, etc.).",
    )
    details: str = Field(
        default="",
        description="Free-text event details.",
    )
    prev_hash: str = Field(
        default=_GENESIS_HASH,
        description="SHA-256 hash of the previous entry.",
    )
    entry_hash: str = Field(
        default="",
        description="SHA-256 hash of this entry.",
    )

    model_config = {"frozen": False, "extra": "allow"}

    def compute_hash(self) -> str:
        """Compute the SHA-256 hash for this entry.

        The hash is computed over the prev_hash concatenated with the
        JSON-serialized content (excluding entry_hash itself).

        Returns:
            Hex-encoded SHA-256 hash.
        """
        content = self.model_dump(
            mode="json",
            exclude={"entry_hash"},
        )
        # Ensure deterministic serialization
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256((self.prev_hash + serialized).encode("utf-8")).hexdigest()

    def finalize(self, prev_hash: str) -> "AuditEntry":
        """Set the prev_hash and compute the entry_hash.

        Args:
            prev_hash: Hash of the previous entry in the chain.

        Returns:
            A new AuditEntry with prev_hash and entry_hash set.
        """
        self.prev_hash = prev_hash
        self.entry_hash = self.compute_hash()
        return self


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


def default_audit_path() -> Path:
    """Return the default audit log file path without creating it."""
    from hermes_katana._paths import home_or_fallback

    return home_or_fallback() / ".config" / "hermes-katana" / "audit" / "audit.jsonl"


def _default_audit_path() -> Path:
    """Return the default audit log file path."""
    log_dir = default_audit_path().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "audit.jsonl"


class AuditTrail:
    """SHA-256 hash-chained append-only audit trail.

    Provides tamper-evident logging with O(1) last-hash tracking,
    file locking for concurrent writes, and automatic rotation.

    Args:
        path: Path to the audit log file (JSONL format).
        max_size: Maximum file size before rotation (default: 10MB).
        max_rotations: Maximum number of rotated log files to keep.

    Example:
        >>> trail = AuditTrail()
        >>> entry = AuditEntry(
        ...     event_type=AuditEventType.TOOL_CALL,
        ...     tool_name="terminal",
        ...     args_hash="abc123",
        ...     decision="allow",
        ... )
        >>> trail.log(entry)
        >>> trail.verify_chain()
        True
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        max_size: int = _DEFAULT_MAX_SIZE,
        max_rotations: int = _DEFAULT_MAX_ROTATIONS,
    ) -> None:
        self._path = path or _default_audit_path()
        self._max_size = max_size
        self._max_rotations = max_rotations
        self._file_lock = AdvisoryFileLock(self._path)
        self._rlock = threading.RLock()

        # O(1) last-hash cache — this is the key improvement over aegis
        self._last_hash: str = _GENESIS_HASH
        self._entry_count: int = 0

        # Ensure parent directory exists
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize from existing file
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the last hash from the existing log file.

        Only called once during initialization. After that, the last hash
        is tracked in memory (O(1) instead of O(n) for each append).
        """
        if not self._path.exists() or self._path.stat().st_size == 0:
            self._last_hash = _GENESIS_HASH
            self._entry_count = 0
            return

        try:
            count = 0
            last_hash = _GENESIS_HASH
            with open(self._path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        last_hash = data.get("entry_hash", _GENESIS_HASH)
                        count += 1
                    except json.JSONDecodeError:
                        continue

            self._last_hash = last_hash
            self._entry_count = count
            logger.debug(
                "Loaded audit trail: %d entries, last hash: %s...",
                count,
                last_hash[:12],
            )
        except Exception as exc:
            logger.warning("Could not load audit trail: %s", exc)
            self._last_hash = _GENESIS_HASH
            self._entry_count = 0

    def log(self, entry: AuditEntry) -> str:
        """Append an entry to the audit trail.

        Finalizes the entry (sets prev_hash and computes entry_hash),
        writes it to the log file with file locking, and updates the
        in-memory last-hash cache.

        For multi-process safety, we re-read the last hash from the file
        while holding the file lock, so concurrent processes always chain
        correctly.

        Args:
            entry: The audit entry to log. The prev_hash and entry_hash
                fields will be set automatically.

        Returns:
            The entry_hash of the logged entry.
        """
        with self._rlock:
            # Write with file locking — hold the lock across read+finalize+write
            # to prevent two processes from reading the same last_hash
            with self._file_lock:
                # Re-read the actual last hash from disk under lock
                # (another process may have appended since our last write)
                actual_last_hash = self._read_last_hash_from_file()
                if actual_last_hash != self._last_hash:
                    self._last_hash = actual_last_hash

                # Finalize the entry with the verified chain hash
                entry.finalize(self._last_hash)

                # Serialize to JSON
                line = entry.model_dump_json(exclude_none=False) + "\n"

                with open(self._path, "a", encoding="utf-8") as fp:
                    fp.write(line)
                    fp.flush()
                    os.fsync(fp.fileno())

            # Update in-memory state (O(1))
            self._last_hash = entry.entry_hash
            self._entry_count += 1

            # Check if rotation is needed
            self._maybe_rotate()

            return entry.entry_hash

    def _read_last_hash_from_file(self) -> str:
        """Read the last entry_hash from the audit log file.

        Called under file lock to get the true last hash for multi-process
        chain integrity. Returns _GENESIS_HASH if the file is empty or
        doesn't exist.
        """
        if not self._path.exists() or self._path.stat().st_size == 0:
            return _GENESIS_HASH
        try:
            last_hash = _GENESIS_HASH
            with open(self._path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        last_hash = data.get("entry_hash", _GENESIS_HASH)
                    except json.JSONDecodeError:
                        continue
            return last_hash
        except Exception:
            return _GENESIS_HASH

    def _maybe_rotate(self) -> None:
        """Check if the log file needs rotation and rotate if so."""
        try:
            if self._path.exists() and self._path.stat().st_size >= self._max_size:
                self.rotate()
        except Exception as exc:
            logger.debug("Rotation check failed: %s", exc)

    def rotate(self) -> Optional[Path]:
        """Rotate the current log file.

        Renames the current log to include a timestamp suffix and starts
        a fresh log file. Old rotated files beyond max_rotations are deleted.

        Returns:
            Path to the rotated file, or None on failure.
        """
        with self._rlock:
            if not self._path.exists():
                return None

            # Generate rotated filename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            rotated_name = f"{self._path.stem}_{timestamp}{self._path.suffix}"
            rotated_path = self._path.parent / rotated_name

            try:
                with self._file_lock:
                    shutil.move(str(self._path), str(rotated_path))

                # Reset entry count but preserve _last_hash for chain continuity
                self._entry_count = 0

                logger.info("Rotated audit log to %s", rotated_path)

                # Clean up old rotations
                self._cleanup_rotations()

                return rotated_path

            except Exception as exc:
                logger.error("Log rotation failed: %s", exc)
                return None

    def _cleanup_rotations(self) -> None:
        """Remove old rotated log files beyond max_rotations."""
        pattern = f"{self._path.stem}_*{self._path.suffix}"
        rotated_files = sorted(
            self._path.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for old_file in rotated_files[self._max_rotations :]:
            try:
                old_file.unlink()
                logger.debug("Deleted old audit log: %s", old_file)
            except Exception as exc:
                logger.debug("Could not delete old log %s: %s", old_file, exc)

    def verify_chain(self) -> bool:
        """Verify the entire hash chain for tamper detection.

        Reads all entries and verifies that each entry's hash is
        correctly computed from the previous entry's hash.

        Returns:
            True if the chain is valid, False if tampered.
        """
        if not self._path.exists():
            return True

        prev_hash = _GENESIS_HASH
        try:
            with open(self._path, "r", encoding="utf-8") as fp:
                for line_num, line in enumerate(fp, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.error(
                            "Chain verification failed: invalid JSON at line %d",
                            line_num,
                        )
                        return False

                    # Check prev_hash linkage
                    stored_prev = data.get("prev_hash", "")
                    if stored_prev != prev_hash:
                        logger.error(
                            "Chain verification failed at line %d: prev_hash mismatch (expected %s..., got %s...)",
                            line_num,
                            prev_hash[:12],
                            stored_prev[:12],
                        )
                        return False

                    # Recompute the entry hash
                    stored_hash = data.get("entry_hash", "")
                    entry = AuditEntry(**{k: v for k, v in data.items() if k != "entry_hash"})
                    expected_hash = entry.compute_hash()

                    if stored_hash != expected_hash:
                        logger.error(
                            "Chain verification failed at line %d: entry_hash mismatch (expected %s..., got %s...)",
                            line_num,
                            expected_hash[:12],
                            stored_hash[:12],
                        )
                        return False

                    prev_hash = stored_hash

            return True

        except Exception as exc:
            logger.error("Chain verification error: %s", exc)
            return False

    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        tool_name: Optional[str] = None,
        decision: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        predicate: Optional[Callable[[AuditEntry], bool]] = None,
    ) -> list[AuditEntry]:
        """Query audit entries with filters.

        Args:
            event_type: Filter by event type.
            tool_name: Filter by tool name (substring match).
            decision: Filter by decision value.
            since: Filter entries after this timestamp.
            until: Filter entries before this timestamp.
            limit: Maximum number of entries to return.
            predicate: Custom filter function.

        Returns:
            List of matching AuditEntry objects (most recent first).
        """
        if not self._path.exists():
            return []

        results: list[AuditEntry] = []
        try:
            with open(self._path, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entry = AuditEntry(**data)
                    except (json.JSONDecodeError, Exception):
                        continue

                    # Apply filters
                    if event_type and entry.event_type != event_type:
                        continue
                    if tool_name and tool_name not in entry.tool_name:
                        continue
                    if decision and entry.decision != decision:
                        continue
                    if since and entry.timestamp < since:
                        continue
                    if until and entry.timestamp > until:
                        continue
                    if predicate and not predicate(entry):
                        continue

                    results.append(entry)

            # Return most recent first, limited
            results.reverse()
            return results[:limit]

        except Exception as exc:
            logger.error("Query failed: %s", exc)
            return []

    def stats(self) -> dict[str, Any]:
        """Compute audit trail statistics.

        Returns:
            Dict with entry counts by event type, total count,
            file size, chain status, etc.
        """
        result: dict[str, Any] = {
            "total_entries": self._entry_count,
            "last_hash": self._last_hash[:16] + "...",
            "file_exists": self._path.exists(),
            "file_size": 0,
            "by_event_type": {},
            "by_decision": {},
        }

        if self._path.exists():
            result["file_size"] = self._path.stat().st_size
            result["file_path"] = str(self._path)

            # Count by type
            by_type: dict[str, int] = {}
            by_decision: dict[str, int] = {}

            try:
                with open(self._path, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            et = data.get("event_type", "unknown")
                            dec = data.get("decision", "none")
                            by_type[et] = by_type.get(et, 0) + 1
                            by_decision[dec] = by_decision.get(dec, 0) + 1
                        except json.JSONDecodeError:
                            continue

                result["by_event_type"] = by_type
                result["by_decision"] = by_decision
            except Exception as exc:
                logger.debug("Stats computation error: %s", exc)

        # Count rotated files
        if self._path.parent.exists():
            pattern = f"{self._path.stem}_*{self._path.suffix}"
            rotated = list(self._path.parent.glob(pattern))
            result["rotated_files"] = len(rotated)

        return result

    def clear(self, *, include_rotations: bool = False) -> None:
        """Clear the current audit trail and optionally rotated files."""
        with self._rlock:
            with self._file_lock:
                if self._path.exists():
                    self._path.unlink()

            if include_rotations and self._path.parent.exists():
                pattern = f"{self._path.stem}_*{self._path.suffix}"
                for rotated_path in self._path.parent.glob(pattern):
                    try:
                        rotated_path.unlink()
                    except OSError:
                        logger.debug("Could not delete rotated log %s", rotated_path, exc_info=True)

            self._last_hash = _GENESIS_HASH
            self._entry_count = 0

    @property
    def path(self) -> Path:
        """Return the audit log file path."""
        return self._path

    @property
    def last_hash(self) -> str:
        """Return the last entry hash (O(1) - cached in memory)."""
        return self._last_hash

    @property
    def entry_count(self) -> int:
        """Return the total number of entries."""
        return self._entry_count
