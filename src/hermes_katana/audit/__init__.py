"""
HermesKatana Audit Module - SHA-256 hash-chained append-only audit log.

Provides tamper-evident audit logging with:
- SHA-256 hash-chained entries (each entry hashes the previous)
- O(1) last-hash tracking (cached in memory, not read from file)
- File locking for concurrent writers
- Automatic rotation when log exceeds size threshold
- Structured entries with Pydantic models
- Chain verification for tamper detection

Usage:
    from hermes_katana.audit import AuditTrail, AuditEntry

    trail = AuditTrail()
    entry = AuditEntry(
        event_type=AuditEventType.TOOL_CALL,
        tool_name="terminal",
        args_hash="abc123",
        decision="allow",
    )
    trail.log(entry)
    assert trail.verify_chain()
"""

from hermes_katana.audit.trail import (
    AuditEntry,
    AuditEventType,
    AuditTrail,
    default_audit_path,
)

__all__ = [
    "AuditTrail",
    "AuditEntry",
    "AuditEventType",
    "default_audit_path",
]
