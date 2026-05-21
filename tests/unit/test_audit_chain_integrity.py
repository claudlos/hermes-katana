"""Regression tests for audit-chain integrity across maintenance actions."""

from __future__ import annotations

import json

from hermes_katana.audit.trail import AuditEntry, AuditEventType, AuditTrail


def test_rotation_preserves_inter_file_chain(audit_path):
    trail = AuditTrail(path=audit_path, max_size=10_000, max_rotations=5)
    first_hash = trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="before"))
    rotated = trail.rotate()

    assert rotated is not None

    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="after"))
    with open(audit_path) as f:
        first_active = json.loads(next(line for line in f if line.strip()))

    assert first_active["prev_hash"] == first_hash
    assert trail.verify_chain() is True


def test_missing_rotated_file_breaks_chain_verification(audit_path):
    trail = AuditTrail(path=audit_path, max_size=10_000, max_rotations=5)
    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="before"))
    rotated = trail.rotate()
    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="after"))

    assert rotated is not None
    rotated.unlink()

    assert trail.verify_chain() is False


def test_clear_records_sentinel_instead_of_unlinking(audit_path):
    trail = AuditTrail(path=audit_path)
    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="before"))
    trail.clear(include_rotations=True)

    events = [json.loads(line)["event_type"] for line in audit_path.read_text().splitlines()]
    assert events[-1] == AuditEventType.TRAIL_CLEARED
    assert trail.verify_chain() is True
