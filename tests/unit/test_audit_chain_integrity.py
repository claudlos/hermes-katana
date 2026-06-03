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


def test_query_and_stats_include_rotated_history(audit_path):
    trail = AuditTrail(path=audit_path, max_size=10_000, max_rotations=5)
    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="before", decision="allow"))
    rotated = trail.rotate()
    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="after", decision="deny"))

    assert rotated is not None

    results = trail.query(event_type=AuditEventType.TOOL_CALL)
    stats = trail.stats()

    assert [entry.tool_name for entry in results] == ["after", "before"]
    assert stats["total_entries"] == 2
    assert stats["by_event_type"]["tool_call"] == 2
    assert stats["by_decision"]["allow"] == 1
    assert stats["by_decision"]["deny"] == 1
    assert stats["rotated_files"] == 1


def test_stats_file_exists_when_only_rotated_history_exists(audit_path):
    trail = AuditTrail(path=audit_path, max_size=10_000, max_rotations=5)
    trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="before", decision="allow"))
    rotated = trail.rotate()

    assert rotated is not None
    assert not audit_path.exists()

    stats = trail.stats()

    assert stats["file_exists"] is True
    assert stats["active_file_size"] == 0
    assert stats["total_entries"] == 1
    assert stats["history_files"] == [str(rotated)]


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
