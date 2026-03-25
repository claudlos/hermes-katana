"""Tests for HermesKatana audit trail."""

from __future__ import annotations

import json


from hermes_katana.audit.trail import (
    AuditEntry,
    AuditEventType,
    AuditTrail,
    _GENESIS_HASH,
)


# ======================================================================
# AuditEntry
# ======================================================================

class TestAuditEntry:
    def test_create_entry(self):
        entry = AuditEntry(
            event_type=AuditEventType.TOOL_CALL,
            tool_name="terminal",
            decision="allow",
        )
        assert entry.event_type == AuditEventType.TOOL_CALL
        assert entry.tool_name == "terminal"
        assert entry.decision == "allow"

    def test_finalize_sets_hashes(self):
        entry = AuditEntry(
            event_type=AuditEventType.TOOL_CALL,
            tool_name="terminal",
        )
        entry.finalize(_GENESIS_HASH)
        assert entry.prev_hash == _GENESIS_HASH
        assert entry.entry_hash != ""
        assert len(entry.entry_hash) == 64  # SHA-256 hex

    def test_compute_hash_deterministic(self):
        entry = AuditEntry(
            event_type=AuditEventType.TOOL_CALL,
            tool_name="terminal",
            decision="allow",
            prev_hash=_GENESIS_HASH,
        )
        hash1 = entry.compute_hash()
        hash2 = entry.compute_hash()
        assert hash1 == hash2

    def test_different_entries_different_hashes(self):
        e1 = AuditEntry(
            event_type=AuditEventType.TOOL_CALL,
            tool_name="terminal",
            prev_hash=_GENESIS_HASH,
        )
        e2 = AuditEntry(
            event_type=AuditEventType.SCAN_RESULT,
            tool_name="scanner",
            prev_hash=_GENESIS_HASH,
        )
        assert e1.compute_hash() != e2.compute_hash()


# ======================================================================
# AuditTrail — append and hash chain
# ======================================================================

class TestAuditTrailAppend:
    def test_append_entry(self, audit_path):
        trail = AuditTrail(path=audit_path)
        entry = AuditEntry(
            event_type=AuditEventType.TOOL_CALL,
            tool_name="terminal",
            decision="allow",
        )
        entry_hash = trail.log(entry)
        assert len(entry_hash) == 64
        assert trail.entry_count == 1
        assert trail.last_hash == entry_hash

    def test_append_multiple_entries(self, audit_path):
        trail = AuditTrail(path=audit_path)
        hashes = []
        for i in range(5):
            entry = AuditEntry(
                event_type=AuditEventType.TOOL_CALL,
                tool_name=f"tool_{i}",
                decision="allow",
            )
            h = trail.log(entry)
            hashes.append(h)

        assert trail.entry_count == 5
        assert trail.last_hash == hashes[-1]
        # All hashes should be unique
        assert len(set(hashes)) == 5

    def test_hash_chain_linked(self, audit_path):
        trail = AuditTrail(path=audit_path)
        e1 = AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="t1")
        h1 = trail.log(e1)

        e2 = AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="t2")
        trail.log(e2)

        # Read raw file and check chain linkage
        with open(audit_path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert lines[0]["prev_hash"] == _GENESIS_HASH
        assert lines[1]["prev_hash"] == h1


# ======================================================================
# Chain verification
# ======================================================================

class TestChainVerification:
    def test_valid_chain_verifies(self, audit_path):
        trail = AuditTrail(path=audit_path)
        for i in range(10):
            entry = AuditEntry(
                event_type=AuditEventType.TOOL_CALL,
                tool_name=f"tool_{i}",
                decision="allow",
            )
            trail.log(entry)

        assert trail.verify_chain() is True

    def test_tampered_entry_detected(self, audit_path):
        trail = AuditTrail(path=audit_path)
        for i in range(5):
            entry = AuditEntry(
                event_type=AuditEventType.TOOL_CALL,
                tool_name=f"tool_{i}",
                decision="allow",
            )
            trail.log(entry)

        # Tamper with the file — modify a decision field
        with open(audit_path) as f:
            lines = f.readlines()

        data = json.loads(lines[2])
        data["decision"] = "TAMPERED"
        lines[2] = json.dumps(data) + "\n"

        with open(audit_path, "w") as f:
            f.writelines(lines)

        assert trail.verify_chain() is False

    def test_empty_trail_verifies(self, audit_path):
        trail = AuditTrail(path=audit_path)
        assert trail.verify_chain() is True


# ======================================================================
# Query with filters
# ======================================================================

class TestAuditQuery:
    def test_query_by_event_type(self, audit_path):
        trail = AuditTrail(path=audit_path)
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="terminal"))
        trail.log(AuditEntry(event_type=AuditEventType.SCAN_RESULT, tool_name="scanner"))
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="write_file"))

        results = trail.query(event_type=AuditEventType.TOOL_CALL)
        assert len(results) == 2
        assert all(e.event_type == AuditEventType.TOOL_CALL for e in results)

    def test_query_by_tool_name(self, audit_path):
        trail = AuditTrail(path=audit_path)
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="terminal"))
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="write_file"))
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name="terminal"))

        results = trail.query(tool_name="terminal")
        assert len(results) == 2

    def test_query_by_decision(self, audit_path):
        trail = AuditTrail(path=audit_path)
        trail.log(AuditEntry(event_type=AuditEventType.POLICY_DECISION, decision="deny"))
        trail.log(AuditEntry(event_type=AuditEventType.POLICY_DECISION, decision="allow"))
        trail.log(AuditEntry(event_type=AuditEventType.POLICY_DECISION, decision="deny"))

        results = trail.query(decision="deny")
        assert len(results) == 2

    def test_query_with_limit(self, audit_path):
        trail = AuditTrail(path=audit_path)
        for i in range(20):
            trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, tool_name=f"t{i}"))

        results = trail.query(limit=5)
        assert len(results) == 5

    def test_query_empty_trail(self, audit_path):
        trail = AuditTrail(path=audit_path)
        results = trail.query()
        assert len(results) == 0


# ======================================================================
# Stats
# ======================================================================

class TestAuditStats:
    def test_stats_counts(self, audit_path):
        trail = AuditTrail(path=audit_path)
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, decision="allow"))
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL, decision="deny"))
        trail.log(AuditEntry(event_type=AuditEventType.SCAN_RESULT, decision="warn"))

        stats = trail.stats()
        assert stats["total_entries"] == 3
        assert stats["file_exists"] is True
        assert stats["by_event_type"]["tool_call"] == 2
        assert stats["by_event_type"]["scan_result"] == 1
        assert stats["by_decision"]["allow"] == 1
        assert stats["by_decision"]["deny"] == 1

    def test_stats_empty_trail(self, audit_path):
        trail = AuditTrail(path=audit_path)
        stats = trail.stats()
        assert stats["total_entries"] == 0

    def test_last_hash_property(self, audit_path):
        trail = AuditTrail(path=audit_path)
        assert trail.last_hash == _GENESIS_HASH
        trail.log(AuditEntry(event_type=AuditEventType.TOOL_CALL))
        assert trail.last_hash != _GENESIS_HASH
