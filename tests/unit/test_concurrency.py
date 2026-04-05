"""Concurrency stress tests for vault, audit trail, and config.

Tests multi-threaded access to ensure thread safety and data integrity
under concurrent operations.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch as mock_patch

import pytest

# ---------------------------------------------------------------------------
# Vault concurrency tests
# ---------------------------------------------------------------------------


class TestVaultConcurrency:
    """Stress-test the vault under concurrent thread access."""

    @pytest.fixture()
    def vault(self, tmp_path):
        """Create a vault with a test master key."""
        import base64
        import secrets

        test_key = secrets.token_bytes(32)
        encoded_key = base64.b64encode(test_key).decode("ascii")

        vault_path = tmp_path / "vault.json"

        with mock_patch.dict(
            os.environ,
            {"HERMES_KATANA_VAULT_KEY": encoded_key},
        ):
            from hermes_katana.vault.store import Vault

            v = Vault(path=vault_path, auto_create=True)
            yield v

    def test_concurrent_set_operations(self, vault):
        """Multiple threads setting different keys should not corrupt the vault."""
        num_threads = 8
        keys_per_thread = 5
        errors = []

        def writer(thread_id: int):
            try:
                for i in range(keys_per_thread):
                    vault.set(f"thread{thread_id}_key{i}", f"value_{thread_id}_{i}")
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=writer, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent set errors: {errors}"

        # Verify all keys exist and are correct
        for tid in range(num_threads):
            for i in range(keys_per_thread):
                val = vault.get(f"thread{tid}_key{i}")
                assert val == f"value_{tid}_{i}"

    def test_concurrent_get_operations(self, vault):
        """Multiple threads reading the same key should all get the right value."""
        vault.set("shared_key", "shared_value")
        num_threads = 16
        results = []
        errors = []

        def reader(thread_id: int):
            try:
                val = vault.get("shared_key")
                results.append(val)
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=reader, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent get errors: {errors}"
        assert all(v == "shared_value" for v in results)
        assert len(results) == num_threads

    def test_concurrent_set_and_get(self, vault):
        """Mixed reads and writes should not corrupt data or raise errors."""
        vault.set("counter_key", "initial")
        errors = []

        def mixed_ops(thread_id: int):
            try:
                for i in range(10):
                    if i % 2 == 0:
                        vault.set(f"mixed_{thread_id}", f"val_{i}")
                    else:
                        try:
                            vault.get(f"mixed_{thread_id}")
                        except Exception:
                            pass  # Key may not exist yet from another thread
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(mixed_ops, tid) for tid in range(8)]
            for f in as_completed(futures):
                f.result()  # propagate exceptions

        assert not errors, f"Mixed operation errors: {errors}"

    def test_concurrent_remove(self, vault):
        """Concurrent removes should not corrupt the vault."""
        # Pre-populate
        for i in range(20):
            vault.set(f"remove_key_{i}", f"val_{i}")

        def remover(key_id: int):
            try:
                vault.remove(f"remove_key_{key_id}")
            except Exception:
                pass  # Another thread may have removed it first

        threads = []
        for i in range(20):
            t = threading.Thread(target=remover, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        # Vault should still be readable
        keys = vault.list_keys()
        assert isinstance(keys, list)


# ---------------------------------------------------------------------------
# Audit trail concurrency tests
# ---------------------------------------------------------------------------


class TestAuditTrailConcurrency:
    """Stress-test the audit trail under concurrent thread access."""

    @pytest.fixture()
    def trail(self, tmp_path):
        from hermes_katana.audit.trail import AuditTrail

        return AuditTrail(path=tmp_path / "audit.jsonl")

    def test_concurrent_log_entries(self, trail):
        """Multiple threads logging entries should produce a valid chain."""
        from hermes_katana.audit.trail import AuditEntry, AuditEventType

        num_threads = 8
        entries_per_thread = 10
        errors = []

        def logger_fn(thread_id: int):
            try:
                for i in range(entries_per_thread):
                    entry = AuditEntry(
                        event_type=AuditEventType.TOOL_CALL,
                        tool_name=f"thread_{thread_id}",
                        details=f"entry_{i}",
                        decision="allow",
                    )
                    trail.log(entry)
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = []
        for tid in range(num_threads):
            t = threading.Thread(target=logger_fn, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Concurrent log errors: {errors}"

        # Verify total entry count
        total_expected = num_threads * entries_per_thread
        count = 0
        with open(trail._path, "r") as fp:
            for line in fp:
                if line.strip():
                    count += 1
        assert count == total_expected, f"Expected {total_expected} entries, got {count}"

    def test_concurrent_log_chain_integrity(self, trail):
        """Hash chain should remain valid after concurrent writes."""
        from hermes_katana.audit.trail import AuditEntry, AuditEventType

        errors = []

        def logger_fn(thread_id: int):
            try:
                for i in range(5):
                    entry = AuditEntry(
                        event_type=AuditEventType.SCAN_RESULT,
                        tool_name=f"scanner_{thread_id}",
                        details=f"finding_{i}",
                    )
                    trail.log(entry)
            except Exception as exc:
                errors.append(f"Thread {thread_id}: {exc}")

        threads = []
        for tid in range(4):
            t = threading.Thread(target=logger_fn, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Log errors: {errors}"

        # The chain must be valid
        assert trail.verify_chain(), "Hash chain integrity broken after concurrent writes"

    def test_concurrent_log_with_threadpool(self, trail):
        """ThreadPoolExecutor-based stress test."""
        from hermes_katana.audit.trail import AuditEntry, AuditEventType

        def log_one(idx: int) -> str:
            entry = AuditEntry(
                event_type=AuditEventType.POLICY_DECISION,
                tool_name="terminal",
                decision="allow" if idx % 2 == 0 else "deny",
                details=f"op_{idx}",
            )
            return trail.log(entry)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(log_one, i) for i in range(30)]
            hashes = [f.result() for f in as_completed(futures)]

        assert len(hashes) == 30
        assert len(set(hashes)) == 30  # All hashes should be unique
        assert trail.verify_chain()


# ---------------------------------------------------------------------------
# Config concurrency & hardening tests
# ---------------------------------------------------------------------------


class TestConfigHardening:
    """Test config validation against malicious/corrupt inputs."""

    def test_path_traversal_in_policy_path(self):
        """policy_path with .. components should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            KatanaConfig(policy_path="/etc/../../../etc/shadow")

    def test_policy_path_directory_rejected(self, tmp_path):
        """policy_path pointing to a directory should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError, match="must be a file"):
            KatanaConfig(policy_path=tmp_path)

    def test_domain_allowlist_rejects_urls(self):
        """Entries with :// should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError, match="looks like a URL"):
            KatanaConfig(domain_allowlist=["https://evil.com"])

    def test_domain_allowlist_rejects_paths(self):
        """Entries with / path components should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError, match="path component"):
            KatanaConfig(domain_allowlist=["example.com/admin"])

    def test_domain_allowlist_rejects_bad_wildcards(self):
        """Only leading *. wildcards are allowed."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError, match="[Ii]nvalid wildcard"):
            KatanaConfig(domain_allowlist=["evil.*.com"])

    def test_domain_allowlist_accepts_valid(self):
        """Valid domain entries should pass."""
        from hermes_katana.config import KatanaConfig

        cfg = KatanaConfig(
            domain_allowlist=[
                "api.example.com",
                "*.internal.co",
                "localhost",
            ]
        )
        assert cfg.domain_allowlist == ["api.example.com", "*.internal.co", "localhost"]

    def test_invalid_preset_rejected(self):
        """Unknown policy_preset should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError):
            KatanaConfig(policy_preset="yolo")

    def test_invalid_log_level_rejected(self):
        """Unknown log_level should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError):
            KatanaConfig(log_level="VERBOSE")

    def test_port_out_of_range(self):
        """Proxy port outside 1-65535 should be rejected."""
        from hermes_katana.config import KatanaConfig

        with pytest.raises(ValueError):
            KatanaConfig(proxy_port=99999)

    def test_corrupt_yaml_falls_back_to_defaults(self, tmp_path):
        """A corrupt config file should result in defaults, not a crash."""
        from hermes_katana.config import load_config

        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text("{{{{not: valid: yaml: [[[", encoding="utf-8")
        cfg = load_config(path=bad_config)
        assert cfg.policy_preset == "balanced"  # default

    def test_malicious_yaml_bomb(self, tmp_path):
        """yaml.safe_load should handle billion-laughs-style input safely."""
        from hermes_katana.config import load_config

        # safe_load won't expand aliases dangerously, but test robustness
        bomb = tmp_path / "bomb.yaml"
        bomb.write_text(
            "a: &a [1,2,3]\nb: &b [*a,*a,*a]\nc: &c [*b,*b,*b]\n",
            encoding="utf-8",
        )
        cfg = load_config(path=bomb)
        # Should load (safe_load handles this) or fall back to defaults
        assert cfg.policy_preset == "balanced"

    def test_env_override_with_invalid_int(self, tmp_path):
        """Invalid integer env var should be skipped, not crash."""
        from hermes_katana.config import load_config

        with mock_patch.dict(os.environ, {"KATANA_PROXY_PORT": "not_a_number"}):
            cfg = load_config(path=tmp_path / "nonexistent.yaml")
            assert cfg.proxy_port == 8443  # default preserved

    def test_valid_policy_path_accepted(self, tmp_path):
        """A valid absolute policy path should be accepted."""
        from hermes_katana.config import KatanaConfig

        policy_file = tmp_path / "custom.yaml"
        policy_file.write_text("rules: []", encoding="utf-8")
        cfg = KatanaConfig(policy_path=policy_file)
        assert cfg.policy_path == policy_file.resolve()
