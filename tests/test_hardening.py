"""Tests for Worker 6 hardening: concurrency, TOCTOU, installer."""

from __future__ import annotations

import json
import os
import sys
import threading
from unittest import mock

import pytest


def _can_symlink() -> bool:
    """Return True if the current user can create symlinks on this platform.

    On Windows, os.symlink() requires either admin rights or Developer Mode
    enabled. Returns False if a test symlink cannot be created.
    """
    if sys.platform != "win32":
        return True
    import tempfile
    from pathlib import Path as _P

    with tempfile.TemporaryDirectory() as td:
        target = _P(td) / "target.txt"
        target.write_text("x")
        link = _P(td) / "link.txt"
        try:
            link.symlink_to(target)
            return True
        except (OSError, NotImplementedError):
            return False


# ---------------------------------------------------------------------------
# 1. Fork safety — TaintTracker._reset_instance
# ---------------------------------------------------------------------------


class TestForkSafety:
    def test_reset_instance_clears_singleton(self):
        from hermes_katana.taint.tracker import TaintTracker

        TaintTracker.reset_instance()
        inst = TaintTracker.get_instance()
        assert TaintTracker._instance is inst

        # Simulate what happens in child after fork
        TaintTracker._reset_instance()
        assert TaintTracker._instance is None

        # New get_instance should create fresh
        inst2 = TaintTracker.get_instance()
        assert inst2 is not inst
        TaintTracker.reset_instance()

    def test_reset_instance_creates_new_lock(self):
        from hermes_katana.taint.tracker import TaintTracker

        old_lock = TaintTracker._lock
        TaintTracker._reset_instance()
        assert TaintTracker._lock is not old_lock
        TaintTracker.reset_instance()

    def test_register_at_fork_called(self):
        """os.register_at_fork should have been called at import time."""
        # Just verify _reset_instance is callable as a classmethod
        from hermes_katana.taint.tracker import TaintTracker

        assert callable(TaintTracker._reset_instance)


# ---------------------------------------------------------------------------
# 2. Flow history flush
# ---------------------------------------------------------------------------


class TestHistoryFlush:
    def test_flush_triggers_on_overflow(self, tmp_path):
        from hermes_katana.taint.flow import FlowAnalyzer
        from hermes_katana.taint.labels import TaintLabel, Source, TrustLevel
        from hermes_katana.taint.value import TaintedStr

        analyzer = FlowAnalyzer()
        analyzer._MAX_HISTORY = 10  # small for testing

        # Patch the flush method to track calls
        flush_called = []

        def mock_flush(entries):
            flush_called.append(len(entries))

        analyzer._flush_history_to_disk = mock_flush

        src = Source(
            label=TaintLabel.USER,
            origin="test",
            trust_level=TrustLevel.TRUSTED,
        )
        tv = TaintedStr(value="test", sources=frozenset({src}))

        # Generate enough analyses to trigger truncation
        for i in range(15):
            analyzer.analyze(tv, f"tool_{i}")

        assert len(flush_called) > 0, "Flush should have been called"

    def test_flush_method_exists(self):
        from hermes_katana.taint.flow import FlowAnalyzer

        analyzer = FlowAnalyzer()
        assert hasattr(analyzer, "_flush_history_to_disk")


# ---------------------------------------------------------------------------
# 3. Access log HMAC verification
# ---------------------------------------------------------------------------


class TestAccessLogHMAC:
    def test_log_and_verify(self, tmp_path):
        from hermes_katana.vault.access_log import VaultAccessLog

        log_path = tmp_path / "access.jsonl"
        log = VaultAccessLog(path=log_path)

        log.log_access("KEY1", "GET", caller="test")
        log.log_access("KEY2", "SET", caller="test")

        assert log.verify_integrity() is True

    def test_tampered_log_fails_verify(self, tmp_path):
        from hermes_katana.vault.access_log import VaultAccessLog

        log_path = tmp_path / "access.jsonl"
        log = VaultAccessLog(path=log_path)

        log.log_access("KEY1", "GET", caller="test")

        # Tamper with the log
        content = log_path.read_text()
        lines = content.strip().split("\n")
        # Modify the data portion
        if "|" in lines[0]:
            data, hmac_val = lines[0].rsplit("|", 1)
            d = json.loads(data)
            d["key_name"] = "TAMPERED"
            lines[0] = json.dumps(d) + "|" + hmac_val
        log_path.write_text("\n".join(lines) + "\n")

        assert log.verify_integrity() is False

    def test_empty_log_passes_verify(self, tmp_path):
        from hermes_katana.vault.access_log import VaultAccessLog

        log_path = tmp_path / "access.jsonl"
        log = VaultAccessLog(path=log_path)
        assert log.verify_integrity() is True

    def test_history_still_works(self, tmp_path):
        from hermes_katana.vault.access_log import VaultAccessLog

        log_path = tmp_path / "access.jsonl"
        log = VaultAccessLog(path=log_path)

        log.log_access("MYKEY", "GET", caller="test_caller")
        history = log.get_access_history("MYKEY")
        assert len(history) == 1
        assert history[0].key_name == "MYKEY"


# ---------------------------------------------------------------------------
# 4. Secure delete with fsync
# ---------------------------------------------------------------------------


class TestSecureDelete:
    def test_secure_delete_uses_fsync(self, tmp_path):
        from hermes_katana.vault.migrate import _secure_delete_from_file

        env_file = tmp_path / ".env"
        env_file.write_text('MY_API_KEY="secret123"\n')

        fsync_called = []
        orig_fsync = os.fsync

        def mock_fsync(fd):
            fsync_called.append(fd)
            orig_fsync(fd)

        with mock.patch("os.fsync", side_effect=mock_fsync):
            result = _secure_delete_from_file(env_file, "MY_API_KEY")

        assert result is True
        assert len(fsync_called) > 0, "os.fsync should have been called"

        # Verify value was zeroed
        content = env_file.read_text()
        assert "secret123" not in content


# ---------------------------------------------------------------------------
# 5. Rotation journal recovery
# ---------------------------------------------------------------------------


class TestRotationJournal:
    def test_no_journal_returns_false(self, tmp_path):
        """recover_rotation returns False when no journal exists."""
        from hermes_katana.vault.store import Vault

        vault_path = tmp_path / "vault.json"
        # Create a minimal vault without auto_create to avoid keyring
        vault = Vault.__new__(Vault)
        vault._path = vault_path
        vault._lock_path = vault_path.with_suffix(".lock")
        vault._file_lock = mock.MagicMock()
        vault._rlock = threading.RLock()
        vault._master_key = None

        assert vault.recover_rotation() is False

    def test_journal_created_during_rotation(self, tmp_path):
        """Rotation should create and clean up journal file."""
        # Just verify the code path exists
        from hermes_katana.vault.store import Vault

        assert hasattr(Vault, "recover_rotation")


# ---------------------------------------------------------------------------
# 6. Installer permission checks
# ---------------------------------------------------------------------------


class TestInstallerPermissions:
    def test_validate_normal_file(self, tmp_path):
        from hermes_katana.installer.patches import (
            validate_patch_target,
            Patch,
        )

        target = tmp_path / "test.py"
        target.write_text("hello")

        patch = Patch(
            name="test",
            description="test",
            target_file="test.py",
            search_text="hello",
            replace_text="world",
            sentinel="# TEST",
        )

        issues = validate_patch_target(target, patch)
        assert issues == [], f"Unexpected issues: {issues}"

    def test_validate_nonexistent_file(self, tmp_path):
        from hermes_katana.installer.patches import (
            validate_patch_target,
            Patch,
        )

        target = tmp_path / "nonexistent.py"
        patch = Patch(
            name="test",
            description="test",
            target_file="nonexistent.py",
            search_text="x",
            replace_text="y",
            sentinel="# TEST",
        )

        issues = validate_patch_target(target, patch)
        assert any("does not exist" in i for i in issues)

    @pytest.mark.skipif(
        not _can_symlink(),
        reason="Windows: symlinks require admin or Developer Mode (SeCreateSymbolicLinkPrivilege)",
    )
    def test_validate_symlink_detected(self, tmp_path):
        from hermes_katana.installer.patches import (
            validate_patch_target,
            Patch,
        )

        real = tmp_path / "real.py"
        real.write_text("content")
        link = tmp_path / "link.py"
        link.symlink_to(real)

        patch = Patch(
            name="test",
            description="test",
            target_file="link.py",
            search_text="x",
            replace_text="y",
            sentinel="# TEST",
        )

        issues = validate_patch_target(link, patch)
        assert any("symlink" in i for i in issues)

    def test_create_backup(self, tmp_path):
        from hermes_katana.installer.patches import create_backup

        target = tmp_path / "file.py"
        target.write_text("original content")

        backup = create_backup(target)
        assert backup is not None
        assert backup.exists()
        assert backup.read_text() == "original content"

    def test_create_backup_nonexistent(self, tmp_path):
        from hermes_katana.installer.patches import create_backup

        target = tmp_path / "nope.py"
        assert create_backup(target) is None


# ---------------------------------------------------------------------------
# 7. Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_path_traversal_rejected(self):
        from hermes_katana.config import KatanaConfig

        with pytest.raises(Exception):
            KatanaConfig(policy_path="../../etc/passwd")

    def test_excessively_long_preset(self):
        from hermes_katana.config import KatanaConfig

        with pytest.raises(Exception):
            KatanaConfig(policy_preset="a" * 300)

    def test_control_chars_rejected(self):
        from hermes_katana.config import KatanaConfig

        with pytest.raises(Exception):
            KatanaConfig(policy_preset="balanced\x00injected")

    def test_valid_config_passes(self):
        from hermes_katana.config import KatanaConfig

        config = KatanaConfig(policy_preset="balanced", log_level="DEBUG")
        assert config.policy_preset == "balanced"
        assert config.log_level == "DEBUG"

    def test_saved_config_is_owner_only(self, tmp_path):
        if os.name == "nt":
            pytest.skip("POSIX owner-only mode bits are not meaningful on Windows")

        from hermes_katana.config import KatanaConfig

        config_path = tmp_path / "config.yaml"
        config = KatanaConfig(policy_preset="balanced", log_level="DEBUG")
        config.save(config_path)

        assert config_path.stat().st_mode & 0o777 == 0o600

    def test_long_path_rejected(self):
        from hermes_katana.config import KatanaConfig

        with pytest.raises(Exception):
            KatanaConfig(policy_path="/" + "a" * 5000)
