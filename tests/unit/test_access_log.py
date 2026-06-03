"""Tests for hermes_katana.vault.access_log."""

from __future__ import annotations

import json
import time

import pytest

from hermes_katana.vault.access_log import AccessEntry, VaultAccessLog


@pytest.fixture
def log_path(tmp_path):
    return tmp_path / "vault_access.jsonl"


@pytest.fixture
def access_log(log_path):
    return VaultAccessLog(path=log_path)


class TestAccessEntry:
    def test_defaults(self):
        entry = AccessEntry(key_name="KEY", operation="GET")
        assert entry.key_name == "KEY"
        assert entry.operation == "GET"
        assert entry.success is True
        assert entry.caller == ""
        assert entry.timestamp > 0

    def test_frozen(self):
        entry = AccessEntry(key_name="K", operation="SET")
        with pytest.raises(AttributeError):
            entry.key_name = "other"  # type: ignore


class TestVaultAccessLog:
    def test_log_and_retrieve(self, access_log):
        access_log.log_access("MY_KEY", "GET", caller="test:func")
        history = access_log.get_access_history("MY_KEY")
        assert len(history) == 1
        assert history[0].key_name == "MY_KEY"
        assert history[0].operation == "GET"
        assert history[0].caller == "test:func"

    def test_multiple_entries(self, access_log):
        access_log.log_access("K1", "GET")
        access_log.log_access("K2", "SET")
        access_log.log_access("K1", "GET")

        k1_history = access_log.get_access_history("K1")
        assert len(k1_history) == 2

        k2_history = access_log.get_access_history("K2")
        assert len(k2_history) == 1

    def test_most_recent_first(self, access_log):
        access_log.log_access("K", "GET", caller="first")
        time.sleep(0.01)
        access_log.log_access("K", "SET", caller="second")

        history = access_log.get_access_history("K")
        assert history[0].caller == "second"
        assert history[1].caller == "first"

    def test_limit(self, access_log):
        for i in range(20):
            access_log.log_access("K", "GET", caller=f"caller_{i}")

        history = access_log.get_access_history("K", limit=5)
        assert len(history) == 5

    def test_get_all_access(self, access_log):
        access_log.log_access("K1", "GET")
        access_log.log_access("K2", "SET")
        access_log.log_access("K3", "DELETE")

        all_entries = access_log.get_all_access()
        assert len(all_entries) == 3

    def test_get_all_with_since(self, access_log):
        access_log.log_access("OLD", "GET")
        time.sleep(0.05)
        cutoff = time.time()
        time.sleep(0.05)
        access_log.log_access("NEW", "SET")

        recent = access_log.get_all_access(since=cutoff)
        assert len(recent) == 1
        assert recent[0].key_name == "NEW"

    def test_failure_logging(self, access_log):
        access_log.log_access("K", "GET", success=False, detail="Key not found")

        history = access_log.get_access_history("K")
        assert len(history) == 1
        assert history[0].success is False
        assert history[0].detail == "Key not found"

    def test_empty_log(self, access_log):
        assert access_log.get_access_history("K") == []
        assert access_log.get_all_access() == []

    def test_clear(self, access_log, log_path):
        access_log.log_access("K", "GET")
        assert log_path.exists()
        access_log.clear()
        assert not log_path.exists()
        assert access_log.get_access_history("K") == []

    def test_file_written_as_jsonl(self, access_log, log_path):
        access_log.log_access("K", "SET", caller="test")
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        # Lines now have HMAC appended: json_data|hmac
        raw_line = lines[0]
        if "|" in raw_line:
            raw_line = raw_line.rsplit("|", 1)[0]
        data = json.loads(raw_line)
        assert data["key_name"] == "K"
        assert data["operation"] == "SET"

    def test_file_written_owner_only(self, access_log, log_path):
        access_log.log_access("K", "SET", caller="test")

        assert log_path.exists()
        assert log_path.stat().st_mode & 0o777 == 0o600

    def test_operations_normalized_to_upper(self, access_log):
        access_log.log_access("K", "get")
        history = access_log.get_access_history("K")
        assert history[0].operation == "GET"
