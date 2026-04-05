"""Tests for hermes_katana.vault.expiry."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hermes_katana.vault.expiry import SecretExpiry


@pytest.fixture
def expiry_path(tmp_path):
    return tmp_path / "vault_expiry.json"


@pytest.fixture
def expiry(expiry_path):
    return SecretExpiry(path=expiry_path)


class TestSecretExpiry:
    def test_set_and_get_expiry(self, expiry):
        expiry.set_expiry("TOKEN", ttl_seconds=3600)
        dt = expiry.get_expiry("TOKEN")
        assert dt is not None
        assert isinstance(dt, datetime)
        # Should be roughly 1 hour from now
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        assert 3500 < delta < 3700

    def test_no_expiry(self, expiry):
        assert expiry.get_expiry("NONEXISTENT") is None

    def test_is_expired_false(self, expiry):
        expiry.set_expiry("TOKEN", ttl_seconds=3600)
        assert expiry.is_expired("TOKEN") is False

    def test_is_expired_true(self, expiry):
        expiry.set_expiry("TOKEN", ttl_seconds=-1)  # Already expired
        assert expiry.is_expired("TOKEN") is True

    def test_is_expired_no_entry(self, expiry):
        assert expiry.is_expired("NONEXISTENT") is False

    def test_check_expired(self, expiry):
        expiry.set_expiry("EXPIRED_1", ttl_seconds=-10)
        expiry.set_expiry("EXPIRED_2", ttl_seconds=-5)
        expiry.set_expiry("VALID", ttl_seconds=3600)

        expired = expiry.check_expired()
        assert "EXPIRED_1" in expired
        assert "EXPIRED_2" in expired
        assert "VALID" not in expired

    def test_check_expired_empty(self, expiry):
        assert expiry.check_expired() == []

    def test_extend_expiry(self, expiry):
        expiry.set_expiry("TOKEN", ttl_seconds=100)
        before = expiry.get_expiry("TOKEN")
        expiry.extend_expiry("TOKEN", additional_seconds=200)
        after = expiry.get_expiry("TOKEN")
        assert after is not None and before is not None
        delta = (after - before).total_seconds()
        assert 195 < delta < 205

    def test_extend_nonexistent_raises(self, expiry):
        with pytest.raises(KeyError):
            expiry.extend_expiry("NONEXISTENT", additional_seconds=100)

    def test_remove_expiry(self, expiry):
        expiry.set_expiry("TOKEN", ttl_seconds=100)
        assert expiry.get_expiry("TOKEN") is not None
        expiry.remove_expiry("TOKEN")
        assert expiry.get_expiry("TOKEN") is None

    def test_remove_nonexistent_no_error(self, expiry):
        expiry.remove_expiry("NONEXISTENT")  # Should not raise

    def test_list_expiries(self, expiry):
        expiry.set_expiry("K1", ttl_seconds=100)
        expiry.set_expiry("K2", ttl_seconds=200)

        listing = expiry.list_expiries()
        assert "K1" in listing
        assert "K2" in listing
        assert isinstance(listing["K1"], datetime)

    def test_list_expiries_empty(self, expiry):
        assert expiry.list_expiries() == {}

    def test_persistence(self, expiry_path):
        """Expiry data persists across instances."""
        e1 = SecretExpiry(path=expiry_path)
        e1.set_expiry("TOKEN", ttl_seconds=3600)

        e2 = SecretExpiry(path=expiry_path)
        assert e2.get_expiry("TOKEN") is not None

    def test_clear(self, expiry, expiry_path):
        expiry.set_expiry("TOKEN", ttl_seconds=100)
        assert expiry_path.exists()
        expiry.clear()
        assert not expiry_path.exists()
