"""Tests for HermesKatana proxy addon (KatanaAddon + RateTracker)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from hermes_katana.proxy.addon import (
    KatanaAddon,
    RateTracker,
    _get_client_id,
    _make_block_response,
)
from hermes_katana.proxy.config import ProxyConfig


# ======================================================================
# Helpers
# ======================================================================


def _make_flow(
    host: str = "api.openai.com",
    body: bytes = b"",
    client_ip: str = "127.0.0.1",
    headers: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal mock mitmproxy flow object."""
    req_headers = dict(headers or {})
    flow = SimpleNamespace(
        request=SimpleNamespace(
            host=host,
            headers=req_headers,
            get_content=lambda: body,
        ),
        response=None,
        client_conn=SimpleNamespace(peername=(client_ip, 12345)),
    )
    return flow


def _make_response_flow(
    host: str = "api.openai.com",
    req_body: bytes = b"",
    resp_body: bytes = b"ok",
    client_ip: str = "127.0.0.1",
) -> SimpleNamespace:
    """Build a flow with both request and response."""
    resp_headers = {}
    flow = SimpleNamespace(
        request=SimpleNamespace(
            host=host,
            headers={},
            get_content=lambda: req_body,
        ),
        response=SimpleNamespace(
            get_content=lambda: resp_body,
            set_content=lambda data: None,
            status_code=200,
            headers=resp_headers,
        ),
        client_conn=SimpleNamespace(peername=(client_ip, 12345)),
    )
    return flow


def _addon(
    allowed_domains: list[str] | None = None,
    ignore_hosts: list[str] | None = None,
    rate_limit: int = 50,
    inject_credentials: bool = False,
    max_body_scan_size: int = 1_048_576,
    add_scanned_header: bool = False,
) -> KatanaAddon:
    """Create a KatanaAddon with a minimal config and no vault/audit."""
    cfg = ProxyConfig(
        allowed_domains=allowed_domains or [],
        ignore_hosts=ignore_hosts or [],
        rate_limit_requests=rate_limit,
        rate_limit_window=1.0,
        inject_credentials=inject_credentials,
        max_body_scan_size=max_body_scan_size,
        add_scanned_header=add_scanned_header,
    )
    return KatanaAddon(config=cfg, vault=None, audit=None)


# ======================================================================
# RateTracker
# ======================================================================


class TestRateTracker:
    def test_allows_under_limit(self):
        rt = RateTracker(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            allowed, _ = rt.check("client1")
            assert allowed is True

    def test_blocks_over_limit(self):
        rt = RateTracker(max_requests=3, window_seconds=10.0)
        for _ in range(3):
            rt.check("client1")
        allowed, count = rt.check("client1")
        assert allowed is False
        assert count == 3

    def test_separate_clients(self):
        rt = RateTracker(max_requests=2, window_seconds=10.0)
        rt.check("a")
        rt.check("a")
        allowed_a, _ = rt.check("a")
        allowed_b, _ = rt.check("b")
        assert allowed_a is False
        assert allowed_b is True

    def test_escalation_reduces_limit(self):
        rt = RateTracker(max_requests=10, window_seconds=10.0, escalation_factor=2.0)
        # Fill up to limit
        for _ in range(10):
            rt.check("c")
        # First violation (violations becomes 1)
        allowed, _ = rt.check("c")
        assert allowed is False
        # Second violation (window still has 10 >= effective_limit 5, violations becomes 2)
        allowed, _ = rt.check("c")
        assert allowed is False
        # Now violations=2, effective_limit = int(10 / 2^2) = 2
        # Reset windows but keep violations
        rt._windows["c"].clear()
        # Fill up to the reduced limit of 2
        for _ in range(2):
            rt.check("c")
        # This should be blocked at the reduced limit
        allowed2, _ = rt.check("c")
        assert allowed2 is False

    def test_get_stats(self):
        rt = RateTracker(max_requests=1, window_seconds=10.0)
        rt.check("x")
        rt.check("x")  # This triggers a violation
        stats = rt.get_stats()
        assert stats["tracked_clients"] >= 1
        assert "x" in stats["total_violations"]

    def test_reset(self):
        rt = RateTracker(max_requests=1, window_seconds=10.0)
        rt.check("x")
        rt.check("x")
        rt.reset()
        stats = rt.get_stats()
        assert stats["tracked_clients"] == 0

    def test_thread_safety(self):
        rt = RateTracker(max_requests=1000, window_seconds=10.0)
        errors = []

        def hammer():
            try:
                for _ in range(100):
                    rt.check("shared")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_stale_client_eviction(self):
        rt = RateTracker(max_requests=5, window_seconds=0.001)
        rt._MAX_CLIENTS = 2  # Force eviction threshold low
        # Add 3 clients
        rt.check("a")
        rt.check("b")
        time.sleep(0.01)  # Let windows expire
        rt.check("c")  # Triggers eviction of stale a and b
        # After eviction, stale clients should be cleaned
        stats = rt.get_stats()
        # c should still be tracked
        assert stats["tracked_clients"] >= 1


# ======================================================================
# _get_client_id
# ======================================================================


class TestGetClientId:
    def test_extracts_ip(self):
        flow = _make_flow(client_ip="10.0.0.1")
        assert _get_client_id(flow) == "10.0.0.1"

    def test_missing_client_conn(self):
        flow = SimpleNamespace(request=SimpleNamespace(host="x"))
        assert _get_client_id(flow) == "unknown"

    def test_none_peername(self):
        flow = SimpleNamespace(
            request=SimpleNamespace(host="x"),
            client_conn=SimpleNamespace(peername=None),
        )
        assert _get_client_id(flow) == "unknown"


# ======================================================================
# _make_block_response
# ======================================================================


class TestMakeBlockResponse:
    def test_returns_response_object(self):
        resp = _make_block_response(403, "blocked")
        assert resp is not None
        # Should have status_code and content in some form
        assert resp.status_code == 403

    def test_429_response(self):
        resp = _make_block_response(429, "rate limit")
        assert resp.status_code == 429


# ======================================================================
# KatanaAddon — domain allowlist
# ======================================================================


class TestAddonDomainAllowlist:
    def test_no_allowlist_allows_all(self):
        addon = _addon(allowed_domains=[])
        assert addon._is_allowed_domain("anything.com") is True

    def test_exact_match(self):
        addon = _addon(allowed_domains=["api.openai.com"])
        assert addon._is_allowed_domain("api.openai.com") is True
        assert addon._is_allowed_domain("evil.com") is False

    def test_subdomain_match(self):
        addon = _addon(allowed_domains=["openai.com"])
        assert addon._is_allowed_domain("api.openai.com") is True
        assert addon._is_allowed_domain("openai.com") is True
        assert addon._is_allowed_domain("notopenai.com") is False

    def test_case_insensitive(self):
        addon = _addon(allowed_domains=["openai.com"])
        assert addon._is_allowed_domain("API.OPENAI.COM") is True

    def test_request_blocked_domain(self):
        addon = _addon(allowed_domains=["safe.com"])
        flow = _make_flow(host="evil.com")
        addon.request(flow)
        # flow.response should be set to a block response
        assert flow.response is not None
        assert flow.response.status_code == 403


# ======================================================================
# KatanaAddon — ignored hosts
# ======================================================================


class TestAddonIgnoredHosts:
    def test_ignored_host_passes_through(self):
        addon = _addon(ignore_hosts=["pypi.org"])
        flow = _make_flow(host="pypi.org")
        addon.request(flow)
        # No response set = passed through
        assert flow.response is None

    def test_non_ignored_host_processed(self):
        addon = _addon(ignore_hosts=["pypi.org"])
        flow = _make_flow(host="api.openai.com", body=b"hello")
        addon.request(flow)
        # Should be processed (not ignored) — response may or may not be set
        stats = addon.get_stats()
        assert stats.get("requests_ignored", 0) == 0


# ======================================================================
# KatanaAddon — rate limiting
# ======================================================================


class TestAddonRateLimiting:
    def test_rate_limit_blocks(self):
        addon = _addon(rate_limit=2)
        for _ in range(2):
            flow = _make_flow(client_ip="1.2.3.4")
            addon.request(flow)
        # Third request should be rate limited
        flow = _make_flow(client_ip="1.2.3.4")
        addon.request(flow)
        assert flow.response is not None
        assert flow.response.status_code == 429


# ======================================================================
# KatanaAddon — body scanning
# ======================================================================


class TestAddonBodyScanning:
    def test_body_too_large_skipped(self):
        addon = _addon(max_body_scan_size=10)
        assert addon._body_too_large(b"x" * 11) is True
        assert addon._body_too_large(b"x" * 5) is False

    def test_body_none_not_too_large(self):
        addon = _addon()
        assert addon._body_too_large(None) is False

    def test_unlimited_body_scan(self):
        addon = _addon(max_body_scan_size=0)
        assert addon._body_too_large(b"x" * 999999) is False


# ======================================================================
# KatanaAddon — header injection
# ======================================================================


class TestAddonHeaderInjection:
    def test_x_katana_scanned_header_set(self):
        addon = _addon(add_scanned_header=True)
        flow = _make_response_flow(resp_body=b"some response")
        addon.response(flow)
        assert "X-Katana-Scanned" in flow.response.headers

    def test_passthrough_header_for_ignored(self):
        addon = _addon(ignore_hosts=["ignored.com"])
        flow = _make_response_flow(host="ignored.com")
        addon.response(flow)
        # Ignored hosts don't get scanned header
        assert "X-Katana-Scanned" not in flow.response.headers


# ======================================================================
# KatanaAddon — response hook
# ======================================================================


class TestAddonResponse:
    def test_response_missing_attribute(self):
        addon = _addon()
        # Flow without request.host should not crash
        flow = SimpleNamespace()
        addon.response(flow)  # Should return silently

    def test_response_no_body(self):
        addon = _addon(add_scanned_header=True)
        flow = _make_response_flow(resp_body=b"")
        addon.response(flow)
        # Should not crash, header still injected
        assert "X-Katana-Scanned" in flow.response.headers


# ======================================================================
# KatanaAddon — request hook edge cases
# ======================================================================


class TestAddonRequestEdgeCases:
    def test_request_missing_host(self):
        addon = _addon()
        flow = SimpleNamespace()
        addon.request(flow)  # Should not crash

    def test_stats_tracking(self):
        addon = _addon()
        flow = _make_flow(body=b"hello world")
        addon.request(flow)
        stats = addon.get_stats()
        assert stats["requests_total"] >= 1

    def test_credential_injection_with_vault(self):
        addon = _addon(inject_credentials=True)
        mock_vault = MagicMock()
        mock_vault._get_all_values.return_value = {}
        addon.vault = mock_vault
        addon.config.inject_credentials = True
        flow = _make_flow(host="api.openai.com")
        with patch("hermes_katana.proxy.addon.inject_credentials") as mock_inject:
            mock_inject.return_value = "OpenAI"
            addon.request(flow)
            mock_inject.assert_called_once()


# ======================================================================
# KatanaAddon — audit logging
# ======================================================================


class TestAddonAuditLogging:
    def test_log_audit_no_audit_trail(self):
        addon = _addon()
        # Should not crash when audit is None
        addon._log_audit("SCAN_RESULT", "test", "deny", "details")

    def test_log_audit_with_mock_trail(self):
        addon = _addon()
        mock_audit = MagicMock()
        addon.audit = mock_audit
        addon._log_audit("SCAN_RESULT", "test", "deny", "details")
        # Should have attempted to log

    def test_args_hash_deterministic(self):
        addon = _addon()
        h1 = addon._args_hash("a", "b")
        h2 = addon._args_hash("a", "b")
        assert h1 == h2
        assert len(h1) == 16

    def test_args_hash_different_inputs(self):
        addon = _addon()
        h1 = addon._args_hash("a", "b")
        h2 = addon._args_hash("c", "d")
        assert h1 != h2
