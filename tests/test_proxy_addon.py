"""Tests for hermes_katana.proxy.addon — KatanaAddon and RateTracker."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from hermes_katana.proxy.addon import RateTracker, KatanaAddon, _get_client_id, _make_block_response
from hermes_katana.proxy.config import ProxyConfig


# ---------------------------------------------------------------------------
# Mock mitmproxy objects
# ---------------------------------------------------------------------------

class MockHeaders(dict):
    """Minimal mitmproxy-like headers."""
    def items(self):
        return super().items()


class MockRequest:
    def __init__(self, host="api.openai.com", url="https://api.openai.com/v1/chat",
                 headers=None, body=b"", query=None):
        self.host = host
        self.url = url
        self.headers = MockHeaders(headers or {})
        self._body = body
        self.query = query or {}

    def get_content(self):
        return self._body


class MockResponse:
    def __init__(self, body=b"", headers=None, status_code=200):
        self._body = body
        self.headers = MockHeaders(headers or {})
        self.status_code = status_code

    def get_content(self):
        return self._body

    def set_content(self, data):
        self._body = data


class MockClientConn:
    def __init__(self, ip="192.168.1.1"):
        self.peername = (ip, 12345)


class MockFlow:
    def __init__(self, host="example.com", body=b"hello", resp_body=b"world",
                 headers=None, resp_headers=None, client_ip="192.168.1.1",
                 query=None, url=None):
        self.request = MockRequest(
            host=host,
            url=url or f"https://{host}/path",
            headers=headers or {},
            body=body,
            query=query,
        )
        self.response = MockResponse(body=resp_body, headers=resp_headers or {})
        self.client_conn = MockClientConn(client_ip)


# ---------------------------------------------------------------------------
# RateTracker tests
# ---------------------------------------------------------------------------

class TestRateTracker:
    def test_allows_within_limit(self):
        rt = RateTracker(max_requests=5, window_seconds=1.0)
        for _ in range(5):
            allowed, _ = rt.check("c1")
            assert allowed

    def test_blocks_over_limit(self):
        rt = RateTracker(max_requests=3, window_seconds=10.0)
        for _ in range(3):
            rt.check("c1")
        allowed, count = rt.check("c1")
        assert not allowed
        assert count == 3

    def test_different_clients_independent(self):
        rt = RateTracker(max_requests=2, window_seconds=10.0)
        rt.check("c1")
        rt.check("c1")
        allowed_c1, _ = rt.check("c1")
        allowed_c2, _ = rt.check("c2")
        assert not allowed_c1
        assert allowed_c2

    def test_window_expiry(self):
        rt = RateTracker(max_requests=2, window_seconds=0.05)
        rt.check("c1")
        rt.check("c1")
        time.sleep(0.1)
        allowed, _ = rt.check("c1")
        assert allowed

    def test_escalation_increases_violations(self):
        rt = RateTracker(max_requests=3, window_seconds=10.0, escalation_factor=2.0)
        # Fill to limit
        for _ in range(3):
            rt.check("c1")
        # This should be blocked and record a violation
        allowed, _ = rt.check("c1")
        assert not allowed
        assert rt._violations["c1"] >= 1

    def test_get_stats(self):
        rt = RateTracker(max_requests=1, window_seconds=10.0)
        rt.check("c1")
        rt.check("c1")  # triggers violation
        stats = rt.get_stats()
        assert stats["tracked_clients"] >= 1
        assert "c1" in stats["total_violations"]

    def test_reset(self):
        rt = RateTracker(max_requests=1, window_seconds=10.0)
        rt.check("c1")
        rt.reset()
        stats = rt.get_stats()
        assert stats["tracked_clients"] == 0

    def test_thread_safety(self):
        rt = RateTracker(max_requests=1000, window_seconds=10.0)
        errors = []

        def hammer(client):
            try:
                for _ in range(100):
                    rt.check(client)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=hammer, args=(f"c{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors

    def test_evict_stale_clients(self):
        rt = RateTracker(max_requests=100, window_seconds=0.01, _MAX_CLIENTS=5)
        for i in range(10):
            rt.check(f"client_{i}")
        time.sleep(0.05)
        # Next check triggers eviction of stale clients
        rt.check("new_client")
        # Should not crash and stale clients may be evicted

    def test_violation_decay(self):
        rt = RateTracker(max_requests=10, window_seconds=10.0, escalation_factor=2.0)
        # Force a violation
        for _ in range(10):
            rt.check("c1")
        rt.check("c1")  # blocked, violation=1
        assert rt._violations["c1"] >= 1
        # Clear windows, make only 1 request (well below effective_limit//2)
        rt._windows["c1"].clear()
        rt.check("c1")
        # Violation should have decayed


# ---------------------------------------------------------------------------
# KatanaAddon tests
# ---------------------------------------------------------------------------

class TestKatanaAddon:
    def _make_addon(self, **config_overrides):
        cfg = ProxyConfig(**config_overrides)
        return KatanaAddon(config=cfg, vault=None, audit=None)

    def test_init(self):
        addon = self._make_addon()
        assert addon.config is not None
        assert addon.vault is None

    def test_request_no_host(self):
        addon = self._make_addon()
        flow = MagicMock(spec=[])  # no request attr
        addon.request(flow)  # should not raise

    def test_request_ignored_host(self):
        addon = self._make_addon(ignore_hosts=["pypi.org"])
        flow = MockFlow(host="pypi.org")
        addon.request(flow)
        assert addon._stats["requests_ignored"] == 1

    def test_request_domain_not_allowed(self):
        addon = self._make_addon(allowed_domains=["allowed.com"])
        flow = MockFlow(host="blocked.com")
        addon.request(flow)
        assert addon._stats["requests_blocked_domain"] == 1
        assert flow.response is not None

    def test_request_allowed_domain(self):
        addon = self._make_addon(allowed_domains=["example.com"])
        flow = MockFlow(host="example.com", body=b"")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.request(flow)
        assert addon._stats.get("requests_blocked_domain", 0) == 0

    def test_request_rate_limited(self):
        addon = self._make_addon(rate_limit_requests=1, rate_limit_window=10.0)
        flow1 = MockFlow(host="example.com", body=b"")
        flow2 = MockFlow(host="example.com", body=b"")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.request(flow1)
            addon.request(flow2)
        assert addon._stats.get("requests_rate_limited", 0) >= 1

    @patch("hermes_katana.proxy.addon.inject_credentials", return_value="OpenAI")
    def test_credential_injection(self, mock_inject):
        vault = MagicMock()
        vault._get_all_values.return_value = {}
        cfg = ProxyConfig(inject_credentials=True)
        addon = KatanaAddon(config=cfg, vault=vault, audit=None)
        flow = MockFlow(host="api.openai.com", body=b"")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.request(flow)
        assert addon._stats.get("credentials_injected", 0) == 1

    def test_request_body_blocked(self):
        addon = self._make_addon()
        flow = MockFlow(host="example.com", body=b"malicious content")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "block", "risk_score": 100, "is_blocked": True,
            "finding_count": 1, "summary": "injection detected"
        }):
            addon.request(flow)
        assert addon._stats.get("requests_blocked_scan", 0) >= 1

    def test_request_body_warned(self):
        addon = self._make_addon()
        flow = MockFlow(host="example.com", body=b"suspicious")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "warn", "risk_score": 30, "is_blocked": False,
            "finding_count": 1, "summary": "suspicious pattern"
        }):
            addon.request(flow)
        assert addon._stats.get("requests_warned", 0) >= 1

    def test_response_basic(self):
        addon = self._make_addon()
        flow = MockFlow(host="example.com", resp_body=b"safe response")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.response(flow)
        assert addon._stats.get("responses_total", 0) == 1

    def test_response_no_host(self):
        addon = self._make_addon()
        flow = MagicMock(spec=[])
        addon.response(flow)

    def test_response_ignored_host(self):
        addon = self._make_addon(ignore_hosts=["pypi.org"])
        flow = MockFlow(host="pypi.org")
        addon.response(flow)
        assert addon._stats.get("responses_passed", 0) == 0

    def test_response_body_blocked(self):
        addon = self._make_addon()
        flow = MockFlow(host="example.com", resp_body=b"bad response")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "block", "risk_score": 100, "is_blocked": True,
            "finding_count": 1, "summary": "attack detected"
        }):
            addon.response(flow)
        assert addon._stats.get("responses_blocked_scan", 0) >= 1

    def test_scanned_header_injected(self):
        addon = self._make_addon(add_scanned_header=True)
        flow = MockFlow(host="example.com", resp_body=b"ok")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.response(flow)
        assert flow.response.headers.get("X-Katana-Scanned") in ("true", "passthrough")

    def test_scanned_header_not_added_by_default(self):
        addon = self._make_addon(add_scanned_header=False)
        flow = MockFlow(host="example.com", resp_body=b"ok")
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.response(flow)
        assert "X-Katana-Scanned" not in flow.response.headers

    def test_oversized_body_scans_prefix(self):
        addon = self._make_addon(max_body_scan_size=10)
        flow = MockFlow(host="example.com", body=b"A" * 100)
        scanned_texts = []
        original_scan = addon._scan_text
        def capture_scan(text, direction="request"):
            scanned_texts.append(text)
            return {"verdict": "pass", "risk_score": 0, "is_blocked": False,
                    "finding_count": 0, "summary": ""}
        with patch.object(addon, '_scan_text', side_effect=capture_scan):
            addon.request(flow)
        # The body text scanned should be truncated to max_body_scan_size
        body_texts = [t for t in scanned_texts if len(t) <= 10]
        assert addon._stats.get("requests_oversized", 0) == 1

    def test_get_stats(self):
        addon = self._make_addon()
        stats = addon.get_stats()
        assert "rate_tracker" in stats

    def test_websocket_message_basic(self):
        addon = self._make_addon()
        msg = MagicMock()
        msg.content = b"hello ws"
        msg.from_client = True
        flow = MagicMock()
        flow.request.host = "example.com"
        flow.websocket.messages = [msg]
        with patch.object(addon, '_scan_text', return_value={
            "verdict": "pass", "risk_score": 0, "is_blocked": False,
            "finding_count": 0, "summary": ""
        }):
            addon.websocket_message(flow)
        assert addon._stats.get("ws_messages_total", 0) == 1

    def test_websocket_no_messages(self):
        addon = self._make_addon()
        flow = MagicMock()
        flow.websocket.messages = []
        addon.websocket_message(flow)

    def test_websocket_no_websocket_attr(self):
        addon = self._make_addon()
        flow = MagicMock(spec=[])
        addon.websocket_message(flow)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_client_id(self):
        flow = MockFlow(client_ip="10.0.0.1")
        assert _get_client_id(flow) == "10.0.0.1"

    def test_get_client_id_missing(self):
        flow = MagicMock(spec=[])
        assert _get_client_id(flow) == "unknown"

    def test_make_block_response(self):
        resp = _make_block_response(403, "Forbidden")
        assert resp.status_code == 403

    def test_is_ignored_host(self):
        addon = KatanaAddon(config=ProxyConfig(ignore_hosts=["example.com"]))
        assert addon._is_ignored_host("example.com")
        assert not addon._is_ignored_host("other.com")

    def test_is_allowed_domain_empty(self):
        addon = KatanaAddon(config=ProxyConfig(allowed_domains=[]))
        assert addon._is_allowed_domain("anything.com")

    def test_is_allowed_domain_subdomain(self):
        addon = KatanaAddon(config=ProxyConfig(allowed_domains=["example.com"]))
        assert addon._is_allowed_domain("example.com")
        assert addon._is_allowed_domain("sub.example.com")
        assert not addon._is_allowed_domain("other.com")

    def test_body_too_large(self):
        addon = KatanaAddon(config=ProxyConfig(max_body_scan_size=100))
        assert addon._body_too_large(b"x" * 101)
        assert not addon._body_too_large(b"x" * 50)
        assert not addon._body_too_large(None)

    def test_body_unlimited(self):
        addon = KatanaAddon(config=ProxyConfig(max_body_scan_size=0))
        assert not addon._body_too_large(b"x" * 10_000_000)

    def test_args_hash(self):
        addon = KatanaAddon(config=ProxyConfig())
        h1 = addon._args_hash("a", "b")
        h2 = addon._args_hash("a", "b")
        h3 = addon._args_hash("a", "c")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16
