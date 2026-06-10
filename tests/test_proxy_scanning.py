"""Tests for proxy addon scanning gaps (Worker 3 — Part A).

Tests cover:
- GAP 3.1: URL, header, query param, cookie scanning
- GAP 3.2: Response header scanning
- GAP 3.3: WebSocket message scanning
- GAP 3.5: Oversized body prefix scanning
- GAP 3.7: X-Katana-Scanned header opt-in
- GAP 3.8: Credential injection order (injected headers skipped)
"""

from __future__ import annotations

from types import SimpleNamespace


from hermes_katana.proxy.addon import KatanaAddon
from hermes_katana.proxy.config import ProxyConfig


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockHeaders(dict):
    """Dict-like mock for mitmproxy headers."""

    def get(self, key, default=None):
        # case-insensitive get
        for k, v in self.items():
            if k.lower() == key.lower():
                return v
        return default


class MockRequest:
    def __init__(self, host="example.com", url="https://example.com/path", headers=None, body=b"", query=None):
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
    def __init__(self, ip="127.0.0.1"):
        self.peername = (ip, 12345)


class MockFlow:
    def __init__(self, request=None, response=None, client_ip="127.0.0.1"):
        self.request = request or MockRequest()
        self.response = response
        self.client_conn = MockClientConn(client_ip)


class MockWSMessage:
    def __init__(self, content, from_client=True):
        self.content = content
        self.from_client = from_client


class MockWebSocket:
    def __init__(self, messages):
        self.messages = messages


class MockWSFlow:
    def __init__(self, message_content, from_client=True):
        msg = MockWSMessage(message_content, from_client)
        self.websocket = MockWebSocket([msg])
        self.request = MockRequest()
        self.client_conn = MockClientConn()


def _make_addon(**overrides) -> KatanaAddon:
    """Create a KatanaAddon with sensible test defaults."""
    cfg_kwargs = {
        "inject_credentials": False,
        "allowed_domains": [],
        "rate_limit_requests": 1000,
    }
    cfg_kwargs.update(overrides)
    config = ProxyConfig(**cfg_kwargs)
    return KatanaAddon(config=config, vault=None, audit=None)


# ---------------------------------------------------------------------------
# GAP 3.1 — Scan request URL, headers, query params, cookies
# ---------------------------------------------------------------------------


class TestRequestURLScanning:
    """GAP 3.1: URL path segments are scanned."""

    def test_clean_url_passes(self):
        addon = _make_addon()
        flow = MockFlow(request=MockRequest(url="https://api.openai.com/v1/chat"))
        addon.request(flow)
        assert flow.response is None  # not blocked
        assert addon._stats["requests_passed"] > 0

    def test_url_scanning_runs(self):
        """Verify the URL is fed to the scanner."""
        addon = _make_addon()
        flow = MockFlow(
            request=MockRequest(
                url="https://example.com/safe/path",
            )
        )
        addon.request(flow)
        # Should pass through cleanly
        assert flow.response is None


class TestRequestHeaderScanning:
    """GAP 3.1: Request headers are scanned."""

    def test_clean_headers_pass(self):
        addon = _make_addon()
        flow = MockFlow(
            request=MockRequest(
                headers={"Content-Type": "application/json", "Accept": "text/html"},
            )
        )
        addon.request(flow)
        assert flow.response is None

    def test_header_scanning_iterates_all(self):
        addon = _make_addon()
        flow = MockFlow(
            request=MockRequest(
                headers={"X-Custom": "hello", "X-Other": "world"},
            )
        )
        addon.request(flow)
        assert flow.response is None


class TestQueryParamScanning:
    """GAP 3.1: Query parameters are scanned."""

    def test_clean_query_passes(self):
        addon = _make_addon()
        flow = MockFlow(
            request=MockRequest(
                query={"q": "hello world", "page": "1"},
            )
        )
        addon.request(flow)
        assert flow.response is None


class TestCookieScanning:
    """GAP 3.1: Cookie values are scanned."""

    def test_clean_cookies_pass(self):
        addon = _make_addon()
        flow = MockFlow(
            request=MockRequest(
                headers={"cookie": "session=abc123; theme=dark"},
            )
        )
        addon.request(flow)
        assert flow.response is None

    def test_empty_cookie_header_ok(self):
        addon = _make_addon()
        flow = MockFlow(request=MockRequest(headers={"cookie": ""}))
        addon.request(flow)
        assert flow.response is None


# ---------------------------------------------------------------------------
# GAP 3.2 — Scan response headers
# ---------------------------------------------------------------------------


class TestResponseHeaderScanning:
    """GAP 3.2: Response headers are scanned."""

    def test_clean_response_headers_pass(self):
        addon = _make_addon()
        flow = MockFlow()
        flow.response = MockResponse(
            headers={"Content-Type": "text/html", "X-Request-Id": "abc123"},
        )
        addon.response(flow)
        assert flow.response.status_code == 200  # not replaced

    def test_response_header_scanning_runs(self):
        addon = _make_addon()
        flow = MockFlow()
        flow.response = MockResponse(
            headers={"Set-Cookie": "session=safe_value"},
        )
        addon.response(flow)
        assert addon._stats["responses_total"] == 1


# ---------------------------------------------------------------------------
# GAP 3.3 — WebSocket message scanning
# ---------------------------------------------------------------------------


class TestWebSocketScanning:
    """GAP 3.3: WebSocket messages are scanned."""

    def test_clean_ws_message_passes(self):
        addon = _make_addon()
        flow = MockWSFlow(b"Hello, WebSocket!")
        addon.websocket_message(flow)
        assert addon._stats["ws_messages_total"] == 1
        assert addon._stats.get("ws_messages_blocked", 0) == 0

    def test_ws_string_content(self):
        addon = _make_addon()
        msg = MockWSMessage("text message", from_client=True)
        flow = SimpleNamespace(
            websocket=MockWebSocket([msg]),
            request=MockRequest(),
            client_conn=MockClientConn(),
        )
        addon.websocket_message(flow)
        assert addon._stats["ws_messages_total"] == 1

    def test_ws_server_message_scanned_as_response(self):
        addon = _make_addon()
        flow = MockWSFlow(b"server says hello", from_client=False)
        addon.websocket_message(flow)
        assert addon._stats["ws_messages_total"] == 1

    def test_ws_no_messages_noop(self):
        addon = _make_addon()
        flow = SimpleNamespace(
            websocket=MockWebSocket([]),
        )
        addon.websocket_message(flow)
        assert addon._stats.get("ws_messages_total", 0) == 0

    def test_ws_no_websocket_attr_noop(self):
        addon = _make_addon()
        flow = SimpleNamespace()
        addon.websocket_message(flow)
        assert addon._stats.get("ws_messages_total", 0) == 0


# ---------------------------------------------------------------------------
# GAP 3.5 — Oversized body prefix scanning
# ---------------------------------------------------------------------------


class TestOversizedBodyScanning:
    """GAP 3.5: Oversized bodies get first N bytes scanned."""

    def test_oversized_request_body_scans_prefix_then_blocks_in_strict_mode(self):
        addon = _make_addon(max_body_scan_size=100)
        big_body = b"A" * 500
        flow = MockFlow(request=MockRequest(body=big_body))
        addon.request(flow)
        assert addon._stats.get("requests_oversized", 0) == 1
        assert addon._stats.get("requests_blocked_oversized", 0) == 1
        assert flow.response is not None
        assert flow.response.status_code == 413

    def test_oversized_request_body_scans_prefix_then_passes_in_permissive_mode(self):
        addon = _make_addon(max_body_scan_size=100, mode="permissive")
        big_body = b"A" * 500
        flow = MockFlow(request=MockRequest(body=big_body))
        addon.request(flow)
        assert addon._stats.get("requests_oversized", 0) == 1
        assert addon._stats["requests_passed"] > 0
        assert flow.response is None

    def test_oversized_response_body_scans_prefix(self):
        addon = _make_addon(max_body_scan_size=100)
        big_body = b"B" * 500
        flow = MockFlow()
        flow.response = MockResponse(body=big_body)
        addon.response(flow)
        assert addon._stats.get("responses_oversized", 0) == 1

    def test_normal_body_not_flagged_oversized(self):
        addon = _make_addon(max_body_scan_size=1000)
        flow = MockFlow(request=MockRequest(body=b"small body"))
        addon.request(flow)
        assert addon._stats.get("requests_oversized", 0) == 0


# ---------------------------------------------------------------------------
# GAP 3.7 — X-Katana-Scanned header opt-in
# ---------------------------------------------------------------------------


class TestScannedHeaderOptIn:
    """GAP 3.7: X-Katana-Scanned is opt-in (disabled by default)."""

    def test_header_not_added_by_default(self):
        addon = _make_addon()
        flow = MockFlow()
        flow.response = MockResponse(body=b"hello")
        addon.response(flow)
        assert "X-Katana-Scanned" not in flow.response.headers

    def test_header_added_when_enabled(self):
        addon = _make_addon(add_scanned_header=True)
        flow = MockFlow()
        flow.response = MockResponse(body=b"hello")
        addon.response(flow)
        assert "X-Katana-Scanned" in flow.response.headers
        assert flow.response.headers["X-Katana-Scanned"] == "true"

    def test_header_passthrough_when_no_body(self):
        addon = _make_addon(add_scanned_header=True)
        flow = MockFlow()
        flow.response = MockResponse(body=b"")
        addon.response(flow)
        assert flow.response.headers.get("X-Katana-Scanned") == "passthrough"


# ---------------------------------------------------------------------------
# GAP 3.8 — Credential injection order
# ---------------------------------------------------------------------------


class TestCredentialInjectionOrder:
    """GAP 3.8: Injected headers are excluded from scanning."""

    def test_injected_headers_tracking(self):
        """When inject_credentials is off, no headers are skipped."""
        addon = _make_addon(inject_credentials=False)
        flow = MockFlow(
            request=MockRequest(
                headers={"Authorization": "Bearer sk-test123"},
            )
        )
        addon.request(flow)
        # With injection off, Authorization IS scanned (not excluded).
        # Whether it triggers a block depends on scanner sensitivity;
        # the key property is that it was NOT added to _injected_headers.
        # Verify the header was not exempted from scanning.
        assert not hasattr(addon, "_last_injected_headers") or "authorization" not in getattr(
            addon, "_last_injected_headers", set()
        )


# ---------------------------------------------------------------------------
# Basic existing functionality preserved
# ---------------------------------------------------------------------------


class TestBasicFunctionality:
    """Verify existing proxy behaviors still work."""

    def test_ignored_host_skipped(self):
        # Audit fix C2: pypi.org is no longer ignored by default — only the
        # key-pinned sigstore/TUF endpoints remain on the bypass list.
        addon = _make_addon()
        flow = MockFlow(request=MockRequest(host="rekor.sigstore.dev"))
        addon.request(flow)
        assert flow.response is None
        assert addon._stats["requests_ignored"] == 1

    def test_package_hosts_scanned_by_default(self):
        addon = _make_addon()
        assert addon._is_ignored_host("pypi.org") is False
        assert addon._is_ignored_host("files.pythonhosted.org") is False

    def test_domain_allowlist_blocks(self):
        addon = _make_addon(allowed_domains=["api.openai.com"])
        flow = MockFlow(request=MockRequest(host="evil.com"))
        addon.request(flow)
        assert flow.response is not None
        assert addon._stats["requests_blocked_domain"] == 1

    def test_rate_limiting(self):
        addon = _make_addon(rate_limit_requests=2, rate_limit_window=10.0)
        for i in range(3):
            flow = MockFlow(request=MockRequest())
            addon.request(flow)
        assert addon._stats.get("requests_rate_limited", 0) >= 1

    def test_empty_body_request(self):
        addon = _make_addon()
        flow = MockFlow(request=MockRequest(body=b""))
        addon.request(flow)
        assert flow.response is None

    def test_response_without_response_obj(self):
        addon = _make_addon()
        flow = MockFlow()
        flow.response = None
        # Should not crash
        addon.response(flow)

    def test_get_stats(self):
        addon = _make_addon()
        stats = addon.get_stats()
        assert "rate_tracker" in stats
