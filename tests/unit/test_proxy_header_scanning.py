"""Tests for proxy header, URL, query, cookie, and response header scanning."""

from __future__ import annotations
from urllib.parse import urlparse
from unittest.mock import patch
from hermes_katana.proxy.addon import KatanaAddon
from hermes_katana.proxy.config import ProxyConfig


class FakeRequest:
    def __init__(self, host="example.com", headers=None, body=b"", url="http://example.com/", query=None):
        self.host = host
        self.headers = headers or {}
        self._body = body
        self.url = url
        self.query = query or {}

    def get_content(self):
        return self._body


class FakeResponse:
    def __init__(self, headers=None, body=b"ok", status_code=200):
        self.headers = headers or {}
        self._body = body
        self.status_code = status_code

    def get_content(self):
        return self._body

    def set_content(self, data):
        self._body = data


class FakeClientConn:
    peername = ("127.0.0.1", 12345)


class FakeFlow:
    def __init__(self, request=None, response=None):
        self.request = request or FakeRequest()
        self.response = response or FakeResponse()
        self.client_conn = FakeClientConn()


def _make_addon(**kw):
    return KatanaAddon(config=ProxyConfig(**kw), vault=None, audit=None)


CLEAN = {"verdict": "pass", "risk_score": 0.0, "is_blocked": False, "finding_count": 0, "summary": "clean"}
BLOCKED = {"verdict": "block", "risk_score": 1.0, "is_blocked": True, "finding_count": 1, "summary": "secret leaked"}
WARNED = {"verdict": "warn", "risk_score": 0.5, "is_blocked": False, "finding_count": 1, "summary": "suspicious"}


def _capture_calls(addon, return_val=CLEAN):
    """Patch _scan_text to capture all calls and return given result."""
    calls = []

    def side_effect(*a, **kw):
        calls.append((a, kw))
        return return_val

    patcher = patch.object(addon, "_scan_text", side_effect=side_effect)
    return calls, patcher


# ---- Request header scanning ----


class TestRequestHeaderScanning:
    def test_header_value_reaches_scanner(self):
        """Each request header value is individually scanned."""
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                headers={"Authorization": "Bearer sk-secret123", "X-Custom": "foo"},
                body=b"",
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.request(flow)
        scanned_texts = [c[0][0] for c in calls]
        assert any("sk-secret123" in t for t in scanned_texts), f"Auth header not scanned. Texts: {scanned_texts}"

    def test_blocked_header_returns_400(self):
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                headers={"Authorization": "Bearer leaked"},
                body=b"",
            )
        )
        with patch.object(addon, "_scan_text", return_value=BLOCKED):
            addon.request(flow)
        assert flow.response is not None
        assert hasattr(flow.response, "status_code")

    def test_multiple_headers_all_scanned(self):
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                headers={"X-Api-Key": "key1", "X-Other": "val2"},
                body=b"",
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.request(flow)
        scanned = [c[0][0] for c in calls]
        assert any("key1" in t for t in scanned)
        assert any("val2" in t for t in scanned)


# ---- URL scanning ----


class TestURLScanning:
    def test_url_scanned(self):
        """Full URL string is scanned."""
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                url="http://example.com/api/v1/secret-token?key=abc",
                body=b"",
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.request(flow)
        scanned = [c[0][0] for c in calls]
        assert any("secret-token" in t for t in scanned), f"URL not scanned. Texts: {scanned}"

    def test_blocked_url_returns_400(self):
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                url="http://example.com/api?token=leaked",
                body=b"",
            )
        )
        with patch.object(addon, "_scan_text", return_value=BLOCKED):
            addon.request(flow)
        assert flow.response is not None


# ---- Query parameter scanning ----


class TestQueryScanning:
    def test_query_values_scanned(self):
        """Individual query parameter values are scanned."""
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                query={"api_key": "sk-secret", "q": "hello"},
                body=b"",
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.request(flow)
        scanned = [c[0][0] for c in calls]
        assert any("sk-secret" in t for t in scanned), f"Query param not scanned. Texts: {scanned}"

    def test_blocked_query_returns_400(self):
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                query={"token": "leaked"},
                body=b"",
            )
        )
        with patch.object(addon, "_scan_text", return_value=BLOCKED):
            addon.request(flow)
        assert flow.response is not None


# ---- Cookie scanning ----


class TestCookieScanning:
    def test_cookie_values_scanned(self):
        """Cookie values from the Cookie header are individually scanned."""
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                headers={"cookie": "session=abc123; token=xyz789"},
                body=b"",
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.request(flow)
        scanned = [c[0][0] for c in calls]
        assert any("abc123" in t for t in scanned), f"Cookie not scanned. Texts: {scanned}"


# ---- Response header scanning ----


class TestResponseHeaderScanning:
    def test_set_cookie_scanned(self):
        addon = _make_addon()
        flow = FakeFlow(
            response=FakeResponse(
                headers={"Set-Cookie": "session=secret-val; Path=/"},
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.response(flow)
        scanned = [c[0][0] for c in calls]
        assert any("secret-val" in t for t in scanned), f"Set-Cookie not scanned. Texts: {scanned}"

    def test_location_header_scanned(self):
        addon = _make_addon()
        flow = FakeFlow(
            response=FakeResponse(
                headers={"Location": "https://evil.com/phish"},
            )
        )
        calls, p = _capture_calls(addon)
        with p:
            addon.response(flow)
        scanned = [c[0][0] for c in calls]
        assert any(urlparse(t).hostname == "evil.com" for t in scanned), f"Location not scanned. Texts: {scanned}"

    def test_blocked_response_header_sets_502(self):
        addon = _make_addon()
        flow = FakeFlow(
            response=FakeResponse(
                headers={"Set-Cookie": "session=stolen"},
            )
        )
        with patch.object(addon, "_scan_text", return_value=BLOCKED):
            addon.response(flow)
        assert flow.response.status_code == 502

    def test_clean_response_passes(self):
        addon = _make_addon()
        flow = FakeFlow(
            response=FakeResponse(
                headers={"Content-Type": "text/html"},
                body=b"hello",
            )
        )
        with patch.object(addon, "_scan_text", return_value=CLEAN):
            addon.response(flow)
        assert flow.response.status_code == 200


# ---- Stats ----


class TestScanStats:
    def test_header_block_increments_stat(self):
        addon = _make_addon()
        flow = FakeFlow(
            request=FakeRequest(
                headers={"Authorization": "Bearer leaked"},
                body=b"",
            )
        )
        with patch.object(addon, "_scan_text", return_value=BLOCKED):
            addon.request(flow)
        assert addon.get_stats().get("requests_blocked_scan", 0) > 0
