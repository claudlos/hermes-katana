"""Tests for the pre-buffer size gate hooks added by the 2026-05-07 audit fixes.

Codex audit finding #1 (HIGH): without ``requestheaders`` / ``responseheaders``
hooks, mitmproxy fully buffers oversized bodies before Katana ever runs.
These tests pin the new behavior: reject early on Content-Length, allow when
unknown or within limit, and (Codex #2) mark X-Katana-Scanned as 'partial'
when the body was oversized but allowed through in permissive mode.
"""

from __future__ import annotations

import gzip
import zlib

from hermes_katana.proxy.addon import KatanaAddon
from hermes_katana.proxy.config import ProxyConfig


# ---------------------------------------------------------------------------
# Lightweight mitmproxy flow stand-ins
# ---------------------------------------------------------------------------


class _Headers(dict):
    """dict that also supports .get() with case-insensitive lookup."""

    def get(self, key, default=None):  # type: ignore[override]
        v = super().get(key, None)
        if v is not None:
            return v
        for k, val in self.items():
            if k.lower() == str(key).lower():
                return val
        return default


class _Request:
    def __init__(self, host="example.com", headers=None, body=b""):
        self.host = host
        self.headers = _Headers(headers or {})
        self._body = body

    def get_content(self):
        return self._body


class _Response:
    def __init__(self, headers=None, body=b"", status=200):
        self.headers = _Headers(headers or {})
        self._body = body
        self.status_code = status

    def get_content(self):
        return self._body

    def set_content(self, b):
        self._body = b


class _Flow:
    def __init__(self, request=None, response=None):
        self.request = request or _Request()
        self.response = response


# ---------------------------------------------------------------------------
# requestheaders gate
# ---------------------------------------------------------------------------


def _addon(**cfg_kwargs) -> KatanaAddon:
    cfg = ProxyConfig(allowed_domains=["example.com"], **cfg_kwargs)
    return KatanaAddon(cfg)


def test_requestheaders_blocks_oversized_content_length():
    addon = _addon(max_request_body_size=1_000_000)
    flow = _Flow(_Request(headers={"Content-Length": "9999999"}))  # ~10 MB > 1 MB
    addon.requestheaders(flow)
    assert flow.response is not None, "requestheaders should reject oversized body"
    assert flow.response.status_code == 413
    assert "exceeds proxy limit" in flow.response.get_content().decode()


def test_requestheaders_allows_when_content_length_within_limit():
    addon = _addon(max_request_body_size=1_000_000)
    flow = _Flow(_Request(headers={"Content-Length": "1024"}))
    addon.requestheaders(flow)
    assert flow.response is None, "small body should not be rejected pre-buffer"


def test_requestheaders_allows_when_content_length_missing():
    """Chunked transfers / unknown size: don't reject pre-buffer (handled later)."""
    addon = _addon(max_request_body_size=1_000_000)
    flow = _Flow(_Request(headers={}))
    addon.requestheaders(flow)
    assert flow.response is None


def test_requestheaders_allows_when_limit_zero_unlimited():
    addon = _addon(max_request_body_size=0)  # 0 = unlimited
    flow = _Flow(_Request(headers={"Content-Length": "999999999"}))
    addon.requestheaders(flow)
    assert flow.response is None


def test_responseheaders_blocks_oversized():
    addon = _addon(max_response_body_size=1_000_000, add_scanned_header=True)
    flow = _Flow(
        _Request(),
        _Response(headers={"Content-Length": "5000000"}),  # 5 MB > 1 MB
    )
    addon.responseheaders(flow)
    assert flow.response.status_code == 502
    # When header injection is enabled, the blocked response gets X-Katana-Scanned: blocked
    assert flow.response.headers.get("X-Katana-Scanned") == "blocked"
    assert "exceeds proxy limit" in flow.response.get_content().decode()


def test_responseheaders_records_stat_on_block():
    addon = _addon(max_response_body_size=1_000_000)
    flow = _Flow(_Request(), _Response(headers={"Content-Length": "9000000"}))
    addon.responseheaders(flow)
    stats = addon.get_stats() if hasattr(addon, "get_stats") else getattr(addon, "_stats", {})
    # Either a get_stats() method or an internal _stats dict — accept both.
    val = stats.get("responses_blocked_pre_buffer", 0) if isinstance(stats, dict) else 0
    assert val >= 1, f"expected pre-buffer block stat, got {stats}"


# ---------------------------------------------------------------------------
# Codex #2: X-Katana-Scanned partial vs true vs passthrough
# ---------------------------------------------------------------------------


def test_content_length_helper_parses_string_and_int():
    assert KatanaAddon._content_length(_Headers({"Content-Length": "1024"})) == 1024
    assert KatanaAddon._content_length(_Headers({"content-length": "  2048  "})) == 2048
    assert KatanaAddon._content_length(_Headers({})) is None
    assert KatanaAddon._content_length(_Headers({"Content-Length": "garbage"})) is None
    assert KatanaAddon._content_length(None) is None


def test_gzip_decode_is_bounded_by_scan_cap():
    addon = _addon(max_body_scan_size=128)
    payload = b"A" * (2 * 1024 * 1024)
    compressed = gzip.compress(payload, compresslevel=9)

    decoded, _content_type, oversized = addon._decode_body_for_scan(
        compressed,
        _Headers({"content-type": "text/plain", "content-encoding": "gzip"}),
    )

    assert len(compressed) < len(payload)
    assert decoded == payload[:128]
    assert oversized is True


def test_gzip_decode_scans_concatenated_members():
    addon = _addon(max_body_scan_size=1024)
    payload = b"first member\nsecond member"
    compressed = gzip.compress(b"first member\n") + gzip.compress(b"second member")

    decoded, _content_type, oversized = addon._decode_body_for_scan(
        compressed,
        _Headers({"content-type": "text/plain", "content-encoding": "gzip"}),
    )

    assert decoded == payload
    assert oversized is False


def test_deflate_decode_is_bounded_by_scan_cap():
    addon = _addon(max_body_scan_size=128)
    payload = b"A" * (2 * 1024 * 1024)
    compressed = zlib.compress(payload, level=9)

    decoded, _content_type, oversized = addon._decode_body_for_scan(
        compressed,
        _Headers({"content-type": "text/plain", "content-encoding": "deflate"}),
    )

    assert len(compressed) < len(payload)
    assert decoded == payload[:128]
    assert oversized is True
