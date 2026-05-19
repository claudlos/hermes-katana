"""Tests for the second-wave audit fixes (Codex #4, #6, #7).

Codex #4 (MED): KatanaScanMiddleware.post_dispatch now redacts tool_output
when ``enforce_output_findings=True`` and a finding fires.

Codex #6 (MED): RateTracker eviction now prunes stale entries from EVERY
tracked window (not just the current client's), so clients that hit the
limit and disappeared no longer hold deques that look non-empty.

Codex #7 (LOW): KatanaAddon.websocket_message gates frame size before
decode + scan; oversized frames are blocked fail-closed in strict mode and
prefix-scanned in permissive mode.
"""

from __future__ import annotations

import time


from hermes_katana.middleware.chain import CallContext, DispatchDecision
from hermes_katana.middleware.integration import KatanaScanMiddleware
from hermes_katana.proxy.addon import KatanaAddon, RateTracker
from hermes_katana.proxy.config import ProxyConfig


# ---------------------------------------------------------------------------
# Codex #4 — output-finding enforcement
# ---------------------------------------------------------------------------


class _FakeScanResult:
    def __init__(self, has_findings: bool, summary: str = "fake summary", risk_score: float = 0.0):
        self.has_findings = has_findings
        self.summary = summary
        self.risk_score = risk_score


def test_scan_middleware_post_dispatch_redacts_when_enforce_on(monkeypatch):
    mw = KatanaScanMiddleware(enforce_output_findings=True)

    def fake_scan_output(*args, **kwargs):
        return _FakeScanResult(has_findings=True, summary="leaked vault token")

    import hermes_katana.scanner as scanner_mod

    monkeypatch.setattr(scanner_mod, "scan_output", fake_scan_output, raising=False)

    ctx = CallContext(tool_name="read_file", args={"path": "/tmp/x"})
    ctx.tool_output = "secret stuff: TOKEN_ABC123"
    mw.post_dispatch(ctx)
    assert "redacted by post-dispatch scanner" in str(ctx.tool_output), ctx.tool_output
    assert ctx.extras.get("output_redacted") is True
    assert "leaked vault token" in ctx.extras.get("output_redacted_reason", "")


def test_scan_middleware_post_dispatch_observes_only_when_enforce_off(monkeypatch):
    mw = KatanaScanMiddleware(enforce_output_findings=False)

    def fake_scan_output(*args, **kwargs):
        return _FakeScanResult(has_findings=True, summary="possible injection")

    import hermes_katana.scanner as scanner_mod

    monkeypatch.setattr(scanner_mod, "scan_output", fake_scan_output, raising=False)

    ctx = CallContext(tool_name="read_file", args={"path": "/tmp/x"})
    original = "tool output with potential issue"
    ctx.tool_output = original
    mw.post_dispatch(ctx)
    # Default off: output unchanged, no redaction marker.
    assert ctx.tool_output == original
    assert ctx.extras.get("output_redacted") is None


def test_scan_middleware_skips_structural_args_but_scans_commands(monkeypatch):
    mw = KatanaScanMiddleware(route_aware=True)
    seen_input: list[str] = []
    seen_command: list[str] = []

    def fake_scan_input(text, *args, **kwargs):
        seen_input.append(text)
        return _FakeScanResult(has_findings=False, summary="clean")

    def fake_scan_command(text, *args, **kwargs):
        seen_command.append(text)
        return _FakeScanResult(has_findings=False, summary="clean command")

    import hermes_katana.scanner as scanner_mod

    monkeypatch.setattr(scanner_mod, "scan_input", fake_scan_input, raising=False)
    monkeypatch.setattr(scanner_mod, "scan_command", fake_scan_command, raising=False)

    ctx = CallContext(
        tool_name="terminal",
        args={"workdir": "/home/carlos/project", "timeout": 30, "command": "printf benchmark"},
    )

    assert mw.pre_dispatch(ctx) == DispatchDecision.ALLOW
    assert seen_input == []
    assert seen_command == ["printf benchmark"]


def test_scan_middleware_still_scans_natural_language_content(monkeypatch):
    mw = KatanaScanMiddleware(route_aware=True)
    seen_input: list[str] = []

    def fake_scan_input(text, *args, **kwargs):
        seen_input.append(text)
        return _FakeScanResult(has_findings=False, summary="clean")

    import hermes_katana.scanner as scanner_mod

    monkeypatch.setattr(scanner_mod, "scan_input", fake_scan_input, raising=False)

    ctx = CallContext(tool_name="write_file", args={"path": "sandbox/output.txt", "content": "normal prose content"})

    assert mw.pre_dispatch(ctx) == DispatchDecision.ALLOW
    assert seen_input == ["normal prose content"]


# ---------------------------------------------------------------------------
# Codex #6 — rate-tracker eviction prunes orphaned clients
# ---------------------------------------------------------------------------


def test_rate_tracker_evicts_clients_with_stale_but_nonempty_windows():
    """Clients that hit the limit and disappeared must be evictable.

    Pre-fix behavior: their deque had stale timestamps (popleft was lazy
    only for the current client). Eviction skipped them because their
    window was non-empty. This test reproduces that scenario and verifies
    the post-fix behavior actually drops them.
    """
    rt = RateTracker(max_requests=5, window_seconds=0.1, escalation_factor=1.0)
    # Force a tiny MAX so eviction triggers in this test.
    rt._MAX_CLIENTS = 5

    # Phase 1: 6 distinct clients each hit once at time T0.
    for i in range(6):
        rt.check(f"client_{i}")

    # All 6 windows non-empty — but the per-client deques have entries that
    # will be stale after window_seconds elapses.
    assert len(rt._windows) == 6

    # Wait past the window so all entries become stale.
    time.sleep(0.15)

    # Now a 7th client triggers eviction (we're over MAX_CLIENTS=5).
    rt.check("client_new")

    # All 6 originals had stale entries; the fix should have pruned them
    # and freed their slots. Tracked-clients should be at most 1 now
    # (the new one), or 0 (if the eviction also caught the new one's
    # current entry — implementation detail).
    tracked = len(rt._windows)
    assert tracked <= 2, f"eviction left too many: {tracked}"


def test_rate_tracker_does_not_evict_active_clients():
    rt = RateTracker(max_requests=10, window_seconds=10.0, escalation_factor=1.0)
    rt._MAX_CLIENTS = 5
    # 6 distinct active clients hitting now; eviction must not drop any
    # because they're all within-window.
    for i in range(6):
        rt.check(f"client_{i}")
    # All should still be present (eviction prunes stale entries; nothing's
    # stale because we're well within window_seconds).
    assert len(rt._windows) == 6


# ---------------------------------------------------------------------------
# Codex #7 — WebSocket size cap
# ---------------------------------------------------------------------------


class _Msg:
    def __init__(self, content, from_client=True):
        self.content = content
        self.from_client = from_client


class _WS:
    def __init__(self, msg):
        self.messages = [msg]


class _Flow:
    def __init__(self, content, from_client=True):
        self.websocket = _WS(_Msg(content, from_client=from_client))


def _addon(**cfg_kwargs):
    cfg = ProxyConfig(allowed_domains=["example.com"], **cfg_kwargs)
    return KatanaAddon(cfg)


def test_websocket_oversized_blocked_in_strict_mode():
    addon = _addon(max_body_scan_size=1024, mode="strict")
    big = b"a" * 2048
    flow = _Flow(big, from_client=True)
    addon.websocket_message(flow)
    # Block marker replaces content.
    assert b"exceeds size limit" in flow.websocket.messages[-1].content
    assert b"blocked fail-closed" in flow.websocket.messages[-1].content


def _stub_scan_pass_through(monkeypatch):
    """Replace KatanaAddon._scan_text with a no-op that returns a benign verdict.

    The under-cap and zero-cap WebSocket tests only assert that the size gate
    did NOT fire; they do not test the downstream scanner pipeline. Hitting
    the real ``_scan_text`` triggers semantic_recall, which lazy-loads
    SentenceTransformer + torch.cuda.is_available() — that probe stalls
    indefinitely on hosts where CUDA driver state is unhealthy and prevents
    these unit tests from completing.
    """

    def _fake_scan(self, text, direction="request"):
        return {
            "verdict": "allow",
            "risk_score": 0.0,
            "is_blocked": False,
            "finding_count": 0,
            "summary": "stubbed",
        }

    monkeypatch.setattr(KatanaAddon, "_scan_text", _fake_scan, raising=True)


def test_websocket_under_cap_proceeds_to_scan(monkeypatch):
    _stub_scan_pass_through(monkeypatch)
    addon = _addon(max_body_scan_size=1024, mode="strict")
    flow = _Flow(b"hello world", from_client=True)
    addon.websocket_message(flow)
    # Under cap, scanner runs (no oversized stat hit). The actual scan
    # outcome depends on real scanner; here we just assert the size gate
    # didn't fire.
    stats = addon.get_stats() if hasattr(addon, "get_stats") else getattr(addon, "_stats", {})
    if isinstance(stats, dict):
        assert stats.get("ws_messages_oversized", 0) == 0


def test_websocket_size_cap_zero_disables_gate(monkeypatch):
    _stub_scan_pass_through(monkeypatch)
    addon = _addon(max_body_scan_size=0, mode="strict")  # 0 = unlimited
    big = b"x" * 5000
    flow = _Flow(big, from_client=True)
    # Should not hit the size gate at all.
    addon.websocket_message(flow)
    # Content should not contain the size-block marker.
    assert b"exceeds size limit" not in flow.websocket.messages[-1].content
