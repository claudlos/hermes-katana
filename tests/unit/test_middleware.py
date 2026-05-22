"""Tests for HermesKatana middleware chain."""

from __future__ import annotations

import json


from hermes_katana.middleware.chain import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
    MiddlewareChain,
)


# ======================================================================
# Test middleware implementations
# ======================================================================


class AllowMiddleware(KatanaMiddleware):
    """Always allows calls."""

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        ctx.extras[f"visited_{self.name}"] = True
        return DispatchDecision.ALLOW


class DenyMiddleware(KatanaMiddleware):
    """Always denies calls."""

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        ctx.extras[f"visited_{self.name}"] = True
        ctx.deny(f"Denied by {self.name}")
        return DispatchDecision.DENY


class EscalateMiddleware(KatanaMiddleware):
    """Always escalates calls."""

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        ctx.extras[f"visited_{self.name}"] = True
        ctx.escalate(f"Escalated by {self.name}")
        return DispatchDecision.ESCALATE


class RecordingMiddleware(KatanaMiddleware):
    """Records pre/post visits for order verification."""

    def __init__(self, name: str, record: list, *, priority: int = 0):
        super().__init__(name=name, priority=priority)
        self.record = record

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        self.record.append(f"pre:{self.name}")
        return DispatchDecision.ALLOW

    def post_dispatch(self, ctx: CallContext) -> None:
        self.record.append(f"post:{self.name}")


class TestKatanaTaintMiddleware:
    def test_find_tainted_checks_dict_keys(self):
        from hermes_katana.middleware.taint_middleware import KatanaTaintMiddleware
        from hermes_katana.taint import Source, TaintTracker

        tainted_key = TaintTracker().register("attacker_key", Source.web("https://example.invalid"))

        found = KatanaTaintMiddleware._find_tainted({tainted_key: "value"})

        assert found == [tainted_key]


# ======================================================================
# Chain execution order
# ======================================================================


class TestChainExecutionOrder:
    def test_pre_dispatch_runs_in_priority_order(self):
        record = []
        chain = MiddlewareChain()
        chain.add(RecordingMiddleware("low", record, priority=10))
        chain.add(RecordingMiddleware("high", record, priority=100))
        chain.add(RecordingMiddleware("mid", record, priority=50))

        ctx = CallContext(tool_name="test")
        chain.execute_pre(ctx)

        pre_events = [r for r in record if r.startswith("pre:")]
        assert pre_events == ["pre:high", "pre:mid", "pre:low"]

    def test_post_dispatch_runs_in_reverse_order(self):
        record = []
        chain = MiddlewareChain()
        chain.add(RecordingMiddleware("low", record, priority=10))
        chain.add(RecordingMiddleware("high", record, priority=100))
        chain.add(RecordingMiddleware("mid", record, priority=50))

        ctx = CallContext(tool_name="test")
        chain.execute_pre(ctx)
        record.clear()  # Clear pre records
        chain.execute_post(ctx)

        post_events = [r for r in record if r.startswith("post:")]
        assert post_events == ["post:low", "post:mid", "post:high"]

    def test_disabled_middleware_skipped(self):
        record = []
        chain = MiddlewareChain()

        mw_enabled = RecordingMiddleware("enabled", record, priority=50)
        mw_disabled = RecordingMiddleware("disabled", record, priority=100)
        mw_disabled.enabled = False

        chain.add(mw_enabled)
        chain.add(mw_disabled)

        ctx = CallContext(tool_name="test")
        chain.execute_pre(ctx)

        assert "pre:enabled" in record
        assert "pre:disabled" not in record


# ======================================================================
# DENY short-circuits
# ======================================================================


class TestDenyShortCircuit:
    def test_deny_stops_chain(self):
        chain = MiddlewareChain()
        chain.add(AllowMiddleware("first", priority=100))
        chain.add(DenyMiddleware("blocker", priority=50))
        chain.add(AllowMiddleware("after_blocker", priority=10))

        ctx = CallContext(tool_name="test")
        decision = chain.execute_pre(ctx)

        assert decision == DispatchDecision.DENY
        assert ctx.is_denied is True
        assert ctx.extras.get("visited_first") is True
        assert ctx.extras.get("visited_blocker") is True
        assert ctx.extras.get("visited_after_blocker") is None  # Never reached

    def test_deny_reason_recorded(self):
        chain = MiddlewareChain()
        chain.add(DenyMiddleware("blocker", priority=50))

        ctx = CallContext(tool_name="test")
        chain.execute_pre(ctx)

        assert len(ctx.deny_reasons) > 0
        assert "blocker" in ctx.deny_reasons[0]

    def test_deny_overrides_escalate(self):
        chain = MiddlewareChain()
        chain.add(EscalateMiddleware("escalator", priority=100))
        chain.add(DenyMiddleware("blocker", priority=50))

        ctx = CallContext(tool_name="test")
        decision = chain.execute_pre(ctx)

        # DENY short-circuits regardless of prior ESCALATE
        assert decision == DispatchDecision.DENY


# ======================================================================
# ESCALATE propagation
# ======================================================================


class TestEscalatePropagation:
    def test_escalate_sticky(self):
        chain = MiddlewareChain()
        chain.add(EscalateMiddleware("escalator", priority=100))
        chain.add(AllowMiddleware("allower", priority=50))

        ctx = CallContext(tool_name="test")
        decision = chain.execute_pre(ctx)

        # ESCALATE is sticky — doesn't get downgraded by ALLOW
        assert decision == DispatchDecision.ESCALATE
        assert ctx.is_escalated is True

    def test_escalate_reason_recorded(self):
        chain = MiddlewareChain()
        chain.add(EscalateMiddleware("esc", priority=50))

        ctx = CallContext(tool_name="test")
        chain.execute_pre(ctx)

        assert len(ctx.escalate_reasons) > 0


# ======================================================================
# Chain management
# ======================================================================


class TestChainManagement:
    def test_add_and_list(self):
        chain = MiddlewareChain()
        chain.add(AllowMiddleware("mw1", priority=10))
        chain.add(AllowMiddleware("mw2", priority=20))

        mws = chain.list_middleware()
        assert len(mws) == 2
        assert mws[0].name == "mw2"  # Higher priority first
        assert mws[1].name == "mw1"

    def test_add_replaces_same_name(self):
        chain = MiddlewareChain()
        chain.add(AllowMiddleware("mw1", priority=10))
        chain.add(DenyMiddleware("mw1", priority=20))

        assert len(chain) == 1

    def test_remove(self):
        chain = MiddlewareChain()
        chain.add(AllowMiddleware("mw1", priority=10))
        chain.add(AllowMiddleware("mw2", priority=20))

        removed = chain.remove("mw1")
        assert removed is True
        assert len(chain) == 1

    def test_remove_nonexistent(self):
        chain = MiddlewareChain()
        removed = chain.remove("nonexistent")
        assert removed is False

    def test_clear(self):
        chain = MiddlewareChain()
        chain.add(AllowMiddleware("mw1"))
        chain.add(AllowMiddleware("mw2"))
        chain.clear()
        assert len(chain) == 0

    def test_len_and_bool(self):
        chain = MiddlewareChain()
        assert len(chain) == 0
        assert bool(chain) is False

        chain.add(AllowMiddleware("mw1"))
        assert len(chain) == 1
        assert bool(chain) is True


# ======================================================================
# CallContext
# ======================================================================


class TestCallContext:
    def test_deny_prevents_downgrade(self):
        ctx = CallContext(tool_name="test")
        ctx.deny("reason 1")
        ctx.escalate("reason 2")  # Should not downgrade from DENY
        assert ctx.decision == DispatchDecision.DENY

    def test_escalate_from_allow(self):
        ctx = CallContext(tool_name="test")
        ctx.escalate("reason")
        assert ctx.decision == DispatchDecision.ESCALATE

    def test_total_middleware_ms(self):
        ctx = CallContext(tool_name="test")
        ctx.timestamps.append(("mw1", 1.5))
        ctx.timestamps.append(("mw2", 2.5))
        assert ctx.total_middleware_ms == 4.0

    def test_call_id_auto_generated(self):
        ctx = CallContext(tool_name="test")
        assert len(ctx.call_id) > 0


# ======================================================================
# Full lifecycle: execute()
# ======================================================================


class TestExecuteLifecycle:
    def test_execute_full_lifecycle(self):
        record = []
        chain = MiddlewareChain()
        chain.add(RecordingMiddleware("taint", record, priority=100))
        chain.add(RecordingMiddleware("scan", record, priority=80))
        chain.add(RecordingMiddleware("policy", record, priority=60))

        def executor(tool_name, args):
            return f"executed {tool_name}"

        ctx = chain.execute("terminal", {"command": "ls"}, tool_executor=executor)

        assert ctx.decision == DispatchDecision.ALLOW
        assert ctx.tool_output == "executed terminal"
        # Verify order: pre in priority order, post in reverse
        assert "pre:taint" in record
        assert "pre:scan" in record
        assert "pre:policy" in record
        assert "post:policy" in record
        assert "post:scan" in record
        assert "post:taint" in record

    def test_execute_denied_skips_tool(self):
        chain = MiddlewareChain()
        chain.add(DenyMiddleware("blocker", priority=100))

        executed = False

        def executor(tool_name, args):
            nonlocal executed
            executed = True
            return "result"

        ctx = chain.execute("terminal", {"command": "rm -rf /"}, tool_executor=executor)

        assert ctx.is_denied is True
        assert executed is False
        assert ctx.tool_output is None


# ======================================================================
# create_default_chain
# ======================================================================


class TestCreateDefaultChain:
    def test_create_default_chain_returns_chain(self):
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain()
        assert isinstance(chain, MiddlewareChain)
        assert len(chain) > 0

    def test_create_default_chain_has_expected_middleware(self):
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain()
        names = {mw.name for mw in chain.list_middleware()}
        # Core middleware
        assert "katana.taint" in names
        assert "katana.scabbard" in names
        assert "katana.protectai" in names
        assert "katana.sentinel" in names
        assert "katana.scan" in names
        # New scanners (mcp, multiturn, rag_injection)
        assert "katana.mcp" in names
        assert "katana.multiturn" in names
        assert "katana.rag_injection" in names
        # Structural, behavioral, policy, audit
        assert "katana.structural" in names
        assert "katana.behavioral" in names
        assert "katana.policy" in names
        assert "katana.audit" in names

    def test_audit_middleware_writes_structured_entries(self, tmp_dir):
        from hermes_katana.audit.trail import AuditTrail
        from hermes_katana.middleware.integration import KatanaAuditMiddleware

        trail = AuditTrail(path=tmp_dir / "audit.jsonl")
        middleware = KatanaAuditMiddleware(audit_trail=trail)
        ctx = CallContext(tool_name="terminal", args={"command": "ls"})

        middleware.pre_dispatch(ctx)
        middleware.post_dispatch(ctx)

        entries = trail.query(limit=10)
        assert len(entries) == 2
        assert all(entry.tool_name == "terminal" for entry in entries)

    def test_denied_pre_dispatch_is_audit_visible(self, tmp_dir):
        from hermes_katana.audit.trail import AuditTrail
        from hermes_katana.middleware.integration import create_default_chain

        trail = AuditTrail(path=tmp_dir / "audit.jsonl")
        chain = create_default_chain({"audit.trail": trail})

        ctx = CallContext(tool_name="terminal", args={"command": "rm -rf /"})
        decision = chain.execute_pre(ctx)

        assert decision == DispatchDecision.DENY
        entries = trail.query(limit=10)
        assert len(entries) == 1
        payload = json.loads(entries[0].details)
        assert payload["phase"] == "short_circuit"
        assert payload["decision"] == "deny"
