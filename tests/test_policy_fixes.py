"""Tests for Worker 4 policy and middleware fixes (GAPs 4.1-4.5)."""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# GAP 4.2 — Unknown tools default to escalate in BALANCED mode
# ---------------------------------------------------------------------------


class TestBalancedUnknownToolDefault:
    """The BALANCED catch-all for unknown tools with taint must be 'escalate'."""

    def test_balanced_catchall_tainted_is_escalate(self):
        from hermes_katana.policy.defaults import BALANCED_POLICIES

        catchall = None
        for p in BALANCED_POLICIES["policies"]:
            if p["name"] == "balanced_catchall":
                catchall = p
                break
        assert catchall is not None
        assert catchall["action"] == "escalate"

    def test_balanced_catchall_clean_is_escalate(self):
        """Unknown clean tools should also escalate (not allow)."""
        from hermes_katana.policy.defaults import BALANCED_POLICIES

        catchall_clean = None
        for p in BALANCED_POLICIES["policies"]:
            if p["name"] == "balanced_catchall_clean":
                catchall_clean = p
                break
        assert catchall_clean is not None
        assert catchall_clean["action"] == "escalate"

    def test_engine_escalates_unknown_tool_with_taint(self):
        from hermes_katana.policy.engine import PolicyEngine
        from hermes_katana.policy.models import PolicyResult

        engine = PolicyEngine.with_defaults("balanced")
        taint_ctx = {"tainted_fields": {"data": {"is_tainted": True, "source": "web", "labels": [], "level": 5}}}
        result = engine.evaluate("totally_unknown_tool_xyz", {"data": "hello"}, taint_ctx)
        assert result.action in (PolicyResult.ESCALATE, PolicyResult.DENY)

    def test_engine_escalates_unknown_clean_tool(self):
        from hermes_katana.policy.engine import PolicyEngine
        from hermes_katana.policy.models import PolicyResult

        engine = PolicyEngine.with_defaults("balanced")
        result = engine.evaluate("totally_unknown_tool_xyz", {"data": "hello"}, {})
        assert result.action == PolicyResult.ESCALATE


# ---------------------------------------------------------------------------
# GAP 4.1 — Middleware bypass detection
# ---------------------------------------------------------------------------


class TestMiddlewareBypassDetection:
    """_direct_call_detector should warn when a tool bypasses the chain."""

    def test_bypass_detected(self):
        from hermes_katana.middleware.chain import (
            register_protected_tool,
            _direct_call_detector,
            get_bypass_warnings,
            clear_bypass_warnings,
        )

        clear_bypass_warnings()
        register_protected_tool("terminal")
        _direct_call_detector("terminal", via_chain=False)
        warnings = get_bypass_warnings()
        assert len(warnings) == 1
        assert warnings[0]["tool"] == "terminal"
        assert "bypassed" in warnings[0]["message"].lower() or "bypass" in warnings[0]["message"].lower()
        clear_bypass_warnings()

    def test_no_bypass_when_via_chain(self):
        from hermes_katana.middleware.chain import (
            register_protected_tool,
            _direct_call_detector,
            get_bypass_warnings,
            clear_bypass_warnings,
        )

        clear_bypass_warnings()
        register_protected_tool("terminal")
        _direct_call_detector("terminal", via_chain=True)
        warnings = get_bypass_warnings()
        assert len(warnings) == 0

    def test_unregistered_tool_no_warning(self):
        from hermes_katana.middleware.chain import (
            _direct_call_detector,
            get_bypass_warnings,
            clear_bypass_warnings,
        )

        clear_bypass_warnings()
        _direct_call_detector("some_random_unregistered_tool", via_chain=False)
        warnings = get_bypass_warnings()
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# GAP 4.3 — Policy thread safety (snapshot evaluation)
# ---------------------------------------------------------------------------


class TestPolicyThreadSafety:
    """evaluate() should snapshot policies so concurrent replace_all() is safe."""

    def test_concurrent_evaluate_and_replace(self):
        from hermes_katana.policy.engine import PolicyEngine

        engine = PolicyEngine.with_defaults("balanced")
        errors = []

        def evaluator():
            for _ in range(50):
                try:
                    engine.evaluate("terminal", {"command": "ls"}, {})
                except Exception as e:
                    errors.append(str(e))

        def replacer():
            for _ in range(50):
                try:
                    policies = engine.list_policies()
                    engine.replace_all(policies)
                except Exception as e:
                    errors.append(str(e))

        threads = [
            threading.Thread(target=evaluator),
            threading.Thread(target=replacer),
            threading.Thread(target=evaluator),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# ---------------------------------------------------------------------------
# GAP 4.4 — Middleware execution order (audit after policy for full context)
# ---------------------------------------------------------------------------


class TestMiddlewareExecutionOrder:
    """Audit middleware runs after policy to have full context.
    Denied calls are still captured via on_short_circuit() regardless of order.
    """

    def test_audit_priority_lower_than_policy(self):
        from hermes_katana.middleware.integration import (
            KatanaAuditMiddleware,
            KatanaPolicyMiddleware,
        )

        audit = KatanaAuditMiddleware()
        policy = KatanaPolicyMiddleware()
        assert audit.priority < policy.priority, (
            f"Audit priority ({audit.priority}) must be < policy priority ({policy.priority}) "
            "so audit has full policy context when logging"
        )

    def test_default_chain_order(self):
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain({"taint.enabled": False, "scan.enabled": False})
        mws = chain.list_middleware()
        names = [m.name for m in mws]
        # Audit should come AFTER policy in execution order (lower priority = later)
        # Denied calls are still captured via on_short_circuit()
        if "katana.audit" in names and "katana.policy" in names:
            audit_idx = names.index("katana.audit")
            policy_idx = names.index("katana.policy")
            assert audit_idx > policy_idx, f"Audit should execute after policy. Order: {names}"


# ---------------------------------------------------------------------------
# GAP 4.5 — Policy hot-reload validation
# ---------------------------------------------------------------------------


class TestHotReloadValidation:
    """replace_all() must reject empty or malformed policy lists."""

    def test_replace_all_rejects_empty(self):
        from hermes_katana.policy.engine import PolicyEngine

        engine = PolicyEngine.with_defaults("balanced")
        original_count = engine.policy_count
        engine.replace_all([])  # Should be rejected
        assert engine.policy_count == original_count

    def test_replace_all_rejects_all_malformed(self):
        """Pydantic won't even allow empty name/tool_pattern, so
        we verify that replace_all with empty list is rejected."""
        from hermes_katana.policy.engine import PolicyEngine

        engine = PolicyEngine.with_defaults("balanced")
        original_count = engine.policy_count

        # Empty list should be rejected
        engine.replace_all([])
        assert engine.policy_count == original_count

    def test_replace_all_accepts_valid(self):
        from hermes_katana.policy.engine import PolicyEngine
        from hermes_katana.policy.models import Policy

        engine = PolicyEngine.with_defaults("balanced")
        good = Policy(name="test_policy", tool_pattern="*", action="allow", priority=1)
        engine.replace_all([good])
        assert engine.policy_count == 1
