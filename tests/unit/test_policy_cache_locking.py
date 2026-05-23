"""Regression tests for policy evaluation cache locking."""

from __future__ import annotations

from hermes_katana.policy.engine import EvaluationResult, PolicyEngine, PolicyResult


class CountingLock:
    def __init__(self) -> None:
        self.enters = 0

    def __enter__(self):
        self.enters += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_cache_put_is_lock_protected():
    engine = PolicyEngine()
    lock = CountingLock()
    engine._lock = lock  # type: ignore[assignment]

    result = EvaluationResult(action=PolicyResult.ALLOW, matched_policy=None, reason="ok", details={})
    engine._cache_put("key", result)

    assert lock.enters == 1
    assert engine._eval_cache["key"] is result


def test_invalidate_cache_is_lock_protected():
    engine = PolicyEngine()
    engine._eval_cache["key"] = EvaluationResult(
        action=PolicyResult.ALLOW,
        matched_policy=None,
        reason="ok",
        details={},
    )
    lock = CountingLock()
    engine._lock = lock  # type: ignore[assignment]

    engine.invalidate_cache()

    assert lock.enters == 1
    assert engine._eval_cache == {}


def test_taint_differing_args_produce_distinct_cache_keys():
    """A tainted vs untainted arg with the same string value must not collide.

    This is the property that makes the cache taint-aware. If both share a
    cache entry, a previously-allowed clean arg could let a later tainted
    arg through without re-evaluation.
    """
    from hermes_katana.taint.labels import Source, TaintLabel
    from hermes_katana.taint.value import TaintedStr

    untainted_args = {"command": "ls /tmp"}
    src = Source(label=TaintLabel.USER, origin="user_input")
    tainted_args = {"command": TaintedStr("ls /tmp", sources=frozenset({src}))}

    key_untainted = PolicyEngine._make_cache_key("terminal", untainted_args, {})
    key_tainted = PolicyEngine._make_cache_key("terminal", tainted_args, {})

    assert key_untainted != key_tainted, (
        "Cache key did not include taint metadata — tainted and untainted args collide."
    )


def test_taint_context_changes_cache_key():
    args = {"command": "ls"}
    ctx_a = {"sensitive": False}
    ctx_b = {"sensitive": True}

    key_a = PolicyEngine._make_cache_key("terminal", args, ctx_a)
    key_b = PolicyEngine._make_cache_key("terminal", args, ctx_b)

    assert key_a != key_b
