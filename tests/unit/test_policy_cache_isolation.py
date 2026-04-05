"""Regression tests for the policy evaluation cache key.

These guard against issue #4: the original cache key used
``json.dumps(..., default=str)`` which silently coerced taint objects
to their string representation. A plain string and a TaintedStr with
the same content produced identical cache keys, so an attacker could
warm the cache with a benign plain-string call, then bypass policy
on a subsequent tainted call with the same content.

Every test here is a policy-bypass attempt. If any assertion fires,
distinct taint contexts are collapsing to the same cache key and
cached ALLOW verdicts could leak through.
"""

from __future__ import annotations

import pytest

from hermes_katana.policy.engine import PolicyEngine
from hermes_katana.taint import Source, TaintedStr


class TestCacheKeyTaintAwareness:
    """Cache key must distinguish plain strings from tainted values
    and distinct taint objects from one another."""

    def test_plain_string_vs_tainted_string_same_content(self):
        """A plain ``str`` and a ``TaintedStr`` with identical content
        must produce different cache keys — otherwise a benign call
        can poison the cache for subsequent tainted calls."""
        plain = "payload"
        tainted = TaintedStr("payload")
        k_plain = PolicyEngine._make_cache_key("terminal", {"arg": plain}, {})
        k_tainted = PolicyEngine._make_cache_key("terminal", {"arg": tainted}, {})
        assert k_plain != k_tainted, (
            "CACHE COLLISION: plain str and TaintedStr with same content hash "
            "to the same key — policy bypass via cache poisoning"
        )

    def test_tainted_string_clean_vs_with_source(self):
        """TaintedStr with the same content but different taint sources
        must produce different cache keys."""
        clean = TaintedStr("payload")
        tainted = TaintedStr("payload", sources=frozenset({Source.web("https://evil.example")}))
        k_clean = PolicyEngine._make_cache_key("terminal", {"arg": clean}, {})
        k_tainted = PolicyEngine._make_cache_key("terminal", {"arg": tainted}, {})
        assert k_clean != k_tainted, (
            "CACHE COLLISION: same-content TaintedStr with and without a web "
            "source hash to the same key — untrusted content would skip policy"
        )

    def test_source_user_vs_web_vs_mcp_distinct(self):
        """Different Source types must produce different cache keys —
        trusted user vs untrusted web vs untrusted mcp should never
        share a cache entry."""
        ctx_user = {"source": Source.user("alice")}
        ctx_web = {"source": Source.web("https://evil.example")}
        ctx_mcp = {"source": Source.mcp("untrusted-mcp")}
        k_user = PolicyEngine._make_cache_key("terminal", {"cmd": "curl"}, ctx_user)
        k_web = PolicyEngine._make_cache_key("terminal", {"cmd": "curl"}, ctx_web)
        k_mcp = PolicyEngine._make_cache_key("terminal", {"cmd": "curl"}, ctx_mcp)
        assert len({k_user, k_web, k_mcp}) == 3, (
            f"CACHE COLLISION across Source types: user={k_user} web={k_web} mcp={k_mcp}"
        )

    def test_source_origin_changes_key(self):
        """Two web Sources with different origins must produce different
        cache keys — an attacker should not be able to reuse a cached
        verdict from a trusted origin for a malicious origin."""
        k_site_a = PolicyEngine._make_cache_key(
            "terminal", {"cmd": "curl"}, {"source": Source.web("https://trusted.example")}
        )
        k_site_b = PolicyEngine._make_cache_key(
            "terminal", {"cmd": "curl"}, {"source": Source.web("https://evil.example")}
        )
        assert k_site_a != k_site_b

    def test_foreign_object_does_not_collide_with_primitive(self):
        """A foreign object whose ``__str__`` / ``__repr__`` happens to
        match a primitive must not produce the same cache key. This is
        the exact bug reported in issue #4."""

        class Spoof:
            def __str__(self):
                return "user"

            def __repr__(self):
                return "user"

        k_primitive = PolicyEngine._make_cache_key("terminal", {"cmd": "rm -rf /"}, {"source": "user"})
        k_spoof = PolicyEngine._make_cache_key("terminal", {"cmd": "rm -rf /"}, {"source": Spoof()})
        assert k_primitive != k_spoof, (
            "CACHE COLLISION: foreign object stringified to same key as "
            "primitive — this is CVE-shaped bypass via cache poisoning"
        )


class TestCacheKeyDeterminism:
    """Cache key must be stable across equivalent inputs or the cache
    never hits and we waste CPU re-evaluating identical calls."""

    def test_same_source_type_produces_stable_key(self):
        """Two Source objects of the same type with matching fields must
        produce identical keys (despite internal timestamp differences).
        The previous key scheme included timestamps, which made every
        new Source object produce a fresh key — cache was useless."""
        s1 = Source.web("https://example.com")
        s2 = Source.web("https://example.com")
        k1 = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {"source": s1})
        k2 = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {"source": s2})
        assert k1 == k2, (
            "cache key changes between equivalent Source objects — likely including timestamp, which defeats caching"
        )

    def test_dict_ordering_does_not_affect_key(self):
        k1 = PolicyEngine._make_cache_key(
            "terminal",
            {"cmd": "ls", "verbose": True, "depth": 3},
            {"source": "user", "trust": "high"},
        )
        k2 = PolicyEngine._make_cache_key(
            "terminal",
            {"depth": 3, "verbose": True, "cmd": "ls"},
            {"trust": "high", "source": "user"},
        )
        assert k1 == k2

    def test_nested_tainted_string_in_list(self):
        """TaintedStr in a nested list must still affect the key."""
        k_plain = PolicyEngine._make_cache_key("terminal", {"argv": ["a", "b", "payload"]}, {})
        k_tainted = PolicyEngine._make_cache_key("terminal", {"argv": ["a", "b", TaintedStr("payload")]}, {})
        assert k_plain != k_tainted


class TestCacheKeyNestedTypesNotCollapsed:
    """Follow-up review (PR #5 CodeRabbit): the first-round fix coerced
    dict keys via ``str()`` (reintroducing the same collision class for
    structured keys) and collapsed list/tuple and set/frozenset into a
    single kind tag. Both would let distinct inputs share a cache entry."""

    def test_dict_int_key_vs_string_key(self):
        """{1: v} and {"1": v} serialize identically after str() —
        they must NOT share a cache entry."""
        k_int_key = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {1: "trusted"})
        k_str_key = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {"1": "trusted"})
        assert k_int_key != k_str_key, (
            "CACHE COLLISION: integer dict key and matching string key hash to the same cache entry"
        )

    def test_dict_spoof_object_key_vs_string_key(self):
        class Spoof:
            def __str__(self):
                return "source"

        k_spoof = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {Spoof(): "trusted"})
        k_string = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {"source": "trusted"})
        assert k_spoof != k_string, (
            "CACHE COLLISION: foreign object dict key stringified to the same value as a legitimate key"
        )

    def test_list_vs_tuple_not_collapsed(self):
        k_list = PolicyEngine._make_cache_key("terminal", {"argv": [1, 2, 3]}, {})
        k_tuple = PolicyEngine._make_cache_key("terminal", {"argv": (1, 2, 3)}, {})
        assert k_list != k_tuple, (
            "list and tuple collapsed to same cache key — evaluate_condition() treats them differently"
        )

    def test_set_vs_frozenset_not_collapsed(self):
        k_set = PolicyEngine._make_cache_key("terminal", {"args": {1, 2, 3}}, {})
        k_frozen = PolicyEngine._make_cache_key("terminal", {"args": frozenset({1, 2, 3})}, {})
        assert k_set != k_frozen

    def test_list_vs_set_not_collapsed(self):
        k_list = PolicyEngine._make_cache_key("terminal", {"argv": [1, 2, 3]}, {})
        k_set = PolicyEngine._make_cache_key("terminal", {"argv": {1, 2, 3}}, {})
        assert k_list != k_set

    def test_mixed_key_types_do_not_raise(self):
        """Sorting must tolerate mixed-type dict keys. The older sorted(
        (str(k), fp(v)) for ...) would TypeError when two keys stringified
        identically and Python then tried to compare the fingerprint
        values across different types."""
        # Should not raise:
        key = PolicyEngine._make_cache_key("terminal", {"cmd": "ls"}, {1: "a", "1": "b", (1,): "c"})
        assert isinstance(key, str) and len(key) == 32


class TestKeyIsolationForBypassPath:
    """Proves that distinct benign vs tainted calls produce distinct cache
    keys — the necessary condition for cache-poisoning bypass to be impossible."""

    def test_plain_call_and_tainted_call_have_distinct_keys(self):
        """Plain-string call and web-tainted call with identical content
        must hit DIFFERENT cache entries."""
        plain_call_key = PolicyEngine._make_cache_key(
            "terminal",
            {"command": "curl https://api.example.com"},
            {"source": "user"},
        )
        tainted_payload = TaintedStr(
            "curl https://api.example.com",
            sources=frozenset({Source.web("https://attacker.example")}),
        )
        tainted_call_key = PolicyEngine._make_cache_key(
            "terminal",
            {"command": tainted_payload},
            {"source": Source.web("https://attacker.example")},
        )
        assert plain_call_key != tainted_call_key, (
            "POLICY BYPASS: plain-string terminal call and web-tainted "
            "terminal call with identical content hash to the same cache "
            "key — the web-tainted call will inherit the plain call's "
            "ALLOW verdict and skip policy"
        )


class TestEvaluateEndToEndCacheIsolation:
    """End-to-end against a real PolicyEngine.evaluate(): after a benign
    call is cached, a subsequent call with a tainted argument MUST create
    a new cache entry rather than returning the benign verdict."""

    def test_evaluate_creates_distinct_cache_entries_for_tainted_args(self):
        """Run both calls through evaluate() and inspect the engine's
        internal cache. The vulnerable version would have a single entry
        after both calls (collision); the fixed version has two entries."""
        engine = PolicyEngine(policies=[])  # no policies → default path
        # Baseline: cache should start empty (the engine may internally
        # warm some caches on construction, so snapshot length instead).
        baseline_size = len(engine._eval_cache)

        # Call 1: plain string command
        r1 = engine.evaluate("terminal", {"command": "ls -la"}, {"source": "user"})
        assert r1 is not None
        after_plain = len(engine._eval_cache)

        # Call 2: same textual content, but tainted with a WEB source.
        # In the vulnerable code, key collided with call 1 → cache hit →
        # r2 would equal r1 without going through command_safety_check.
        tainted = TaintedStr("ls -la", sources=frozenset({Source.web("https://attacker.example")}))
        engine.evaluate(
            "terminal",
            {"command": tainted},
            {"source": Source.web("https://attacker.example")},
        )
        after_tainted = len(engine._eval_cache)

        # The tainted call MUST add a new cache entry — otherwise it hit
        # the existing entry and skipped policy re-evaluation.
        assert after_tainted == after_plain + 1, (
            f"BYPASS: tainted call reused the cached entry from the benign "
            f"call (cache size {after_plain} → {after_tainted}, expected "
            f"{after_plain + 1}). Baseline was {baseline_size}."
        )

    def test_evaluate_reuses_cache_for_equivalent_sources_despite_timestamps(self):
        """Same tool, same args, equivalent Sources constructed at
        different times MUST hit the same cache entry. The pre-fix scheme
        included Source.timestamp, so every call created a fresh entry
        (cache effectively disabled)."""
        engine = PolicyEngine(policies=[])
        baseline = len(engine._eval_cache)

        engine.evaluate("terminal", {"command": "ls"}, {"src": Source.user("alice")})
        after_first = len(engine._eval_cache)

        # Second call: equivalent Source object (new timestamp, but same
        # label/origin/trust) — must hit the entry created above.
        engine.evaluate("terminal", {"command": "ls"}, {"src": Source.user("alice")})
        after_second = len(engine._eval_cache)

        assert after_second == after_first, (
            f"equivalent Source objects produced different cache keys "
            f"(cache grew {after_first} → {after_second}); timestamp is "
            f"leaking into the fingerprint, defeating caching. "
            f"Baseline was {baseline}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
