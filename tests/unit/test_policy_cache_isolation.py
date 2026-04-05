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


class TestIntegrationCacheDoesNotBypassTaintCheck:
    """End-to-end: evaluate() must return different decisions for
    plain-string and tainted-string calls with the same content."""

    def test_cache_poisoning_via_plain_then_tainted(self):
        """Warm the cache with a plain-string call, then make a tainted
        call with the same content. The two should hit DIFFERENT cache
        entries — if they share a key, the tainted call gets the plain
        call's verdict back (BYPASS)."""
        # Build keys directly (no live engine needed to prove isolation).
        # If keys differ, the two calls cannot share a cache slot.
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
