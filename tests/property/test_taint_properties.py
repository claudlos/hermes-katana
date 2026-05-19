"""Property-based invariant tests for TaintedStr taint propagation.

Inspired by CaMeL (arXiv 2503.18813): if taint can ever silently drop while
an operation is applied, an attacker can launder an untrusted payload into a
trusted sink. These tests assert four invariants on TaintedStr:

1. **Monotonic labels** — every str method (upper, lower, strip, replace,
   format, encode/decode roundtrip, slice, concat, join, split) either
   PRESERVES or SHRINKS the label set. It may never silently add fewer
   labels than it started with when no fresh sources are introduced.
2. **Char-level provenance** — concat/slice preserve per-character taint.
   After ``a + b``, the first ``len(a)`` chars carry a's sources and the
   rest carry b's sources.
3. **Trust floor** — combining two Sources keeps MINIMUM trust: if any
   source is UNTRUSTED the combined value is_untrusted(); trusted-only
   holds only when every contributing source is TRUSTED.
4. **Idempotence** — tainting a TaintedStr with an already-present Source
   is a no-op (set semantics).

Run: pytest tests/property/test_taint_properties.py -q
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from hermes_katana.taint import (
    Source,
    TaintedStr,
    TaintLabel,
    TrustLevel,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Keep timestamp fixed to make Sources hashable-stable across construction.
_FIXED_TS = 1_700_000_000.0

# Web & MCP are UNTRUSTED; USER & SYSTEM are TRUSTED by default; TOOL/FILE/MEMORY
# are CONDITIONAL.
_UNTRUSTED_LABELS = (
    TaintLabel.WEB_CONTENT,
    TaintLabel.MCP,
    TaintLabel.MCP_TOOL_DESCRIPTION,
    TaintLabel.MCP_TOOL_RESULT,
    TaintLabel.UNKNOWN,
)
_TRUSTED_LABELS = (TaintLabel.USER, TaintLabel.SYSTEM)
_CONDITIONAL_LABELS = (
    TaintLabel.TOOL_OUTPUT,
    TaintLabel.FILE_CONTENT,
    TaintLabel.MEMORY,
    TaintLabel.AGENT,
)


def _src(label: TaintLabel, trust: TrustLevel, origin: str = "test") -> Source:
    return Source(label=label, origin=origin, timestamp=_FIXED_TS, trust_level=trust)


@st.composite
def source_sets(draw, min_size: int = 1, max_size: int = 3) -> frozenset[Source]:
    """Build a small frozenset of Sources with distinct origins."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    out: set[Source] = set()
    for i in range(n):
        label = draw(st.sampled_from(list(TaintLabel)))
        trust = draw(st.sampled_from(list(TrustLevel)))
        out.add(_src(label, trust, origin=f"o{i}"))
    return frozenset(out)


@st.composite
def tainted_strs(
    draw,
    min_len: int = 0,
    max_len: int = 80,
    min_sources: int = 1,
) -> TaintedStr:
    text = draw(st.text(min_size=min_len, max_size=max_len))
    srcs = draw(source_sets(min_size=min_sources))
    return TaintedStr(text, sources=srcs)


# Strings that don't change length under upper/lower/strip(None) in a way
# that triggers edge cases — still exercise full unicode.
_TEXT = st.text(min_size=0, max_size=60)
_NONEMPTY_TEXT = st.text(min_size=1, max_size=60)


# ---------------------------------------------------------------------------
# Invariant 1: monotonic labels — operations without new sources preserve labels
# ---------------------------------------------------------------------------


class TestMonotonicLabels:
    """Every op without a fresh tainted arg must preserve the label set."""

    @given(ts=tainted_strs())
    @settings(max_examples=80, deadline=None)
    def test_upper_preserves_labels(self, ts: TaintedStr) -> None:
        out = ts.upper()
        assert out.sources == ts.sources
        assert out.labels == ts.labels

    @given(ts=tainted_strs())
    @settings(max_examples=80, deadline=None)
    def test_lower_preserves_labels(self, ts: TaintedStr) -> None:
        out = ts.lower()
        assert out.sources == ts.sources
        assert out.labels == ts.labels

    @given(ts=tainted_strs(min_len=1))
    @settings(max_examples=80, deadline=None)
    def test_strip_preserves_or_shrinks_labels(self, ts: TaintedStr) -> None:
        out = ts.strip()
        # strip() can only narrow to chars that remain; its source set must
        # be a SUBSET of the original (monotonic shrinkage).
        assert out.sources.issubset(ts.sources)
        # If something survives stripping, it must still carry at least one
        # source — NEVER silent untainting of non-empty content.
        if len(out) > 0 and ts.sources:
            # char_taint may be empty-default-frozenset only if every
            # surviving char was overridden, but sources set must be non-empty
            # whenever content exists AND originally carried taint.
            assert out.sources, (
                f"strip() silently dropped taint from non-empty residue: orig={ts.sources!r} -> out empty"
            )

    @given(
        ts=tainted_strs(min_len=1),
        old=_NONEMPTY_TEXT,
        new=_TEXT,
    )
    @settings(max_examples=60, deadline=None)
    def test_replace_preserves_labels(self, ts: TaintedStr, old: str, new: str) -> None:
        out = ts.replace(old, new)
        # replace(old, new) may only spread existing sources; no new taint.
        assert out.sources == ts.sources

    @given(ts=tainted_strs(min_len=1), i=st.integers(min_value=-30, max_value=30))
    @settings(max_examples=80, deadline=None)
    def test_index_slice_preserves_or_shrinks(self, ts: TaintedStr, i: int) -> None:
        assume(-len(ts) <= i < len(ts))
        out = ts[i]  # single char
        # Single-char taint must be a subset of original
        assert out.sources.issubset(ts.sources)
        # Never silently untaint a character that came from a tainted string
        if ts.sources:
            assert out.sources, "Single-char indexing silently dropped taint"

    @given(
        ts=tainted_strs(min_len=2),
        a=st.integers(min_value=0, max_value=40),
        b=st.integers(min_value=0, max_value=40),
    )
    @settings(max_examples=80, deadline=None)
    def test_range_slice_preserves_or_shrinks(self, ts: TaintedStr, a: int, b: int) -> None:
        out = ts[a:b]
        assert out.sources.issubset(ts.sources)
        if len(out) > 0 and ts.sources:
            assert out.sources, "Slice silently dropped taint"

    @given(ts=tainted_strs(min_len=1))
    @settings(max_examples=40, deadline=None)
    def test_format_spec_preserves_labels(self, ts: TaintedStr) -> None:
        out = ts.__format__("")  # f"{ts}" with no spec
        assert out.sources == ts.sources

    @given(ts=tainted_strs(min_len=1))
    @settings(max_examples=40, deadline=None)
    def test_encode_decode_roundtrip_does_not_silently_drop(self, ts: TaintedStr) -> None:
        # encode returns plain bytes (taint is lost by design — and WARNED).
        # The contract: the library warns; we verify it emits bytes identical
        # to the raw string. This test documents the known drop site.
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            raw = ts.encode("utf-8")
            # Must warn that taint was dropped
            assert any("taint" in str(w.message).lower() for w in caught), (
                "encode() dropped taint without emitting a warning"
            )
        # Raw-str equivalence (go through str directly to avoid the warning).
        assert raw == str.__str__(ts).encode("utf-8")

    @given(ts=tainted_strs(min_len=1), sep=st.sampled_from([None, " ", ",", "."]))
    @settings(max_examples=40, deadline=None)
    def test_split_parts_preserve_labels_subset(self, ts: TaintedStr, sep) -> None:
        parts = ts.split(sep)
        for p in parts:
            assert p.sources.issubset(ts.sources)
            if len(p) > 0 and ts.sources:
                assert p.sources, "split() silently dropped taint from a part"


# ---------------------------------------------------------------------------
# Invariant 2: char-level provenance is preserved through concat / slice
# ---------------------------------------------------------------------------


class TestCharLevelProvenance:
    @given(a=tainted_strs(min_len=1), b=tainted_strs(min_len=1))
    @settings(max_examples=60, deadline=None)
    def test_concat_splits_preserve_each_side(self, a: TaintedStr, b: TaintedStr) -> None:
        ab = a + b
        la, lb = len(a), len(b)
        assert len(ab) == la + lb
        # Union invariant on the full string
        assert ab.sources == a.sources | b.sources
        # Slice back out each side. The invariant is NEVER UNDER-TAINT:
        # each side must retain at least the sources of its contributing half
        # (the scanner may safely over-approximate with union but never drop).
        left = ab[:la]
        right = ab[la:]
        assert left.sources.issuperset(a.sources), "Char-level provenance broken: left half lost sources from a"
        assert right.sources.issuperset(b.sources), "Char-level provenance broken: right half lost sources from b"
        # Over-approximation bound: no source from outside (a|b) appears
        assert left.sources.issubset(a.sources | b.sources)
        assert right.sources.issubset(a.sources | b.sources)

    @given(a=tainted_strs(min_len=1), b=tainted_strs(min_len=1))
    @settings(max_examples=40, deadline=None)
    def test_concat_slice_index_carries_correct_side(self, a: TaintedStr, b: TaintedStr) -> None:
        ab = a + b
        la = len(a)
        # First char must retain a's taint (over-approx allowed)
        assert ab[0].sources.issuperset(a.sources)
        # Last char must retain b's taint
        assert ab[-1].sources.issuperset(b.sources)
        # Exact boundary: char at index la is first of b, retains b's taint
        assert ab[la].sources.issuperset(b.sources)

    @given(
        a=tainted_strs(min_len=1),
        b=tainted_strs(min_len=1),
        c=tainted_strs(min_len=1),
    )
    @settings(max_examples=30, deadline=None)
    def test_concat_associative_on_sources(self, a: TaintedStr, b: TaintedStr, c: TaintedStr) -> None:
        left = (a + b) + c
        right = a + (b + c)
        assert left.sources == right.sources
        assert str(left) == str(right)


# ---------------------------------------------------------------------------
# Invariant 3: trust level FLOOR (minimum) across combinations
# ---------------------------------------------------------------------------


class TestTrustFloor:
    @given(
        untrusted_label=st.sampled_from(_UNTRUSTED_LABELS),
        trusted_label=st.sampled_from(_TRUSTED_LABELS),
        a_text=_NONEMPTY_TEXT,
        b_text=_NONEMPTY_TEXT,
    )
    @settings(max_examples=40, deadline=None)
    def test_untrusted_dominates_on_concat(
        self,
        untrusted_label: TaintLabel,
        trusted_label: TaintLabel,
        a_text: str,
        b_text: str,
    ) -> None:
        trusted = TaintedStr(
            a_text,
            sources=frozenset({_src(trusted_label, TrustLevel.TRUSTED)}),
        )
        untrusted = TaintedStr(
            b_text,
            sources=frozenset({_src(untrusted_label, TrustLevel.UNTRUSTED)}),
        )
        out = trusted + untrusted
        assert out.is_untrusted(), "Trust floor violated: trusted + untrusted was not flagged untrusted"
        assert not out.is_trusted(), "Trust floor violated: combined value still reports is_trusted()"

    @given(
        l1=st.sampled_from(_TRUSTED_LABELS),
        l2=st.sampled_from(_TRUSTED_LABELS),
        a_text=_NONEMPTY_TEXT,
        b_text=_NONEMPTY_TEXT,
    )
    @settings(max_examples=20, deadline=None)
    def test_trusted_plus_trusted_stays_trusted(self, l1, l2, a_text: str, b_text: str) -> None:
        a = TaintedStr(a_text, sources=frozenset({_src(l1, TrustLevel.TRUSTED)}))
        b = TaintedStr(b_text, sources=frozenset({_src(l2, TrustLevel.TRUSTED)}))
        out = a + b
        assert out.is_trusted()
        assert not out.is_untrusted()

    @given(
        labels=st.lists(st.sampled_from(list(TaintLabel)), min_size=1, max_size=4, unique=True),
        trusts=st.lists(st.sampled_from(list(TrustLevel)), min_size=1, max_size=4),
        text=_NONEMPTY_TEXT,
    )
    @settings(max_examples=40, deadline=None)
    def test_is_untrusted_iff_any_untrusted(self, labels, trusts, text: str) -> None:
        n = min(len(labels), len(trusts))
        srcs = frozenset(_src(labels[i], trusts[i], origin=f"o{i}") for i in range(n))
        ts = TaintedStr(text, sources=srcs)
        any_untrusted = any(s.trust_level is TrustLevel.UNTRUSTED for s in srcs)
        all_trusted = all(s.trust_level is TrustLevel.TRUSTED for s in srcs)
        assert ts.is_untrusted() == any_untrusted
        assert ts.is_trusted() == all_trusted


# ---------------------------------------------------------------------------
# Invariant 4: idempotence — adding the same source twice is a no-op
# ---------------------------------------------------------------------------


class TestIdempotence:
    @given(text=_NONEMPTY_TEXT, label=st.sampled_from(list(TaintLabel)))
    @settings(max_examples=40, deadline=None)
    def test_duplicate_source_is_noop(self, text: str, label: TaintLabel) -> None:
        s = _src(label, TrustLevel.UNTRUSTED, origin="dup")
        once = TaintedStr(text, sources=frozenset({s}))
        twice = TaintedStr(text, sources=frozenset({s, s}))  # set collapses
        assert once.sources == twice.sources
        assert once.labels == twice.labels

    @given(text=_NONEMPTY_TEXT)
    @settings(max_examples=40, deadline=None)
    def test_self_concat_preserves_sources(self, text: str) -> None:
        ts = TaintedStr(
            text,
            sources=frozenset({_src(TaintLabel.WEB_CONTENT, TrustLevel.UNTRUSTED)}),
        )
        out = ts + ts
        # Self-union of a set equals the set
        assert out.sources == ts.sources


# ---------------------------------------------------------------------------
# Regression guard: scanning a tainted string must not silently untaint it.
# ---------------------------------------------------------------------------


class TestScanDoesNotUntaint:
    @given(ts=tainted_strs(min_len=1))
    @settings(max_examples=30, deadline=None)
    def test_repr_and_upper_chain_keeps_sources(self, ts: TaintedStr) -> None:
        chained = ts.upper().lower().replace("", "")  # no-op chain
        assert chained.sources == ts.sources
