"""Tests for taint laundering fixes (Worker 1).

Covers all 12 fixes from .task-w1-taint.md.
"""

import logging

import pytest

from hermes_katana.taint.labels import Source, TaintLabel, TrustLevel
from hermes_katana.taint.value import (
    TaintedBytes,
    TaintedDict,
    TaintedList,
    TaintedStr,
    TaintedValue,
    taint_aware_fstring,
)
from hermes_katana.taint.flow import (
    CRITICAL_SINKS,
    FlowAnalyzer,
    FlowDecision,
    FlowRule,
)


def _src(label=TaintLabel.WEB_CONTENT, trust=TrustLevel.UNTRUSTED):
    return frozenset({Source(label=label, trust_level=trust, origin="test")})


def _has_label(sources, label=TaintLabel.WEB_CONTENT):
    return any(s.label == label for s in sources)


# ---------------------------------------------------------------------------
# Fix 1: __format__ override
# ---------------------------------------------------------------------------


class TestFormat:
    def test_format_method_returns_tainted(self):
        """Direct __format__ call returns TaintedStr (CPython coerces via builtins)."""
        t = TaintedStr("hello", sources=_src())
        result = t.__format__("")
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources)

    def test_format_spec_works(self):
        t = TaintedStr("hi", sources=_src())
        result = t.__format__(">10")
        assert isinstance(result, TaintedStr)
        assert result.value == "        hi"
        assert _has_label(result.sources)

    def test_format_method_preserves_taint(self):
        tmpl = TaintedStr("say {}", sources=_src())
        arg = TaintedStr("hello", sources=_src(TaintLabel.MCP))
        result = tmpl.format(arg)
        assert isinstance(result, TaintedStr)
        assert result.value == "say hello"
        assert _has_label(result.sources, TaintLabel.WEB_CONTENT)
        assert _has_label(result.sources, TaintLabel.MCP)


# ---------------------------------------------------------------------------
# Fix 2: __str__ returns TaintedStr (self)
# ---------------------------------------------------------------------------


class TestStr:
    def test_str_method_returns_tainted(self):
        """Direct __str__ call returns TaintedStr."""
        t = TaintedStr("secret", sources=_src())
        result = t.__str__()
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources)

    def test_str_value_correct(self):
        t = TaintedStr("abc", sources=_src())
        assert str(t) == "abc"


# ---------------------------------------------------------------------------
# Fix 3: __repr__ returns TaintedStr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_method_returns_tainted(self):
        """Direct __repr__ call returns TaintedStr."""
        t = TaintedStr("data", sources=_src())
        result = t.__repr__()
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources)
        assert "TaintedStr" in result.value

    def test_repr_contains_info(self):
        t = TaintedStr("data", sources=_src())
        r = repr(t)
        assert "TaintedStr" in r
        assert "data" in r


# ---------------------------------------------------------------------------
# Fix 4: __mod__ and __rmod__
# ---------------------------------------------------------------------------


class TestModFormatting:
    def test_mod_preserves_taint(self):
        t = TaintedStr("value is %s", sources=_src())
        result = t % "foo"
        assert isinstance(result, TaintedStr)
        assert result.value == "value is foo"
        assert _has_label(result.sources)

    def test_rmod_preserves_taint(self):
        t = TaintedStr("injected", sources=_src())
        result = "data: %s" % t
        assert isinstance(result, TaintedStr)
        assert result.value == "data: injected"
        assert _has_label(result.sources)

    def test_mod_merges_sources(self):
        src1 = _src(TaintLabel.WEB_CONTENT)
        src2 = _src(TaintLabel.MCP)
        t = TaintedStr("fmt %s", sources=src1)
        arg = TaintedStr("val", sources=src2)
        result = t % arg
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources, TaintLabel.WEB_CONTENT)
        assert _has_label(result.sources, TaintLabel.MCP)

    def test_mod_with_tuple_args(self):
        t = TaintedStr("%s and %s", sources=_src())
        result = t % ("a", "b")
        assert isinstance(result, TaintedStr)
        assert result.value == "a and b"


# ---------------------------------------------------------------------------
# Fix 5: encode() preserves taint via TaintedBytes
# ---------------------------------------------------------------------------
#
# Historically TaintedStr.encode() returned plain bytes and emitted a
# warning, which opened a codec-laundering evasion path (tainted → .encode
# → base64 → .decode strips taint). That behaviour was replaced with
# TaintedBytes propagation — see tests/test_codec_evasion.py for the full
# attack surface.


class TestEncode:
    def test_encode_returns_tainted_bytes(self):
        t = TaintedStr("hello", sources=_src())
        result = t.encode("utf-8")
        assert isinstance(result, TaintedBytes)
        assert bytes(result) == b"hello"
        assert _has_label(result.sources)

    def test_encode_decode_roundtrip_preserves_taint(self):
        t = TaintedStr("hello", sources=_src())
        round_tripped = t.encode("utf-8").decode("utf-8")
        assert isinstance(round_tripped, TaintedStr)
        assert _has_label(round_tripped.sources)

    def test_encode_with_non_default_encoding(self):
        t = TaintedStr("caf\u00e9", sources=_src())
        result = t.encode("latin-1")
        assert isinstance(result, TaintedBytes)
        assert bytes(result) == b"caf\xe9"
        assert _has_label(result.sources)


# ---------------------------------------------------------------------------
# Fix 6: split() cursor-based
# ---------------------------------------------------------------------------


class TestSplitCursor:
    def test_split_repeated_substrings(self):
        t = TaintedStr("aa|aa|aa", sources=_src())
        parts = t.split("|")
        assert [p.value for p in parts] == ["aa", "aa", "aa"]
        for p in parts:
            assert isinstance(p, TaintedStr)
            assert p.sources

    def test_split_whitespace(self):
        t = TaintedStr("  a  b  c  ", sources=_src())
        parts = t.split()
        assert [p.value for p in parts] == ["a", "b", "c"]
        for p in parts:
            assert isinstance(p, TaintedStr)


# ---------------------------------------------------------------------------
# Fix 7: strip() direct offset
# ---------------------------------------------------------------------------


class TestStripOffset:
    def test_strip_ambiguous(self):
        t = TaintedStr("xxyxx", sources=_src())
        result = t.strip("x")
        assert result.value == "y"
        assert isinstance(result, TaintedStr)
        assert result.sources

    def test_strip_whitespace(self):
        t = TaintedStr("  hello  ", sources=_src())
        result = t.strip()
        assert result.value == "hello"
        assert isinstance(result, TaintedStr)


# ---------------------------------------------------------------------------
# Fix 8: TaintedList/TaintedDict __getitem__ wrapping
# ---------------------------------------------------------------------------


class TestContainerGetitem:
    def test_list_getitem_wraps(self):
        tl = TaintedList([1, 2, 3], sources=_src())
        item = tl[0]
        assert isinstance(item, TaintedValue)
        assert item.value == 1
        assert _has_label(item.sources)

    def test_dict_getitem_wraps(self):
        td = TaintedDict({"key": "val"}, sources=_src())
        item = td["key"]
        assert isinstance(item, TaintedValue)
        assert item.value == "val"
        assert _has_label(item.sources)

    def test_list_getitem_preserves_existing_taint(self):
        inner = TaintedValue(value=42, sources=_src(TaintLabel.MCP))
        tl = TaintedList([inner], sources=_src())
        item = tl[0]
        assert isinstance(item, TaintedValue)
        assert _has_label(item.sources, TaintLabel.MCP)


# ---------------------------------------------------------------------------
# Fix 9: unwrap() audit trail
# ---------------------------------------------------------------------------


class TestUnwrapAudit:
    def test_unwrap_logs_warning(self, caplog):
        t = TaintedValue(value="secret", sources=_src())
        with caplog.at_level(logging.WARNING):
            result = t.unwrap()
        assert result == "secret"
        assert "Taint stripped" in caplog.text

    def test_unwrap_with_reason(self, caplog):
        t = TaintedValue(value="data", sources=_src())
        with caplog.at_level(logging.WARNING):
            t.unwrap(reason="sending to trusted API")
        assert "sending to trusted API" in caplog.text

    def test_unwrap_audit_false_no_log(self, caplog):
        t = TaintedValue(value="data", sources=_src())
        with caplog.at_level(logging.WARNING):
            t.unwrap(audit=False)
        assert "Taint stripped" not in caplog.text

    def test_unwrap_no_sources_no_log(self, caplog):
        t = TaintedValue(value="clean")
        with caplog.at_level(logging.WARNING):
            t.unwrap()
        assert "Taint stripped" not in caplog.text


# ---------------------------------------------------------------------------
# Fix 10: Critical sinks expanded
# ---------------------------------------------------------------------------


class TestCriticalSinks:
    @pytest.mark.parametrize(
        "sink",
        [
            "subprocess",
            "os.system",
            "exec",
            "eval",
            "http_request",
            "fetch",
            "api_call",
            "browser_type",
            "browser_click",
            "cronjob",
            "skill_manage",
        ],
    )
    def test_new_sinks_present(self, sink):
        assert sink in CRITICAL_SINKS


# ---------------------------------------------------------------------------
# Fix 11: Default FlowAnalyzer decision is ASK_USER
# ---------------------------------------------------------------------------


class TestDefaultDecision:
    def test_default_is_ask_user(self):
        analyzer = FlowAnalyzer()
        src = frozenset(
            {
                Source(
                    label=TaintLabel.AGENT,
                    trust_level=TrustLevel.UNTRUSTED,
                    origin="test",
                )
            }
        )
        val = TaintedValue(value="x", sources=src)
        result = analyzer.check(val, "some_unknown_tool_xyz")
        assert result == FlowDecision.ASK_USER


# ---------------------------------------------------------------------------
# Fix 12: fnmatch glob patterns in FlowRule
# ---------------------------------------------------------------------------


class TestFnmatchGlob:
    def test_glob_pattern_matching(self):
        rule = FlowRule(
            source_labels=frozenset({TaintLabel.WEB_CONTENT}),
            target_tools=frozenset({"memory_*"}),
            decision=FlowDecision.DENY,
        )
        assert rule.matches_tool("memory_write")
        assert rule.matches_tool("memory_delete")
        assert not rule.matches_tool("send_message")

    def test_question_mark_glob(self):
        rule = FlowRule(
            source_labels=frozenset({TaintLabel.WEB_CONTENT}),
            target_tools=frozenset({"file_?rite"}),
            decision=FlowDecision.DENY,
        )
        assert rule.matches_tool("file_write")
        assert not rule.matches_tool("file_read")

    def test_exact_match_still_works(self):
        rule = FlowRule(
            source_labels=frozenset({TaintLabel.WEB_CONTENT}),
            target_tools=frozenset({"terminal"}),
            decision=FlowDecision.DENY,
        )
        assert rule.matches_tool("terminal")
        assert not rule.matches_tool("terminal2")


# ---------------------------------------------------------------------------
# F-string laundering — known limitation + taint_aware_fstring mitigation
# ---------------------------------------------------------------------------


class TestFStringLaundering:
    """Document and test the f-string taint laundering limitation.

    CPython's BUILD_STRING opcode (used when an f-string has surrounding
    literal text) assembles parts via an internal C-level join that bypasses
    Python's __add__ / __format__ overrides.  The result is a plain str, not
    a TaintedStr.  Only a bare `f"{tainted}"` (single expression, no adjacent
    literals) calls __format__ and preserves the TaintedStr type.
    """

    def test_bare_fstring_preserves_type(self):
        """f"{tainted}" with NO surrounding text preserves TaintedStr type."""
        t = TaintedStr("evil", sources=_src())
        result = f"{t}"
        assert isinstance(result, TaintedStr), "Bare f-string should preserve TaintedStr (calls __format__ only)"
        assert _has_label(result.sources)

    def test_fstring_with_surrounding_literals_launders_taint(self):
        """f"prefix {tainted} suffix" — taint IS laundered (CPython BUILD_STRING).

        This test documents the known limitation.  The result is a plain str
        and taint is lost.  Use taint_aware_fstring() instead.
        """
        t = TaintedStr("evil", sources=_src())
        result = f"Context: {t}"
        # BUILD_STRING produces a plain str — TaintedStr type is lost
        assert not isinstance(result, TaintedStr), (
            "f-string with surrounding literals should launder taint (this test documents the known limitation)"
        )

    def test_taint_aware_fstring_positional(self):
        """taint_aware_fstring() preserves taint with positional placeholders."""
        t = TaintedStr("evil", sources=_src())
        result = taint_aware_fstring("Context: {}", t)
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources)
        assert result.value == "Context: evil"

    def test_taint_aware_fstring_keyword(self):
        """taint_aware_fstring() preserves taint with keyword placeholders."""
        t = TaintedStr("injected", sources=_src(TaintLabel.MCP))
        result = taint_aware_fstring("Hello {name}!", name=t)
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources, TaintLabel.MCP)
        assert result.value == "Hello injected!"

    def test_taint_aware_fstring_merges_multiple_sources(self):
        """taint_aware_fstring() merges taint from multiple tainted args."""
        t1 = TaintedStr("web", sources=_src(TaintLabel.WEB_CONTENT))
        t2 = TaintedStr("mcp", sources=_src(TaintLabel.MCP))
        result = taint_aware_fstring("{} and {}", t1, t2)
        assert isinstance(result, TaintedStr)
        assert _has_label(result.sources, TaintLabel.WEB_CONTENT)
        assert _has_label(result.sources, TaintLabel.MCP)
        assert result.value == "web and mcp"

    def test_taint_aware_fstring_plain_args_pass_through(self):
        """taint_aware_fstring() with plain (non-tainted) args produces clean output."""
        result = taint_aware_fstring("Hello {}!", "world")
        assert result.value == "Hello world!"
        assert not result.sources  # no taint sources
