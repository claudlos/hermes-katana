"""Tests for HermesKatana taint tracking system."""

from __future__ import annotations

import pytest

from hermes_katana.taint.labels import (
    Source,
    TaintLabel,
    TrustLevel,
    default_trust_for,
)
from hermes_katana.taint.value import (
    TaintedDict,
    TaintedList,
    TaintedStr,
    TaintedValue,
    collect_sources,
    taint_aware_json_dumps,
    unwrap,
)


# ======================================================================
# TaintLabel enum
# ======================================================================


class TestTaintLabel:
    def test_all_labels_exist(self):
        # Core labels
        required = {"USER", "SYSTEM", "TOOL_OUTPUT", "WEB_CONTENT", "FILE_CONTENT", "MEMORY", "MCP", "AGENT", "UNKNOWN"}
        # Extended labels added in sweep (research docs 01/03)
        extended = {
            "MCP_TOOL_DESCRIPTION",
            "MCP_TOOL_RESULT",
            "MCP_RESOURCE",
            "MCP_PROMPT",
            "AGENT_DELEGATED",
            "CROSS_SESSION",
        }
        actual = {label.name for label in TaintLabel}
        assert required.issubset(actual), f"Missing core labels: {required - actual}"
        assert extended.issubset(actual), f"Missing extended labels: {extended - actual}"

    def test_labels_are_unique(self):
        values = [label.value for label in TaintLabel]
        assert len(values) == len(set(values))

    def test_default_trust_user(self):
        assert default_trust_for(TaintLabel.USER) == TrustLevel.TRUSTED

    def test_default_trust_web(self):
        assert default_trust_for(TaintLabel.WEB_CONTENT) == TrustLevel.UNTRUSTED

    def test_default_trust_mcp(self):
        assert default_trust_for(TaintLabel.MCP) == TrustLevel.UNTRUSTED

    def test_default_trust_tool_output(self):
        assert default_trust_for(TaintLabel.TOOL_OUTPUT) == TrustLevel.CONDITIONAL

    def test_default_trust_mcp_tool_description(self):
        """MCP tool descriptions are the highest-risk MCP label — must be UNTRUSTED."""
        assert default_trust_for(TaintLabel.MCP_TOOL_DESCRIPTION) == TrustLevel.UNTRUSTED

    def test_default_trust_agent_delegated(self):
        assert default_trust_for(TaintLabel.AGENT_DELEGATED) == TrustLevel.CONDITIONAL

    def test_default_trust_cross_session(self):
        assert default_trust_for(TaintLabel.CROSS_SESSION) == TrustLevel.CONDITIONAL


# ======================================================================
# Source factory methods
# ======================================================================


class TestSource:
    def test_user_factory(self):
        s = Source.user("cli_input")
        assert s.label == TaintLabel.USER
        assert s.origin == "cli_input"
        assert s.trust_level == TrustLevel.TRUSTED

    def test_web_factory(self):
        s = Source.web("https://example.com")
        assert s.label == TaintLabel.WEB_CONTENT
        assert s.origin == "https://example.com"
        assert s.trust_level == TrustLevel.UNTRUSTED

    def test_tool_factory(self):
        s = Source.tool("browser_navigate")
        assert s.label == TaintLabel.TOOL_OUTPUT
        assert s.origin == "browser_navigate"
        assert s.trust_level == TrustLevel.CONDITIONAL

    def test_mcp_factory(self):
        s = Source.mcp("remote_server")
        assert s.label == TaintLabel.MCP
        assert s.origin == "remote_server"
        assert s.trust_level == TrustLevel.UNTRUSTED

    def test_file_factory(self):
        s = Source.file("/etc/passwd")
        assert s.label == TaintLabel.FILE_CONTENT
        assert s.origin == "/etc/passwd"
        assert s.trust_level == TrustLevel.CONDITIONAL

    def test_memory_factory(self):
        s = Source.memory("user_prefs")
        assert s.label == TaintLabel.MEMORY
        assert s.origin == "user_prefs"
        assert s.trust_level == TrustLevel.CONDITIONAL

    def test_agent_factory(self):
        s = Source.agent("claude")
        assert s.label == TaintLabel.AGENT
        assert s.origin == "claude"
        assert s.trust_level == TrustLevel.CONDITIONAL

    def test_unknown_factory(self):
        s = Source.unknown("mystery")
        assert s.label == TaintLabel.UNKNOWN
        assert s.origin == "mystery"
        assert s.trust_level == TrustLevel.UNTRUSTED

    def test_factory_with_metadata(self):
        s = Source.web("https://example.com", status="200", content_type="text/html")
        assert s.metadata == {"status": "200", "content_type": "text/html"}

    def test_source_has_timestamp(self):
        s = Source.user()
        assert s.timestamp > 0

    def test_source_is_frozen(self):
        s = Source.user()
        with pytest.raises(AttributeError):
            s.label = TaintLabel.WEB_CONTENT  # type: ignore


# ======================================================================
# TaintedValue
# ======================================================================


class TestTaintedValue:
    def test_creation(self, user_source):
        tv = TaintedValue(value=42, sources=frozenset({user_source}))
        assert tv.value == 42
        assert tv.unwrap() == 42
        assert TaintLabel.USER in tv.labels

    def test_is_trusted_all_trusted(self, user_source):
        tv = TaintedValue(value="data", sources=frozenset({user_source}))
        assert tv.is_trusted() is True

    def test_is_trusted_no_sources(self):
        tv = TaintedValue(value="data")
        assert tv.is_trusted() is False

    def test_is_trusted_untrusted_source(self, web_source):
        tv = TaintedValue(value="data", sources=frozenset({web_source}))
        assert tv.is_trusted() is False
        assert tv.is_untrusted() is True

    def test_is_public_no_readers(self, user_source):
        tv = TaintedValue(value="data", sources=frozenset({user_source}))
        assert tv.is_public() is True

    def test_is_public_with_readers(self, user_source):
        from hermes_katana.taint.labels import Reader

        r = Reader.trusted_only("terminal")
        tv = TaintedValue(value="data", sources=frozenset({user_source}), readers=frozenset({r}))
        assert tv.is_public() is False

    def test_derive(self, user_source, web_source):
        tv1 = TaintedValue(value="a", sources=frozenset({user_source}))
        tv2 = TaintedValue(value="b", sources=frozenset({web_source}))
        derived = tv1.derive("ab", tv2)
        assert derived.value == "ab"
        assert TaintLabel.USER in derived.labels
        assert TaintLabel.WEB_CONTENT in derived.labels
        assert len(derived.dependencies) == 2

    def test_merge_metadata(self, user_source, web_source):
        tv1 = TaintedValue(value="hello", sources=frozenset({user_source}))
        tv2 = TaintedValue(value="world", sources=frozenset({web_source}))
        merged = tv1.merge_metadata(tv2)
        assert merged.value == "hello"  # value unchanged
        assert TaintLabel.USER in merged.labels
        assert TaintLabel.WEB_CONTENT in merged.labels

    def test_has_label(self, user_source):
        tv = TaintedValue(value="x", sources=frozenset({user_source}))
        assert tv.has_label(TaintLabel.USER) is True
        assert tv.has_label(TaintLabel.WEB_CONTENT) is False

    def test_bool_truthy(self, user_source):
        tv = TaintedValue(value="hello", sources=frozenset({user_source}))
        assert bool(tv) is True

    def test_bool_falsy(self, user_source):
        tv = TaintedValue(value="", sources=frozenset({user_source}))
        assert bool(tv) is False

    def test_equality(self, user_source):
        tv1 = TaintedValue(value="hello", sources=frozenset({user_source}))
        tv2 = TaintedValue(value="hello", sources=frozenset({user_source}))
        assert tv1 == tv2
        assert tv1 == "hello"


# ======================================================================
# TaintedStr — character-level taint
# ======================================================================


class TestTaintedStr:
    def test_basic_creation(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        assert ts.value == "hello"
        assert str(ts) == "hello"
        assert len(ts) == 5

    def test_concat_two_tainted(self, user_source, web_source):
        ts1 = TaintedStr("hello", sources=frozenset({user_source}))
        ts2 = TaintedStr(" world", sources=frozenset({web_source}))
        result = ts1 + ts2
        assert result.value == "hello world"
        assert TaintLabel.USER in result.labels
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_concat_tainted_plus_plain(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        result = ts + " world"
        assert result.value == "hello world"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_radd_plain_plus_tainted(self, web_source):
        ts = TaintedStr("world", sources=frozenset({web_source}))
        result = "hello " + ts
        assert result.value == "hello world"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_slicing(self, web_source):
        ts = TaintedStr("hello world", sources=frozenset({web_source}))
        sliced = ts[0:5]
        assert sliced.value == "hello"
        assert isinstance(sliced, TaintedStr)
        assert TaintLabel.WEB_CONTENT in sliced.labels

    def test_indexing(self, web_source):
        ts = TaintedStr("abc", sources=frozenset({web_source}))
        ch = ts[0]
        assert ch.value == "a"
        assert isinstance(ch, TaintedStr)

    def test_upper(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        result = ts.upper()
        assert result.value == "HELLO"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_lower(self, web_source):
        ts = TaintedStr("HELLO", sources=frozenset({web_source}))
        result = ts.lower()
        assert result.value == "hello"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_strip(self, web_source):
        ts = TaintedStr("  hello  ", sources=frozenset({web_source}))
        result = ts.strip()
        assert result.value == "hello"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_split(self, web_source):
        ts = TaintedStr("hello world foo", sources=frozenset({web_source}))
        parts = ts.split(" ")
        assert len(parts) == 3
        assert all(isinstance(p, TaintedStr) for p in parts)
        assert parts[0].value == "hello"
        assert parts[1].value == "world"

    def test_replace_propagates(self, web_source):
        ts = TaintedStr("hello world", sources=frozenset({web_source}))
        result = ts.replace("world", "earth")
        assert result.value == "hello earth"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_contains(self, web_source):
        ts = TaintedStr("hello world", sources=frozenset({web_source}))
        assert "hello" in ts
        assert "xyz" not in ts

    def test_iter(self, web_source):
        ts = TaintedStr("ab", sources=frozenset({web_source}))
        chars = list(ts)
        assert len(chars) == 2
        assert chars[0].value == "a"
        assert chars[1].value == "b"


# ======================================================================
# TaintedList — per-item taint
# ======================================================================


class TestTaintedList:
    def test_creation(self, web_source):
        tl = TaintedList(value=[1, 2, 3], sources=frozenset({web_source}))
        assert len(tl) == 3
        assert tl[0] == 1

    def test_per_item_taint(self, web_source, user_source):
        tl = TaintedList(value=["a", "b"], sources=frozenset({web_source}))
        tl.set_item_sources(0, frozenset({user_source}))
        assert user_source in tl.get_item_sources(0)
        assert web_source in tl.get_item_sources(1)  # falls back to container sources

    def test_append_tainted(self, web_source, user_source):
        tl = TaintedList(value=[], sources=frozenset({web_source}))
        tl.append_tainted("new_item", frozenset({user_source}))
        assert len(tl) == 1
        assert user_source in tl.get_item_sources(0)

    def test_all_sources(self, web_source, user_source):
        tl = TaintedList(value=[1, 2], sources=frozenset({web_source}))
        tl.set_item_sources(0, frozenset({user_source}))
        all_src = tl.all_sources()
        assert web_source in all_src
        assert user_source in all_src

    def test_setitem(self, web_source):
        tl = TaintedList(value=[1, 2, 3], sources=frozenset({web_source}))
        tl[1] = 99
        assert tl[1] == 99

    def test_delitem(self, web_source):
        tl = TaintedList(value=[1, 2, 3], sources=frozenset({web_source}))
        del tl[0]
        assert len(tl) == 2

    def test_insert(self, web_source):
        tl = TaintedList(value=[1, 3], sources=frozenset({web_source}))
        tl.insert(1, 2)
        assert tl[1] == 2


# ======================================================================
# TaintedDict — per-key taint
# ======================================================================


class TestTaintedDict:
    def test_creation(self, web_source):
        td = TaintedDict(value={"a": 1, "b": 2}, sources=frozenset({web_source}))
        assert len(td) == 2
        assert td["a"] == 1

    def test_per_key_taint(self, web_source, user_source):
        td = TaintedDict(value={"safe": "ok", "unsafe": "evil"}, sources=frozenset({web_source}))
        td.set_key_sources("safe", frozenset({user_source}))
        assert user_source in td.get_key_sources("safe")
        assert web_source in td.get_key_sources("unsafe")

    def test_setitem(self, web_source):
        td = TaintedDict(value={}, sources=frozenset({web_source}))
        td["key"] = "value"
        assert td["key"] == "value"

    def test_delitem(self, web_source):
        td = TaintedDict(value={"a": 1}, sources=frozenset({web_source}))
        del td["a"]
        assert len(td) == 0

    def test_contains(self, web_source):
        td = TaintedDict(value={"a": 1}, sources=frozenset({web_source}))
        assert "a" in td
        assert "b" not in td

    def test_all_sources(self, web_source, user_source):
        td = TaintedDict(value={"a": 1}, sources=frozenset({web_source}))
        td.set_key_sources("a", frozenset({user_source}))
        all_src = td.all_sources()
        assert web_source in all_src
        assert user_source in all_src


# ======================================================================
# unwrap() and collect_sources()
# ======================================================================


class TestUtilities:
    def test_unwrap_tainted_value(self, user_source):
        tv = TaintedValue(value=42, sources=frozenset({user_source}))
        assert unwrap(tv) == 42

    def test_unwrap_tainted_str(self, user_source):
        ts = TaintedStr("hello", sources=frozenset({user_source}))
        assert unwrap(ts) == "hello"

    def test_unwrap_tainted_list(self, user_source):
        tl = TaintedList(value=[1, 2, 3], sources=frozenset({user_source}))
        assert unwrap(tl) == [1, 2, 3]

    def test_unwrap_tainted_dict(self, user_source):
        td = TaintedDict(value={"a": 1}, sources=frozenset({user_source}))
        assert unwrap(td) == {"a": 1}

    def test_unwrap_plain_value(self):
        assert unwrap(42) == 42
        assert unwrap("hello") == "hello"

    def test_collect_sources_single(self, user_source):
        tv = TaintedValue(value="data", sources=frozenset({user_source}))
        srcs = collect_sources(tv)
        assert user_source in srcs

    def test_collect_sources_nested_list(self, user_source, web_source):
        inner = TaintedValue(value="inner", sources=frozenset({web_source}))
        tl = TaintedList(value=[inner], sources=frozenset({user_source}))
        srcs = collect_sources(tl)
        assert user_source in srcs

    def test_collect_sources_nested_dict(self, user_source, web_source):
        inner = TaintedValue(value="inner", sources=frozenset({web_source}))
        td = TaintedDict(value={"k": inner}, sources=frozenset({user_source}))
        srcs = collect_sources(td)
        assert user_source in srcs

    def test_collect_sources_plain(self):
        srcs = collect_sources("plain_string")
        assert len(srcs) == 0

    def test_collect_sources_tainted_str(self, web_source):
        ts = TaintedStr("data", sources=frozenset({web_source}))
        srcs = collect_sources(ts)
        assert web_source in srcs


# ======================================================================
# Taint Laundering Prevention — TaintedStr subclasses str
# ======================================================================


class TestTaintLaunderingPrevention:
    """Verify that TaintedStr IS a str subclass so it passes isinstance
    checks everywhere str is expected, and that method-level operations
    preserve taint.  Note: CPython's C-level str()/repr()/format()/f-strings
    coerce returns to plain str — this is expected and unavoidable."""

    def test_tainted_str_is_str_subclass(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        assert isinstance(ts, str)

    def test_isinstance_str_passes(self, web_source):
        """The key win: TaintedStr passes isinstance(x, str) checks."""
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        assert isinstance(ts, str)
        # Can be used anywhere a str is expected without conversion
        assert ts == "hello"

    def test_direct_use_preserves_taint(self, web_source):
        """Using TaintedStr directly (without str()) keeps taint."""
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        # Direct method calls preserve taint
        result = ts.upper()
        assert isinstance(result, TaintedStr)
        assert TaintLabel.WEB_CONTENT in result.labels
        assert result == "HELLO"

    def test_chained_operations(self, web_source):
        """Chaining TaintedStr methods preserves taint throughout."""
        ts = TaintedStr("hello world", sources=frozenset({web_source}))
        result = ts.upper().replace("WORLD", "EARTH").strip()
        assert isinstance(result, TaintedStr)
        assert TaintLabel.WEB_CONTENT in result.labels
        assert result == "HELLO EARTH"

    def test_concat_preserves_taint(self, web_source, user_source):
        ts1 = TaintedStr("hello", sources=frozenset({web_source}))
        ts2 = TaintedStr(" world", sources=frozenset({user_source}))
        result = ts1 + ts2
        assert isinstance(result, TaintedStr)
        assert TaintLabel.WEB_CONTENT in result.labels
        assert TaintLabel.USER in result.labels

    def test_mod_formatting(self, web_source, user_source):
        template = TaintedStr("Hello %s!", sources=frozenset({web_source}))
        name = TaintedStr("world", sources=frozenset({user_source}))
        result = template % name
        assert isinstance(result, TaintedStr)
        assert result == "Hello world!"
        assert TaintLabel.WEB_CONTENT in result.labels
        assert TaintLabel.USER in result.labels

    def test_json_dumps_preserves_taint(self, web_source):
        ts = TaintedStr("secret", sources=frozenset({web_source}))
        result = taint_aware_json_dumps({"key": ts})
        assert isinstance(result, TaintedStr)
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_value_property_backward_compat(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        assert ts.value == "hello"
        assert isinstance(ts.value, str)

    def test_encode_tainted(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        result = ts.encode_tainted()
        assert isinstance(result, TaintedValue)
        assert result.value == b"hello"
        assert TaintLabel.WEB_CONTENT in result.labels

    def test_unwrap_tainted_str_returns_plain_str(self, web_source):
        ts = TaintedStr("hello", sources=frozenset({web_source}))
        result = unwrap(ts)
        assert result == "hello"
        assert type(result) is str  # NOT TaintedStr

    def test_split_join_roundtrip(self, web_source):
        ts = TaintedStr("a,b,c", sources=frozenset({web_source}))
        sep = TaintedStr(",", sources=frozenset({web_source}))
        parts = ts.split(",")
        assert all(isinstance(p, TaintedStr) for p in parts)
        rejoined = sep.join(parts)
        assert isinstance(rejoined, TaintedStr)
        assert rejoined == "a,b,c"
        assert TaintLabel.WEB_CONTENT in rejoined.labels
