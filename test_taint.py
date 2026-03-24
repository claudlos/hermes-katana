#!/usr/bin/env python3
"""Integration tests for the taint tracking engine."""
import sys
sys.path.insert(0, "src")

from hermes_katana.taint import (
    TaintLabel, TrustLevel, Source, Reader,
    TaintedValue, TaintedStr, TaintedList, TaintedDict,
    TaintTracker, TaintPolicy,
    FlowDecision, FlowRule, FlowAnalysis, FlowAnalyzer,
    unwrap, collect_sources, default_trust_for,
)

def test_imports():
    print("[OK] All imports successful")

def test_labels():
    assert len(TaintLabel) == 9
    assert TaintLabel.USER.name == "USER"
    assert TrustLevel.TRUSTED.name == "TRUSTED"
    assert default_trust_for(TaintLabel.USER) == TrustLevel.TRUSTED
    assert default_trust_for(TaintLabel.WEB_CONTENT) == TrustLevel.UNTRUSTED
    print("[OK] Labels and enums")

def test_source_factories():
    s = Source.user()
    assert s.label == TaintLabel.USER
    assert s.trust_level == TrustLevel.TRUSTED

    s = Source.web("https://evil.com")
    assert s.label == TaintLabel.WEB_CONTENT
    assert s.trust_level == TrustLevel.UNTRUSTED
    assert s.origin == "https://evil.com"

    s = Source.tool("search")
    assert s.label == TaintLabel.TOOL_OUTPUT
    assert s.trust_level == TrustLevel.CONDITIONAL

    s = Source.mcp("server1")
    assert s.label == TaintLabel.MCP
    assert s.trust_level == TrustLevel.UNTRUSTED

    s = Source.file("/etc/passwd")
    assert s.label == TaintLabel.FILE_CONTENT

    s = Source.memory("key1")
    assert s.label == TaintLabel.MEMORY

    s = Source.agent("claude")
    assert s.label == TaintLabel.AGENT

    s = Source.unknown()
    assert s.label == TaintLabel.UNKNOWN
    assert s.trust_level == TrustLevel.UNTRUSTED

    print("[OK] Source factories")

def test_reader():
    r = Reader.unrestricted("admin")
    assert r.may_access(frozenset({TaintLabel.WEB_CONTENT, TaintLabel.MCP}))

    r = Reader.trusted_only("terminal")
    assert r.may_access(frozenset({TaintLabel.USER}))
    assert not r.may_access(frozenset({TaintLabel.WEB_CONTENT}))
    print("[OK] Reader access control")

def test_tainted_value():
    src = Source.web("https://x.com")
    tv = TaintedValue(value=42, sources=frozenset({src}))
    assert tv.value == 42
    assert tv.is_untrusted()
    assert not tv.is_trusted()
    assert tv.has_label(TaintLabel.WEB_CONTENT)
    assert not tv.has_label(TaintLabel.USER)

    # Derive
    tv2 = tv.derive(84)
    assert tv2.value == 84
    assert tv2.has_label(TaintLabel.WEB_CONTENT)
    assert tv in tv2.dependencies

    # Merge metadata
    src2 = Source.user()
    tv3 = TaintedValue(value="hello", sources=frozenset({src2}))
    tv4 = tv.merge_metadata(tv3)
    assert tv4.value == 42  # unchanged
    assert tv4.has_label(TaintLabel.WEB_CONTENT)
    assert tv4.has_label(TaintLabel.USER)

    # Unwrap
    assert tv.unwrap() == 42

    print("[OK] TaintedValue core")

def test_tainted_str():
    src1 = Source.user()
    src2 = Source.web("https://evil.com")

    ts1 = TaintedStr("Hello ", sources=frozenset({src1}))
    ts2 = TaintedStr("World", sources=frozenset({src2}))

    # Concatenation
    ts3 = ts1 + ts2
    assert ts3.value == "Hello World"
    assert ts3.has_label(TaintLabel.USER)
    assert ts3.has_label(TaintLabel.WEB_CONTENT)

    # radd with plain str
    ts4 = "Prefix: " + ts1
    assert ts4.value == "Prefix: Hello "
    assert ts4.has_label(TaintLabel.USER)

    # Slicing
    ts5 = ts3[0:5]
    assert ts5.value == "Hello"

    ts6 = ts3[6:]
    assert ts6.value == "World"

    # Character access
    ch = ts3[0]
    assert ch.value == "H"

    # String methods
    assert ts1.upper().value == "HELLO "
    assert ts1.lower().value == "hello "
    assert TaintedStr("  hi  ", sources=frozenset({src1})).strip().value == "hi"

    # Split
    parts = TaintedStr("a,b,c", sources=frozenset({src1})).split(",")
    assert len(parts) == 3
    assert parts[0].value == "a"
    assert parts[1].value == "b"

    # Replace
    ts_r = TaintedStr("hello world", sources=frozenset({src1})).replace("world", "earth")
    assert ts_r.value == "hello earth"

    # Join
    sep = TaintedStr(", ", sources=frozenset({src1}))
    joined = sep.join([
        TaintedStr("a", sources=frozenset({src1})),
        TaintedStr("b", sources=frozenset({src2})),
    ])
    assert joined.value == "a, b"
    assert joined.has_label(TaintLabel.USER)
    assert joined.has_label(TaintLabel.WEB_CONTENT)

    # Format
    template = TaintedStr("Hello {}", sources=frozenset({src1}))
    formatted = template.format(TaintedStr("evil", sources=frozenset({src2})))
    assert formatted.value == "Hello evil"
    assert formatted.has_label(TaintLabel.WEB_CONTENT)

    # Contains
    assert "Hello" in ts1
    assert "xyz" not in ts1

    # len, bool, str
    assert len(ts1) == 6
    assert bool(ts1)
    assert str(ts1) == "Hello "

    print("[OK] TaintedStr operations")

def test_tainted_list():
    src = Source.tool("search")
    tl = TaintedList([1, 2, 3], sources=frozenset({src}))
    assert len(tl) == 3
    assert tl[0] == 1
    tl.append_tainted(4, frozenset({Source.web("x")}))
    assert len(tl) == 4
    all_src = tl.all_sources()
    assert any(s.label == TaintLabel.WEB_CONTENT for s in all_src)
    print("[OK] TaintedList")

def test_tainted_dict():
    src = Source.mcp("server")
    td = TaintedDict({"a": 1, "b": 2}, sources=frozenset({src}))
    assert td["a"] == 1
    assert len(td) == 2
    assert "a" in td
    td["c"] = 3
    td.set_key_sources("c", frozenset({Source.user()}))
    assert any(s.label == TaintLabel.USER for s in td.get_key_sources("c"))
    del td["b"]
    assert len(td) == 2
    print("[OK] TaintedDict")

def test_unwrap():
    src = Source.user()
    tv = TaintedValue(42, sources=frozenset({src}))
    assert unwrap(tv) == 42

    tl = TaintedList([1, 2], sources=frozenset({src}))
    assert unwrap(tl) == [1, 2]

    td = TaintedDict({"k": "v"}, sources=frozenset({src}))
    assert unwrap(td) == {"k": "v"}

    # Nested
    assert unwrap([tv, {"x": tv}]) == [42, {"x": 42}]
    print("[OK] unwrap")

def test_collect_sources():
    s1 = Source.user()
    s2 = Source.web("x")
    tv1 = TaintedValue(1, sources=frozenset({s1}))
    tv2 = TaintedValue(2, sources=frozenset({s2}))
    srcs = collect_sources([tv1, tv2])
    assert s1 in srcs
    assert s2 in srcs
    print("[OK] collect_sources")

def test_flow_analyzer():
    analyzer = FlowAnalyzer()

    # Untrusted web -> terminal = DENY
    web_val = TaintedValue("rm -rf /", sources=frozenset({Source.web("evil.com")}))
    assert analyzer.check(web_val, "terminal") == FlowDecision.DENY

    # Untrusted MCP -> send_message = DENY
    mcp_val = TaintedValue("spam", sources=frozenset({Source.mcp("rogue")}))
    assert analyzer.check(mcp_val, "send_message") == FlowDecision.DENY

    # Trusted user -> terminal = ALLOW
    user_val = TaintedValue("echo hi", sources=frozenset({Source.user()}))
    assert analyzer.check(user_val, "terminal") == FlowDecision.ALLOW

    # Tool output -> terminal = ASK_USER
    tool_val = TaintedValue("data", sources=frozenset({Source.tool("search")}))
    assert analyzer.check(tool_val, "terminal") == FlowDecision.ASK_USER

    # Agent -> terminal = QUARANTINE
    agent_val = TaintedValue("cmd", sources=frozenset({Source.agent()}))
    assert analyzer.check(agent_val, "terminal") == FlowDecision.QUARANTINE

    # Anything -> non-critical tool = ALLOW
    assert analyzer.check(web_val, "read_file") == FlowDecision.ALLOW

    # Mixed: web + user -> terminal = DENY (worst wins)
    mixed = TaintedValue("x", sources=frozenset({Source.web("x"), Source.user()}))
    result = analyzer.check(mixed, "terminal")
    assert result == FlowDecision.DENY

    print("[OK] Flow analyzer rules")

def test_flow_analysis_detail():
    analyzer = FlowAnalyzer()
    web_val = TaintedValue("payload", sources=frozenset({Source.web("evil.com")}))
    analysis = analyzer.analyze(web_val, "terminal")
    assert analysis.decision == FlowDecision.DENY
    assert analysis.is_blocked()
    assert len(analysis.matched_rules) > 0
    assert TaintLabel.WEB_CONTENT in analysis.labels_present
    assert analysis.tool_name == "terminal"
    assert len(analysis.reasoning) > 0
    print("[OK] Flow analysis detail")

def test_strict_mode():
    analyzer = FlowAnalyzer(strict_mode=True)
    # No matching rule for a random label/tool combo -> ASK_USER in strict
    val = TaintedValue(1, sources=frozenset({Source.system()}))
    # System -> non-critical: rule 3 matches (ALLOW)
    assert analyzer.check(val, "some_random_tool") == FlowDecision.ALLOW
    print("[OK] Strict mode")

def test_tracker_singleton():
    TaintTracker.reset_instance()
    t1 = TaintTracker.get_instance()
    t2 = TaintTracker.get_instance()
    assert t1 is t2
    TaintTracker.reset_instance()
    print("[OK] Tracker singleton")

def test_tracker_register_and_flow():
    TaintTracker.reset_instance()
    tracker = TaintTracker.get_instance()

    web_data = tracker.register("evil payload", Source.web("https://evil.com"))
    assert isinstance(web_data, TaintedStr)
    assert web_data.value == "evil payload"
    assert web_data.is_untrusted()

    decision = tracker.check_flow(web_data, "terminal")
    assert decision == FlowDecision.DENY

    user_data = tracker.register("echo hello", Source.user())
    decision = tracker.check_flow(user_data, "terminal")
    assert decision == FlowDecision.ALLOW

    assert tracker.stats.values_registered == 2
    assert tracker.stats.flow_checks == 2
    assert tracker.stats.flow_denied == 1
    assert tracker.stats.flow_allowed == 1

    TaintTracker.reset_instance()
    print("[OK] Tracker register and flow")

def test_tracker_propagate():
    TaintTracker.reset_instance()
    tracker = TaintTracker.get_instance()

    v1 = tracker.register("web", Source.web("x"))
    v2 = tracker.register("user", Source.user())
    combined = tracker.propagate("web+user", v1, v2)

    assert combined.has_label(TaintLabel.WEB_CONTENT)
    assert combined.has_label(TaintLabel.USER)
    assert tracker.check_flow(combined, "terminal") == FlowDecision.DENY

    assert tracker.stats.values_propagated == 1

    TaintTracker.reset_instance()
    print("[OK] Tracker propagation")

def test_tracker_taint_chain():
    TaintTracker.reset_instance()
    tracker = TaintTracker.get_instance()

    v1 = tracker.register("a", Source.web("site1"))
    v2 = tracker.register("b", Source.tool("search"))
    v3 = tracker.propagate("ab", v1, v2)
    v4 = tracker.propagate("abc", v3, tracker.register("c", Source.user()))

    chain = tracker.get_taint_chain(v4)
    labels_in_chain = {s.label for s in chain}
    assert TaintLabel.WEB_CONTENT in labels_in_chain
    assert TaintLabel.TOOL_OUTPUT in labels_in_chain
    assert TaintLabel.USER in labels_in_chain

    TaintTracker.reset_instance()
    print("[OK] Taint chain reconstruction")

def test_tracker_scoped():
    TaintTracker.reset_instance()
    with TaintTracker.scoped() as tracker:
        tv = tracker.register("temp", Source.unknown())
        assert tracker.tracked_count == 1
    # After scope, tracker is cleared
    print("[OK] Scoped tracker")

def test_tracker_check_args():
    TaintTracker.reset_instance()
    tracker = TaintTracker.get_instance()

    safe = tracker.register("ok", Source.user())
    unsafe = tracker.register("bad", Source.web("evil"))

    assert tracker.check_args_flow("terminal", cmd=safe) == FlowDecision.ALLOW
    assert tracker.check_args_flow("terminal", cmd=unsafe) == FlowDecision.DENY
    assert tracker.check_args_flow("terminal", cmd=safe, data=unsafe) == FlowDecision.DENY

    TaintTracker.reset_instance()
    print("[OK] Tracker check_args_flow")

def test_custom_rules():
    custom_rule = FlowRule(
        source_labels=frozenset({TaintLabel.AGENT}),
        target_tools=frozenset({"custom_tool"}),
        decision=FlowDecision.DENY,
        reason="Custom deny rule",
        priority=200,
    )
    analyzer = FlowAnalyzer(rules=[custom_rule])
    val = TaintedValue("x", sources=frozenset({Source.agent()}))
    assert analyzer.check(val, "custom_tool") == FlowDecision.DENY
    assert analyzer.check(val, "other_tool") == FlowDecision.ALLOW  # default
    print("[OK] Custom flow rules")

def test_taint_policy_alias():
    assert TaintPolicy is FlowAnalyzer
    policy = TaintPolicy()
    assert isinstance(policy, FlowAnalyzer)
    print("[OK] TaintPolicy alias")

def main():
    tests = [
        test_imports,
        test_labels,
        test_source_factories,
        test_reader,
        test_tainted_value,
        test_tainted_str,
        test_tainted_list,
        test_tainted_dict,
        test_unwrap,
        test_collect_sources,
        test_flow_analyzer,
        test_flow_analysis_detail,
        test_strict_mode,
        test_tracker_singleton,
        test_tracker_register_and_flow,
        test_tracker_propagate,
        test_tracker_taint_chain,
        test_tracker_scoped,
        test_tracker_check_args,
        test_custom_rules,
        test_taint_policy_alias,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")

if __name__ == "__main__":
    main()
