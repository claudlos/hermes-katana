"""Unit tests for the structural meta-scanner and middleware integration."""

from __future__ import annotations

import json

from hermes_katana.scanner.structural import (
    ContentType,
    StructuralFlag,
    StructuralReport,
    detect_content_type,
    detect_structural,
)


# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------


class TestDetectContentType:
    """Tests for content type classification heuristics."""

    def test_html_basic(self):
        assert detect_content_type("<div>hello</div>") == ContentType.HTML

    def test_html_doctype(self):
        assert detect_content_type("<!DOCTYPE html><html><body></body></html>") == ContentType.HTML

    def test_html_script_tag(self):
        assert detect_content_type('<script>alert("xss")</script>') == ContentType.HTML

    def test_html_case_insensitive(self):
        assert detect_content_type("<DIV>test</DIV>") == ContentType.HTML

    def test_pdf_magic(self):
        assert detect_content_type("%PDF-1.7 some content") == ContentType.PDF

    def test_pdf_old_version(self):
        assert detect_content_type("%PDF-1.4 ...") == ContentType.PDF

    def test_markdown_headers_and_links(self):
        text = "# Title\n\nSome text with [a link](http://example.com)\n"
        assert detect_content_type(text) == ContentType.MARKDOWN

    def test_markdown_needs_two_indicators(self):
        # Single indicator should NOT match as markdown
        assert detect_content_type("# Just a heading") == ContentType.PLAIN

    def test_markdown_bold_and_list(self):
        text = "**bold text** and\n- list item\n- another item\n"
        assert detect_content_type(text) == ContentType.MARKDOWN

    def test_plain_text(self):
        assert detect_content_type("Hello, this is just normal text.") == ContentType.PLAIN

    def test_empty_string(self):
        assert detect_content_type("") == ContentType.PLAIN

    def test_whitespace_only(self):
        assert detect_content_type("   \n\t  ") == ContentType.PLAIN

    def test_pdf_takes_priority_over_html(self):
        # PDF magic at start should win even if HTML tags follow
        assert detect_content_type("%PDF-1.7 <div>hi</div>") == ContentType.PDF


# ---------------------------------------------------------------------------
# StructuralReport
# ---------------------------------------------------------------------------


class TestStructuralReport:
    """Tests for the risk report data model."""

    def test_default_report(self):
        r = StructuralReport()
        assert r.content_type == "plain"
        assert r.flags == []
        assert r.structural_score == 0.0
        assert r.pattern_matches == 0
        assert r.bloom_hits == 0

    def test_to_dict(self):
        flag = StructuralFlag(
            type="hidden_text",
            location="css_display_none",
            excerpt="secret stuff",
            severity="high",
        )
        r = StructuralReport(
            content_type="html",
            flags=[flag],
            structural_score=0.82,
            pattern_matches=3,
            bloom_hits=2,
        )
        d = r.to_dict()
        assert d["content_type"] == "html"
        assert len(d["flags"]) == 1
        assert d["flags"][0]["type"] == "hidden_text"
        assert d["structural_score"] == 0.82
        assert d["pattern_matches"] == 3
        assert d["bloom_hits"] == 2

    def test_to_json_is_valid(self):
        flag = StructuralFlag(
            type="layer_mismatch",
            location="page_2",
            excerpt="hidden layer",
            severity="medium",
        )
        r = StructuralReport(
            content_type="pdf",
            flags=[flag],
            structural_score=0.5,
            pattern_matches=1,
            bloom_hits=0,
        )
        raw = r.to_json()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert parsed["content_type"] == "pdf"
        assert isinstance(parsed["flags"], list)

    def test_to_json_roundtrip_keys(self):
        r = StructuralReport()
        parsed = json.loads(r.to_json())
        expected_keys = {"content_type", "flags", "structural_score", "pattern_matches", "bloom_hits"}
        assert set(parsed.keys()) == expected_keys


# ---------------------------------------------------------------------------
# detect_structural (end-to-end, no sub-scanners available)
# ---------------------------------------------------------------------------


class TestDetectStructural:
    """Tests for the main entry point when sub-scanners are absent."""

    def test_plain_text_fast_path(self):
        report = detect_structural("Just some normal text, nothing special here.")
        assert report.content_type == "plain"
        assert report.flags == []
        assert report.structural_score == 0.0

    def test_html_input_returns_html_type(self):
        report = detect_structural("<div style='display:none'>hidden</div>")
        assert report.content_type == "html"
        # Without html_diff installed, no flags expected
        # Score should be 0 or whatever bloom produces
        assert isinstance(report.structural_score, float)

    def test_pdf_input_returns_pdf_type(self):
        report = detect_structural("%PDF-1.7 fake pdf content here")
        assert report.content_type == "pdf"

    def test_markdown_input_returns_markdown_type(self):
        text = "# Heading\n\n**bold** and [link](http://x.com)\n- item\n"
        report = detect_structural(text)
        assert report.content_type == "markdown"

    def test_empty_input(self):
        report = detect_structural("")
        assert report.content_type == "plain"
        assert report.structural_score == 0.0
        assert report.flags == []

    def test_report_is_json_serializable(self):
        report = detect_structural("<html><body>test</body></html>")
        raw = report.to_json()
        parsed = json.loads(raw)
        assert parsed["content_type"] == "html"


# ---------------------------------------------------------------------------
# Middleware integration — structural middleware in the chain
# ---------------------------------------------------------------------------


class TestStructuralMiddlewareIntegration:
    """Test that KatanaStructuralMiddleware wires into the chain correctly."""

    def test_structural_middleware_in_default_chain(self):
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain(
            {
                "taint.enabled": False,
                "scan.enabled": False,
                "policy.enabled": False,
                "audit.enabled": False,
                "structural.enabled": True,
            }
        )
        names = [m.name for m in chain.list_middleware()]
        assert "katana.structural" in names

    def test_structural_middleware_order(self):
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain()
        names = [m.name for m in chain.list_middleware()]
        # structural (pri=70) should come after scan (pri=80), before policy (pri=60)
        scan_idx = names.index("katana.scan")
        struct_idx = names.index("katana.structural")
        policy_idx = names.index("katana.policy")
        assert scan_idx < struct_idx < policy_idx

    def test_structural_middleware_allows_plain_text(self):
        from hermes_katana.middleware.chain import CallContext, DispatchDecision
        from hermes_katana.middleware.integration import KatanaStructuralMiddleware

        mw = KatanaStructuralMiddleware()
        ctx = CallContext(tool_name="terminal", args={"command": "ls -la"})
        decision = mw.pre_dispatch(ctx)
        assert decision == DispatchDecision.ALLOW

    def test_structural_middleware_disabled(self):
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain({"structural.enabled": False})
        mw = next(m for m in chain.list_middleware() if m.name == "katana.structural")
        assert not mw.enabled

    def test_existing_chain_not_broken(self):
        """Verify the full default chain still builds with all 7 middleware."""
        from hermes_katana.middleware.integration import create_default_chain

        chain = create_default_chain()
        assert len(chain) == 12
        names = [m.name for m in chain.list_middleware()]
        assert "katana.taint" in names
        assert "katana.scabbard" in names
        assert "katana.sentinel" in names
        assert "katana.scan" in names
        assert "katana.structural" in names
        assert "katana.policy" in names
        assert "katana.audit" in names
