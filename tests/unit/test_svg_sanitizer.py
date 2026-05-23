"""
Unit tests for the SVG sanitizer scanner.

Tests cover:
- SVG <script> tag injection
- <foreignObject> elements
- SVG event handler attributes
- SVG <use> with external xlink/href references
- Embedded executable data URIs
- Base64 data URIs
- javascript: URI scheme
- SVG animate/set targeting security attributes
- Clean inputs (false positive checks)
"""

from hermes_katana.scanner.svg_sanitizer import (
    SVGSanitizerFinding,
    SVGSanitizerSeverity,
    scan_svg,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def categories(text: str) -> set[str]:
    return {f.category for f in scan_svg(text)}


def has_category(text: str, cat: str) -> bool:
    return cat in categories(text)


# ---------------------------------------------------------------------------
# 1. Script injection
# ---------------------------------------------------------------------------


def test_script_tag_basic():
    svg = "<svg><script>alert(1)</script></svg>"
    assert has_category(svg, "script_injection")


def test_script_tag_with_type_attr():
    svg = '<svg><script type="text/javascript">evil()</script></svg>'
    assert has_category(svg, "script_injection")


def test_script_tag_severity_critical():
    svg = "<svg><script>x=1</script></svg>"
    findings = scan_svg(svg)
    script = [f for f in findings if f.category == "script_injection"]
    assert script
    assert script[0].severity == SVGSanitizerSeverity.CRITICAL


def test_script_tag_with_namespace():
    svg = "<svg:script>bad()</svg:script>"
    assert has_category(svg, "script_injection")


# ---------------------------------------------------------------------------
# 2. foreignObject
# ---------------------------------------------------------------------------


def test_foreign_object_detected():
    svg = '<svg><foreignObject width="100" height="100"><body>hi</body></foreignObject></svg>'
    assert has_category(svg, "foreign_object")


def test_foreign_object_severity_high():
    findings = scan_svg("<svg><foreignObject/></svg>")
    fo = [f for f in findings if f.category == "foreign_object"]
    assert fo
    assert fo[0].severity == SVGSanitizerSeverity.HIGH


def test_foreign_object_with_js():
    svg = "<svg><foreignObject><script>bad()</script></foreignObject></svg>"
    cats = categories(svg)
    assert "foreign_object" in cats
    assert "script_injection" in cats


# ---------------------------------------------------------------------------
# 3. Event handlers
# ---------------------------------------------------------------------------


def test_onload_handler():
    svg = '<svg onload="evil()">'
    assert has_category(svg, "event_handler")


def test_onclick_handler():
    svg = '<circle onclick="steal()">'
    assert has_category(svg, "event_handler")


def test_onerror_handler():
    svg = '<image onerror="pwn()">'
    assert has_category(svg, "event_handler")


def test_event_handler_severity_high():
    findings = scan_svg('<svg onmouseover="x()">')
    eh = [f for f in findings if f.category == "event_handler"]
    assert eh
    assert eh[0].severity == SVGSanitizerSeverity.HIGH


def test_multiple_event_handlers():
    svg = '<svg onload="a()" onclick="b()">'
    findings = [f for f in scan_svg(svg) if f.category == "event_handler"]
    assert len(findings) >= 2


# ---------------------------------------------------------------------------
# 4. External use/xlink references
# ---------------------------------------------------------------------------


def test_use_xlink_external():
    svg = '<use xlink:href="https://evil.com/sprite.svg#icon"/>'
    assert has_category(svg, "external_reference")


def test_use_href_external():
    svg = '<use href="http://attacker.com/payload.svg#x"/>'
    assert has_category(svg, "external_reference")


def test_use_local_anchor_not_flagged():
    # Local fragment references (#id) are safe
    svg = '<use xlink:href="#local-icon"/>'
    assert not has_category(svg, "external_reference")


# ---------------------------------------------------------------------------
# 5. Executable data URIs
# ---------------------------------------------------------------------------


def test_data_uri_html():
    svg = '<a href="data:text/html,<script>alert(1)</script>">'
    assert has_category(svg, "data_uri_executable")


def test_data_uri_javascript_mime():
    svg = '<a href="data:application/javascript,evil()">'
    assert has_category(svg, "data_uri_executable")


def test_data_uri_exec_severity_critical():
    findings = scan_svg('<image src="data:text/html,<h1>x</h1>">')
    ev = [f for f in findings if f.category == "data_uri_executable"]
    assert ev
    assert ev[0].severity == SVGSanitizerSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 6. Base64 data URIs
# ---------------------------------------------------------------------------


def test_data_uri_base64_image_flagged():
    svg = '<image xlink:href="data:image/png;base64,ABC123==">'
    assert has_category(svg, "data_uri_base64")


def test_data_uri_base64_severity_medium():
    findings = scan_svg('<image src="data:image/svg+xml;base64,PHN2Zz4=">')
    b64 = [f for f in findings if f.category == "data_uri_base64"]
    assert b64
    assert b64[0].severity == SVGSanitizerSeverity.MEDIUM


# ---------------------------------------------------------------------------
# 7. javascript: URI
# ---------------------------------------------------------------------------


def test_javascript_uri_href():
    svg = '<a href="javascript:alert(1)">'
    assert has_category(svg, "javascript_uri")


def test_javascript_uri_severity_critical():
    findings = scan_svg('<a href="javascript:void(0)">')
    js = [f for f in findings if f.category == "javascript_uri"]
    assert js
    assert js[0].severity == SVGSanitizerSeverity.CRITICAL


# ---------------------------------------------------------------------------
# 8. Animate/set security attributes
# ---------------------------------------------------------------------------


def test_animate_href_attribute():
    svg = '<animate attributeName="href" to="javascript:bad()"/>'
    assert has_category(svg, "animate_security_attr")


def test_animate_event_handler_attribute():
    svg = '<animate attributeName="onload" to="evil()"/>'
    cats = categories(svg)
    # May match animate_event_handler and/or animate_security_attr
    assert "animate_event_handler" in cats or "animate_security_attr" in cats


def test_set_element_security_attr():
    svg = '<set attributeName="src" to="javascript:x"/>'
    assert has_category(svg, "animate_security_attr")


def test_animate_event_severity_high():
    findings = scan_svg('<animate attributeName="onclick" to="bad()"/>')
    ev = [f for f in findings if f.category == "animate_event_handler"]
    if ev:
        assert ev[0].severity == SVGSanitizerSeverity.HIGH


# ---------------------------------------------------------------------------
# 9. Clean inputs — false positive checks
# ---------------------------------------------------------------------------


def test_clean_svg_no_findings():
    svg = '<svg xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="40"/></svg>'
    assert scan_svg(svg) == []


def test_plain_text_no_findings():
    assert scan_svg("Hello, world!") == []


def test_benign_css_class_no_findings():
    html = '<div class="onclick-button">Click me</div>'
    # "onclick-button" is NOT an event handler attribute
    findings = [f for f in scan_svg(html) if f.category == "event_handler"]
    assert findings == []


def test_local_use_reference_no_external_flag():
    svg = """
    <svg>
      <defs><symbol id="icon"><circle r="5"/></symbol></defs>
      <use href="#icon"/>
    </svg>
    """
    assert not has_category(svg, "external_reference")


# ---------------------------------------------------------------------------
# 10. Return type / structure checks
# ---------------------------------------------------------------------------


def test_returns_list():
    assert isinstance(scan_svg(""), list)


def test_finding_fields():
    findings = scan_svg('<svg onload="x()">')
    assert findings
    f = findings[0]
    assert isinstance(f, SVGSanitizerFinding)
    assert isinstance(f.category, str)
    assert isinstance(f.match, str)
    assert isinstance(f.severity, SVGSanitizerSeverity)
    assert isinstance(f.description, str)
    assert 0.0 <= f.confidence <= 1.0
