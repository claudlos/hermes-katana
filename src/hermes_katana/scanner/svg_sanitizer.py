"""
SVG sanitizer scanner for HermesKatana.

Detects SVG-based content attacks:
- <script> tags inside SVG
- foreignObject elements (can contain arbitrary HTML/JS)
- SVG event handlers (onload, onclick, onerror, etc.)
- SVG use/xlink references to external resources
- Embedded data URIs with executable content
- SVG animate/set elements that modify security-relevant attributes

Speed: Microseconds for typical SVG content.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "SVGSanitizerSeverity",
    "SVGSanitizerFinding",
    "scan_svg",
]


class SVGSanitizerSeverity(str, Enum):
    """Severity of SVG sanitizer findings."""

    CRITICAL = "critical"
    """Script execution or external code loading."""

    HIGH = "high"
    """Event handlers or foreignObject with JS."""

    MEDIUM = "medium"
    """Suspicious use/xlink or data URI."""

    LOW = "low"
    """Animate/set targeting security attributes."""


@dataclass(frozen=True, slots=True)
class SVGSanitizerFinding:
    """A single SVG sanitizer finding.

    Attributes:
        category: The attack category detected.
        match: The matched content snippet.
        severity: How severe this finding is.
        description: Human-readable explanation.
        confidence: Detection confidence 0.0-1.0.
    """

    category: str
    match: str
    severity: SVGSanitizerSeverity
    description: str
    confidence: float = 0.9


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# <script> tags inside SVG (with or without namespace)
_SCRIPT_TAG_RE = re.compile(
    r"<\s*(?:svg:)?script[\s>]",
    re.IGNORECASE,
)

# foreignObject element (can contain HTML with inline JS)
_FOREIGN_OBJECT_RE = re.compile(
    r"<\s*(?:svg:)?foreignObject[\s>/]",
    re.IGNORECASE,
)

# SVG event handler attributes (on* attributes)
_EVENT_HANDLER_RE = re.compile(
    r"""\bon\w+\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+)""",
    re.IGNORECASE,
)

# SVG use element with xlink:href or href pointing to external resource
_USE_EXTERNAL_RE = re.compile(
    r"""<\s*(?:svg:)?use\b[^>]*(?:xlink:href|href)\s*=\s*["'](?!#)([^"']+)["']""",
    re.IGNORECASE,
)

# href/xlink:href containing a data URI with executable content
_DATA_URI_EXEC_RE = re.compile(
    r"""(?:href|src|action|xlink:href)\s*=\s*["']?\s*data\s*:\s*(?:text/html|image/svg\+xml|application/(?:javascript|x-javascript|ecmascript)|text/javascript)""",
    re.IGNORECASE,
)

# data: URI with base64 encoded content (could be executable)
_DATA_URI_BASE64_RE = re.compile(
    r"""(?:href|src|xlink:href)\s*=\s*["']?\s*data\s*:[^;]+;\s*base64\s*,""",
    re.IGNORECASE,
)

# javascript: URI scheme
_JAVASCRIPT_URI_RE = re.compile(
    r"""(?:href|src|action|xlink:href)\s*=\s*["']?\s*javascript\s*:""",
    re.IGNORECASE,
)

_CSS_JAVASCRIPT_URL_RE = re.compile(
    r"""url\s*\(\s*["']?\s*javascript\s*:""",
    re.IGNORECASE,
)

_EXTERNAL_RESOURCE_RE = re.compile(
    r"""<\s*(?:svg:)?(?:image|a|script|iframe|embed|object)\b[^>]*(?:href|src|xlink:href)\s*=\s*["']\s*https?://[^"']+["']""",
    re.IGNORECASE,
)

# SVG animate/set targeting security-sensitive attributes
_ANIMATE_SECURITY_ATTR_RE = re.compile(
    r"""<\s*(?:svg:)?(?:animate|animateTransform|set)\b[^>]*\battributeName\s*=\s*["']?\s*(?:href|xlink:href|src|action|formaction|data|srcdoc|onload|onclick|onerror)""",
    re.IGNORECASE,
)

# SVG animate targeting any on* event handler attribute
_ANIMATE_EVENT_ATTR_RE = re.compile(
    r"""<\s*(?:svg:)?(?:animate|animateTransform|set)\b[^>]*\battributeName\s*=\s*["']?\s*on\w+""",
    re.IGNORECASE,
)


def scan_svg(text: str) -> list[SVGSanitizerFinding]:
    """Scan text for SVG-based attack patterns.

    Detects script injection, event handlers, foreignObject elements,
    external use/xlink references, executable data URIs, and malicious
    animate/set attribute targets.

    Args:
        text: The text content to scan (HTML, SVG, or mixed).

    Returns:
        List of :class:`SVGSanitizerFinding` instances, empty if clean.

    Example:
        >>> findings = scan_svg('<svg><script>alert(1)</script></svg>')
        >>> findings[0].category
        'script_injection'
    """
    findings: list[SVGSanitizerFinding] = []
    scan_text = html.unescape(text)

    # 1. <script> tags inside SVG
    for m in _SCRIPT_TAG_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="script_injection",
                match=m.group(0).strip(),
                severity=SVGSanitizerSeverity.CRITICAL,
                description="SVG <script> tag enables JavaScript execution",
                confidence=0.98,
            )
        )

    # 2. foreignObject element
    for m in _FOREIGN_OBJECT_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="foreign_object",
                match=m.group(0).strip(),
                severity=SVGSanitizerSeverity.HIGH,
                description="SVG <foreignObject> can embed arbitrary HTML/JS content",
                confidence=0.9,
            )
        )

    # 3. Event handler attributes
    for m in _EVENT_HANDLER_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="event_handler",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.HIGH,
                description=f"SVG event handler attribute: {m.group(0).split('=')[0].strip()}",
                confidence=0.85,
            )
        )

    # 4. use element with external xlink/href
    for m in _USE_EXTERNAL_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="external_reference",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.MEDIUM,
                description=f"SVG <use> references external resource: {m.group(1)[:50]}",
                confidence=0.85,
            )
        )

    # 5. data: URIs with executable MIME types
    for m in _DATA_URI_EXEC_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="data_uri_executable",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.CRITICAL,
                description="Executable content in data URI (HTML/JS MIME type)",
                confidence=0.95,
            )
        )

    # 6. data: URIs with base64 encoding (potentially executable)
    for m in _DATA_URI_BASE64_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="data_uri_base64",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.MEDIUM,
                description="Base64-encoded data URI — may contain executable content",
                confidence=0.75,
            )
        )

    # 7. javascript: URI scheme
    for m in _JAVASCRIPT_URI_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="javascript_uri",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.CRITICAL,
                description="javascript: URI scheme enables script execution",
                confidence=0.98,
            )
        )

    # 8. CSS url(javascript:...) patterns
    for m in _CSS_JAVASCRIPT_URL_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="css_javascript_url",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.CRITICAL,
                description="CSS url(javascript:...) inside SVG can execute script",
                confidence=0.95,
            )
        )

    # 9. External executable/embed resource references
    for m in _EXTERNAL_RESOURCE_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="external_resource",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.MEDIUM,
                description="SVG element references an external resource",
                confidence=0.78,
            )
        )

    # 10. animate/set targeting security-sensitive named attributes
    for m in _ANIMATE_SECURITY_ATTR_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="animate_security_attr",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.LOW,
                description="SVG animate/set targets a security-relevant attribute",
                confidence=0.8,
            )
        )

    # 11. animate/set targeting on* event handler attributes
    for m in _ANIMATE_EVENT_ATTR_RE.finditer(scan_text):
        findings.append(
            SVGSanitizerFinding(
                category="animate_event_handler",
                match=m.group(0)[:80],
                severity=SVGSanitizerSeverity.HIGH,
                description="SVG animate/set can inject an event handler attribute",
                confidence=0.9,
            )
        )

    return findings
