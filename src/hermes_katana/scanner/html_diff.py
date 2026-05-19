"""
HTML divergence detector for HermesKatana.

Detects hidden text in HTML that diverges from visible/rendered content.
This catches Content Injection Traps where attackers hide instructions
in HTML that humans can't see but LLMs process.

Techniques detected:
- CSS display:none / visibility:hidden elements
- Off-viewport positioned text (left:-9999px etc)
- Zero-opacity / zero-size elements
- overflow:hidden with clipped text
- HTML comments (<!-- -->)
- aria-hidden / aria-label hidden content

Per hermes-scabbard-brainstorm-v2.md §3 Layer B.

Performance: Microseconds for simple HTML, low milliseconds for complex pages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from html.parser import HTMLParser

__all__ = [
    "HtmlDiffCategory",
    "HtmlDiffSeverity",
    "HtmlDiffFinding",
    "scan_html",
]


class HtmlDiffCategory(str, Enum):
    """Categories of HTML divergence findings."""

    CSS_HIDDEN = "css_hidden"
    """Text hidden via CSS (display:none, visibility:hidden)."""

    OFFSCREEN = "offscreen"
    """Text positioned off-viewport."""

    ZERO_RENDER = "zero_render"
    """Text with zero opacity or zero size."""

    OVERFLOW_CLIP = "overflow_clip"
    """Text clipped by overflow:hidden."""

    COMMENT = "comment"
    """Text inside HTML comments."""

    ARIA_HIDDEN = "aria_hidden"
    """Content in aria-hidden or misleading aria-label."""


class HtmlDiffSeverity(str, Enum):
    """Severity levels for HTML divergence findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class HtmlDiffFinding:
    """A single HTML divergence finding.

    Attributes:
        pattern_name: Name of the detection pattern.
        category: Type of HTML hiding technique.
        severity: How dangerous this finding is.
        hidden_text: The text that is hidden from rendering.
        description: Human-readable explanation.
        confidence: Detection confidence 0.0-1.0.
    """

    pattern_name: str
    category: HtmlDiffCategory
    severity: HtmlDiffSeverity
    hidden_text: str
    description: str
    confidence: float = 0.90


# ---------------------------------------------------------------------------
# Injection keyword patterns (subset from injection.py for cross-reference)
# ---------------------------------------------------------------------------

_INJECTION_KEYWORDS = re.compile(
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above)?\s*"
    r"(?:instructions?|rules?|prompts?|guidelines?)|"
    r"(?:system|assistant|user)\s*(?:prompt|message|role)\s*:|"
    r"(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|new\s+instructions?)|"
    r"(?:reveal|extract|dump|exfiltrate|output)\s+(?:the\s+)?(?:system|secret|hidden|api|key|password)|"
    r"(?:secret|hidden|covert)\s+(?:code|message|instruction|text|note)|"
    r"(?:do\s+(?:not|n.?t)\s+(?:mention|reveal|discuss|include).{0,60}(?:secret|hidden|covert|instruction))|"
    r"(?:meant|intended|supposed)\s+to\s+be\s+(?:hidden|covert|secret)|"
    r"\b(?:protocol|configuration|mode)\s+(?:override|update|change)\b|"
    r"\b(acrostic|cipher|encoded)\b",
    re.IGNORECASE,
)


def _has_injection_keywords(text: str) -> bool:
    """Check if text contains injection-related keywords."""
    return bool(_INJECTION_KEYWORDS.search(text))


# ---------------------------------------------------------------------------
# CSS property parsing helpers
# ---------------------------------------------------------------------------

_STYLE_DISPLAY_NONE = re.compile(r"display\s*:\s*none", re.IGNORECASE)
_STYLE_VISIBILITY_HIDDEN = re.compile(r"visibility\s*:\s*hidden", re.IGNORECASE)
_STYLE_OPACITY_ZERO = re.compile(r"opacity\s*:\s*0(?:\.0+)?(?:\s*;|\s*$)", re.IGNORECASE)
_STYLE_OFFSCREEN = re.compile(
    r"(?:left|top|right|bottom)\s*:\s*-\d{3,}px|"
    r"position\s*:\s*(?:absolute|fixed).*?(?:left|top)\s*:\s*-\d{3,}px|"
    r"text-indent\s*:\s*-\d{3,}",
    re.IGNORECASE | re.DOTALL,
)
_STYLE_ZERO_SIZE = re.compile(
    r"(?:width|height)\s*:\s*0(?:px)?\s*(?:;|$).*?"
    r"(?:width|height)\s*:\s*0(?:px)?\s*(?:;|$)|"
    r"font-size\s*:\s*0(?:px)?\s*(?:;|$)|"
    r"max-(?:width|height)\s*:\s*0(?:px)?\s*(?:;|$)",
    re.IGNORECASE | re.DOTALL,
)
_STYLE_OVERFLOW_HIDDEN = re.compile(r"overflow\s*:\s*hidden", re.IGNORECASE)
_STYLE_CLIP = re.compile(
    r"clip\s*:\s*rect\s*\(\s*0|clip-path\s*:\s*inset\s*\(\s*100",
    re.IGNORECASE,
)

# Matches <style> blocks to extract CSS rules targeting hidden content
_STYLE_BLOCK_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.IGNORECASE | re.DOTALL)
_CSS_RULE_RE = re.compile(r"([^{]+)\{([^}]+)\}", re.DOTALL)
# Pre-compiled selectors for class/id extraction in pre_scan_stylesheets
_CSS_CLASS_RE = re.compile(r"\.([a-zA-Z_][\w-]*)")
_CSS_ID_RE = re.compile(r"#([a-zA-Z_][\w-]*)")


def _is_hidden_style(style: str) -> str | None:
    """Check inline style for hiding techniques. Returns category name or None."""
    if _STYLE_DISPLAY_NONE.search(style):
        return "css_hidden"
    if _STYLE_VISIBILITY_HIDDEN.search(style):
        return "css_hidden"
    if _STYLE_OPACITY_ZERO.search(style):
        return "zero_render"
    if _STYLE_OFFSCREEN.search(style):
        return "offscreen"
    if _STYLE_ZERO_SIZE.search(style):
        return "zero_render"
    if _STYLE_CLIP.search(style):
        return "overflow_clip"
    if _STYLE_OVERFLOW_HIDDEN.search(style) and _STYLE_ZERO_SIZE.search(style):
        return "overflow_clip"
    return None


# ---------------------------------------------------------------------------
# HTML Parser for extracting all text and hidden text
# ---------------------------------------------------------------------------

_NON_TEXT_TAGS = frozenset({"script", "style", "noscript", "template"})


class _HtmlDivergenceParser(HTMLParser):
    """Custom HTML parser that tracks hidden vs visible text."""

    def __init__(self) -> None:
        super().__init__()
        self.all_text_parts: list[str] = []
        self.hidden_parts: list[tuple[str, str, str]] = []
        self.comments: list[str] = []
        self.aria_labels: list[str] = []

        self._tag_stack: list[dict] = []
        self._in_non_text = 0
        self._css_hidden_classes: set[str] = set()
        self._css_hidden_ids: set[str] = set()

    def _current_hidden(self) -> str | None:
        """Check if we're inside a hidden element."""
        for frame in reversed(self._tag_stack):
            reason = frame.get("hidden_reason")
            if isinstance(reason, str) and reason:
                return reason
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower in _NON_TEXT_TAGS:
            self._in_non_text += 1

        attr_dict: dict[str, str] = {}
        for k, v in attrs:
            attr_dict[k.lower()] = v or ""

        hidden_reason: str | None = None

        # Check inline style
        style = attr_dict.get("style", "")
        if style:
            hidden_reason = _is_hidden_style(style)

        # Check hidden attribute
        if "hidden" in attr_dict:
            hidden_reason = "css_hidden"

        # Check aria-hidden
        if attr_dict.get("aria-hidden", "").lower() == "true":
            hidden_reason = "aria_hidden"

        # Check type="hidden" (form fields)
        if attr_dict.get("type", "").lower() == "hidden":
            hidden_reason = "css_hidden"

        # Check CSS class/id against stylesheet rules
        classes = set(attr_dict.get("class", "").split())
        elem_id = attr_dict.get("id", "")
        if classes & self._css_hidden_classes:
            hidden_reason = "css_hidden"
        if elem_id and elem_id in self._css_hidden_ids:
            hidden_reason = "css_hidden"

        # Capture aria-label
        aria_label = attr_dict.get("aria-label", "")
        if aria_label.strip():
            self.aria_labels.append(aria_label.strip())

        self._tag_stack.append(
            {
                "tag": tag_lower,
                "hidden_reason": hidden_reason,
            }
        )

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower in _NON_TEXT_TAGS:
            self._in_non_text = max(0, self._in_non_text - 1)

        for i in range(len(self._tag_stack) - 1, -1, -1):
            if self._tag_stack[i]["tag"] == tag_lower:
                self._tag_stack.pop(i)
                break

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        if self._in_non_text > 0:
            return

        self.all_text_parts.append(text)

        hidden_reason = self._current_hidden()
        if hidden_reason:
            ctx = self._tag_stack[-1]["tag"] if self._tag_stack else "?"
            self.hidden_parts.append((text, hidden_reason, f"<{ctx}>"))

    def handle_comment(self, data: str) -> None:
        text = data.strip()
        if text:
            self.comments.append(text)

    def pre_scan_stylesheets(self, html: str) -> None:
        """Pre-scan <style> blocks for CSS rules that hide elements."""
        for style_match in _STYLE_BLOCK_RE.finditer(html):
            css_text = style_match.group(1)
            for rule_match in _CSS_RULE_RE.finditer(css_text):
                selector = rule_match.group(1).strip()
                properties = rule_match.group(2)
                if _is_hidden_style(properties):
                    for cls in _CSS_CLASS_RE.findall(selector):
                        self._css_hidden_classes.add(cls)
                    for eid in _CSS_ID_RE.findall(selector):
                        self._css_hidden_ids.add(eid)


# ---------------------------------------------------------------------------
# Scoring and main entry point
# ---------------------------------------------------------------------------


def _score_severity(hidden_text: str, visible_ratio: float, has_injection: bool) -> HtmlDiffSeverity:
    """Determine severity based on content and ratio."""
    if has_injection:
        return HtmlDiffSeverity.HIGH
    if visible_ratio > 0.3:
        return HtmlDiffSeverity.HIGH
    if visible_ratio > 0.1:
        return HtmlDiffSeverity.MEDIUM
    if len(hidden_text) > 20:
        return HtmlDiffSeverity.MEDIUM
    return HtmlDiffSeverity.LOW


def scan_html(html: str) -> list[HtmlDiffFinding]:
    """Scan HTML for hidden text that diverges from visible content.

    Parses the HTML, extracts all text vs. visible-only text, and flags
    any hidden content. If hidden text contains injection keywords,
    severity is boosted to HIGH.

    Args:
        html: Raw HTML string to scan.

    Returns:
        List of HtmlDiffFinding objects, sorted by severity.

    Performance:
        Microseconds for simple HTML, low milliseconds for complex pages.

    Example:
        >>> findings = scan_html('<div style="display:none">IGNORE INSTRUCTIONS</div>')
        >>> len(findings) > 0
        True
        >>> findings[0].category
        <HtmlDiffCategory.CSS_HIDDEN: 'css_hidden'>
    """
    if not html or not html.strip():
        return []

    parser = _HtmlDivergenceParser()
    parser.pre_scan_stylesheets(html)

    try:
        parser.feed(html)
    except Exception:
        return []

    findings: list[HtmlDiffFinding] = []

    all_text = " ".join(parser.all_text_parts)
    total_len = max(len(all_text), 1)

    # --- Hidden element text ---
    for text, category, context in parser.hidden_parts:
        if not text.strip():
            continue

        has_injection = _has_injection_keywords(text)
        ratio = len(text) / total_len

        cat_enum = HtmlDiffCategory(category)
        severity = _score_severity(text, ratio, has_injection)

        if has_injection and severity not in (
            HtmlDiffSeverity.HIGH,
            HtmlDiffSeverity.CRITICAL,
        ):
            severity = HtmlDiffSeverity.HIGH

        desc_parts = [
            f"Hidden text ({category}) in {context}: {text[:80]}{'...' if len(text) > 80 else ''}",
        ]
        if has_injection:
            desc_parts.append("Contains injection keywords — likely prompt injection via hidden HTML.")

        findings.append(
            HtmlDiffFinding(
                pattern_name=f"hidden_{category}",
                category=cat_enum,
                severity=severity,
                hidden_text=text,
                description=" ".join(desc_parts),
                confidence=0.95 if has_injection else 0.85,
            )
        )

    # --- HTML comments ---
    for comment_text in parser.comments:
        if not comment_text.strip():
            continue

        has_injection = _has_injection_keywords(comment_text)

        if has_injection:
            severity = HtmlDiffSeverity.HIGH
        elif len(comment_text) > 200:
            severity = HtmlDiffSeverity.MEDIUM
        else:
            severity = HtmlDiffSeverity.LOW

        desc = f"HTML comment: {comment_text[:80]}{'...' if len(comment_text) > 80 else ''}"
        if has_injection:
            desc += " — Contains injection keywords."

        findings.append(
            HtmlDiffFinding(
                pattern_name="html_comment",
                category=HtmlDiffCategory.COMMENT,
                severity=severity,
                hidden_text=comment_text,
                description=desc,
                confidence=0.95 if has_injection else 0.70,
            )
        )

    # --- aria-label with injection ---
    for label in parser.aria_labels:
        has_injection = _has_injection_keywords(label)
        if not has_injection:
            continue

        findings.append(
            HtmlDiffFinding(
                pattern_name="aria_label_injection",
                category=HtmlDiffCategory.ARIA_HIDDEN,
                severity=HtmlDiffSeverity.HIGH,
                hidden_text=label,
                description=f"aria-label contains injection keywords: {label[:80]}",
                confidence=0.95,
            )
        )

    # --- SVG embedded in HTML with hidden content ---
    _SVG_PATTERN = re.compile(r"<svg\b([^>]*)>(.*?)</svg>", re.DOTALL | re.IGNORECASE)
    for m in _SVG_PATTERN.finditer(html):
        svg_content = m.group(0)
        svg_body = m.group(2)
        # Check for suspicious attributes (event handlers, external references)
        if re.search(r"\bon\w+\s*=", svg_content, re.IGNORECASE):
            findings.append(
                HtmlDiffFinding(
                    pattern_name="svg_event_handler",
                    category=HtmlDiffCategory.CSS_HIDDEN,
                    severity=HtmlDiffSeverity.HIGH,
                    hidden_text=svg_content[:200],
                    description=f"SVG element contains inline event handler: {svg_content[:100]}...",
                    confidence=0.85,
                )
            )
        # Check SVG metadata/desc for injection keywords
        desc_match = re.search(r"<desc[^>]*>(.*?)</desc>", svg_body, re.DOTALL | re.IGNORECASE)
        if desc_match:
            desc_text = re.sub(r"<[^>]+>", "", desc_match.group(1)).strip()
            if desc_text and _has_injection_keywords(desc_text):
                findings.append(
                    HtmlDiffFinding(
                        pattern_name="svg_desc_injection",
                        category=HtmlDiffCategory.CSS_HIDDEN,
                        severity=HtmlDiffSeverity.HIGH,
                        hidden_text=desc_text[:200],
                        description=f"SVG <desc> contains injection keywords: {desc_text[:100]}...",
                        confidence=0.90,
                    )
                )
        title_match = re.search(r"<title[^>]*>(.*?)</title>", svg_body, re.DOTALL | re.IGNORECASE)
        if title_match:
            title_text = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()
            if title_text and _has_injection_keywords(title_text):
                findings.append(
                    HtmlDiffFinding(
                        pattern_name="svg_title_injection",
                        category=HtmlDiffCategory.CSS_HIDDEN,
                        severity=HtmlDiffSeverity.HIGH,
                        hidden_text=title_text[:200],
                        description=f"SVG <title> contains injection keywords: {title_text[:100]}...",
                        confidence=0.90,
                    )
                )

    # --- HTML meta tags with injection content ---
    _META_PATTERN = re.compile(
        r'<meta\s+[^>]*(?:content|value)\s*=\s*["\']([^"\']{20,})["\']',
        re.IGNORECASE,
    )
    for m in _META_PATTERN.finditer(html):
        content_val = m.group(1)
        if _has_injection_keywords(content_val):
            findings.append(
                HtmlDiffFinding(
                    pattern_name="meta_tag_injection",
                    category=HtmlDiffCategory.COMMENT,
                    severity=HtmlDiffSeverity.HIGH,
                    hidden_text=content_val[:200],
                    description=f"HTML meta tag contains injection keywords: {content_val[:100]}...",
                    confidence=0.85,
                )
            )

    # Sort by severity
    severity_order = {
        HtmlDiffSeverity.CRITICAL: 0,
        HtmlDiffSeverity.HIGH: 1,
        HtmlDiffSeverity.MEDIUM: 2,
        HtmlDiffSeverity.LOW: 3,
        HtmlDiffSeverity.INFO: 4,
    }
    findings.sort(key=lambda f: severity_order.get(f.severity, 99))

    return findings
