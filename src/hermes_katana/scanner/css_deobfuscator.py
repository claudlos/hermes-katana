"""
CSS deobfuscator for HermesKatana Scabbard.

Detects CSS-based content hiding and obfuscation techniques:
- text-indent:-9999px (classic screen-reader-only / hidden content trick)
- clip-path: rect(0,0,0,0) (complete visual hiding)
- position:absolute with off-screen coordinates
- ::before / ::after pseudo-elements with content injection
- @font-face character remapping (Xiong et al. malicious font attack)
- overflow:hidden with height restrictions
- font-size:0, color:transparent, opacity:0
- CSS custom properties hiding content

Catches rendering-layer attacks where content exists in the DOM
but is visually hidden from the user while still being processed
by AI agents.

Speed: Microseconds for typical CSS content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "CSSDeobfuscatorSeverity",
    "CSSDeobfuscatorFinding",
    "detect_css_deobfuscate",
]


class CSSDeobfuscatorSeverity(str, Enum):
    """Severity of CSS deobfuscation findings."""

    CRITICAL = "critical"
    """Substantial hidden content likely malicious."""

    HIGH = "high"
    """Hidden content detected, likely injection."""

    MEDIUM = "medium"
    """Content hiding technique detected."""

    LOW = "low"
    """Minor hiding technique, likely benign."""


@dataclass(frozen=True, slots=True)
class CSSDeobfuscatorFinding:
    """A single CSS deobfuscation finding.

    Attributes:
        technique: The CSS hiding technique detected.
        location: Where in the CSS/HTML this was found.
        style_content: The CSS property/value that hides content.
        hidden_preview: Preview of the hidden content (if extractable).
        severity: How severe this finding is.
        description: Human-readable explanation.
        confidence: Detection confidence 0.0-1.0.
    """

    technique: str
    location: str
    style_content: str
    hidden_preview: str
    severity: CSSDeobfuscatorSeverity
    description: str
    confidence: float = 0.85


# -----------------------------------------------------------------------
# CSS hiding pattern definitions
# Each: (technique_name, pattern_regex, severity, confidence, description)
# -----------------------------------------------------------------------

_CSS_HIDING_PATTERNS: list[tuple[str, str, CSSDeobfuscatorSeverity, float, str]] = [
    # Text-indent hiding
    (
        "text_indent_screen_reader",
        r"text-indent\s*:\s*-?\s*9999\s*(?:px|em|rem|%)?",
        CSSDeobfuscatorSeverity.HIGH,
        0.85,
        "text-indent:-9999px is a classic technique to hide text visually while keeping it accessible to screen readers (and AI parsers). May indicate hidden instructions.",
    ),
    (
        "text_indent_large_negative",
        r"text-indent\s*:\s*-?\s*\d{4,}\s*(?:px|em|rem)?",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.75,
        "Large negative text-indent hides content off-screen.",
    ),
    # Clip-path hiding
    (
        "clip_path_zero",
        r"clip(-path)?\s*:\s*rect\s*\(\s*0\s*(?:px)?\s*,\s*0\s*(?:px)?\s*,\s*0\s*(?:px)?\s*,\s*0\s*(?:px)?\s*\)",
        CSSDeobfuscatorSeverity.HIGH,
        0.88,
        "clip-path:rect(0,0,0,0) completely hides an element visually while keeping it in the DOM.",
    ),
    (
        "clip_path_similar",
        r"clip(-path)?\s*:\s*rect\s*\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\s*\)",
        CSSDeobfuscatorSeverity.HIGH,
        0.88,
        "clip-path with all-zero values hides content completely.",
    ),
    # Position off-screen
    (
        "position_absolute_offscreen",
        r"position\s*:\s*absolute\s*[^;{]*(?:left\s*:\s*-?\s*\d{3,}|top\s*:\s*-?\s*\d{3,}|right\s*:\s*-?\s*\d{3,}|bottom\s*:\s*-?\s*\d{3,})",
        CSSDeobfuscatorSeverity.HIGH,
        0.80,
        "position:absolute with large off-screen coordinates places content beyond the visible viewport.",
    ),
    (
        "position_fixed_offscreen",
        r"position\s*:\s*fixed\s*[^;{]*(?:left\s*:\s*-?\s*\d{3,}|top\s*:\s*-?\s*\d{3,})",
        CSSDeobfuscatorSeverity.HIGH,
        0.80,
        "position:fixed with off-screen coordinates hides content from view.",
    ),
    # Overflow hiding
    (
        "overflow_hidden_clip",
        r"overflow\s*:\s*hidden\s*[^;{]*(?:height\s*:\s*\d+(?:px|em)?|max-height\s*:\s*\d+(?:px|em)?)",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.70,
        "overflow:hidden with small height may clip text content.",
    ),
    (
        "overflow_auto_scroll",
        r"overflow\s*:\s*auto\s*[^;{]*(?:height\s*:\s*\d+px)",
        CSSDeobfuscatorSeverity.LOW,
        0.50,
        "overflow:auto with explicit height may hide content in scrollable container.",
    ),
    # Size hiding
    (
        "font_size_zero",
        r"font-size\s*:\s*0\s*(?:px|em|rem)?",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.70,
        "font-size:0 makes text invisible while keeping it in the DOM.",
    ),
    (
        "line_height_zero",
        r"line-height\s*:\s*0\s*(?:px|em|rem)?",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.65,
        "line-height:0 makes text lines invisible.",
    ),
    # Color hiding
    (
        "color_transparent",
        r"(?:color|background(?:-color)?)\s*:\s*transparent",
        CSSDeobfuscatorSeverity.LOW,
        0.40,
        "transparent color may hide text (low confidence alone).",
    ),
    (
        "color_fg_bg_match",
        r"(?:color\s*:\s*#[Ff]{6}|background(?:-color)?\s*:\s*#[Ff]{6})",
        CSSDeobfuscatorSeverity.LOW,
        0.50,
        "Color set to #FFFFFF on white background may be invisible.",
    ),
    # Opacity hiding
    (
        "opacity_zero",
        r"opacity\s*:\s*0(?:\.\d+)?",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.70,
        "opacity:0 completely hides an element visually.",
    ),
    # Display hiding
    (
        "display_none",
        r"display\s*:\s*none",
        CSSDeobfuscatorSeverity.HIGH,
        0.85,
        "display:none removes element from render tree but it may still exist in the DOM for AI parsing.",
    ),
    (
        "visibility_hidden",
        r"visibility\s*:\s*hidden",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.65,
        "visibility:hidden hides element visually but keeps it in layout flow.",
    ),
]


# Pseudo-element content injection patterns
_PSEUDO_PATTERNS: list[tuple[str, str, CSSDeobfuscatorSeverity, float, str]] = [
    (
        "before_content_injection",
        r"::?\s*before\s*[^;{]*content\s*:\s*['\"]?",
        CSSDeobfuscatorSeverity.HIGH,
        0.80,
        "::before pseudo-element with content property can inject hidden text via CSS.",
    ),
    (
        "after_content_injection",
        r"::?\s*after\s*[^;{]*content\s*:\s*['\"]?",
        CSSDeobfuscatorSeverity.HIGH,
        0.80,
        "::after pseudo-element with content property can inject hidden text via CSS.",
    ),
    (
        "placeholder_content",
        r"::?\s*placeholder\s*[^;{]*content\s*:\s*['\"]?",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.70,
        "::placeholder pseudo-element with content property may hide data.",
    ),
]


# @font-face character remapping detection
_FONT_PATTERNS: list[tuple[str, str, CSSDeobfuscatorSeverity, float, str]] = [
    (
        "font_face_unicode_range",
        r"@font-face\s*\{[^}]*unicode-range\s*:\s*U\+",
        CSSDeobfuscatorSeverity.MEDIUM,
        0.70,
        "@font-face with unicode-range may use character remapping for homoglyph attacks (Xiong et al.).",
    ),
    (
        "font_face_descriptor_override",
        r"@font-face\s*\{[^}]*(?:ascent|descent|baseline|src)\s*:\s*[^;]+",
        CSSDeobfuscatorSeverity.LOW,
        0.50,
        "@font-face descriptor override may manipulate font metrics.",
    ),
]


# Pre-compiled utility patterns used by helper functions
_STYLE_TAG_RE = re.compile(r"<style[^>]*>(.*?)</style>", re.DOTALL | re.IGNORECASE)
_INLINE_STYLE_RE = re.compile(r'style\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)
_PSEUDO_RULE_RE = re.compile(r"([^{}]+)\s*::?\s*(before|after)\s*\{([^}]*)\}", re.IGNORECASE)
_CONTENT_PROP_RE = re.compile(r"content\s*:", re.IGNORECASE)
_CONTENT_VAL_RE = re.compile(r"content\s*:\s*['\"]?([^'\";}]+)['\"]?", re.IGNORECASE)
_FONT_FACE_RE = re.compile(r"@font-face\s*\{([^}]+)\}", re.IGNORECASE)
_UNICODE_RANGE_RE = re.compile(r"unicode-range\s*:\s*([^;}]+)", re.IGNORECASE)
_EXTERNAL_LINK_RE = re.compile(
    r'<link[^>]+rel\s*=\s*["\']stylesheet["\'][^>]+href\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# Pre-compiled (pattern_re, name, severity, confidence, description) tuples
_CompiledPattern = tuple  # (re.Pattern, str, CSSDeobfuscatorSeverity, float, str)

_CSS_HIDING_PATTERNS_COMPILED: list[_CompiledPattern] = [
    (re.compile(p, re.IGNORECASE), n, s, c, d) for n, p, s, c, d in _CSS_HIDING_PATTERNS
]
_PSEUDO_PATTERNS_COMPILED: list[_CompiledPattern] = [
    (re.compile(p, re.IGNORECASE), n, s, c, d) for n, p, s, c, d in _PSEUDO_PATTERNS
]
_FONT_PATTERNS_COMPILED: list[_CompiledPattern] = [
    (re.compile(p, re.IGNORECASE), n, s, c, d) for n, p, s, c, d in _FONT_PATTERNS
]


def _extract_css_blocks(html: str) -> list[tuple[str, str]]:
    """Extract CSS from style tags and return (css_content, location)."""
    return [(m.group(1), f"<style> at offset {m.start()}") for m in _STYLE_TAG_RE.finditer(html)]


def _extract_inline_styles(html: str) -> list[tuple[str, str, int]]:
    """Extract inline style attributes and return (style_value, selector_context, offset)."""
    return [(m.group(1), "inline style attribute", m.start()) for m in _INLINE_STYLE_RE.finditer(html)]


def _scan_css_for_patterns(
    css_content: str,
    location: str,
    patterns: list[_CompiledPattern],
) -> list[CSSDeobfuscatorFinding]:
    """Scan CSS content for pre-compiled hiding patterns."""
    findings: list[CSSDeobfuscatorFinding] = []

    for pattern_re, name, severity, confidence, description in patterns:
        for match in pattern_re.finditer(css_content):
            findings.append(
                CSSDeobfuscatorFinding(
                    technique=name,
                    location=location,
                    style_content=match.group(),
                    hidden_preview="",
                    severity=severity,
                    description=description,
                    confidence=confidence,
                )
            )

    return findings


def _scan_for_pseudo_element_injection(
    css_blocks: list[tuple[str, str]],
) -> list[CSSDeobfuscatorFinding]:
    """Detect CSS pseudo-element content injection from pre-extracted blocks."""
    findings: list[CSSDeobfuscatorFinding] = []
    _skip = {"", "none", "normal", '""', "''"}

    for block, _location in css_blocks:
        for match in _PSEUDO_RULE_RE.finditer(block):
            selector = match.group(1).strip()
            pseudo = match.group(2)
            content_block = match.group(3)

            if not _CONTENT_PROP_RE.search(content_block):
                continue
            content_match = _CONTENT_VAL_RE.search(content_block)
            content_value = content_match.group(1) if content_match else "unknown"

            if content_value not in _skip:
                findings.append(
                    CSSDeobfuscatorFinding(
                        technique=f"pseudo_element_content_{pseudo}",
                        location=f"CSS rule: {selector}::{pseudo}",
                        style_content=f"content: {content_value}",
                        hidden_preview=content_value[:100],
                        severity=CSSDeobfuscatorSeverity.HIGH,
                        description=f"::{pseudo} pseudo-element injects text content via CSS. This text is invisible to users but visible to AI parsers.",
                        confidence=0.85,
                    )
                )

    return findings


def _scan_font_face_remapping(
    css_blocks: list[tuple[str, str]],
) -> list[CSSDeobfuscatorFinding]:
    """Detect @font-face character remapping attacks from pre-extracted blocks."""
    findings: list[CSSDeobfuscatorFinding] = []

    for block, _location in css_blocks:
        for match in _FONT_FACE_RE.finditer(block):
            font_rule = match.group(1)
            unicode_match = _UNICODE_RANGE_RE.search(font_rule)
            if unicode_match:
                unicode_range = unicode_match.group(1)
                findings.append(
                    CSSDeobfuscatorFinding(
                        technique="font_face_unicode_range",
                        location="@font-face rule",
                        style_content=f"unicode-range: {unicode_range}",
                        hidden_preview="",
                        severity=CSSDeobfuscatorSeverity.MEDIUM,
                        description="@font-face with unicode-range may use character remapping for homoglyph attacks (Xiong et al. malicious font attack). Character rendering may differ from what AI reads.",
                        confidence=0.70,
                    )
                )

    return findings


def detect_css_deobfuscate(html: str) -> list[CSSDeobfuscatorFinding]:
    """Detect CSS-based content hiding and obfuscation.

    Analyzes CSS in HTML documents for:
    - Text hiding techniques (text-indent, clip-path, etc.)
    - Pseudo-element content injection
    - Font-based character remapping attacks
    - Off-screen positioning

    Args:
        html: The HTML content containing CSS to analyze.

    Returns:
        List of CSSDeobfuscatorFinding objects for each detected technique.
    """
    findings: list[CSSDeobfuscatorFinding] = []

    if not html or not html.strip():
        return findings

    # Extract CSS blocks
    css_blocks = _extract_css_blocks(html)

    # Scan each block with pre-compiled patterns (blocks extracted once above)
    for css_content, location in css_blocks:
        findings.extend(_scan_css_for_patterns(css_content, location, _CSS_HIDING_PATTERNS_COMPILED))
        findings.extend(_scan_css_for_patterns(css_content, location, _PSEUDO_PATTERNS_COMPILED))
        findings.extend(_scan_css_for_patterns(css_content, location, _FONT_PATTERNS_COMPILED))

    # Scan inline styles
    for style_value, location, _offset in _extract_inline_styles(html):
        findings.extend(_scan_css_for_patterns(style_value, location, _CSS_HIDING_PATTERNS_COMPILED))

    # Detect pseudo-element content injection (reuse already-extracted blocks)
    findings.extend(_scan_for_pseudo_element_injection(css_blocks))

    # Detect font-face remapping (reuse already-extracted blocks)
    findings.extend(_scan_font_face_remapping(css_blocks))

    # Check for <link rel="stylesheet"> with href
    external_styles = _EXTERNAL_LINK_RE.findall(html)
    if external_styles:
        findings.append(
            CSSDeobfuscatorFinding(
                technique="external_stylesheet",
                location="external <link> stylesheet(s)",
                style_content=f"{len(external_styles)} external stylesheet(s): {', '.join(external_styles[:3])}",
                hidden_preview="[requires external fetch]",
                severity=CSSDeobfuscatorSeverity.LOW,
                description="External stylesheets referenced - full hiding analysis requires fetching external CSS.",
                confidence=0.30,
            )
        )

    return findings
