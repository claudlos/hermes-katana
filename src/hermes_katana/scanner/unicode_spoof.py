"""
Advanced Unicode spoof scanner for HermesKatana.

Extends unicode.py with additional detection techniques:
- Combining character abuse (stacked diacritics)
- Variation selectors used for fingerprinting/steganography
- Mathematical alphanumeric symbols (U+1D400-U+1D7FF) bypassing ASCII filters
- Enclosed alphanumerics (①②③ / Ⓐⓑ) bypassing ASCII filters
- Full bidi attack detection (re-exports from unicode.py)
- Mixed-script confusables (re-exports from unicode.py)
- Invisible characters (re-exports from unicode.py)
- Tag characters (re-exports from unicode.py)

Performance: O(n) single-pass scanning with precompiled patterns.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .unicode import (
    UnicodeFinding,
    normalize_and_scan,
)

__all__ = [
    "SpoofCategory",
    "SpoofSeverity",
    "SpoofFinding",
    "scan_unicode_spoof",
    "scan_unicode_spoof_full",
    "VARIATION_SELECTOR_RANGES",
    "MATH_ALPHA_RANGE",
    "ENCLOSED_ALPHA_RANGES",
    "normalize_spoof",
]


class SpoofCategory(str, Enum):
    """Categories of advanced Unicode spoof attacks."""

    COMBINING_ABUSE = "combining_abuse"
    """Excessive stacked combining/diacritic marks (zalgo-style)."""

    VARIATION_SELECTOR = "variation_selector"
    """Variation selector characters used for fingerprinting or steganography."""

    MATH_ALPHA = "math_alpha"
    """Mathematical alphanumeric symbols (𝐀𝐁𝐂) bypassing ASCII filters."""

    ENCLOSED_ALPHA = "enclosed_alpha"
    """Enclosed alphanumerics (①②Ⓐⓑ) bypassing ASCII filters."""

    BIDI_OVERRIDE = "bidi_override"
    """Bidirectional override (re-detected from unicode.py for unified reporting)."""

    MIXED_SCRIPT = "mixed_script"
    """Mixed-script confusables (re-detected from unicode.py)."""

    INVISIBLE = "invisible"
    """Invisible characters: zero-width, BOM, soft-hyphen, word-joiner."""

    TAG_CHAR = "tag_char"
    """Unicode Tags block (U+E0001-U+E007F) — invisible LLM instructions."""


class SpoofSeverity(str, Enum):
    """Severity levels for spoof findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class SpoofFinding:
    """A single advanced Unicode spoof finding.

    Attributes:
        category: The type of spoof attack detected.
        severity: How dangerous this finding is.
        description: Human-readable explanation of the finding.
        position: (start, end) character positions in the original text.
        matched_text: The problematic text (repr form for invisible chars).
        char_names: Unicode names of the problematic characters.
        recommendation: Suggested action to take.
        count: Number of offending characters in this finding.
    """

    category: SpoofCategory
    severity: SpoofSeverity
    description: str
    position: tuple[int, int]
    matched_text: str
    char_names: list[str] = field(default_factory=list)
    recommendation: str = "Strip or reject this input"
    count: int = 1


# ---------------------------------------------------------------------------
# Variation selectors
# VS1-VS16:   U+FE00–U+FE0F  (text/emoji presentation selectors)
# VS17-VS256: U+E0100–U+E01EF (ideographic variation selectors)
# These are legitimate for emoji/CJK but can be abused for fingerprinting
# by embedding arbitrary bit patterns invisible to humans.
# ---------------------------------------------------------------------------
VARIATION_SELECTOR_RANGES: list[tuple[int, int, str]] = [
    (0xFE00, 0xFE0F, "VS1-VS16 (presentation selectors)"),
    (0xE0100, 0xE01EF, "VS17-VS256 (ideographic variation selectors)"),
]

# Precompile pattern for VS1-VS16 (BMP range, easy to include in char class)
_VS_BMP_PATTERN = re.compile("[\ufe00-\ufe0f]")
# VS17-VS256 are in supplementary planes; check via ordinal in a loop


def _has_variation_selector(cp: int) -> bool:
    """Return True if code point is a variation selector."""
    return (0xFE00 <= cp <= 0xFE0F) or (0xE0100 <= cp <= 0xE01EF)


# ---------------------------------------------------------------------------
# Mathematical alphanumeric symbols: U+1D400–U+1D7FF
# Includes math bold, italic, fraktur, script, double-struck, monospace, etc.
# All visually resemble Latin/Greek letters but are distinct code points.
# Attack: use 𝐩𝐚𝐬𝐬𝐰𝐨𝐫𝐝 instead of "password" to bypass keyword filters.
# ---------------------------------------------------------------------------
MATH_ALPHA_RANGE = (0x1D400, 0x1D7FF)


# Approximate mapping from math alpha block to base ASCII (a-z, A-Z, 0-9).
# The block has 26-letter alphabets in ~16 styles; we normalise to ASCII.
# Reference: Unicode standard, block "Mathematical Alphanumeric Symbols"
def _math_alpha_to_ascii(cp: int) -> Optional[str]:
    """Map a math alphanumeric code point to its ASCII base character."""
    if not (0x1D400 <= cp <= 0x1D7FF):
        return None
    # Letter ranges: each alphabet is 26 upper + 26 lower = 52 chars
    # Multiple alphabets packed sequentially (bold, italic, bold-italic,
    # script, bold-script, fraktur, double-struck, bold-fraktur, sans-serif,
    # sans-serif-bold, sans-serif-italic, sans-serif-bold-italic, monospace)
    # Offsets for uppercase alphabets within the block
    upper_starts = [
        0x1D400,  # Bold capital A
        0x1D434,  # Italic capital A
        0x1D468,  # Bold italic capital A
        0x1D49C,  # Script capital A (sparse — some use other blocks)
        0x1D4D0,  # Bold script capital A
        0x1D504,  # Fraktur capital A (sparse)
        0x1D538,  # Double-struck capital A (sparse)
        0x1D56C,  # Bold fraktur capital A
        0x1D5A0,  # Sans-serif capital A
        0x1D5D4,  # Sans-serif bold capital A
        0x1D608,  # Sans-serif italic capital A
        0x1D63C,  # Sans-serif bold italic capital A
        0x1D670,  # Monospace capital A
    ]
    lower_starts = [
        0x1D41A,  # Bold small a
        0x1D44E,  # Italic small a
        0x1D482,  # Bold italic small a
        0x1D4B6,  # Script small a (sparse)
        0x1D4EA,  # Bold script small a
        0x1D51E,  # Fraktur small a (sparse)
        0x1D552,  # Double-struck small a (sparse)
        0x1D586,  # Bold fraktur small a
        0x1D5BA,  # Sans-serif small a
        0x1D5EE,  # Sans-serif bold small a
        0x1D622,  # Sans-serif italic small a
        0x1D656,  # Sans-serif bold italic small a
        0x1D68A,  # Monospace small a
    ]
    for start in upper_starts:
        if start <= cp < start + 26:
            return chr(ord("A") + (cp - start))
    for start in lower_starts:
        if start <= cp < start + 26:
            return chr(ord("a") + (cp - start))
    # Digit ranges (bold, double-struck, sans-serif, sans-serif-bold, monospace)
    digit_starts = [0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6]
    for start in digit_starts:
        if start <= cp < start + 10:
            return chr(ord("0") + (cp - start))
    return None


# ---------------------------------------------------------------------------
# Enclosed alphanumerics
# U+2460–U+24FF  : ①②③ ... Ⓐⓑⓒ ... ⑴⑵ ...
# U+1F100–U+1F1FF: 🄀🄁 ... 🅐🅑 ...
# U+2776–U+2793  : ❶❷❸ (dingbat filled)
# These look like ordinary letters/numbers but are separate code points.
# ---------------------------------------------------------------------------
ENCLOSED_ALPHA_RANGES: list[tuple[int, int, str]] = [
    (0x2460, 0x24FF, "Enclosed Alphanumerics"),
    (0x1F100, 0x1F1FF, "Enclosed Alphanumeric Supplement"),
    (0x2776, 0x2793, "Dingbat enclosed numbers"),
]

# Build pattern for BMP enclosed range
_ENCLOSED_BMP_PATTERN = re.compile("[\u2460-\u24ff\u2776-\u2793]")


def _is_enclosed_alpha(cp: int) -> bool:
    """Return True if code point is an enclosed alphanumeric."""
    return (0x2460 <= cp <= 0x24FF) or (0x1F100 <= cp <= 0x1F1FF) or (0x2776 <= cp <= 0x2793)


# Approximate mapping for the most common enclosed digits/letters
def _enclosed_to_ascii(cp: int) -> Optional[str]:
    """Map an enclosed alphanumeric to its base character where possible."""
    # Circled digits 1-20: U+2460-U+2473
    if 0x2460 <= cp <= 0x2473:
        return str(cp - 0x2460 + 1) if (cp - 0x2460 + 1) <= 9 else None
    # Circled uppercase A-Z: U+24B6-U+24CF
    if 0x24B6 <= cp <= 0x24CF:
        return chr(ord("A") + (cp - 0x24B6))
    # Circled lowercase a-z: U+24D0-U+24E9
    if 0x24D0 <= cp <= 0x24E9:
        return chr(ord("a") + (cp - 0x24D0))
    return None


# ---------------------------------------------------------------------------
# Combining character abuse (Zalgo-style / stacked diacritics)
# Normal text: 0-2 combining marks per base char.
# Attack text: 5+ combining marks per base char (zalgo/leet obfuscation).
# We report runs of 4+ consecutive combining characters.
# ---------------------------------------------------------------------------
_COMBINING_THRESHOLD = 4  # combining marks in a row = suspicious


def _is_combining(ch: str) -> bool:
    """Return True if character is a combining/diacritic mark."""
    cat = unicodedata.category(ch)
    return cat in ("Mn", "Mc", "Me")  # non-spacing, spacing-combining, enclosing


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def _detect_combining_abuse(text: str) -> list[SpoofFinding]:
    """Detect excessive stacked combining characters (zalgo / diacritic abuse).

    A run of 4+ consecutive combining marks on a single base character is
    indicative of obfuscation or filter-bypass attempts.
    """
    findings: list[SpoofFinding] = []
    run_start = -1
    run_count = 0
    run_chars: list[str] = []

    for i, ch in enumerate(text):
        if _is_combining(ch):
            if run_start == -1:
                run_start = i
            run_count += 1
            run_chars.append(ch)
        else:
            if run_count >= _COMBINING_THRESHOLD:
                severity = SpoofSeverity.HIGH if run_count >= 8 else SpoofSeverity.MEDIUM
                try:
                    names = [unicodedata.name(c, f"U+{ord(c):04X}") for c in run_chars[:5]]
                except Exception:
                    names = [f"U+{ord(c):04X}" for c in run_chars[:5]]
                findings.append(
                    SpoofFinding(
                        category=SpoofCategory.COMBINING_ABUSE,
                        severity=severity,
                        description=(
                            f"Combining character abuse: {run_count} stacked diacritic "
                            f"marks at position {run_start}. Excessive combining marks "
                            "are used for zalgo-style obfuscation or to corrupt filter "
                            "keyword matching."
                        ),
                        position=(run_start, i),
                        matched_text=repr(text[run_start:i]),
                        char_names=names,
                        recommendation="Strip combining characters beyond 1 per base char",
                        count=run_count,
                    )
                )
            run_start = -1
            run_count = 0
            run_chars = []

    # Handle trailing run
    if run_count >= _COMBINING_THRESHOLD:
        severity = SpoofSeverity.HIGH if run_count >= 8 else SpoofSeverity.MEDIUM
        try:
            names = [unicodedata.name(c, f"U+{ord(c):04X}") for c in run_chars[:5]]
        except Exception:
            names = [f"U+{ord(c):04X}" for c in run_chars[:5]]
        end = len(text)
        findings.append(
            SpoofFinding(
                category=SpoofCategory.COMBINING_ABUSE,
                severity=severity,
                description=(
                    f"Combining character abuse: {run_count} stacked diacritic "
                    f"marks at position {run_start}. Excessive combining marks "
                    "are used for zalgo-style obfuscation or to corrupt filter "
                    "keyword matching."
                ),
                position=(run_start, end),
                matched_text=repr(text[run_start:end]),
                char_names=names,
                recommendation="Strip combining characters beyond 1 per base char",
                count=run_count,
            )
        )
    return findings


def _detect_variation_selectors(text: str) -> list[SpoofFinding]:
    """Detect variation selector characters used for fingerprinting.

    Variation selectors are invisible characters that change glyph appearance.
    When used outside their intended purpose (emoji/CJK presentation),
    they can encode hidden bits in apparently normal text — allowing unique
    fingerprinting of every copy of a document.
    """
    findings: list[SpoofFinding] = []
    positions: list[int] = []

    for i, ch in enumerate(text):
        cp = ord(ch)
        if _has_variation_selector(cp):
            positions.append(i)

    if not positions:
        return []

    # Group into a single finding (fingerprint is document-level)
    count = len(positions)
    severity = SpoofSeverity.HIGH if count >= 8 else SpoofSeverity.MEDIUM if count >= 3 else SpoofSeverity.LOW
    # Identify which ranges are present
    bmp_count = sum(1 for i in positions if 0xFE00 <= ord(text[i]) <= 0xFE0F)
    sup_count = count - bmp_count
    range_desc = []
    if bmp_count:
        range_desc.append(f"{bmp_count} VS1-VS16 (U+FE00-U+FE0F)")
    if sup_count:
        range_desc.append(f"{sup_count} VS17-VS256 (U+E0100-U+E01EF)")

    # Check if the VS appear outside emoji/CJK context (simple heuristic:
    # if the preceding char is ASCII, the VS is almost certainly abuse)
    abuse_count = 0
    for pos in positions:
        if pos > 0 and ord(text[pos - 1]) < 0x2E80:
            # Base char is ASCII/Latin — no legitimate VS use here
            abuse_count += 1

    findings.append(
        SpoofFinding(
            category=SpoofCategory.VARIATION_SELECTOR,
            severity=severity,
            description=(
                f"Variation selector characters detected: {count} occurrence(s) "
                f"({', '.join(range_desc)}). "
                f"{abuse_count}/{count} appear after non-CJK/emoji base characters, "
                "indicating potential document fingerprinting or steganographic payload."
            ),
            position=(positions[0], positions[-1] + 1),
            matched_text=repr("".join(text[p] for p in positions[:8])),
            char_names=[f"U+{ord(text[p]):04X}" for p in positions[:5]],
            recommendation=(
                "Strip variation selectors from non-emoji/non-CJK contexts. "
                "Presence in ASCII text is a strong fingerprinting indicator."
            ),
            count=count,
        )
    )
    return findings


def _detect_math_alpha(text: str) -> list[SpoofFinding]:
    """Detect mathematical alphanumeric symbols (U+1D400-U+1D7FF).

    These look like ordinary Latin/Greek letters (bold, italic, fraktur, etc.)
    but are different code points. Attackers use them to spell out injection
    keywords (e.g., 𝐢𝐠𝐧𝐨𝐫𝐞 𝐩𝐫𝐞𝐯𝐢𝐨𝐮𝐬) while bypassing ASCII-based filters.
    """
    findings: list[SpoofFinding] = []
    positions: list[int] = []
    decoded_chars: list[str] = []

    for i, ch in enumerate(text):
        cp = ord(ch)
        if MATH_ALPHA_RANGE[0] <= cp <= MATH_ALPHA_RANGE[1]:
            positions.append(i)
            ascii_ch = _math_alpha_to_ascii(cp)
            decoded_chars.append(ascii_ch if ascii_ch else "?")

    if not positions:
        return []

    count = len(positions)
    decoded_preview = "".join(decoded_chars[:40])
    severity = SpoofSeverity.HIGH if count >= 5 else SpoofSeverity.MEDIUM if count >= 2 else SpoofSeverity.LOW

    findings.append(
        SpoofFinding(
            category=SpoofCategory.MATH_ALPHA,
            severity=severity,
            description=(
                f"Mathematical alphanumeric symbols detected: {count} character(s) "
                f"(U+1D400-U+1D7FF). These visually resemble Latin letters but bypass "
                f"ASCII-based keyword filters. Decoded content: '{decoded_preview}'"
            ),
            position=(positions[0], positions[-1] + 1),
            matched_text=repr("".join(text[p] for p in positions[:8])),
            char_names=[f"U+{ord(text[p]):05X}" for p in positions[:5]],
            recommendation=("Normalize math alphanumeric symbols to their ASCII equivalents before keyword scanning."),
            count=count,
        )
    )
    return findings


def _detect_enclosed_alpha(text: str) -> list[SpoofFinding]:
    """Detect enclosed alphanumeric characters (①②③ / Ⓐⓑ / 🅐🅑).

    These represent digits and letters inside circles, squares, or parentheses.
    Attackers use them to spell out words while bypassing ASCII filters.
    Example: 'ⓘⓖⓝⓞⓡⓔ ⓟⓡⓔⓥⓘⓞⓤⓢ' decodes to 'ignore previous'.
    """
    findings: list[SpoofFinding] = []
    positions: list[int] = []
    decoded_chars: list[str] = []

    for i, ch in enumerate(text):
        cp = ord(ch)
        if _is_enclosed_alpha(cp):
            positions.append(i)
            ascii_ch = _enclosed_to_ascii(cp)
            decoded_chars.append(ascii_ch if ascii_ch else "?")

    if not positions:
        return []

    count = len(positions)
    decoded_preview = "".join(decoded_chars[:40])
    severity = SpoofSeverity.HIGH if count >= 5 else SpoofSeverity.MEDIUM if count >= 2 else SpoofSeverity.LOW

    findings.append(
        SpoofFinding(
            category=SpoofCategory.ENCLOSED_ALPHA,
            severity=severity,
            description=(
                f"Enclosed alphanumeric characters detected: {count} character(s). "
                f"These visually represent letters/digits (①②Ⓐⓑ) but bypass ASCII "
                f"keyword filters. Decoded content: '{decoded_preview}'"
            ),
            position=(positions[0], positions[-1] + 1),
            matched_text=repr("".join(text[p] for p in positions[:8])),
            char_names=[f"U+{ord(text[p]):04X}" for p in positions[:5]],
            recommendation=("Normalize enclosed alphanumerics to their base characters before keyword scanning."),
            count=count,
        )
    )
    return findings


# ---------------------------------------------------------------------------
# Normalize spoof characters to ASCII equivalents
# ---------------------------------------------------------------------------


def normalize_spoof(text: str) -> str:
    """Normalize advanced spoof characters in text.

    In addition to the normalization in unicode.py, this:
    - Replaces math alphanumeric symbols with ASCII equivalents
    - Replaces enclosed alphanumerics with ASCII equivalents
    - Strips variation selectors
    - Strips excessive combining characters (keeps at most 1 per base)

    Call after normalize_text() from unicode.py for full normalization.
    """
    result: list[str] = []
    prev_was_base = False
    combining_count = 0

    for ch in text:
        cp = ord(ch)

        # Strip variation selectors
        if _has_variation_selector(cp):
            continue

        # Math alphanumeric → ASCII
        if MATH_ALPHA_RANGE[0] <= cp <= MATH_ALPHA_RANGE[1]:
            ascii_ch = _math_alpha_to_ascii(cp)
            result.append(ascii_ch if ascii_ch else ch)
            prev_was_base = True
            combining_count = 0
            continue

        # Enclosed alphanumeric → ASCII
        if _is_enclosed_alpha(cp):
            ascii_ch = _enclosed_to_ascii(cp)
            result.append(ascii_ch if ascii_ch else ch)
            prev_was_base = True
            combining_count = 0
            continue

        # Combining character abuse: keep only 1 per base char
        if _is_combining(ch):
            if prev_was_base and combining_count == 0:
                result.append(ch)
                combining_count += 1
            # Else: drop additional combining chars
            continue

        # Normal character
        result.append(ch)
        prev_was_base = True
        combining_count = 0

    return "".join(result)


# ---------------------------------------------------------------------------
# Main scanner entry points
# ---------------------------------------------------------------------------


def scan_unicode_spoof(text: str) -> list[SpoofFinding]:
    """Scan for advanced Unicode spoof techniques only (new detectors).

    Covers:
    - Combining character abuse
    - Variation selectors
    - Mathematical alphanumeric symbols
    - Enclosed alphanumerics

    For full Unicode scanning (including bidi, zero-width, homoglyphs from
    the base scanner), use scan_unicode_spoof_full().

    Args:
        text: Input text to scan.

    Returns:
        List of SpoofFinding objects sorted by position.
    """
    if not text:
        return []

    findings: list[SpoofFinding] = []
    findings.extend(_detect_combining_abuse(text))
    findings.extend(_detect_variation_selectors(text))
    findings.extend(_detect_math_alpha(text))
    findings.extend(_detect_enclosed_alpha(text))

    findings.sort(key=lambda f: f.position[0])
    return findings


def scan_unicode_spoof_full(text: str) -> tuple[str, list[UnicodeFinding], list[SpoofFinding]]:
    """Full Unicode spoof scan: base scanner + advanced techniques.

    Runs both the base unicode.py scanner and the new advanced spoof detectors.

    Args:
        text: Input text to scan.

    Returns:
        Tuple of (normalized_text, base_findings, spoof_findings).
        normalized_text has had both base and spoof normalization applied.
    """
    if not text:
        return "", [], []

    # Base scan (bidi, zero-width, homoglyphs, mixed-script, tags)
    normalized, base_findings = normalize_and_scan(text)

    # Advanced spoof scan on original text
    spoof_findings = scan_unicode_spoof(text)

    # Apply spoof normalization on top of base normalization
    fully_normalized = normalize_spoof(normalized)

    return fully_normalized, base_findings, spoof_findings
