"""
Unicode attack detector for HermesKatana.

Detects malicious use of Unicode characters including:
- Bidirectional text overrides (source code trojan attacks, CVE-2021-42574)
- Zero-width characters (invisible payload smuggling)
- Homoglyph attacks (visual spoofing of identifiers/URLs)
- Invisible/control characters (hidden commands/text)
- Mixed-script attacks (Cyrillic+Latin confusion)

Performance: O(n) single-pass scanning with precompiled lookup tables.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "UnicodeCategory",
    "UnicodeSeverity",
    "UnicodeFinding",
    "normalize_text",
    "normalize_and_scan",
    "scan_unicode",
    "BIDI_CHARS",
    "ZERO_WIDTH_CHARS",
    "CONFUSABLE_MAP",
    "BIDI_PATTERN",
    "ZERO_WIDTH_PATTERN",
    "SCRIPT_GROUPS",
    "SCRIPT_NEUTRAL_CATEGORIES",
    "SUSPICIOUS_CONTROL_RANGES",
]


class UnicodeCategory(str, Enum):
    """Categories of Unicode-based attacks."""

    BIDI_OVERRIDE = "bidi_override"
    """Bidirectional text override characters that can reorder displayed text."""

    ZERO_WIDTH = "zero_width"
    """Zero-width characters that are invisible but present in text."""

    HOMOGLYPH = "homoglyph"
    """Characters that visually resemble other characters from different scripts."""

    INVISIBLE_CHAR = "invisible_char"
    """Control or formatting characters that are not visible."""

    MIXED_SCRIPT = "mixed_script"
    """Text mixing characters from multiple scripts (e.g., Cyrillic + Latin)."""

    CONTROL_CHAR = "control_char"
    """Unexpected control characters in text content."""


class UnicodeSeverity(str, Enum):
    """Severity levels for Unicode findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class UnicodeFinding:
    """A single Unicode attack finding.

    Attributes:
        category: The type of Unicode attack detected.
        severity: How dangerous this finding is.
        description: Human-readable explanation of the finding.
        position: (start, end) character positions in the original text.
        matched_text: The problematic text (repr form for invisible chars).
        char_names: Unicode names of the problematic characters.
        recommendation: Suggested action to take.
    """

    category: UnicodeCategory
    severity: UnicodeSeverity
    description: str
    position: tuple[int, int]
    matched_text: str
    char_names: list[str] = field(default_factory=list)
    recommendation: str = "Strip or reject this input"


# ---------------------------------------------------------------------------
# Bidirectional override characters
# These can reorder displayed text, hiding malicious code in source files.
# See: CVE-2021-42574 "Trojan Source" attack
# ---------------------------------------------------------------------------
BIDI_CHARS: dict[int, str] = {
    0x200E: "LEFT-TO-RIGHT MARK (LRM)",
    0x200F: "RIGHT-TO-LEFT MARK (RLM)",
    0x200B: "ZERO WIDTH SPACE",  # Also zero-width, but bidi-relevant
    0x202A: "LEFT-TO-RIGHT EMBEDDING (LRE)",
    0x202B: "RIGHT-TO-LEFT EMBEDDING (RLE)",
    0x202C: "POP DIRECTIONAL FORMATTING (PDF)",
    0x202D: "LEFT-TO-RIGHT OVERRIDE (LRO)",
    0x202E: "RIGHT-TO-LEFT OVERRIDE (RLO)",
    0x2066: "LEFT-TO-RIGHT ISOLATE (LRI)",
    0x2067: "RIGHT-TO-LEFT ISOLATE (RLI)",
    0x2068: "FIRST STRONG ISOLATE (FSI)",
    0x2069: "POP DIRECTIONAL ISOLATE (PDI)",
    0x061C: "ARABIC LETTER MARK (ALM)",
}

BIDI_PATTERN = re.compile(
    "[" + "".join(chr(cp) for cp in BIDI_CHARS) + "]"
)

# ---------------------------------------------------------------------------
# Zero-width characters
# Used to smuggle invisible payloads, watermark text, or bypass filters.
# ---------------------------------------------------------------------------
ZERO_WIDTH_CHARS: dict[int, str] = {
    0x200B: "ZERO WIDTH SPACE (ZWSP)",
    0x200C: "ZERO WIDTH NON-JOINER (ZWNJ)",
    0x200D: "ZERO WIDTH JOINER (ZWJ)",
    0xFEFF: "ZERO WIDTH NO-BREAK SPACE (BOM/ZWNBSP)",
    0x2060: "WORD JOINER",
    0x2061: "FUNCTION APPLICATION",
    0x2062: "INVISIBLE TIMES",
    0x2063: "INVISIBLE SEPARATOR",
    0x2064: "INVISIBLE PLUS",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x00AD: "SOFT HYPHEN",
}

ZERO_WIDTH_PATTERN = re.compile(
    "[" + "".join(chr(cp) for cp in ZERO_WIDTH_CHARS) + "]"
)

# ---------------------------------------------------------------------------
# Unicode Tags block (U+E0000–U+E007F)
# CRITICAL: deprecated language-tag characters that are invisible in virtually
# all renderers but fully readable by LLMs. Can encode arbitrary ASCII text.
# Documented attack vector: arXiv:2603.00164 "Reverse CAPTCHA" (Feb 2026).
# U+E0041 = tag 'A', U+E0042 = tag 'B', etc.
# ---------------------------------------------------------------------------
UNICODE_TAGS_START = 0xE0000
UNICODE_TAGS_END = 0xE007F

def _has_unicode_tags(text: str) -> list[int]:
    """Return positions of Unicode Tags block characters in text."""
    return [i for i, c in enumerate(text) if UNICODE_TAGS_START <= ord(c) <= UNICODE_TAGS_END]

def _decode_unicode_tags(text: str) -> str:
    """Decode Unicode Tags block characters to their ASCII equivalents."""
    return "".join(
        chr(ord(c) - UNICODE_TAGS_START) if UNICODE_TAGS_START <= ord(c) <= UNICODE_TAGS_END else c
        for c in text
    )

# ---------------------------------------------------------------------------
# ZWJ binary-encoding detection
# Sequences of 8+ ZWSP/ZWNJ can encode arbitrary binary data (1 byte = 8 chars)
# ---------------------------------------------------------------------------
_ZW_BINARY_CHARS = {"\u200b", "\u200c"}  # ZWSP=0, ZWNJ=1

def _max_zw_binary_run(text: str) -> int:
    """Return the length of the longest consecutive ZW binary-encoding run."""
    max_run = cur = 0
    for c in text:
        if c in _ZW_BINARY_CHARS:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run

# ---------------------------------------------------------------------------
# Comprehensive homoglyph / confusable character mapping
# Maps visually similar characters from non-Latin scripts to their Latin
# equivalents. Used to detect visual spoofing in URLs, identifiers, etc.
# ---------------------------------------------------------------------------
CONFUSABLE_MAP: dict[str, str] = {
    # Cyrillic -> Latin (most dangerous: entire words can be spoofed)
    "\u0410": "A",  # А -> A
    "\u0412": "B",  # В -> B
    "\u0421": "C",  # С -> C
    "\u0415": "E",  # Е -> E
    "\u041D": "H",  # Н -> H
    "\u041A": "K",  # К -> K
    "\u041C": "M",  # М -> M
    "\u041E": "O",  # О -> O
    "\u0420": "P",  # Р -> P
    "\u0422": "T",  # Т -> T
    "\u0425": "X",  # Х -> X
    "\u0430": "a",  # а -> a
    "\u0435": "e",  # е -> e
    "\u043E": "o",  # о -> o
    "\u0440": "p",  # р -> p
    "\u0441": "c",  # с -> c
    "\u0443": "y",  # у -> y
    "\u0445": "x",  # х -> x
    "\u04BB": "h",  # һ -> h
    "\u0456": "i",  # і -> i
    "\u0458": "j",  # ј -> j
    "\u0455": "s",  # ѕ -> s
    "\u04CF": "l",  # ӏ -> l
    "\u051B": "q",  # ԛ -> q
    "\u051D": "w",  # ԝ -> w
    # Greek -> Latin
    "\u0391": "A",  # Α -> A
    "\u0392": "B",  # Β -> B
    "\u0395": "E",  # Ε -> E
    "\u0396": "Z",  # Ζ -> Z
    "\u0397": "H",  # Η -> H
    "\u0399": "I",  # Ι -> I
    "\u039A": "K",  # Κ -> K
    "\u039C": "M",  # Μ -> M
    "\u039D": "N",  # Ν -> N
    "\u039F": "O",  # Ο -> O
    "\u03A1": "P",  # Ρ -> P
    "\u03A4": "T",  # Τ -> T
    "\u03A5": "Y",  # Υ -> Y
    "\u03A7": "X",  # Χ -> X
    "\u03BF": "o",  # ο -> o
    "\u03B1": "a",  # α -> a  (less similar but used in attacks)
    # Fullwidth Latin (used to bypass ASCII filters)
    "\uFF21": "A",
    "\uFF22": "B",
    "\uFF23": "C",
    "\uFF24": "D",
    "\uFF25": "E",
    "\uFF26": "F",
    "\uFF27": "G",
    "\uFF28": "H",
    "\uFF29": "I",
    "\uFF2A": "J",
    "\uFF2B": "K",
    "\uFF2C": "L",
    "\uFF2D": "M",
    "\uFF2E": "N",
    "\uFF2F": "O",
    "\uFF30": "P",
    "\uFF31": "Q",
    "\uFF32": "R",
    "\uFF33": "S",
    "\uFF34": "T",
    "\uFF35": "U",
    "\uFF36": "V",
    "\uFF37": "W",
    "\uFF38": "X",
    "\uFF39": "Y",
    "\uFF3A": "Z",
    # Mathematical/special lookalikes
    "\u2219": ".",  # ∙ -> .
    "\u2024": ".",  # ․ -> .
    "\u2215": "/",  # ∕ -> /
    "\u2044": "/",  # ⁄ -> /
    "\u2236": ":",  # ∶ -> :
    "\uFE68": "\\",  # ﹨ -> \
    "\uFF0F": "/",  # ／ -> /
    "\uFF3C": "\\",  # ＼ -> \
    # Common number confusables
    "\u04E0": "3",  # Ӡ (looks like 3 in some fonts)
    "\u01B7": "3",  # Ʒ -> 3
    "\u0417": "3",  # З -> 3
}

# Reverse lookup: build set of confusable codepoints for fast detection
_CONFUSABLE_CODEPOINTS = frozenset(ord(c) for c in CONFUSABLE_MAP)

# ---------------------------------------------------------------------------
# Script categories for mixed-script detection
# ---------------------------------------------------------------------------
SCRIPT_GROUPS: dict[str, set[str]] = {
    "Latin": {"LATIN"},
    "Cyrillic": {"CYRILLIC"},
    "Greek": {"GREEK"},
    "Armenian": {"ARMENIAN"},
    "Georgian": {"GEORGIAN"},
    "Cherokee": {"CHEROKEE"},
}

# Characters that are common across scripts and should be ignored
# in mixed-script analysis (digits, punctuation, symbols)
SCRIPT_NEUTRAL_CATEGORIES = frozenset({
    "Nd",  # Decimal number
    "Pc",  # Connector punctuation
    "Pd",  # Dash punctuation
    "Pe",  # Close punctuation
    "Pf",  # Final punctuation
    "Pi",  # Initial punctuation
    "Po",  # Other punctuation
    "Ps",  # Open punctuation
    "Sc",  # Currency symbol
    "Sk",  # Modifier symbol
    "Sm",  # Math symbol
    "So",  # Other symbol
    "Zs",  # Space separator
    "Zl",  # Line separator
    "Zp",  # Paragraph separator
    "Cc",  # Control
    "Cf",  # Format
})

# ---------------------------------------------------------------------------
# Control characters that should not appear in normal text
# ---------------------------------------------------------------------------
SUSPICIOUS_CONTROL_RANGES: list[tuple[int, int, str]] = [
    (0x0000, 0x0008, "NULL/control block"),  # NUL through BS (except TAB/LF/CR)
    (0x000B, 0x000B, "Vertical tab"),
    (0x000E, 0x001F, "Control block (SO through US)"),
    (0x007F, 0x007F, "DELETE"),
    (0x0080, 0x009F, "C1 control block"),
    (0xFFF0, 0xFFF8, "Specials block"),
    (0xFFFE, 0xFFFF, "Non-characters"),
]


def _get_script(char: str) -> Optional[str]:
    """Get the Unicode script name for a character.

    Returns None for characters in neutral categories (digits, punctuation).
    """
    if unicodedata.category(char) in SCRIPT_NEUTRAL_CATEGORIES:
        return None
    try:
        name = unicodedata.name(char, "")
    except ValueError:
        return None
    # Extract script from character name (first word is usually the script)
    for script_name, prefixes in SCRIPT_GROUPS.items():
        for prefix in prefixes:
            if name.startswith(prefix):
                return script_name
    if name.startswith("CJK"):
        return "CJK"
    return "Other"


def _detect_bidi(text: str) -> list[UnicodeFinding]:
    """Detect bidirectional override characters.

    These characters can reorder displayed text, enabling "Trojan Source"
    attacks where code appears benign but executes maliciously.
    """
    findings: list[UnicodeFinding] = []
    for match in BIDI_PATTERN.finditer(text):
        cp = ord(match.group())
        char_name = BIDI_CHARS.get(cp, f"U+{cp:04X}")
        findings.append(UnicodeFinding(
            category=UnicodeCategory.BIDI_OVERRIDE,
            severity=UnicodeSeverity.CRITICAL,
            description=(
                f"Bidirectional override character detected: {char_name}. "
                "This can reorder displayed text to hide malicious content "
                "(Trojan Source attack, CVE-2021-42574)."
            ),
            position=(match.start(), match.end()),
            matched_text=repr(match.group()),
            char_names=[char_name],
            recommendation="Strip bidirectional override characters from input",
        ))
    return findings


def _detect_zero_width(text: str) -> list[UnicodeFinding]:
    """Detect zero-width characters.

    Zero-width characters are invisible but can:
    - Smuggle hidden payloads past filters
    - Watermark text to track leaks
    - Split keywords to bypass pattern matching
    - Create visually identical but technically different strings
    """
    findings: list[UnicodeFinding] = []
    # Group consecutive zero-width chars for cleaner reporting
    runs: list[tuple[int, int, list[str]]] = []
    current_start = -1

    for match in ZERO_WIDTH_PATTERN.finditer(text):
        cp = ord(match.group())
        char_name = ZERO_WIDTH_CHARS.get(cp, f"U+{cp:04X}")

        if current_start >= 0 and match.start() == runs[-1][1]:
            # Extend current run
            runs[-1] = (runs[-1][0], match.end(), runs[-1][2] + [char_name])
        else:
            runs.append((match.start(), match.end(), [char_name]))

    for start, end, names in runs:
        count = len(names)
        severity = (
            UnicodeSeverity.HIGH if count > 3
            else UnicodeSeverity.MEDIUM if count > 1
            else UnicodeSeverity.LOW
        )
        findings.append(UnicodeFinding(
            category=UnicodeCategory.ZERO_WIDTH,
            severity=severity,
            description=(
                f"{'Cluster of ' + str(count) + ' z' if count > 1 else 'Z'}"
                f"ero-width character{'s' if count > 1 else ''} detected. "
                "These invisible characters can smuggle hidden content, "
                "bypass keyword filters, or watermark text."
            ),
            position=(start, end),
            matched_text=repr(text[start:end]),
            char_names=names,
            recommendation="Strip zero-width characters from input",
        ))
    return findings


def _detect_homoglyphs(text: str) -> list[UnicodeFinding]:
    """Detect homoglyph/confusable characters.

    Homoglyphs are characters from different scripts that look visually
    identical or very similar. Attackers use them to create convincing
    visual spoofs of URLs, identifiers, or commands.

    Example: "gооgle.com" with Cyrillic 'о' instead of Latin 'o'.
    """
    findings: list[UnicodeFinding] = []
    for i, ch in enumerate(text):
        if ord(ch) in _CONFUSABLE_CODEPOINTS:
            latin_equiv = CONFUSABLE_MAP[ch]
            try:
                char_name = unicodedata.name(ch, f"U+{ord(ch):04X}")
            except ValueError:
                char_name = f"U+{ord(ch):04X}"
            findings.append(UnicodeFinding(
                category=UnicodeCategory.HOMOGLYPH,
                severity=UnicodeSeverity.HIGH,
                description=(
                    f"Confusable character '{ch}' ({char_name}) "
                    f"looks like Latin '{latin_equiv}'. "
                    "This can be used for visual spoofing attacks on URLs, "
                    "identifiers, or commands."
                ),
                position=(i, i + 1),
                matched_text=ch,
                char_names=[char_name],
                recommendation=(
                    f"Replace with Latin equivalent '{latin_equiv}' "
                    "or reject input"
                ),
            ))
    return findings


def _detect_invisible_chars(text: str) -> list[UnicodeFinding]:
    """Detect invisible and control characters.

    Control characters (except TAB, LF, CR) should not appear in normal
    text input. Their presence may indicate:
    - Smuggled commands
    - Terminal injection attacks
    - Data corruption
    - Deliberate filter evasion
    """
    findings: list[UnicodeFinding] = []
    allowed = {0x09, 0x0A, 0x0D}  # TAB, LF, CR

    for i, ch in enumerate(text):
        cp = ord(ch)
        if cp in allowed:
            continue

        for range_start, range_end, range_name in SUSPICIOUS_CONTROL_RANGES:
            if range_start <= cp <= range_end:
                try:
                    char_name = unicodedata.name(ch, f"U+{cp:04X}")
                except ValueError:
                    char_name = f"U+{cp:04X}"
                findings.append(UnicodeFinding(
                    category=UnicodeCategory.CONTROL_CHAR,
                    severity=UnicodeSeverity.MEDIUM,
                    description=(
                        f"Suspicious control character U+{cp:04X} "
                        f"({range_name}) at position {i}. "
                        "Control characters in text input may indicate "
                        "terminal injection or filter evasion."
                    ),
                    position=(i, i + 1),
                    matched_text=repr(ch),
                    char_names=[char_name],
                    recommendation="Strip control characters from input",
                ))
                break
    return findings


def _detect_mixed_script(text: str) -> list[UnicodeFinding]:
    """Detect mixed-script text (e.g., Cyrillic characters in Latin text).

    Mixing scripts within a single word or identifier is a strong signal
    of a homoglyph attack. Legitimate multilingual text typically doesn't
    mix scripts within individual words.

    We analyze word-by-word to reduce false positives on multilingual text
    that legitimately contains different scripts in different words.
    """
    findings: list[UnicodeFinding] = []
    # Split into word-like tokens
    word_pattern = re.compile(r"[\w]+", re.UNICODE)

    for match in word_pattern.finditer(text):
        word = match.group()
        if len(word) < 2:
            continue

        scripts_found: dict[str, list[int]] = {}
        for i, ch in enumerate(word):
            script = _get_script(ch)
            if script and script != "Other":
                scripts_found.setdefault(script, []).append(i)

        # Check for suspicious combinations
        suspicious_combos = {
            ("Latin", "Cyrillic"),
            ("Latin", "Greek"),
            ("Latin", "Armenian"),
            ("Latin", "Cherokee"),
        }

        script_names = set(scripts_found.keys())
        for s1, s2 in suspicious_combos:
            if s1 in script_names and s2 in script_names:
                findings.append(UnicodeFinding(
                    category=UnicodeCategory.MIXED_SCRIPT,
                    severity=UnicodeSeverity.HIGH,
                    description=(
                        f"Mixed-script word detected: '{word}' contains both "
                        f"{s1} and {s2} characters. This is a strong indicator "
                        "of a homoglyph/visual spoofing attack."
                    ),
                    position=(match.start(), match.end()),
                    matched_text=word,
                    char_names=[
                        f"{s1} chars at indices {scripts_found[s1]}",
                        f"{s2} chars at indices {scripts_found[s2]}",
                    ],
                    recommendation="Reject mixed-script words or normalize to single script",
                ))
    return findings


def _detect_unicode_tags(text: str) -> list[UnicodeFinding]:
    """Detect Unicode Tags block characters (U+E0000–U+E007F).

    CRITICAL severity: these deprecated characters are invisible in virtually
    all text renderers but can be read by LLMs, allowing hidden instructions
    to be embedded in any text. Each tag character maps to an ASCII character
    (U+E0041 = 'A', U+E0042 = 'B', etc.), making full message encoding possible.

    Attack vector documented in arXiv:2603.00164 (Feb 2026).
    """
    positions = _has_unicode_tags(text)
    if not positions:
        return []

    # Decode the hidden payload for reporting
    decoded = _decode_unicode_tags("".join(text[p] for p in positions[:64]))
    decoded_preview = repr(decoded[:40]) if decoded.strip() else "(non-printable)"

    return [UnicodeFinding(
        category=UnicodeCategory.INVISIBLE_CHAR,
        severity=UnicodeSeverity.CRITICAL,
        description=(
            f"Unicode Tags block characters detected at {len(positions)} position(s). "
            f"These deprecated characters (U+E0000–U+E007F) are invisible in text "
            f"renderers but readable by LLMs, allowing hidden instruction embedding. "
            f"Decoded content preview: {decoded_preview}"
        ),
        position=(positions[0], positions[-1] + 1),
        matched_text=repr("".join(text[p] for p in positions[:10])),
        char_names=[f"UNICODE TAG (U+{ord(text[p]):05X})" for p in positions[:5]],
        recommendation=(
            "Reject or strip all Unicode Tags block characters (U+E0000–U+E007F). "
            "These have no legitimate use in modern text."
        ),
    )]


def _detect_zw_binary_encoding(text: str) -> list[UnicodeFinding]:
    """Detect potential ZWJ/ZWSP binary-encoding payloads.

    Sequences of 8+ consecutive ZWSP (U+200B) and ZWNJ (U+200C) characters
    can encode arbitrary binary data (1 bit each, 8 chars = 1 byte). This
    allows hiding complete messages in otherwise normal text.
    """
    max_run = _max_zw_binary_run(text)
    if max_run < 8:
        return []

    full_bytes = max_run // 8
    return [UnicodeFinding(
        category=UnicodeCategory.ZERO_WIDTH,
        severity=UnicodeSeverity.HIGH,
        description=(
            f"Potential ZWJ binary-encoding detected: {max_run} consecutive "
            f"ZWSP/ZWNJ characters (≈ {full_bytes} encoded byte(s)). "
            f"This pattern can hide arbitrary binary payloads in plain text."
        ),
        position=(0, len(text)),
        matched_text=f"<{max_run} ZW binary chars>",
        char_names=["ZERO WIDTH SPACE (ZWSP)", "ZERO WIDTH NON-JOINER (ZWNJ)"],
        recommendation="Strip all zero-width characters; flag content for manual review.",
    )]


def normalize_text(text: str) -> str:
    """Normalize text by replacing confusable characters and stripping
    dangerous Unicode.

    This produces a "canonical" version of the text where:
    - Homoglyphs are replaced with their Latin equivalents
    - Zero-width characters are removed
    - Bidirectional overrides are removed
    - Control characters (except TAB/LF/CR) are removed
    - Unicode is NFC-normalized

    Returns:
        Normalized text safe for comparison and pattern matching.
    """
    # NFC normalize first (catches NFD decomposition bypass attempts)
    text = unicodedata.normalize("NFC", text)

    result: list[str] = []
    allowed_control = {0x09, 0x0A, 0x0D}

    for ch in text:
        cp = ord(ch)

        # Strip Unicode Tags block characters (U+E0000-U+E007F) — CRITICAL
        # These encode ASCII invisibly and are fully readable by LLMs.
        if UNICODE_TAGS_START <= cp <= UNICODE_TAGS_END:
            continue

        # Replace homoglyphs
        if ch in CONFUSABLE_MAP:
            result.append(CONFUSABLE_MAP[ch])
            continue

        # Strip bidi overrides
        if cp in BIDI_CHARS:
            continue

        # Strip zero-width chars
        if cp in ZERO_WIDTH_CHARS:
            continue

        # Strip dangerous control chars
        is_control = False
        for range_start, range_end, _ in SUSPICIOUS_CONTROL_RANGES:
            if range_start <= cp <= range_end and cp not in allowed_control:
                is_control = True
                break
        if is_control:
            continue

        result.append(ch)

    return "".join(result)


def normalize_and_scan(text: str) -> tuple[str, list[UnicodeFinding]]:
    """Normalize text and scan for Unicode attacks in a single pass.

    This is the primary entry point for the Unicode scanner. It:
    1. Scans for all categories of Unicode attacks
    2. Returns a normalized version of the text with attacks removed

    Args:
        text: The input text to scan and normalize.

    Returns:
        A tuple of (normalized_text, list_of_findings).
        The normalized text has dangerous Unicode removed/replaced.
        Findings are sorted by position.

    Performance:
        ~O(n) where n is the length of text. All detection is done via
        precompiled patterns and lookup tables. Typical throughput:
        >1M characters/second.

    Example:
        >>> normalized, findings = normalize_and_scan("hello\\u200Bworld")
        >>> normalized
        'helloworld'
        >>> len(findings)
        1
        >>> findings[0].category
        <UnicodeCategory.ZERO_WIDTH: 'zero_width'>
    """
    if not text:
        return "", []

    findings: list[UnicodeFinding] = []

    # Run all detectors
    findings.extend(_detect_unicode_tags(text))    # CRITICAL — check first
    findings.extend(_detect_zw_binary_encoding(text))
    findings.extend(_detect_bidi(text))
    findings.extend(_detect_zero_width(text))
    findings.extend(_detect_homoglyphs(text))
    findings.extend(_detect_invisible_chars(text))
    findings.extend(_detect_mixed_script(text))

    # Sort by position
    findings.sort(key=lambda f: f.position[0])

    # Normalize
    normalized = normalize_text(text)

    return normalized, findings


def scan_unicode(text: str) -> list[UnicodeFinding]:
    """Scan text for Unicode attacks without normalizing.

    Use this when you only need findings without the normalized text.

    Args:
        text: The input text to scan.

    Returns:
        List of UnicodeFinding objects, sorted by position.
    """
    _, findings = normalize_and_scan(text)
    return findings
