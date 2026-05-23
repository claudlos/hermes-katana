"""Stage 1: Preprocessing Normalizer.

Neutralises character-level evasion attacks before any model sees the text.
Pure Python (stdlib only).  Target latency: <1 ms.

This single module defeats the entire class of tokeniser-level bypass attacks
that Meta Prompt-Guard-86M and every DeBERTa classifier are vulnerable to.
"""

from __future__ import annotations

import base64
import html
import json
import pathlib
import re
import unicodedata
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Homoglyph map: confusable characters -> ASCII
# ---------------------------------------------------------------------------

HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic -> Latin
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u043d": "h",
    "\u0422": "T",
    "\u041c": "M",
    "\u0410": "A",
    "\u0412": "B",
    "\u0415": "E",
    "\u041a": "K",
    "\u041d": "H",
    "\u041e": "O",
    "\u0420": "P",
    "\u0421": "C",
    "\u0423": "Y",
    "\u0425": "X",
    # Greek -> Latin
    "\u03b1": "a",
    "\u03b5": "e",
    "\u03b9": "i",
    "\u03bf": "o",
    "\u03c1": "p",
    "\u03c5": "u",
    "\u0391": "A",
    "\u0392": "B",
    "\u0395": "E",
    "\u0397": "H",
    "\u0399": "I",
    "\u039a": "K",
    "\u039c": "M",
    "\u039d": "N",
    "\u039f": "O",
    "\u03a1": "P",
    "\u03a4": "T",
    "\u03a5": "Y",
    "\u03a7": "X",
    "\u0396": "Z",
    # Mathematical symbols
    "\u2205": "0",  # empty set -> 0
    "\u2113": "l",  # script l
    # Fullwidth -> ASCII
    "\uff01": "!",
    "\uff1f": "?",
    "\uff0e": ".",
    "\uff0c": ",",
}
# Build fullwidth ASCII range (U+FF01-U+FF5E -> U+0021-U+007E)
for _i in range(0xFF01, 0xFF5F):
    HOMOGLYPH_MAP[chr(_i)] = chr(_i - 0xFEE0)

# Load supplemental mappings from JSON (without overriding the confusables above)
_HOMOGLYPH_JSON = pathlib.Path(__file__).parent / "data" / "homoglyph_map.json"
try:
    _json_entries: dict[str, str] = json.loads(_HOMOGLYPH_JSON.read_text(encoding="utf-8"))
    for _k, _v in _json_entries.items():
        if _k not in HOMOGLYPH_MAP:
            HOMOGLYPH_MAP[_k] = _v
except (OSError, json.JSONDecodeError):
    pass

# Zero-width and invisible characters to strip
INVISIBLE_CHARS: re.Pattern[str] = re.compile(
    "["
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u200e"  # left-to-right mark
    "\u200f"  # right-to-left mark
    "\u00ad"  # soft hyphen
    "\u2060"  # word joiner
    "\u2061"  # function application
    "\u2062"  # invisible times
    "\u2063"  # invisible separator
    "\u2064"  # invisible plus
    "\ufeff"  # byte order mark / zero-width no-break space
    "\u034f"  # combining grapheme joiner
    "\u061c"  # arabic letter mark
    "\u115f"  # hangul choseong filler
    "\u1160"  # hangul jungseong filler
    "\u17b4"  # khmer vowel inherent aq
    "\u17b5"  # khmer vowel inherent aa
    "\u180e"  # mongolian vowel separator
    "\u3164"  # hangul filler
    "\uffa0"  # halfwidth hangul filler
    "]+"
)

BASE64_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
HEX_PATTERN: re.Pattern[str] = re.compile(r"(?:0x)?[0-9a-fA-F]{16,}")
URL_ENCODED_PATTERN: re.Pattern[str] = re.compile(r"(?:%[0-9a-fA-F]{2}){4,}")
HTML_COMMENT: re.Pattern[str] = re.compile(r"<!--.*?-->", re.DOTALL)
CHAR_SPACING: re.Pattern[str] = re.compile(r"(?:[a-zA-Z] ){4,}[a-zA-Z]")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NormalizedResult:
    """Result of text normalisation."""

    text: str
    original_text: str
    flags: dict[str, bool] = field(default_factory=dict)
    decoded_segments: list[str] = field(default_factory=list)
    hidden_content: list[str] = field(default_factory=list)

    @property
    def has_anomalies(self) -> bool:
        return any(self.flags.values())

    @property
    def anomaly_count(self) -> int:
        return sum(1 for v in self.flags.values() if v)


# ---------------------------------------------------------------------------
# Individual normalisers
# ---------------------------------------------------------------------------


def normalize_homoglyphs(text: str) -> tuple[str, bool]:
    """Replace confusable Unicode characters with ASCII equivalents."""
    changed = False
    chars: list[str] = []
    for ch in text:
        replacement = HOMOGLYPH_MAP.get(ch)
        if replacement is not None:
            chars.append(replacement)
            changed = True
        else:
            chars.append(ch)
    return "".join(chars), changed


def strip_invisible(text: str) -> tuple[str, bool]:
    """Remove zero-width and invisible Unicode characters."""
    result = INVISIBLE_CHARS.sub("", text)
    return result, result != text


def detect_and_decode_base64(text: str) -> tuple[str, list[str], bool]:
    """Find base64-encoded segments, decode them, append decoded text."""
    decoded_segments: list[str] = []
    found = False

    def _decode_match(m: re.Match[str]) -> str:
        nonlocal found
        segment = m.group(0)
        try:
            decoded_bytes = base64.b64decode(segment)
            decoded_text = decoded_bytes.decode("utf-8", errors="ignore")
            printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c.isspace()) / max(len(decoded_text), 1)
            if printable_ratio > 0.7 and len(decoded_text) > 5:
                found = True
                decoded_segments.append(decoded_text)
                return f"{segment} [DECODED: {decoded_text}]"
        except Exception:  # noqa: BLE001
            pass
        return segment

    result = BASE64_PATTERN.sub(_decode_match, text)
    return result, decoded_segments, found


def detect_and_decode_hex(text: str) -> tuple[str, list[str], bool]:
    """Find hex-encoded segments, decode them."""
    decoded_segments: list[str] = []
    found = False

    def _decode_match(m: re.Match[str]) -> str:
        nonlocal found
        segment = m.group(0).lstrip("0x")
        try:
            decoded_bytes = bytes.fromhex(segment)
            decoded_text = decoded_bytes.decode("utf-8", errors="ignore")
            printable_ratio = sum(1 for c in decoded_text if c.isprintable() or c.isspace()) / max(len(decoded_text), 1)
            if printable_ratio > 0.7 and len(decoded_text) > 3:
                found = True
                decoded_segments.append(decoded_text)
                return f"{m.group(0)} [DECODED: {decoded_text}]"
        except Exception:  # noqa: BLE001
            pass
        return m.group(0)

    result = HEX_PATTERN.sub(_decode_match, text)
    return result, decoded_segments, found


def detect_and_decode_url_encoding(text: str) -> tuple[str, list[str], bool]:
    """Find URL-encoded segments, decode them."""
    import urllib.parse

    decoded_segments: list[str] = []
    found = False

    def _decode_match(m: re.Match[str]) -> str:
        nonlocal found
        segment = m.group(0)
        try:
            decoded = urllib.parse.unquote(segment)
            if decoded != segment:
                found = True
                decoded_segments.append(decoded)
                return f"{segment} [DECODED: {decoded}]"
        except Exception:  # noqa: BLE001
            pass
        return segment

    result = URL_ENCODED_PATTERN.sub(_decode_match, text)
    return result, decoded_segments, found


def collapse_char_spacing(text: str) -> tuple[str, bool]:
    """Detect and collapse ``i g n o r e`` -> ``ignore`` style evasion."""
    found = bool(CHAR_SPACING.search(text))
    if found:

        def _collapse(m: re.Match[str]) -> str:
            return m.group(0).replace(" ", "")

        text = CHAR_SPACING.sub(_collapse, text)
    return text, found


def extract_hidden_content(text: str) -> tuple[str, list[str]]:
    """Extract content from HTML comments and similar hiding spots."""
    hidden: list[str] = []
    for m in HTML_COMMENT.finditer(text):
        content = m.group(0)[4:-3].strip()
        if content:
            hidden.append(content)
    cleaned = HTML_COMMENT.sub("", text)
    return cleaned, hidden


def apply_rot13(text: str) -> str:
    """Apply ROT13 decoding."""
    import codecs

    return codecs.decode(text, "rot_13")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def normalize(text: str, *, aggressive: bool = True) -> NormalizedResult:
    """Full normalisation pipeline.

    Args:
        text: Raw input text.
        aggressive: If *True*, apply all normalisations including encoding
            detection.  If *False*, only apply safe normalisations (unicode,
            invisible chars).

    Returns:
        :class:`NormalizedResult` with normalised text, flags, and decoded
        segments.
    """
    original = text
    flags: dict[str, bool] = {}
    all_decoded: list[str] = []
    all_hidden: list[str] = []

    # 1. Unicode NFKC
    text = unicodedata.normalize("NFKC", text)

    # 2. HTML entity decoding
    text = html.unescape(text)

    # 3. Strip invisible characters
    text, invisible_found = strip_invisible(text)
    flags["invisible_chars"] = invisible_found

    # 4. Homoglyph normalisation
    text, homoglyph_found = normalize_homoglyphs(text)
    flags["homoglyphs"] = homoglyph_found

    # 5. Character spacing collapse
    text, spacing_found = collapse_char_spacing(text)
    flags["char_spacing"] = spacing_found

    if aggressive:
        # 6. Extract hidden content
        text, hidden = extract_hidden_content(text)
        all_hidden.extend(hidden)
        flags["hidden_content"] = len(hidden) > 0

        # 7. Base64 detection + inline decoding
        text, b64_decoded, b64_found = detect_and_decode_base64(text)
        all_decoded.extend(b64_decoded)
        flags["base64_encoded"] = b64_found

        # 8. Hex detection + inline decoding
        text, hex_decoded, hex_found = detect_and_decode_hex(text)
        all_decoded.extend(hex_decoded)
        flags["hex_encoded"] = hex_found

        # 9. URL-encoding detection + inline decoding
        text, url_decoded, url_found = detect_and_decode_url_encoding(text)
        all_decoded.extend(url_decoded)
        flags["url_encoded"] = url_found

    # 10. Whitespace normalisation
    text = re.sub(r"\s+", " ", text).strip()

    # 11. Anomalous whitespace ratio in original
    if len(original) > 10:
        space_ratio = original.count(" ") / len(original)
        flags["whitespace_anomaly"] = space_ratio > 0.4

    return NormalizedResult(
        text=text,
        original_text=original,
        flags=flags,
        decoded_segments=all_decoded,
        hidden_content=all_hidden,
    )
