"""
Decoder Scanner for HermesKatana.

Recursively decodes obfuscated payloads (base64, hex, ROT13, URL-encoding,
Unicode escapes, HTML entities) and re-scans decoded plaintext with the
injection scanner. Handles nested encodings up to configurable depth.

Performance target: 2-5ms per prompt (depth=3, 2-3 blobs typical).
Dependencies: Python stdlib only.
"""

from __future__ import annotations

import base64
import codecs
import html
import re
import string
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "DecoderCategory",
    "DecoderFinding",
    "decode_and_scan",
    "detect_encoded_blobs",
]


class DecoderCategory(str, Enum):
    """Categories of encoding obfuscation detected."""

    BASE64 = "base64"
    BASE32 = "base32"
    HEX = "hex"
    ROT13 = "rot13"
    CAESAR = "caesar"
    URL_ENCODING = "url_encoding"
    HTML_ENTITIES = "html_entities"
    UNICODE_ESCAPES = "unicode_escapes"
    NESTED = "nested"


@dataclass(frozen=True, slots=True)
class DecoderFinding:
    """A finding from the decoder scanner.

    Attributes:
        strategy: Always 'decoded'.
        confidence: Confidence from 0.0 to 1.0 (propagated from inner scan,
                    discounted by 0.1 per decode layer).
        matched_text: The decoded plaintext that triggered inner scan.
        position: (start, end) character positions of the encoded blob
                  in the ORIGINAL input text.
        category: Type of encoding that was decoded.
        encoding_chain: Encoding types applied (outermost first).
        inner_findings: Findings from re-scanning the decoded text.
        pattern_name: Name of the encoding pattern matched.
        description: Human-readable explanation.
    """

    strategy: str
    confidence: float
    matched_text: str
    position: tuple[int, int]
    category: DecoderCategory
    encoding_chain: tuple[str, ...]
    inner_findings: tuple
    pattern_name: str = ""
    description: str = ""


# ---------------------------------------------------------------------------
# Encoding detection patterns (precompiled at module load)
# ---------------------------------------------------------------------------

# Base64: 11+ chars of base64 alphabet plus optional padding — base64 of an
# 8-byte payload like "rm -rf /" is 11 alphabet chars + "=". The previous 20+
# floor skipped short encoded commands entirely (audit finding D2);
# _is_textlike + the inner scanners gate false positives.
_RE_BASE64 = re.compile(r"[A-Za-z0-9+/\-_]{11,}={0,2}")

# Explicit cipher instruction markers (ROT13, Caesar, Base32)
_RE_CIPHER_INSTRUCTION = re.compile(
    r"(?:"
    r"(?:rot[- ]?(?:13|n)|caesar(?:\s+cipher)?(?:\s+with\s+(?:shift|key|offset)\s+\d+)?|"
    r"atbash|vigenere|shift\s+(?:cipher|by\s+\d+)|"
    r"base[- ]?32\s+(?:decode|encoded?|cipher)|"
    r"decode\s+(?:using|with|via)\s+(?:rot|caesar|base32|shift))"
    r")",
    re.IGNORECASE,
)

# Hex: 12+ hex chars (with optional 0x prefix) or \xNN sequences — 6 bytes,
# enough for short encoded commands (audit finding D2)
_RE_HEX_BLOCK = re.compile(r"(?:0x)?[0-9a-fA-F]{12,}")
_RE_HEX_ESCAPES = re.compile(r"(?:\\x[0-9a-fA-F]{2}){4,}")

# URL-encoding: 2+ percent-encoded bytes
_RE_URL_ENCODED = re.compile(r"(?:%[0-9a-fA-F]{2}){2,}")

# HTML entities: 1+ consecutive entities
_RE_HTML_ENTITIES = re.compile(r"(?:&#\d+;|&#x[0-9a-fA-F]+;|&\w+;){1,}")

# Unicode escapes: 2+ consecutive \\uXXXX
_RE_UNICODE_ESCAPES = re.compile(r"(?:\\u[0-9a-fA-F]{4}){2,}")

# Base32: 16+ chars of base32 alphabet (A-Z, 2-7), optional padding
# (single definition — an earlier duplicate silently shadowed a stricter one)
_RE_BASE32 = re.compile(r"[A-Z2-7]{16,}={0,6}")

# ROT13/Caesar explicit instruction markers
_RE_ROT13_INSTRUCTION = re.compile(
    r"(?:rot13|rot-13|caesar\s+(?:cipher|shift|decode|encrypt))\s*(?:decode|decrypt|this|the\s+following|of|:)\s*[\"']?([A-Za-z][A-Za-z\s]{10,})[\"']?",
    re.IGNORECASE,
)

# Caesar cipher with explicit shift: "caesar shift 3: ..." or "shift by 13: ..."
_RE_CAESAR_SHIFT = re.compile(
    r"(?:caesar|shift|rotate)\s+(?:by\s+|of\s+|=\s*)?(\d{1,2})\s*[:\-]\s*[\"']?([A-Za-z][A-Za-z\s]{10,})[\"']?",
    re.IGNORECASE,
)

MAX_DECODE_DEPTH = 3
MIN_INNER_CONFIDENCE = 0.5

# Printable character threshold for decoded output
_PRINTABLE_THRESHOLD = 0.6

# Known false-positive prefixes (JWT, image data markers, etc.)
_FP_PREFIXES = (
    "eyJ",  # JWT tokens (base64 of '{"')
    "/9j/",  # JPEG base64
    "iVBOR",  # PNG base64
    "R0lGO",  # GIF base64
    "AAAA",  # various binary formats
    "MIIB",  # X.509 certificates
    "MIIC",  # X.509 certificates
    "MIID",  # X.509 certificates
    "ssh-",  # SSH keys
)


def _is_textlike(decoded: str) -> bool:
    """Check if decoded string looks like text, not binary noise.

    Returns True if >60% printable and contains at least one space
    or common word boundary character.
    """
    if not decoded:
        return False
    printable_set = set(string.printable)
    printable_count = sum(1 for c in decoded if c in printable_set)
    ratio = printable_count / len(decoded)
    if ratio < _PRINTABLE_THRESHOLD:
        return False
    # Must have some word-like structure (space, punctuation, or length > 4)
    if len(decoded) < 4:
        return False
    return True


def _is_false_positive_base64(blob: str) -> bool:
    """Check if a base64 blob is a known false positive (JWT, image, cert)."""
    for prefix in _FP_PREFIXES:
        if blob.startswith(prefix):
            return True
    return False


def detect_encoded_blobs(text: str) -> list[tuple[DecoderCategory, int, int, str]]:
    """Identify candidate encoded blobs in text.

    Returns list of (category, start_pos, end_pos, blob_text) tuples.
    """
    candidates: list[tuple[DecoderCategory, int, int, str]] = []

    # Check for explicit cipher instruction markers (ROT13, Caesar, Base32)
    for m in _RE_CIPHER_INSTRUCTION.finditer(text):
        instr = m.group().lower()
        if "base" in instr and "32" in instr:
            cat = DecoderCategory.BASE32
        elif "rot" in instr or "caesar" in instr or "shift" in instr or "atbash" in instr:
            cat = DecoderCategory.ROT13
        else:
            cat = DecoderCategory.ROT13
        candidates.append((cat, m.start(), m.end(), m.group()))

    for m in _RE_BASE64.finditer(text):
        blob = m.group()
        if _is_false_positive_base64(blob):
            continue
        # base64 length must be multiple of 4 (or close with padding)
        stripped = blob.rstrip("=")
        pad_len = (4 - len(stripped) % 4) % 4
        if pad_len <= 2:
            candidates.append((DecoderCategory.BASE64, m.start(), m.end(), blob))

    # Base32: uppercase only, length multiple of 8 (or padded), not already caught as base64
    for m in _RE_BASE32.finditer(text):
        blob = m.group()
        # Skip if this overlaps with a base64 candidate (base64 is superset)
        if any(c[0] == DecoderCategory.BASE64 and c[1] <= m.start() < c[2] for c in candidates):
            continue
        # Base32 payload must be uppercase-only and decodable
        stripped = blob.rstrip("=")
        pad_len = (8 - len(stripped) % 8) % 8
        if pad_len <= 6 and len(stripped) >= 8:
            candidates.append((DecoderCategory.BASE32, m.start(), m.end(), blob))

    for m in _RE_HEX_BLOCK.finditer(text):
        blob = m.group()
        raw = blob[2:] if blob.startswith("0x") else blob
        if len(raw) % 2 == 0:
            candidates.append((DecoderCategory.HEX, m.start(), m.end(), blob))

    for m in _RE_HEX_ESCAPES.finditer(text):
        candidates.append((DecoderCategory.HEX, m.start(), m.end(), m.group()))

    for m in _RE_URL_ENCODED.finditer(text):
        candidates.append((DecoderCategory.URL_ENCODING, m.start(), m.end(), m.group()))
    if "%" in text:
        percent_escapes = list(re.finditer(r"%[0-9a-fA-F]{2}", text))
        if percent_escapes:
            decoded = urllib.parse.unquote(text)
            if decoded != text:
                candidates.append((DecoderCategory.URL_ENCODING, 0, len(text), text))

    for m in _RE_HTML_ENTITIES.finditer(text):
        candidates.append((DecoderCategory.HTML_ENTITIES, m.start(), m.end(), m.group()))

    for m in _RE_UNICODE_ESCAPES.finditer(text):
        candidates.append((DecoderCategory.UNICODE_ESCAPES, m.start(), m.end(), m.group()))

    return candidates


def _caesar_shift(text: str, shift: int) -> str:
    """Apply a Caesar cipher shift to all alpha characters."""
    result = []
    for ch in text:
        if ch.isupper():
            result.append(chr((ord(ch) - ord("A") - shift) % 26 + ord("A")))
        elif ch.islower():
            result.append(chr((ord(ch) - ord("a") - shift) % 26 + ord("a")))
        else:
            result.append(ch)
    return "".join(result)


# Approximate letter frequencies for English (for Caesar scoring)
_EN_FREQ = {
    "e": 12.7,
    "t": 9.1,
    "a": 8.2,
    "o": 7.5,
    "i": 7.0,
    "n": 6.7,
    "s": 6.3,
    "h": 6.1,
    "r": 6.0,
    "d": 4.3,
    "l": 4.0,
    "c": 2.8,
    "u": 2.8,
    "m": 2.4,
    "w": 2.4,
    "f": 2.2,
    "g": 2.0,
    "y": 2.0,
    "p": 1.9,
    "b": 1.5,
    "v": 1.0,
    "k": 0.8,
    "j": 0.2,
    "x": 0.2,
    "q": 0.1,
    "z": 0.1,
}


def _english_score(text: str) -> float:
    """Score text on how English-like it is (higher = more English)."""
    lower = text.lower()
    total = sum(1 for c in lower if c.isalpha())
    if total == 0:
        return 0.0
    return sum(_EN_FREQ.get(c, 0.0) for c in lower if c.isalpha()) / total


def _try_decode(blob: str, category: DecoderCategory) -> Optional[str]:
    """Attempt to decode a single blob. Returns decoded text or None."""
    try:
        if category == DecoderCategory.BASE64:
            # Fix padding and handle URL-safe base64 (-_ → +/)
            blob = blob.replace("-", "+").replace("_", "/")
            padded = blob + "=" * ((4 - len(blob) % 4) % 4)
            raw = base64.b64decode(padded, validate=True)
            decoded = raw.decode("utf-8", errors="replace")
        elif category == DecoderCategory.BASE32:
            # Fix padding to multiple of 8
            stripped = blob.rstrip("=")
            pad_len = (8 - len(stripped) % 8) % 8
            padded = stripped + "=" * pad_len
            raw = base64.b32decode(padded, casefold=False)
            decoded = raw.decode("utf-8", errors="replace")
        elif category == DecoderCategory.CAESAR:
            # Generic Caesar — try all 25 shifts, return the most English-like
            best = None
            best_score = 0.0
            for shift in range(1, 26):
                candidate = _caesar_shift(blob, shift)
                score = _english_score(candidate)
                if score > best_score:
                    best_score = score
                    best = candidate
            decoded = best if best else blob
        elif category == DecoderCategory.HEX:
            if blob.startswith("0x") or blob.startswith("0X"):
                hex_str = blob[2:]
            elif "\\x" in blob:
                # \xNN escape sequences
                hex_str = blob.replace("\\x", "")
            else:
                hex_str = blob
            raw = bytes.fromhex(hex_str)
            decoded = raw.decode("utf-8", errors="replace")
        elif category == DecoderCategory.ROT13:
            decoded = codecs.decode(blob, "rot_13")
        elif category == DecoderCategory.URL_ENCODING:
            decoded = urllib.parse.unquote(blob)
        elif category == DecoderCategory.HTML_ENTITIES:
            decoded = html.unescape(blob)
        elif category == DecoderCategory.UNICODE_ESCAPES:
            decoded = blob.encode("raw_unicode_escape").decode("unicode_escape")
        else:
            return None
    except Exception:
        return None

    if not _is_textlike(decoded):
        return None
    return decoded


def decode_and_scan(
    text: str,
    *,
    depth: int = 0,
    max_depth: int = MAX_DECODE_DEPTH,
    vault_values: Optional[set[str]] = None,
    _parent_chain: tuple[str, ...] = (),
    _original_positions: Optional[tuple[int, int]] = None,
) -> list[DecoderFinding]:
    """Recursively decode encoded blobs and re-scan decoded plaintext.

    Args:
        text: Input text to scan for encoded payloads.
        depth: Current recursion depth (internal use).
        max_depth: Maximum decode recursion depth.

    Returns:
        List of DecoderFinding with decoded attack details.
    """
    if not text or depth > max_depth:
        return []

    # Lazy import to avoid circular dependency
    from hermes_katana.scanner.commands import detect_dangerous_command
    from hermes_katana.scanner.injection import detect_injection
    from hermes_katana.scanner.secrets import scan_for_secrets

    findings: list[DecoderFinding] = []
    blobs = detect_encoded_blobs(text)

    for category, start, end, blob in blobs:
        decoded = _try_decode(blob, category)
        if decoded is None:
            continue

        chain = _parent_chain + (category.value,)
        pos = _original_positions if _original_positions else (start, end)

        # Re-scan decoded text through the same high-risk scanner classes used
        # by the top-level scan paths. Encoded command/secret payloads should
        # not need an injection phrase to become visible.
        injection_findings = detect_injection(decoded)
        high_conf_injection = [f for f in injection_findings if f.confidence >= MIN_INNER_CONFIDENCE]
        command_findings = detect_dangerous_command(decoded)
        secret_findings = scan_for_secrets(decoded, vault_values)
        inner_findings = [*high_conf_injection, *command_findings, *secret_findings]

        if inner_findings:
            # Discount confidence by 0.1 per decode layer
            discount = 0.1 * len(chain)
            max_inner_conf = max(getattr(f, "confidence", 0.9) for f in inner_findings)
            adjusted_conf = max(max_inner_conf - discount, 0.1)

            result_category = DecoderCategory.NESTED if len(chain) > 1 else category
            if high_conf_injection:
                finding_type = "injection"
            elif command_findings:
                finding_type = "dangerous command"
            else:
                finding_type = "secret"

            findings.append(
                DecoderFinding(
                    strategy="decoded",
                    confidence=round(adjusted_conf, 2),
                    matched_text=decoded,
                    position=pos,
                    category=result_category,
                    encoding_chain=chain,
                    inner_findings=tuple(inner_findings),
                    pattern_name=f"decoded_{category.value}",
                    description=(
                        f"Decoded {' -> '.join(chain)} payload contains {finding_type}: {inner_findings[0].description}"
                    ),
                )
            )

        # Recurse on decoded text for nested encodings
        if depth < max_depth:
            nested = decode_and_scan(
                decoded,
                depth=depth + 1,
                max_depth=max_depth,
                vault_values=vault_values,
                _parent_chain=chain,
                _original_positions=pos,
            )
            findings.extend(nested)

    return findings
