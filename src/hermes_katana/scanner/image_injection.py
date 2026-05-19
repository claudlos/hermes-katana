"""
Image injection scanner for HermesKatana.

Detects hidden injections embedded in image file metadata:

1. JPEG COM (comment) markers with suspicious content
2. JPEG APP1 (EXIF) and APP13 (XMP/IPTC) metadata with suspicious keywords
3. PNG tEXt/iTXt/zTXt chunks with suspicious content
4. GIF comment extension scanning
5. SVG masquerading as raster images (SVG embedded in PNG/JPEG)
6. Large/unusually sized metadata payloads

Pure Python struct-based parsing — no Pillow, no OCR.

Speed: Sub-millisecond for typical images (<10 MB).
"""

from __future__ import annotations

import struct
import zlib
import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ImageInjectionSeverity",
    "ImageInjectionFinding",
    "detect_image_injection",
    "detect_image_injection_bytes",
]


# ---------------------------------------------------------------------------
# Severity & Finding dataclass
# ---------------------------------------------------------------------------


class ImageInjectionSeverity(str, Enum):
    """Severity of image injection findings."""

    CRITICAL = "critical"
    """Hidden malicious content detected in image metadata."""

    HIGH = "high"
    """Suspicious hidden content in image metadata."""

    MEDIUM = "medium"
    """Hidden content detected, needs review."""

    LOW = "low"
    """Minor hidden elements, likely benign."""


@dataclass(frozen=True, slots=True)
class ImageInjectionFinding:
    """A single image injection detection finding.

    Attributes:
        layer_type: Which image format/layer the content was found in
                   (e.g. "jpeg_com", "png_text", "gif_comment", "svg_masquerade",
                    "large_metadata", "exif", "xmp").
        content: The suspicious content itself (truncated to 500 chars).
        location: Where in the file this was found (e.g. byte offset).
        severity: How severe this finding is.
        description: Human-readable explanation.
        confidence: Detection confidence 0.0-1.0.
    """

    layer_type: str
    content: str
    location: str
    severity: ImageInjectionSeverity
    description: str
    confidence: float


# ---------------------------------------------------------------------------
# Injection keywords — scanned in all metadata fields
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r, re.IGNORECASE)
    for r in [
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
        r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions?|prompts?|rules?)",
        r"forget\s+everything",
        r"(new|updated|real)\s+instructions?\s*[:=]",
        r"override\s+(all\s+)?(previous\s+)?(instructions?|rules?|safety)",
        r"you\s+are\s+now\s+(a\s+)?(new|different|unrestricted)",
        r"(pretend|act)\s+as\s+(an?\s+)?(unrestricted|unfiltered|jailbroken)",
        r"(system|assistant|user|human)\s*:\s*",
        r"<system\s*>|<assistant\s*>|<user\s*>",
        r"```\s*(system|prompt|instruction)",
        r"bypass\s+(safety|restriction|filter|policy)",
        r"(reveal|show|print)\s+(your\s+)?(system\s+)?(prompt|instructions?|secrets?)",
        r"<IMPORTANT>",
        r"END\s+OF\s+(SYSTEM\s+)?PROMPT",
        r"do\s+anything\s+now",
        r"(inject|exfiltrate|extract)\s+(system|data|secret|credential)",
        r"ignore\s+all\s+previous",
    ]
]

# Threshold for "large" metadata as fraction of total file size
_LARGE_METADATA_RATIO = 0.3

# Absolute size above which any single metadata field is suspicious (bytes)
_METADATA_SIZE_WARN = 64 * 1024  # 64 KB

# Maximum content length stored in findings
_MAX_CONTENT_LEN = 500


def _match_injection(text: str) -> list[tuple[str, re.Pattern]]:
    """Return list of (matched_text, pattern) for any injection keyword found."""
    matches = []
    for pat in _INJECTION_PATTERNS:
        m = pat.search(text)
        if m:
            matches.append((m.group(0), pat))
    return matches


def _severity_for_confidence(confidence: float) -> ImageInjectionSeverity:
    if confidence >= 0.9:
        return ImageInjectionSeverity.CRITICAL
    elif confidence >= 0.75:
        return ImageInjectionSeverity.HIGH
    elif confidence >= 0.5:
        return ImageInjectionSeverity.MEDIUM
    return ImageInjectionSeverity.LOW


# ---------------------------------------------------------------------------
# JPEG scanning
# ---------------------------------------------------------------------------


def _scan_jpeg(data: bytes) -> list[ImageInjectionFinding]:
    """Scan JPEG bytes for hidden injections in metadata."""
    findings = []
    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        return findings  # not a JPEG

    i = 2
    metadata_total = 0
    while i < len(data) - 1:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]

        # Standalone markers (no length)
        if marker in (0xD8, 0xD9):
            i += 2
            continue

        # Sos (start of scan) — consume rest as image data
        if marker == 0xDA:
            break

        # Any segment with a length
        if i + 3 >= len(data):
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if i + 2 + seg_len > len(data):
            break
        seg_data = data[i + 4 : i + 2 + seg_len]

        layer_type = None
        text_payload = None

        if marker == 0xFE:
            # COM comment
            layer_type = "jpeg_com"
            try:
                text_payload = seg_data.decode("latin-1")
            except Exception:
                text_payload = ""

        elif marker == 0xE1:
            # APP1 — EXIF or XMP
            if seg_data.startswith(b"Exif\0"):
                layer_type = "exif"
                # EXIF header is 4 bytes, rest is TIFF data
                # Scan raw bytes for text-like injection keywords
                text_payload = _exif_text_extract(seg_data[4:])
            elif seg_data.startswith(b"http://ns.adobe.com/xap/"):
                layer_type = "xmp"
                try:
                    text_payload = seg_data.decode("utf-8", errors="ignore")
                except Exception:
                    text_payload = ""

        elif 0xE0 <= marker <= 0xEF:
            # APP0–APP15 generic metadata
            layer_type = f"jpeg_app{marker - 0xE0}"
            try:
                # Attempt UTF-8 decode for text-like metadata
                text_payload = seg_data.decode("utf-8", errors="ignore")
            except Exception:
                text_payload = ""

        if text_payload:
            metadata_total += len(text_payload)
            findings.extend(_check_text_payload(text_payload, layer_type, f"offset {i}"))

        i += 2 + seg_len

    # Large metadata check
    if metadata_total > len(data) * _LARGE_METADATA_RATIO or metadata_total > _METADATA_SIZE_WARN:
        findings.append(
            ImageInjectionFinding(
                layer_type="large_metadata",
                content=f"Total metadata size: {metadata_total} bytes ({len(data)} byte file)",
                location="jpeg",
                severity=ImageInjectionSeverity.HIGH,
                description=f"Unusually large JPEG metadata payload ({metadata_total} bytes) — may hide steganographic content",
                confidence=0.75,
            )
        )

    return findings


def _exif_text_extract(tiff_data: bytes) -> str:
    """Extract readable text from raw EXIF/TIFF data."""
    text_parts = []
    try:
        # Scan for embedded null-terminated strings in EXIF data
        # EXIF strings are often in ASCII with null terminators
        current = bytearray()
        for b in tiff_data:
            if 0x20 <= b < 0x7F or b in (0x09, 0x0A, 0x0D):
                current.append(b)
            else:
                if len(current) >= 4:
                    try:
                        txt = bytes(current).decode("ascii", errors="ignore").strip()
                        if txt:
                            text_parts.append(txt)
                    except Exception:
                        pass
                current = bytearray()
        if len(current) >= 4:
            try:
                txt = bytes(current).decode("ascii", errors="ignore").strip()
                if txt:
                    text_parts.append(txt)
            except Exception:
                pass
    except Exception:
        pass
    return "\n".join(text_parts)


# ---------------------------------------------------------------------------
# PNG scanning
# ---------------------------------------------------------------------------


def _scan_png(data: bytes) -> list[ImageInjectionFinding]:
    """Scan PNG bytes for hidden injections in tEXt/iTXt/zTXt chunks."""
    findings = []
    if len(data) < 8 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return findings  # not a PNG

    i = 8
    metadata_total = 0
    while i < len(data) - 12:
        chunk_len = struct.unpack(">I", data[i : i + 4])[0]
        chunk_type = data[i + 4 : i + 8]
        if i + 12 + chunk_len > len(data):
            break
        chunk_data = data[i + 8 : i + 8 + chunk_len]

        layer_type = None
        text_payload = None

        if chunk_type == b"tEXt":
            layer_type = "png_text"
            # tEXt: keyword (null-terminated) + \0 + text
            null_pos = chunk_data.find(b"\0")
            if null_pos >= 0:
                text_payload = chunk_data[null_pos + 1 :].decode("latin-1", errors="ignore")

        elif chunk_type == b"iTXt":
            layer_type = "png_itext"
            # iTXt: keyword + \0 + compression flag + compression method + language tag + \0 + translated keyword + \0 + text
            null_pos = chunk_data.find(b"\0")
            if null_pos >= 0:
                rest = chunk_data[null_pos + 1 :]
                if len(rest) >= 2:
                    # compression flag at rest[0], compression method at rest[1]
                    # language tag starts at rest[2]; find its null terminator
                    lang_end = rest[2:].find(b"\0")
                    if lang_end >= 0:
                        # translated keyword starts after lang tag + lang_null + compressed_flag + compressed_method
                        translated_start = 2 + lang_end + 1
                        # find null terminator of translated keyword
                        tk_end = rest[translated_start:].find(b"\0")
                        if tk_end >= 0:
                            text_start = translated_start + tk_end + 1
                            if text_start < len(rest):
                                text_payload = rest[text_start:].decode("utf-8", errors="ignore")

        elif chunk_type == b"zTXt":
            layer_type = "png_ztext"
            null_pos = chunk_data.find(b"\0")
            if null_pos >= 0:
                rest = chunk_data[null_pos + 1 :]
                if rest and rest[0] == 0:
                    # Compression method 0 = zlib
                    compressed = rest[1:]
                    try:
                        text_payload = zlib.decompress(compressed).decode("latin-1", errors="ignore")
                    except Exception:
                        # Fall back: try raw
                        text_payload = compressed.decode("latin-1", errors="ignore")

        elif chunk_type == b"IEND":
            break

        if text_payload:
            metadata_total += len(text_payload)
            findings.extend(
                _check_text_payload(
                    text_payload, layer_type, f"PNG chunk {chunk_type.decode('latin-1', errors='?')} at offset {i}"
                )
            )

        i += 12 + chunk_len

    if metadata_total > _METADATA_SIZE_WARN:
        findings.append(
            ImageInjectionFinding(
                layer_type="large_metadata",
                content=f"Total PNG text chunks: {metadata_total} bytes",
                location="png",
                severity=ImageInjectionSeverity.HIGH,
                description=f"Unusually large PNG text metadata ({metadata_total} bytes) — possible steganographic payload",
                confidence=0.75,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# GIF scanning
# ---------------------------------------------------------------------------


def _scan_gif(data: bytes) -> list[ImageInjectionFinding]:
    """Scan GIF bytes for comment extensions with suspicious content."""
    findings = []
    # GIF89a has comment extension: 0x21 0xFE followed by sub-block sizes
    i = 0
    metadata_total = 0
    if data.startswith(b"GIF87a"):
        version = "87a"
    elif data.startswith(b"GIF89a"):
        version = "89a"
    else:
        return findings

    # Skip header (6 bytes) + Logical Screen Descriptor (7 bytes)
    # LSD = width(2) + height(2) + packed(1) + bg_color(1) + aspect(1) = 7
    if len(data) < 13:
        return findings
    lsd_offset = 6  # after header
    i = lsd_offset + 7  # start of first extension/block after LSD = 13
    if version == "89a":
        # Packed byte is at lsd_offset + 4 = byte 10 in the file
        # (width[2] + height[2] = 4 bytes after header)
        packed_byte = data[lsd_offset + 4]  # = data[10]
        gct_flag = (packed_byte >> 7) & 1
        if gct_flag:
            gct_size = 3 * (2 ** ((packed_byte & 0x07) + 1))
            i += gct_size

    while i < len(data) - 2:
        if data[i] == 0x21 and data[i + 1] == 0xFE:
            # Comment extension
            i += 2
            comment_bytes = bytearray()
            while i < len(data):
                sub_block_size = data[i]
                i += 1
                if sub_block_size == 0:
                    break
                comment_bytes.extend(data[i : i + sub_block_size])
                i += sub_block_size
            try:
                text_payload = comment_bytes.decode("latin-1", errors="ignore")
            except Exception:
                text_payload = ""
            if text_payload:
                metadata_total += len(text_payload)
                findings.extend(_check_text_payload(text_payload, "gif_comment", f"offset {i - len(comment_bytes)}"))
        elif data[i] == 0x21 and data[i + 1] == 0x3B:
            # Trailer
            break
        elif data[i] == 0x2C:
            # Image descriptor — skip to image data
            if i + 9 < len(data):
                i += 9  # LSD + image descriptor
                # Local color table if present
                lct_flag = (data[i - 1] >> 7) & 1
                if lct_flag:
                    lct_size = 3 * (2 ** ((data[i - 1] & 0x07) + 1))
                    i += 1 + lct_size
                # Skip LZW minimum code size + sub-blocks
                if i < len(data):
                    i += 1  # LZW minimum code size
                    while i < len(data):
                        sub_block_size = data[i]
                        i += 1
                        if sub_block_size == 0:
                            break
                        i += sub_block_size
        else:
            i += 1

    if metadata_total > _METADATA_SIZE_WARN:
        findings.append(
            ImageInjectionFinding(
                layer_type="large_metadata",
                content=f"GIF comment extension size: {metadata_total} bytes",
                location="gif",
                severity=ImageInjectionSeverity.HIGH,
                description=f"Unusually large GIF comment ({metadata_total} bytes) — possible hidden payload",
                confidence=0.75,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# SVG masquerade detection
# ---------------------------------------------------------------------------


def _scan_svg_masquerade(data: bytes) -> list[ImageInjectionFinding]:
    """Detect SVG content masquerading inside PNG/JPEG files."""
    findings = []
    # Look for SVG XML declaration or <svg> tag anywhere in the file
    # This catches SVG injected into "image" files that should only contain raster data
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return findings

    # SVG signature patterns
    svg_sigs = [
        b"<svg",
        b"<SVG",
        b"<?xml",
        b"<!DOCTYPE svg",
        b"<!DOCTYPE SVG",
    ]

    for sig in svg_sigs:
        pos = 0
        while True:
            idx = text.find(sig.decode("utf-8", errors="replace"), pos)
            if idx < 0:
                break
            # Confirm it's actually an SVG root by checking for xmlns
            # Look in a window around the match
            start = max(0, idx - 20)
            end = min(len(text), idx + 200)
            window = text[start:end]
            if "xmlns" in window or "svg" in window.lower():
                # Suspicious: SVG content found in an image file
                # Extract surrounding text for context
                ctx_start = max(0, idx)
                ctx_end = min(len(text), idx + 100)
                ctx = text[ctx_start:ctx_end].strip()
                findings.append(
                    ImageInjectionFinding(
                        layer_type="svg_masquerade",
                        content=ctx[:_MAX_CONTENT_LEN],
                        location=f"offset {idx}",
                        severity=ImageInjectionSeverity.CRITICAL,
                        description="SVG content detected inside a raster image file — potential SVG-based injection or polyglot",
                        confidence=0.85,
                    )
                )
                break  # one finding per file for SVG masquerade is enough
            pos = idx + 1

    return findings


# ---------------------------------------------------------------------------
# Common payload checker
# ---------------------------------------------------------------------------


def _check_text_payload(text: str, layer_type: str, location: str) -> list[ImageInjectionFinding]:
    """Scan decoded text for injection keywords and return findings."""
    if not text or len(text) < 4:
        return []

    findings = []
    matches = _match_injection(text)
    for matched, _ in matches:
        conf = 0.85 if len(matched) > 20 else 0.7
        findings.append(
            ImageInjectionFinding(
                layer_type=layer_type,
                content=matched[:_MAX_CONTENT_LEN],
                location=location,
                severity=_severity_for_confidence(conf),
                description=f"Injection keyword found in {layer_type} metadata",
                confidence=conf,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_image_injection(content: str) -> list[ImageInjectionFinding]:
    """Scan a text string for embedded image data URIs (base64 images).

    This handles the case where an image is embedded as a data URI in a
    prompt. The caller passes the full text; this function extracts and scans
    any base64-encoded image data found inline.

    For direct image file bytes, use detect_image_injection_bytes() instead.

    Returns:
        List of ImageInjectionFinding objects.
    """
    findings = []

    # Match data URI image patterns
    data_uri_pattern = re.compile(
        r"data:image\/(\w+);base64,([A-Za-z0-9+/=\s]{20,})",
        re.IGNORECASE,
    )
    for m in data_uri_pattern.finditer(content):
        m.group(1).lower()
        b64_data = re.sub(r"\s", "", m.group(2))
        try:
            import base64

            img_bytes = base64.b64decode(b64_data + "==")
            findings.extend(detect_image_injection_bytes(img_bytes))
        except Exception:
            pass

    return findings


def detect_image_injection_bytes(data: bytes) -> list[ImageInjectionFinding]:
    """Scan raw image bytes for hidden injection payloads.

    Automatically detects format (JPEG, PNG, GIF, or unknown) and scans
    the appropriate metadata structures.

    Args:
        data: Raw bytes of an image file.

    Returns:
        List of ImageInjectionFinding objects. Empty list means no findings.
    """
    if len(data) < 8:
        return []

    # Auto-detect format
    if data[0:2] == b"\xff\xd8":
        findings = _scan_jpeg(data)
    elif data[0:8] == b"\x89PNG\r\n\x1a\n":
        findings = _scan_png(data)
    elif data.startswith((b"GIF87a", b"GIF89a")):
        findings = _scan_gif(data)
    else:
        findings = []

    # SVG masquerade is checked regardless of format (polyglot files)
    svg_findings = _scan_svg_masquerade(data)
    findings.extend(svg_findings)

    return findings
