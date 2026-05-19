"""
Media steganography scanner for HermesKatana.

Lightweight statistical detection of steganographic injection in media files:
- LSB (Least Significant Bit) anomaly detection in images
- Chi-square test for LSB plane randomness
- RS (Regular/Singular) analysis for pixel group correlations
- Audio spectral/metadata anomaly heuristics
- File size inflation detection
- Known stego tool signatures (OpenStego, Steghide, Invisible Secrets, etc.)

Usage::

    from hermes_katana.scanner.stego_scanner import StegoReport, scan_stego

    report = scan_stego(image_bytes)
    print(report.stego_score)  # 0.0-1.0
    if report.has_findings:
        for flag in report.flags:
            print(flag.type, flag.severity)
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

__all__ = [
    "StegoCategory",
    "StegoSeverity",
    "StegoFlag",
    "StegoReport",
    "scan_stego",
    "lsb_anomaly_score",
    "chi_square_lsb",
    "rs_analysis",
    "detect_stego_signatures",
    "file_size_anomaly_ratio",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StegoCategory(str, Enum):
    """Categories of steganographic indicators."""

    LSB_ANOMALY = "lsb_anomaly"
    """Unusual LSB distribution suggesting embedded data."""

    CHI_SQUARE = "chi_square"
    """Chi-square test detected non-random LSB patterns."""

    RS_ANALYSIS = "rs_analysis"
    """Regular-Singular analysis found pixel group anomalies."""

    AUDIO_ANOMALY = "audio_anomaly"
    """Unusual audio frequency/metadata patterns."""

    SIZE_ANOMALY = "size_anomaly"
    """File size inconsistent with declared content type."""

    SIGNATURE_HIT = "signature_hit"
    """Known steganography tool signature detected."""


class StegoSeverity(str, Enum):
    """Severity levels for stego findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ---------------------------------------------------------------------------
# Finding & Report
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StegoFlag:
    """A single steganography detection finding."""

    type: str
    """Short tag identifying the finding type."""

    category: StegoCategory
    """Which detection method produced this flag."""

    severity: StegoSeverity
    """Severity of the finding."""

    description: str
    """Human-readable description of the finding."""

    confidence: float = 0.5
    """Detection confidence 0.0-1.0."""

    location: Optional[str] = None
    """Where in the file the anomaly was detected."""

    value: Optional[float] = None
    """Raw numeric value (e.g. chi-square statistic)."""


@dataclass
class StegoReport:
    """Report from steganography scanning a media file."""

    flags: list[StegoFlag] = field(default_factory=list)
    stego_score: float = 0.0
    """Aggregate steganography likelihood 0.0-1.0."""

    media_type: str = "unknown"
    """Detected media type (image, audio, video, unknown)."""

    file_size: int = 0
    """File size in bytes."""

    sample_size: int = 0
    """Number of bytes/pixels actually analysed."""

    metadata: dict = field(default_factory=dict)

    def has_findings(self) -> bool:
        return bool(self.flags)

    @property
    def is_suspicious(self) -> bool:
        return self.stego_score >= 0.5

    def to_dict(self) -> dict:
        return {
            "stego_score": round(self.stego_score, 4),
            "media_type": self.media_type,
            "file_size": self.file_size,
            "sample_size": self.sample_size,
            "has_findings": self.has_findings(),
            "flags": [
                {
                    "type": f.type,
                    "category": f.category.value,
                    "severity": f.severity.value,
                    "description": f.description,
                    "confidence": round(f.confidence, 4),
                    "location": f.location,
                    "value": f.value,
                }
                for f in self.flags
            ],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Media type detection
# ---------------------------------------------------------------------------

# Magic bytes for common image formats
_IMAGE_SIGNATURES = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"RIFF", "wav"),  # RIFF can be WAV or AVI
    (b"\x00\x00\x01\x00", "ico"),
    (b"II*\x00", "tiff"),  # Little-endian TIFF
    (b"MM\x00*", "tiff"),  # Big-endian TIFF
    (b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", "raw"),  # ambiguous
]

# Known steganography tool magic/signature bytes
_STEGO_SIGNATURES: list[tuple[bytes, str, str, StegoSeverity]] = [
    # Steghide magic bytes (after JPEG DHT segment)
    (b"\xfe\xff", "steghide", "Steghide magic marker", StegoSeverity.HIGH),
    # OpenStego watermark signature
    (b"OpenStego", "openstego", "OpenStego signature found", StegoSeverity.HIGH),
    # Invisible Secrets 4 marker
    (b"IS4", "invisible_secrets", "Invisible Secrets 4 marker", StegoSeverity.HIGH),
    # Camouflage signature
    (b"!CAM", "camouflage", "Camouflage marker", StegoSeverity.MEDIUM),
    # Digital Invisible Ink Palette
    (b"DIIP", "diip", "Digital Invisible Ink marker", StegoSeverity.MEDIUM),
    # SecretSharp marker
    (b"SSIG", "secretsharp", "SecretSharp signature", StegoSeverity.MEDIUM),
    # JSteg comment marker (jpeg)
    (b"JSteg", "jsteg", "JSteg comment marker", StegoSeverity.MEDIUM),
    # OutGuess signature
    (b"OutGuess", "outguess", "OutGuess signature", StegoSeverity.MEDIUM),
    # OpenPuff signature (tries to find the 3-marker sequence)
    (b"OpenPuff", "openpuff", "OpenPuff signature", StegoSeverity.HIGH),
    # Appended null data padding pattern (common stego)
    (b"\x00\x00\x00", "null_padding", "Unusual null-byte padding", StegoSeverity.LOW),
]

# Known PNG tEXt chunks used by stego tools
_STEGO_PNG_CHUNKS = re.compile(
    b"(?<=\\x00)((?:steg|hidetext|comment|stuff|stego|hide)[^\x00]{0,50}\\x00)",
    re.IGNORECASE,
)

# Audio metadata patterns suggesting hidden data
_AUDIO_STEGO_PATTERNS = [
    (b"SSRE", "audio_manipulation"),  # unusual marker
    (b"MICRO", "hidden_audio"),  # embedded audio marker
]


def _detect_media_type(data: bytes) -> str:
    """Detect media type from magic bytes."""
    for magic, media_type in _IMAGE_SIGNATURES:
        if data[: len(magic)] == magic:
            # RIFF can be WAV or AVI - check sub-format
            if magic == b"RIFF":
                if data[8:12] == b"WAVE":
                    return "audio"
                return "video"
            return media_type
    # Check for audio formats by extension-like content
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio"
    return "unknown"


# ---------------------------------------------------------------------------
# LSB Anomaly Detection
# ---------------------------------------------------------------------------

# Thresholds
_LSB_UNIFORM_THRESHOLD = 0.52
"""If LSB distribution deviates >52% from 50/50, flag as suspicious."""

_LSB_CHI_SQUARE_CRITICAL = 0.001
"""p-value below which chi-square result is considered suspicious."""

_RS_THRESHOLD = 0.02
"""RS analysis: if |R - S| difference exceeds this, flag."""


def _extract_lsb_bytes(data: bytes, sample_size: int = 65536) -> list[int]:
    """Extract LSB of each byte as 0/1 list, up to sample_size bytes."""
    actual = min(sample_size, len(data))
    return [(b & 1) for b in data[:actual]]


def lsb_anomaly_score(lsb_bits: list[int]) -> float:
    """Compute how anomalous the LSB distribution is.

    A natural image typically has roughly 50/50 distribution of 0s and 1s
    in the LSB plane (due to natural variation). Steganography often
    shifts this distribution toward an even more uniform state.

    Returns a score 0.0-1.0:
      < 0.3: normal  (near 50/50)
      0.3-0.6: mild deviation
      > 0.6: strongly suspicious
    """
    if len(lsb_bits) < 8:
        return 0.0

    ones = sum(lsb_bits)
    len(lsb_bits) - ones
    ratio = ones / len(lsb_bits)

    # Deviation from 0.5; 0.04 deviation = score 0.8
    deviation = abs(ratio - 0.5)
    score = min(deviation * 10.0, 1.0)
    return round(score, 4)


def chi_square_lsb(lsb_bits: list[int]) -> tuple[float, float]:
    """Chi-square test on LSB plane.

    Computes chi-square statistic testing whether the LSBs are uniformly
    distributed (0 and 1 equally likely).

    Returns:
        (chi_square_statistic, p_value)

    Interpretation:
        High chi-square / low p-value → LSBs are NOT uniformly distributed
        → suspicious (stego often creates near-uniform LSBs)
        → a VERY low chi-square (near 0) can also indicate stego
        because perfectly uniform LSBs are themselves suspicious.
    """
    if len(lsb_bits) < 4:
        return 0.0, 1.0

    ones = sum(lsb_bits)
    zeros = len(lsb_bits) - ones
    expected = len(lsb_bits) / 2

    # Chi-square statistic
    chi2 = ((zeros - expected) ** 2) / expected + ((ones - expected) ** 2) / expected
    chi2 = round(chi2, 4)

    # Approximate p-value using chi-square CDF with 1 degree of freedom
    # Using the regularized gamma function approximation
    p_value = _chi2_survival(chi2, df=1)

    return chi2, p_value


def _chi2_survival(x: float, df: int) -> float:
    """Approximate p-value for chi-square statistic.

    Uses the lower incomplete gamma function approximation for the
    chi-square CDF, then returns 1 - CDF = survival (right-tail p-value).
    """
    if x <= 0:
        return 1.0
    if df <= 0:
        return 1.0

    try:
        import math

        # Use Welch's approximation for large x
        if x > df * 10:
            # Very large chi-square → very small p-value
            return 0.0

        # Wilson-Hilferty approximation
        z = (math.pow(x / df, 1.0 / 3.0) - (1.0 - 2.0 / (9 * df))) / math.sqrt(2.0 / (9 * df))
        import math

        # Standard normal CDF approximation
        p = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
        return max(0.0, min(1.0, 1.0 - p))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# RS (Regular-Singular) Analysis
# ---------------------------------------------------------------------------


def rs_analysis(pixels: bytes, block_size: int = 4) -> float:
    """Perform RS (Regular-Singular) analysis on pixel data.

    RS analysis divides the image into groups of pixels and classifies each
    group as Regular (R), Singular (S), or Unusable (U) based on discrimination
    functions. Steganography shifts the R/S ratio in characteristic ways.

    Args:
        pixels: Raw pixel bytes (grayscale or first channel of each pixel).
        block_size: Number of pixels per group (default 4).

    Returns:
        RS anomaly score 0.0-1.0. Values near 0 = normal image,
        values far from 0 suggest steganographic modification.

    This is a simplified implementation: we compute f1 (discrimination) and
    f-1 for groups, classify R/S, and return the normalised R-S difference.
    """
    if len(pixels) < block_size * 4:
        return 0.0

    def discriminant(vals: list[int]) -> float:
        """Sum of absolute differences from local median."""
        if not vals:
            return 0.0
        m = sum(vals) / len(vals)
        return sum(abs(v - m) for v in vals)

    def flip(vals: list[int], mask: int) -> list[int]:
        """Apply ±1 flip operation to values based on mask bits."""
        out = []
        for i, v in enumerate(vals):
            bit = (mask >> (i % 8)) & 1
            if bit:
                out.append(v ^ 1)  # invert LSB
            else:
                out.append(v)
        return out

    r_count = 0
    s_count = 0
    n_r = 0
    n_s = 0

    # Process in blocks
    step = block_size * 4
    for i in range(0, len(pixels) - step, step):
        block = list(pixels[i : i + step])
        # Use every 4th byte as a simplified "channel"
        channel = block[::4][:block_size]

        f1 = discriminant(channel)
        f_neg1 = discriminant(flip(channel, 0x55))  # mask 01010101

        # Classification threshold
        threshold = 0.0

        if f1 > f_neg1 + threshold:
            r_count += 1
        elif f_neg1 > f1 + threshold:
            s_count += 1

        n_r += 1
        n_s += 1

    if n_r + n_s == 0:
        return 0.0

    # RS anomaly: how much does |R - S| / (R + S) deviate from a normal image?
    total = r_count + s_count
    if total == 0:
        return 0.0

    r_ratio = r_count / total
    s_ratio = s_count / total
    rs_diff = abs(r_ratio - s_ratio)

    # Normal images have some natural R-S difference; large deviations are suspicious
    score = min(rs_diff * 5.0, 1.0)
    return round(score, 4)


# ---------------------------------------------------------------------------
# File size anomaly
# ---------------------------------------------------------------------------


# Expected compression ratios by media type (uncompressed_size / file_size)
# A very high ratio (large uncompressed relative to file) suggests normal compression.
# A very low ratio (file close to uncompressed size) can suggest appended data.
_MEDIA_COMPRESSION_RATIOS = {
    "jpeg": (2.0, 15.0),  # JPEG is already compressed; ratio of declared size to actual
    "png": (1.0, 5.0),  # PNG lossless but compressed
    "gif": (1.0, 4.0),
    "bmp": (8.0, 10.0),  # BMP is uncompressed, ratio ~8-10× for photo content
    "audio": (1.0, 3.0),
    "video": (2.0, 10.0),
    "unknown": (1.0, 20.0),
}


def file_size_anomaly_ratio(
    data: bytes, media_type: str, declared_dimensions: Optional[tuple[int, int]] = None
) -> float:
    """Check if file size is unusually large for its declared dimensions.

    If dimensions are provided, compute expected size for uncompressed storage
    and compare to actual. Steganographic payloads can inflate file size.

    Returns:
        Anomaly score 0.0-1.0.

    Args:
        data: File bytes.
        media_type: Detected media type.
        declared_dimensions: Optional (width, height) tuple for images.
    """
    score = 0.0
    ratios = _MEDIA_COMPRESSION_RATIOS.get(media_type, (1.0, 20.0))

    if declared_dimensions and media_type in ("jpeg", "png", "gif", "bmp"):
        w, h = declared_dimensions
        # Estimate uncompressed size (3 bytes per pixel for RGB)
        if media_type == "bmp":
            expected_raw = w * h * 3
        else:
            # For compressed formats, estimate raw for comparison
            expected_raw = w * h * 3

        if expected_raw > 0:
            ratio = len(data) / expected_raw
            # Very small ratio (file much smaller than raw) = normal compression
            # Very large ratio (file close to or larger than raw) = suspicious
            if ratio > ratios[1]:
                score = min((ratio - ratios[1]) / ratios[1], 1.0)
            elif ratio < ratios[0]:
                score = 0.0

    # Generic check: if file is extremely large for its type (no dimensions known)
    # We use a simple heuristic: compare to a minimum reasonable size
    min_reasonable_sizes = {
        "jpeg": 1024,
        "png": 256,
        "gif": 128,
        "bmp": 1024,
        "audio": 4096,
        "video": 10240,
    }
    min_size = min_reasonable_sizes.get(media_type, 512)

    if len(data) < min_size:
        # Very small file for its type
        score = 0.2

    return round(score, 4)


# ---------------------------------------------------------------------------
# Steganography tool signature detection
# ---------------------------------------------------------------------------


def detect_stego_signatures(data: bytes, media_type: str) -> list[StegoFlag]:
    """Scan for known steganography tool signatures.

    Args:
        data: File bytes.
        media_type: Detected media type.

    Returns:
        List of StegoFlags for each detected signature.
    """
    flags = []

    # Check raw byte signatures
    for magic, tool_name, description, severity in _STEGO_SIGNATURES:
        pos = data.find(magic)
        if pos != -1:
            flags.append(
                StegoFlag(
                    type=f"sig_{tool_name}",
                    category=StegoCategory.SIGNATURE_HIT,
                    severity=severity,
                    description=description,
                    confidence=0.95,
                    location=f"offset_{pos}",
                    value=float(pos),
                )
            )

    # PNG tEXt chunk scan
    if media_type == "png":
        for match in _STEGO_PNG_CHUNKS.finditer(data):
            chunk_data = match.group(0)
            if len(chunk_data) > 3:
                flags.append(
                    StegoFlag(
                        type="png_stego_chunk",
                        category=StegoCategory.SIGNATURE_HIT,
                        severity=StegoSeverity.MEDIUM,
                        description="Suspicious PNG text chunk possibly from stego tool",
                        confidence=0.75,
                        location=f"offset_{match.start()}",
                    )
                )

    # JPEG COM and APPx segment scan for hidden data markers
    if media_type == "jpeg":
        # Look for unusual APP segments (stego tools often use APP11, APP12, etc.)
        # APP11=0xEB, APP12=0xEC, APP13=0xED, APP14=0xEE, APP15=0xEF
        unusual_app = re.compile(b"\xff\xeb|\xff\xec|\xff\xed|\xff\xee|\xff\xef", re.DOTALL)
        for match in unusual_app.finditer(data):
            # Check if segment has unusual size/content
            pos = match.start()
            if pos + 4 < len(data):
                seg_len = struct.unpack(">H", data[pos + 2 : pos + 4])[0]
                # Very large or very small segments can indicate hidden data
                if seg_len < 2 or seg_len > 65500:
                    flags.append(
                        StegoFlag(
                            type="jpeg_unusual_app",
                            category=StegoCategory.SIGNATURE_HIT,
                            severity=StegoSeverity.MEDIUM,
                            description=f"Unusual JPEG APP segment (length={seg_len})",
                            confidence=0.6,
                            location=f"offset_{pos}",
                            value=float(seg_len),
                        )
                    )

    # Check for appended zlib streams (common in LSB-in-PNG stego)
    # Scan for zlib headers (0x78 0x9c or 0x78 0x01 etc.) in unusual locations
    zlib_header_positions = []
    for i in range(len(data) - 2):
        if data[i] == 0x78 and data[i + 1] in (0x9C, 0x01, 0xDA, 0x5E):
            zlib_header_positions.append(i)

    if len(zlib_header_positions) > 5:
        flags.append(
            StegoFlag(
                type="multiple_zlib_headers",
                category=StegoCategory.SIGNATURE_HIT,
                severity=StegoSeverity.MEDIUM,
                description=f"Multiple zlib compression headers found ({len(zlib_header_positions)}), unusual for standard media",
                confidence=0.65,
            )
        )

    return flags


# ---------------------------------------------------------------------------
# Audio anomaly detection
# ---------------------------------------------------------------------------


def _audio_metadata_anomalies(data: bytes) -> list[StegoFlag]:
    """Detect audio-specific steganographic anomalies."""
    flags = []

    # WAV file header check
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        # Parse WAV header
        if len(data) > 44:
            try:
                # Extract fmt chunk
                pos = 12
                while pos < min(len(data) - 8, 1000):
                    chunk_id = data[pos : pos + 4]
                    chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
                    if chunk_id == b"fmt ":
                        if chunk_size < 16:
                            flags.append(
                                StegoFlag(
                                    type="wav_truncated_fmt",
                                    category=StegoCategory.AUDIO_ANOMALY,
                                    severity=StegoSeverity.LOW,
                                    description="WAV format chunk is smaller than minimum",
                                    confidence=0.5,
                                    location="wav_header",
                                )
                            )
                        break
                    pos += 8 + chunk_size

                # Look for hidden data appended after fact chunk
                # Find data chunk
                pos = 12
                while pos < len(data) - 8:
                    chunk_id = data[pos : pos + 4]
                    chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
                    if chunk_id == b"data":
                        # data_size in header vs actual remaining bytes
                        expected = chunk_size
                        actual = len(data) - (pos + 8)
                        if actual > expected + 100:
                            flags.append(
                                StegoFlag(
                                    type="wav_extra_data",
                                    category=StegoCategory.AUDIO_ANOMALY,
                                    severity=StegoSeverity.MEDIUM,
                                    description=f"WAV file has {actual - expected} extra bytes after declared audio data",
                                    confidence=0.7,
                                    location=f"offset_{pos + 8 + expected}",
                                    value=float(actual - expected),
                                )
                            )
                        break
                    pos += 8 + chunk_size
            except Exception:
                pass

    # MP3 stego markers (MP3Stego uses specific frames)
    if len(data) > 10 and (data[:3] == b"ID3" or data[:2] == b"\xff\xfb"):
        # Look for unusual frame sizes or padding
        # This is a lightweight heuristic
        null_paddings = data.count(b"\x00\x00\x00\x00")
        if null_paddings > 10:
            flags.append(
                StegoFlag(
                    type="mp3_unusual_padding",
                    category=StegoCategory.AUDIO_ANOMALY,
                    severity=StegoSeverity.LOW,
                    description=f"MP3 contains {null_paddings} null-padding blocks, possibly hidden data",
                    confidence=0.4,
                )
            )

    return flags


# ---------------------------------------------------------------------------
# Image-specific analysis
# ---------------------------------------------------------------------------


def _image_analysis(data: bytes, media_type: str, sample_size: int = 65536) -> list[StegoFlag]:
    """Run all image-specific steganography analyses."""
    flags = []

    # Extract LSBs from image data (skip header for compressed formats)
    header_skip = 0
    if media_type == "jpeg":
        # Skip JPEG header - find start of scan (SOS segment)
        sos = data.find(b"\xff\xda")  # Start of Scan
        header_skip = sos + 2 if sos != -1 else 0
    elif media_type == "png":
        # Skip PNG signature + IHDR chunk
        header_skip = 8 + 25 if len(data) > 33 else 8
    elif media_type == "gif":
        # Skip GIF header (13 bytes + global color table if present)
        header_skip = 13
        if len(data) > 10 and data[10] & 0x80:
            color_table_size = 3 * (2 ** ((data[10] & 0x07) + 1))
            header_skip += color_table_size

    lsb_bits = _extract_lsb_bytes(data[header_skip:], sample_size=sample_size)

    # 1. LSB anomaly score
    lsb_score = lsb_anomaly_score(lsb_bits)
    if lsb_score > 0.5:
        severity = StegoSeverity.HIGH if lsb_score > 0.7 else StegoSeverity.MEDIUM
        flags.append(
            StegoFlag(
                type="lsb_distribution_anomaly",
                category=StegoCategory.LSB_ANOMALY,
                severity=severity,
                description=f"LSB distribution deviation={lsb_score:.3f} from expected 50/50",
                confidence=lsb_score,
                value=lsb_score,
            )
        )

    # 2. Chi-square test
    if len(lsb_bits) >= 256:
        chi2_stat, p_value = chi_square_lsb(lsb_bits)
        # Suspicious if p-value is very low (non-uniform) OR very close to 1.0
        # (perfectly uniform LSBs are also suspicious for natural images)
        is_suspicious = p_value < _LSB_CHI_SQUARE_CRITICAL or p_value > 0.9999
        if is_suspicious:
            # Confidence based on extremeness of the result
            if p_value < _LSB_CHI_SQUARE_CRITICAL:
                conf = min(0.5 + (0.5 * (1.0 - p_value / _LSB_CHI_SQUARE_CRITICAL)), 0.99)
            else:
                conf = min(0.5 + 0.5 * p_value, 0.99)

            flags.append(
                StegoFlag(
                    type="chi_square_lsb",
                    category=StegoCategory.CHI_SQUARE,
                    severity=StegoSeverity.HIGH if conf > 0.8 else StegoSeverity.MEDIUM,
                    description=f"Chi-square test: stat={chi2_stat:.4f}, p={p_value:.6f}",
                    confidence=round(conf, 4),
                    value=chi2_stat,
                )
            )

    # 3. RS analysis
    if len(lsb_bits) >= 256:
        rs_score = rs_analysis(data[header_skip:], block_size=4)
        if rs_score > 0.3:
            flags.append(
                StegoFlag(
                    type="rs_pixel_anomaly",
                    category=StegoCategory.RS_ANALYSIS,
                    severity=StegoSeverity.HIGH if rs_score > 0.6 else StegoSeverity.MEDIUM,
                    description=f"RS analysis anomaly score={rs_score:.4f}",
                    confidence=rs_score,
                    value=rs_score,
                )
            )

    return flags


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

# Severity weights for aggregation
_SEVERITY_WEIGHT = {
    StegoSeverity.CRITICAL: 1.0,
    StegoSeverity.HIGH: 0.8,
    StegoSeverity.MEDIUM: 0.5,
    StegoSeverity.LOW: 0.2,
    StegoSeverity.INFO: 0.05,
}


def scan_stego(
    data: bytes,
    media_type: Optional[str] = None,
    declared_dimensions: Optional[tuple[int, int]] = None,
    sample_size: int = 65536,
) -> StegoReport:
    """Scan media bytes for steganographic injection.

    Args:
        data: Raw bytes of the media file.
        media_type: Optional media type hint ('jpeg', 'png', 'gif', 'bmp',
            'audio', 'video'). If None, auto-detected from magic bytes.
        declared_dimensions: Optional (width, height) for images, used in
            size anomaly detection.
        sample_size: Max bytes to sample for LSB/statistical analysis
            (default 64KB). Set lower for very large files.

    Returns:
        A :class:`StegoReport` with findings and aggregate score.

    Example::

        with open("suspect.jpg", "rb") as f:
            report = scan_stego(f.read())
        if report.is_suspicious:
            print(f"Stego score: {report.stego_score}")
            for flag in report.flags:
                print(f"  [{flag.severity.value}] {flag.type}: {flag.description}")
    """
    report = StegoReport(
        file_size=len(data),
        sample_size=sample_size,
    )

    if len(data) < 8:
        report.metadata["error"] = "file_too_small"
        return report

    # Detect media type
    detected_type = media_type or _detect_media_type(data)
    report.media_type = detected_type

    # 1. Signature detection (all types)
    sig_flags = detect_stego_signatures(data, detected_type)
    report.flags.extend(sig_flags)

    # 2. File size anomaly
    size_score = file_size_anomaly_ratio(data, detected_type, declared_dimensions)
    if size_score > 0.3:
        report.flags.append(
            StegoFlag(
                type="size_inflation",
                category=StegoCategory.SIZE_ANOMALY,
                severity=StegoSeverity.MEDIUM,
                description=f"File size anomaly ratio={size_score:.4f} for {detected_type}",
                confidence=size_score,
                value=size_score,
            )
        )

    # 3. Image-specific analysis
    if detected_type in ("jpeg", "png", "gif", "bmp"):
        img_flags = _image_analysis(data, detected_type, sample_size=sample_size)
        report.flags.extend(img_flags)

    # 4. Audio-specific analysis
    elif detected_type == "audio":
        audio_flags = _audio_metadata_anomalies(data)
        report.flags.extend(audio_flags)

    # Compute aggregate score
    if report.flags:
        # Weighted sum with diminishing returns for multiple flags
        total_weight = 0.0
        for f in report.flags:
            w = _SEVERITY_WEIGHT.get(f.severity, 0.3)
            total_weight += w * f.confidence

        # Diminishing returns: extra flags count less
        report.stego_score = min(total_weight / (1.0 + 0.25 * (len(report.flags) - 1)), 1.0)
        report.stego_score = round(report.stego_score, 4)
    else:
        report.stego_score = 0.0

    # Add summary metadata
    report.metadata["total_flags"] = len(report.flags)
    report.metadata["header_skipped"] = 0
    if detected_type == "jpeg":
        sos = data.find(b"\xff\xda")
        report.metadata["header_skipped"] = sos + 2 if sos != -1 else 0
    elif detected_type == "png":
        report.metadata["header_skipped"] = 33 if len(data) > 33 else 8

    return report
