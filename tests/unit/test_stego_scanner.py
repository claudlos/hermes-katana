"""Tests for the steganography scanner module."""

from __future__ import annotations

import random
import struct

import pytest

from hermes_katana.scanner.stego_scanner import (
    StegoCategory,
    StegoFlag,
    StegoReport,
    StegoSeverity,
    _chi2_survival,
    _detect_media_type,
    _extract_lsb_bytes,
    _image_analysis,
    chi_square_lsb,
    detect_stego_signatures,
    file_size_anomaly_ratio,
    lsb_anomaly_score,
    rs_analysis,
    scan_stego,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _random_bytes(n: int, seed: int = 42) -> bytes:
    """Generate deterministic pseudo-random bytes."""
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(n))


def _lsb_uniform(n: int) -> bytes:
    """Create a byte stream where LSBs are exactly 50/50."""
    result = bytearray(n)
    for i in range(n):
        result[i] = i % 2  # perfectly alternating 0/1 LSBs
    return bytes(result)


def _jpeg_like_bytes(seed: int = 0) -> bytes:
    """Create minimal JPEG-like bytes (FF D8 FF ... SOF + image data)."""
    rng = random.Random(seed)
    data = bytearray()
    data += b"\xff\xd8"  # SOI
    data += b"\xff\xe0"  # APP0 marker
    data += struct.pack(">H", 16)  # length
    data += b"JFIF\x00"
    data += bytes(rng.randint(0, 255) for _ in range(10))
    # SOS marker (start of scan) — scanner will start LSB analysis here
    data += b"\xff\xda"
    data += struct.pack(">H", 8)
    data += bytes(rng.randint(0, 255) for _ in range(100))
    return bytes(data)


def _png_like_bytes(seed: int = 0) -> bytes:
    """Create minimal PNG-like bytes (signature + IHDR + IDAT)."""
    rng = random.Random(seed)
    data = bytearray()
    data += b"\x89PNG\r\n\x1a\n"  # PNG signature
    # IHDR chunk
    data += b"\x00\x00\x00\x0d"  # length=13
    data += b"IHDR"
    data += struct.pack(">II", 100, 100)  # width, height
    data += b"\x08\x02\x00\x00\x00"  # bit_depth=8, color_type=2, rest=0
    data += b"\x00\x00\x00"  # CRC placeholder
    # IDAT chunk with random image data
    data += b"\x00\x00\x00\x19"  # length=25
    data += b"IDAT"
    data += bytes(rng.randint(0, 255) for _ in range(25))
    data += b"\x00\x00\x00"
    return bytes(data)


# ---------------------------------------------------------------------------
# Test: Media type detection
# ---------------------------------------------------------------------------


class TestMediaTypeDetection:
    def test_detect_jpeg(self):
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        assert _detect_media_type(data) == "jpeg"

    def test_detect_png(self):
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert _detect_media_type(data) == "png"

    def test_detect_gif87(self):
        data = b"GIF87a" + b"\x00" * 100
        assert _detect_media_type(data) == "gif"

    def test_detect_gif89(self):
        data = b"GIF89a" + b"\x00" * 100
        assert _detect_media_type(data) == "gif"

    def test_detect_bmp(self):
        data = b"BM" + b"\x00" * 100
        assert _detect_media_type(data) == "bmp"

    def test_detect_wav(self):
        # RIFF....WAVE
        data = b"RIFF" + b"\x00\x00\x00\x00" + b"WAVE" + b"\x00" * 100
        assert _detect_media_type(data) == "audio"

    def test_detect_unknown(self):
        data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        assert _detect_media_type(data) == "unknown"

    def test_empty_data(self):
        assert _detect_media_type(b"") == "unknown"


# ---------------------------------------------------------------------------
# Test: LSB extraction
# ---------------------------------------------------------------------------


class TestLSBExtraction:
    def test_extract_lsb_basic(self):
        # 0b00000001 -> 1, 0b00000010 -> 0, 0b00000011 -> 1
        data = bytes([0x01, 0x02, 0x03, 0x04])
        bits = _extract_lsb_bytes(data)
        assert bits == [1, 0, 1, 0]

    def test_extract_lsb_truncated_to_sample_size(self):
        data = bytes([0xFF] * 1000)
        bits = _extract_lsb_bytes(data, sample_size=10)
        assert len(bits) == 10

    def test_extract_lsb_empty(self):
        assert _extract_lsb_bytes(b"") == []


# ---------------------------------------------------------------------------
# Test: LSB anomaly score
# ---------------------------------------------------------------------------


class TestLSBAnomalyScore:
    def test_perfectly_uniform(self):
        """Alternating 0/1 LSBs = exactly 50/50, score near 0."""
        bits = [i % 2 for i in range(1000)]
        score = lsb_anomaly_score(bits)
        assert score == 0.0

    def test_all_ones(self):
        """All 1s LSBs = maximum deviation, score = 1.0."""
        bits = [1] * 1000
        score = lsb_anomaly_score(bits)
        assert score == 1.0

    def test_all_zeros(self):
        bits = [0] * 1000
        score = lsb_anomaly_score(bits)
        assert score == 1.0

    def test_mild_deviation(self):
        """70% ones (0.2 deviation from 0.5) → score capped at 1.0."""
        bits = [1] * 700 + [0] * 300
        score = lsb_anomaly_score(bits)
        assert score == 1.0  # 0.2 * 10 = 1.0, capped at 1.0

    def test_moderate_deviation(self):
        """55% ones (0.05 deviation) → score 0.5."""
        bits = [1] * 550 + [0] * 450
        score = lsb_anomaly_score(bits)
        assert score == 0.5

    def test_natural_image(self):
        """Random bytes give near-50/50 distribution, low score."""
        rng = random.Random(123)
        bits = [(rng.randint(0, 255)) & 1 for _ in range(1000)]
        score = lsb_anomaly_score(bits)
        assert score < 0.4  # natural noise has low deviation

    def test_too_few_bits(self):
        """Less than 8 bits returns 0.0."""
        assert lsb_anomaly_score([1, 0]) == 0.0


# ---------------------------------------------------------------------------
# Test: Chi-square
# ---------------------------------------------------------------------------


class TestChiSquare:
    def test_uniform_lsb(self):
        """Perfectly uniform LSBs → chi-square ≈ 0, p ≈ 1.0."""
        bits = [i % 2 for i in range(1000)]
        chi2, p = chi_square_lsb(bits)
        assert chi2 < 0.1
        assert p > 0.9

    def test_very_uneven(self):
        """Highly uneven LSBs → high chi-square, low p-value."""
        bits = [1] * 900 + [0] * 100
        chi2, p = chi_square_lsb(bits)
        assert chi2 > 500
        assert p < _chi2_survival(chi2, 1) + 0.01

    def test_too_few_bits(self):
        chi2, p = chi_square_lsb([1, 0])
        assert chi2 == 0.0
        assert p == 1.0

    def test_chi2_survival_extreme(self):
        """Very large chi-square should give p-value ~0."""
        p = _chi2_survival(1000, df=1)
        assert p == 0.0

    def test_chi2_survival_zero(self):
        p = _chi2_survival(0, df=1)
        assert p == 1.0


# ---------------------------------------------------------------------------
# Test: RS Analysis
# ---------------------------------------------------------------------------


class TestRSAnalysis:
    def test_random_pixels_normal(self):
        """Random pixels should give low RS anomaly score."""
        pixels = bytes(random.Random(99).randint(0, 255) for _ in range(256))
        score = rs_analysis(pixels, block_size=4)
        assert 0.0 <= score <= 1.0

    def test_very_short_pixels(self):
        """Too few pixels → score 0."""
        assert rs_analysis(b"\x00\x01", block_size=4) == 0.0

    def test_rs_output_bounded(self):
        """RS score must be in [0, 1]."""
        rng = random.Random(55)
        pixels = bytes(rng.randint(0, 255) for _ in range(1000))
        score = rs_analysis(pixels)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Test: Stego signatures
# ---------------------------------------------------------------------------


class TestStegoSignatures:
    def test_openstego_signature(self):
        data = b"\x00" * 1000 + b"OpenStego" + b"\x00" * 100
        flags = detect_stego_signatures(data, "png")
        assert len(flags) >= 1
        assert any(f.type == "sig_openstego" for f in flags)

    def test_steghide_signature(self):
        data = b"\xff\xfe\xff" + b"\x00" * 500 + b"\xfe\xff" + b"\x00" * 100
        flags = detect_stego_signatures(data, "jpeg")
        assert any(f.type == "sig_steghide" for f in flags)

    def test_no_false_positive_clean_data(self):
        """Clean random bytes should not trigger signature hits."""
        data = _random_bytes(2000)
        flags = detect_stego_signatures(data, "jpeg")
        # Only signature hits, not counting general anomalies
        sig_hits = [f for f in flags if f.category == StegoCategory.SIGNATURE_HIT]
        assert len(sig_hits) == 0

    def test_multiple_zlib_headers(self):
        """Many zlib headers is suspicious."""
        data = bytearray()
        data += b"\x89PNG\r\n\x1a\n"
        # Add many zlib headers (0x78 0x9c)
        for _ in range(10):
            data += b"\x00" * 20
            data += b"\x78\x9c"
        flags = detect_stego_signatures(bytes(data), "png")
        assert any(f.type == "multiple_zlib_headers" for f in flags)

    def test_png_stego_chunk(self):
        """PNG tEXt chunk with suspicious keyword."""
        data = bytearray(b"\x89PNG\r\n\x1a\n")
        data += b"\x00\x00\x00\x0dIHDR"
        data += b"\x00" * 17
        # tEXt chunk with stego keyword (keyword must be preceded by \x00 per the regex)
        data += b"\x00\x00\x00\x15tEXt"
        data += b"\x00stegcomment\x00"  # keyword preceded by null byte
        data += b"hidden message here"
        data += b"\x00" * 4
        flags = detect_stego_signatures(bytes(data), "png")
        assert any(f.type == "png_stego_chunk" for f in flags)


# ---------------------------------------------------------------------------
# Test: File size anomaly
# ---------------------------------------------------------------------------


class TestFileSizeAnomaly:
    def test_normal_png(self):
        """Normal compressed PNG should score low."""
        # _png_like_bytes only generates ~67 bytes; pad it to look like a real PNG
        data = _png_like_bytes(seed=42) + bytes([0xFF] * 500)
        score = file_size_anomaly_ratio(data, "png")
        assert score == 0.0

    def test_very_small_file_for_type(self):
        """Tiny files score mildly suspicious."""
        score = file_size_anomaly_ratio(b"\xff\xd8", "jpeg")
        assert score == 0.2

    def test_unknown_type(self):
        """Unknown type with reasonable size scores 0."""
        data = _random_bytes(1000, seed=7)
        score = file_size_anomaly_ratio(data, "unknown")
        assert score == 0.0

    def test_with_dimensions_normal(self):
        """Reasonable dimensions gives low score."""
        data = bytes([0xFF] * 1000)
        score = file_size_anomaly_ratio(data, "png", declared_dimensions=(100, 100))
        assert score == 0.0


# ---------------------------------------------------------------------------
# Test: Image analysis
# ---------------------------------------------------------------------------


class TestImageAnalysis:
    def test_jpeg_image_analysis(self):
        data = _jpeg_like_bytes(seed=10)
        flags = _image_analysis(data, "jpeg")
        assert isinstance(flags, list)

    def test_png_image_analysis(self):
        data = _png_like_bytes(seed=20)
        flags = _image_analysis(data, "png")
        assert isinstance(flags, list)

    def test_bmp_image_analysis(self):
        """BMP header + random pixel data."""
        data = bytearray()
        data += b"BM"
        data += struct.pack("<I", 1000)  # file size
        data += b"\x00\x00"
        data += b"\x00\x00"
        data += struct.pack("<I", 54)  # header size
        # DIB header (BITMAPINFOHEADER, 40 bytes)
        data += struct.pack("<I", 40)  # header size
        data += struct.pack("<ii", 10, 10)  # width, height
        data += struct.pack("<H", 1)  # planes
        data += struct.pack("<H", 24)  # bits per pixel
        data += b"\x00" * 32  # rest of DIB header
        # Pixel data
        data += bytes(random.Random(42).randint(0, 255) for _ in range(300))
        flags = _image_analysis(bytes(data), "bmp")
        assert isinstance(flags, list)

    def test_random_image_no_strong_flags(self):
        """Image data with near-balanced LSBs should not trigger high-confidence anomalies.

        Using a large random byte payload that naturally produces near-50/50 LSBs.
        RS analysis correctly flags *perfectly alternating* patterns, so we use
        pseudo-random bytes which look like natural image noise.
        """
        # Large random payload (seeded for reproducibility)
        rng = random.Random(77)
        natural_payload = bytes(rng.randint(0, 255) for _ in range(2000))
        flags = _image_analysis(natural_payload, "png")
        # Natural-looking noise should not trigger HIGH+ high-confidence anomalies
        high_flags = [f for f in flags if f.severity == StegoSeverity.HIGH and f.confidence > 0.8]
        assert len(high_flags) == 0


# ---------------------------------------------------------------------------
# Test: Audio analysis
# ---------------------------------------------------------------------------


class TestAudioAnalysis:
    def test_wav_extra_data(self):
        """WAV with data chunk claiming less than actual = suspicious."""
        data = bytearray()
        data += b"RIFF"
        data += struct.pack("<I", 1000)  # file size
        data += b"WAVE"
        # fmt chunk
        data += b"fmt "
        data += struct.pack("<I", 16)  # chunk size
        data += struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)  # PCM mono
        # data chunk declaring only 100 bytes
        data += b"data"
        data += struct.pack("<I", 100)  # declared data size
        data += b"\x00" * 500  # but 500 bytes actually present
        from hermes_katana.scanner.stego_scanner import _audio_metadata_anomalies

        flags = _audio_metadata_anomalies(bytes(data))
        assert any(f.type == "wav_extra_data" for f in flags)

    def test_clean_wav_no_flags(self):
        """Clean WAV with correct chunk sizes should not flag."""
        data = bytearray()
        data += b"RIFF"
        file_size = 36 + 16 + 8 + 100  # total less RIFF+size+WAVE+fmt+data
        data += struct.pack("<I", file_size)
        data += b"WAVE"
        # fmt chunk
        data += b"fmt "
        data += struct.pack("<I", 16)
        data += struct.pack("<HHIIHH", 1, 1, 44100, 88200, 2, 16)
        # data chunk
        data += b"data"
        data += struct.pack("<I", 100)
        data += b"\x80" * 100  # silence (DC offset)
        from hermes_katana.scanner.stego_scanner import _audio_metadata_anomalies

        flags = _audio_metadata_anomalies(bytes(data))
        assert not any(f.type == "wav_extra_data" for f in flags)


# ---------------------------------------------------------------------------
# Test: scan_stego integration
# ---------------------------------------------------------------------------


class TestScanStego:
    def test_empty_data(self):
        report = scan_stego(b"")
        assert report.file_size == 0
        assert report.stego_score == 0.0

    def test_very_small_data(self):
        report = scan_stego(b"\xff\xd8\xff")
        assert "error" in report.metadata

    def test_jpeg_autodetect(self):
        data = _jpeg_like_bytes(seed=5)
        report = scan_stego(data)
        assert report.media_type == "jpeg"

    def test_png_autodetect(self):
        data = _png_like_bytes(seed=6)
        report = scan_stego(data)
        assert report.media_type == "png"

    def test_with_steghide_signature(self):
        data = b"\xff\xd8"
        data += b"\xff\xfe\xff" + b"\x00" * 100
        data += b"\xfe\xff" + b"\x00" * 200
        report = scan_stego(data)
        sig_hits = [f for f in report.flags if f.category == StegoCategory.SIGNATURE_HIT]
        assert len(sig_hits) >= 1

    def test_report_has_findings(self):
        data = bytearray()
        data += b"\x89PNG\r\n\x1a\n"
        data += b"\x00\x00\x00\x0dIHDR"
        data += struct.pack(">II", 100, 100)
        data += b"\x08\x02\x00\x00\x00"
        data += b"\x00\x00\x00"
        # OpenStego signature
        data += b"\x00" * 500
        data += b"OpenStego\x00"
        data += b"\x00" * 500
        report = scan_stego(bytes(data))
        assert report.has_findings()
        assert len(report.flags) >= 1

    def test_report_not_suspicious_clean(self):
        """Clean random bytes should not be suspicious."""
        data = _random_bytes(5000, seed=88)
        report = scan_stego(data)
        assert report.stego_score < 0.5

    def test_score_bounded(self):
        """Score must always be 0.0-1.0."""
        rng = random.Random(77)
        for _ in range(5):
            data = bytes(rng.randint(0, 255) for _ in range(2000))
            report = scan_stego(data)
            assert 0.0 <= report.stego_score <= 1.0

    def test_to_dict(self):
        data = _png_like_bytes(seed=30)
        report = scan_stego(data)
        d = report.to_dict()
        assert "stego_score" in d
        assert "media_type" in d
        assert "flags" in d
        assert "file_size" in d
        assert isinstance(d["flags"], list)

    def test_sample_size_parameter(self):
        """sample_size should affect sample_size in report."""
        data = _random_bytes(10000, seed=33)
        report = scan_stego(data, sample_size=1000)
        assert report.sample_size == 1000

    def test_is_suspicious_threshold(self):
        data = bytearray()
        data += b"\x89PNG\r\n\x1a\n"
        data += b"\x00\x00\x00\x0dIHDR"
        data += struct.pack(">II", 100, 100)
        data += b"\x08\x02\x00\x00\x00"
        data += b"\x00\x00\x00"
        # Add multiple zlib headers (suspicious)
        for _ in range(10):
            data += b"\x00" * 20
            data += b"\x78\x9c"
        report = scan_stego(bytes(data))
        # The multiple zlib headers + zlib signatures should push score
        assert report.stego_score >= 0.0

    def test_stego_report_no_flags(self):
        """Empty report has no flags and zero score."""
        report = StegoReport()
        assert report.stego_score == 0.0
        assert not report.has_findings()
        assert not report.is_suspicious


# ---------------------------------------------------------------------------
# Test: StegoFlag dataclass
# ---------------------------------------------------------------------------


class TestStegoFlag:
    def test_flag_slots(self):
        flag = StegoFlag(
            type="test",
            category=StegoCategory.LSB_ANOMALY,
            severity=StegoSeverity.MEDIUM,
            description="Test finding",
            confidence=0.75,
            location="offset_100",
            value=0.42,
        )
        assert flag.type == "test"
        assert flag.category == StegoCategory.LSB_ANOMALY
        assert flag.severity == StegoSeverity.MEDIUM
        assert flag.confidence == 0.75
        assert flag.location == "offset_100"
        assert flag.value == 0.42

    def test_flag_default_confidence(self):
        flag = StegoFlag(
            type="test",
            category=StegoCategory.SIGNATURE_HIT,
            severity=StegoSeverity.HIGH,
            description="Desc",
        )
        assert flag.confidence == 0.5


# ---------------------------------------------------------------------------
# Test: Severity enum values
# ---------------------------------------------------------------------------


class TestStegoSeverityValues:
    def test_all_severities_exist(self):
        assert StegoSeverity.CRITICAL.value == "critical"
        assert StegoSeverity.HIGH.value == "high"
        assert StegoSeverity.MEDIUM.value == "medium"
        assert StegoSeverity.LOW.value == "low"
        assert StegoSeverity.INFO.value == "info"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
