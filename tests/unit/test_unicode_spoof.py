"""
Unit tests for unicode_spoof.py — advanced Unicode spoof scanner.

Coverage:
- Combining character abuse (stacked diacritics / zalgo)
- Variation selectors (fingerprinting / steganography)
- Mathematical alphanumeric symbols (math bold/italic/fraktur bypassing filters)
- Enclosed alphanumerics (①②③ / Ⓐⓑ bypassing filters)
- normalize_spoof() normalization
- scan_unicode_spoof_full() integration with base scanner
- Clean-text false-positive guard
"""

from __future__ import annotations


from hermes_katana.scanner.unicode_spoof import (
    SpoofCategory,
    SpoofSeverity,
    scan_unicode_spoof,
    scan_unicode_spoof_full,
    normalize_spoof,
    _detect_combining_abuse,
    _detect_variation_selectors,
    _detect_math_alpha,
    _detect_enclosed_alpha,
    _math_alpha_to_ascii,
    _enclosed_to_ascii,
)


# ---------------------------------------------------------------------------
# Combining character abuse
# ---------------------------------------------------------------------------


class TestCombiningAbuse:
    def test_zalgo_text_detected(self):
        """Zalgo-style text with many stacked combining marks should be flagged."""
        # Base 'a' followed by 6 combining grave accents
        zalgo = "a" + "\u0300" * 6 + "b"
        findings = _detect_combining_abuse(zalgo)
        assert len(findings) == 1
        assert findings[0].category == SpoofCategory.COMBINING_ABUSE
        assert findings[0].count == 6

    def test_severity_high_for_many_marks(self):
        """8+ combining marks → HIGH severity."""
        zalgo = "x" + "\u0300" * 10
        findings = _detect_combining_abuse(zalgo)
        assert findings[0].severity == SpoofSeverity.HIGH

    def test_severity_medium_for_moderate_marks(self):
        """4-7 combining marks → MEDIUM severity."""
        zalgo = "x" + "\u0300" * 5
        findings = _detect_combining_abuse(zalgo)
        assert findings[0].severity == SpoofSeverity.MEDIUM

    def test_normal_text_no_findings(self):
        """Normal text with no combining marks should produce no findings."""
        assert _detect_combining_abuse("Hello, world!") == []

    def test_single_accent_not_flagged(self):
        """A single combining accent is legitimate; should not be flagged."""
        # café: e + combining acute
        assert _detect_combining_abuse("caf\u00e9") == []

    def test_two_combining_marks_not_flagged(self):
        """2-3 combining marks are within normal range."""
        text = "a\u0300\u0301\u0302"  # 3 combining marks
        assert _detect_combining_abuse(text) == []

    def test_threshold_boundary(self):
        """Exactly 4 combining marks hits the detection threshold."""
        text = "a" + "\u0300" * 4 + "b"
        findings = _detect_combining_abuse(text)
        assert len(findings) == 1
        assert findings[0].count == 4

    def test_multiple_zalgo_clusters(self):
        """Multiple separate zalgo clusters are each reported."""
        text = "a" + "\u0300" * 5 + " b" + "\u0301" * 5
        findings = _detect_combining_abuse(text)
        assert len(findings) == 2

    def test_position_reported_correctly(self):
        """Position should point to the start of the combining run."""
        text = "abc" + "\u0300" * 5 + "def"
        findings = _detect_combining_abuse(text)
        assert findings[0].position[0] == 3  # after 'abc'


# ---------------------------------------------------------------------------
# Variation selectors
# ---------------------------------------------------------------------------


class TestVariationSelectors:
    def test_vs1_detected(self):
        """VS1 (U+FE00) after ASCII char should be detected."""
        text = "a\ufe00b"  # VS1 after 'a'
        findings = _detect_variation_selectors(text)
        assert len(findings) == 1
        assert findings[0].category == SpoofCategory.VARIATION_SELECTOR

    def test_multiple_vs_high_severity(self):
        """8+ variation selectors → HIGH severity."""
        text = "".join("a" + chr(0xFE00 + i) for i in range(8))
        findings = _detect_variation_selectors(text)
        assert findings[0].severity == SpoofSeverity.HIGH
        assert findings[0].count >= 8

    def test_few_vs_low_severity(self):
        """1-2 variation selectors → LOW severity."""
        text = "a\ufe00"
        findings = _detect_variation_selectors(text)
        assert findings[0].severity == SpoofSeverity.LOW

    def test_no_vs_no_findings(self):
        """Text without variation selectors → no findings."""
        assert _detect_variation_selectors("Hello, world!") == []

    def test_count_in_finding(self):
        """The count field reflects total number of variation selectors."""
        text = "a\ufe00b\ufe01c\ufe02"
        findings = _detect_variation_selectors(text)
        assert findings[0].count == 3

    def test_description_mentions_abuse_count(self):
        """Description should mention how many VS appear after ASCII chars."""
        text = "a\ufe00b\ufe01"
        findings = _detect_variation_selectors(text)
        assert "2/2" in findings[0].description


# ---------------------------------------------------------------------------
# Mathematical alphanumeric symbols
# ---------------------------------------------------------------------------


class TestMathAlpha:
    # U+1D400 = MATHEMATICAL BOLD CAPITAL A
    # U+1D41A = MATHEMATICAL BOLD SMALL A
    def test_math_bold_A_detected(self):
        """Math bold A (U+1D400) should be detected."""
        text = "\U0001d400"  # 𝐀
        findings = _detect_math_alpha(text)
        assert len(findings) == 1
        assert findings[0].category == SpoofCategory.MATH_ALPHA

    def test_math_word_decoded(self):
        """Math bold 'ignore' should be decoded in the description."""
        # 𝐢𝐠𝐧𝐨𝐫𝐞 = U+1D422 U+1D420 U+1D427 U+1D428 U+1D42B U+1D41E
        ignore = "\U0001d422\U0001d420\U0001d427\U0001d428\U0001d42b\U0001d41e"
        findings = _detect_math_alpha(ignore)
        assert len(findings) == 1
        assert "ignore" in findings[0].description.lower()

    def test_severity_high_for_many(self):
        """5+ math alpha chars → HIGH severity."""
        text = "\U0001d400\U0001d401\U0001d402\U0001d403\U0001d404\U0001d405"
        findings = _detect_math_alpha(text)
        assert findings[0].severity == SpoofSeverity.HIGH

    def test_normal_latin_not_detected(self):
        """Normal Latin text should not trigger math alpha detection."""
        assert _detect_math_alpha("Hello world") == []

    def test_math_alpha_to_ascii_uppercase(self):
        """_math_alpha_to_ascii should map U+1D400-U+1D419 to A-Z."""
        assert _math_alpha_to_ascii(0x1D400) == "A"
        assert _math_alpha_to_ascii(0x1D419) == "Z"

    def test_math_alpha_to_ascii_lowercase(self):
        """_math_alpha_to_ascii should map U+1D41A-U+1D433 to a-z."""
        assert _math_alpha_to_ascii(0x1D41A) == "a"
        assert _math_alpha_to_ascii(0x1D433) == "z"

    def test_math_alpha_to_ascii_digits(self):
        """_math_alpha_to_ascii should map bold digits U+1D7CE-U+1D7D7 to 0-9."""
        assert _math_alpha_to_ascii(0x1D7CE) == "0"
        assert _math_alpha_to_ascii(0x1D7D7) == "9"

    def test_count_field(self):
        """count field reflects total number of math alpha characters."""
        text = "\U0001d400\U0001d401\U0001d402"
        findings = _detect_math_alpha(text)
        assert findings[0].count == 3


# ---------------------------------------------------------------------------
# Enclosed alphanumerics
# ---------------------------------------------------------------------------


class TestEnclosedAlpha:
    def test_circled_digits_detected(self):
        """Circled digits ①②③ should be detected."""
        text = "①②③"
        findings = _detect_enclosed_alpha(text)
        assert len(findings) == 1
        assert findings[0].category == SpoofCategory.ENCLOSED_ALPHA

    def test_circled_letters_detected(self):
        """Circled uppercase letters Ⓐ-Ⓩ should be detected."""
        text = "ⓘⓖⓝⓞⓡⓔ"  # U+24D8 U+24D6 U+24DD U+24DE U+24C7 U+24D4
        findings = _detect_enclosed_alpha(text)
        assert len(findings) == 1

    def test_normal_digits_not_detected(self):
        """Regular ASCII digits should not be detected."""
        assert _detect_enclosed_alpha("123 abc") == []

    def test_enclosed_to_ascii_circled_digit(self):
        """_enclosed_to_ascii should map ① (U+2460) to '1'."""
        assert _enclosed_to_ascii(0x2460) == "1"

    def test_enclosed_to_ascii_uppercase(self):
        """_enclosed_to_ascii should map Ⓐ (U+24B6) to 'A'."""
        assert _enclosed_to_ascii(0x24B6) == "A"

    def test_enclosed_to_ascii_lowercase(self):
        """_enclosed_to_ascii should map ⓐ (U+24D0) to 'a'."""
        assert _enclosed_to_ascii(0x24D0) == "a"

    def test_severity_high_many_chars(self):
        """5+ enclosed chars → HIGH severity."""
        text = "①②③④⑤"
        findings = _detect_enclosed_alpha(text)
        assert findings[0].severity == SpoofSeverity.HIGH

    def test_count_field(self):
        """count field reflects total number of enclosed chars."""
        text = "①②③"
        findings = _detect_enclosed_alpha(text)
        assert findings[0].count == 3

    def test_description_shows_decoded(self):
        """Description should include the decoded content."""
        text = "①②③"
        findings = _detect_enclosed_alpha(text)
        assert "Decoded content" in findings[0].description


# ---------------------------------------------------------------------------
# normalize_spoof()
# ---------------------------------------------------------------------------


class TestNormalizeSpoof:
    def test_variation_selectors_stripped(self):
        """Variation selectors should be removed."""
        text = "a\ufe00b\ufe01c"
        assert normalize_spoof(text) == "abc"

    def test_math_alpha_normalized(self):
        """Math bold A (U+1D400) should become 'A'."""
        assert normalize_spoof("\U0001d400") == "A"

    def test_math_alpha_word_normalized(self):
        """Math bold 'ignore' should become 'ignore'."""
        ignore = "\U0001d422\U0001d420\U0001d427\U0001d428\U0001d42b\U0001d41e"
        assert normalize_spoof(ignore) == "ignore"

    def test_enclosed_digit_normalized(self):
        """① should become '1'."""
        assert normalize_spoof("①") == "1"

    def test_enclosed_letter_normalized(self):
        """Ⓐ should become 'A'."""
        assert normalize_spoof("\u24b6") == "A"

    def test_combining_abuse_stripped(self):
        """Excessive combining marks should be trimmed to 1."""
        text = "a" + "\u0300" * 6
        normalized = normalize_spoof(text)
        # Should have 'a' + 1 combining mark
        assert len(normalized) == 2
        assert normalized[0] == "a"

    def test_normal_text_unchanged(self):
        """Normal ASCII text should be returned unchanged."""
        text = "Hello, world! 123"
        assert normalize_spoof(text) == text

    def test_single_combining_preserved(self):
        """A single combining mark on a base char is kept."""
        text = "e\u0301"  # é (decomposed)
        normalized = normalize_spoof(text)
        assert normalized == "e\u0301"


# ---------------------------------------------------------------------------
# scan_unicode_spoof() — top-level scanner
# ---------------------------------------------------------------------------


class TestScanUnicodeSpoof:
    def test_empty_string(self):
        """Empty string returns empty list."""
        assert scan_unicode_spoof("") == []

    def test_clean_text_no_findings(self):
        """Normal English text should produce no findings."""
        assert scan_unicode_spoof("The quick brown fox jumps over the lazy dog.") == []

    def test_mixed_attacks_all_detected(self):
        """Multiple attack types in one string are all reported."""
        text = (
            "a\ufe00"  # variation selector
            + "①②③"  # enclosed alphanumerics
            + "\U0001d400"  # math bold A
            + "a"
            + "\u0300" * 5  # combining abuse
        )
        findings = scan_unicode_spoof(text)
        categories = {f.category for f in findings}
        assert SpoofCategory.VARIATION_SELECTOR in categories
        assert SpoofCategory.ENCLOSED_ALPHA in categories
        assert SpoofCategory.MATH_ALPHA in categories
        assert SpoofCategory.COMBINING_ABUSE in categories

    def test_sorted_by_position(self):
        """Findings should be sorted by start position."""
        text = "①" + "a" + "\u0300" * 5
        findings = scan_unicode_spoof(text)
        positions = [f.position[0] for f in findings]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# scan_unicode_spoof_full() — integrated scanner
# ---------------------------------------------------------------------------


class TestScanUnicodeSpoofFull:
    def test_returns_three_tuple(self):
        """Should return (normalized_text, base_findings, spoof_findings)."""
        result = scan_unicode_spoof_full("hello")
        assert len(result) == 3
        normalized, base_findings, spoof_findings = result
        assert isinstance(normalized, str)
        assert isinstance(base_findings, list)
        assert isinstance(spoof_findings, list)

    def test_empty_string(self):
        """Empty string should return empty results."""
        normalized, base, spoof = scan_unicode_spoof_full("")
        assert normalized == ""
        assert base == []
        assert spoof == []

    def test_bidi_detected_by_base(self):
        """Bidi RLO (U+202E) should appear in base_findings."""
        text = "Hello\u202eworld"
        _, base_findings, _ = scan_unicode_spoof_full(text)
        assert any(f.category.value == "bidi_override" for f in base_findings)

    def test_math_alpha_detected_by_spoof(self):
        """Math bold A should appear in spoof_findings."""
        text = "\U0001d400BC"
        _, _, spoof_findings = scan_unicode_spoof_full(text)
        assert any(f.category == SpoofCategory.MATH_ALPHA for f in spoof_findings)

    def test_normalization_applied_fully(self):
        """Normalized output should have bidi removed and math alpha converted."""
        text = "\u202e\U0001d400"  # RLO + math bold A
        normalized, _, _ = scan_unicode_spoof_full(text)
        assert "\u202e" not in normalized  # bidi stripped by base
        assert "\U0001d400" not in normalized  # math alpha normalised
        assert "A" in normalized  # math bold A → 'A'

    def test_zero_width_detected_by_base(self):
        """Zero-width space should appear in base findings."""
        text = "hello\u200bworld"
        _, base_findings, _ = scan_unicode_spoof_full(text)
        assert any(f.category.value == "zero_width" for f in base_findings)

    def test_tag_chars_detected_by_base(self):
        """Unicode tag characters should appear in base findings."""
        # U+E0041 = TAG LATIN CAPITAL LETTER A
        text = "hello\U000e0041world"
        _, base_findings, _ = scan_unicode_spoof_full(text)
        assert any(f.category.value == "invisible_char" for f in base_findings)
