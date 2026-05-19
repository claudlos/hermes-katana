"""
PDF JavaScript deep-scanner for HermesKatana Scabbard.

Extends pdf_layers with deeper detection of:
- PDF /JS and /JavaScript action entries
- OpenAction triggers (auto-execute on open)
- Named actions (GoTo, Launch, URI, SubmitForm)
- AcroForm JavaScript events
- Embedded file streams with executable content
- Incremental update injection (appended malicious objects)

Two-tier detection:
1. Regex-based structural analysis (always available, no dependencies)
2. Optional pymupdf (fitz) deep parse when available

Speed: Microseconds for regex tier; milliseconds for pymupdf tier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PDFJSSeverity",
    "PDFJSFinding",
    "detect_pdf_js",
]

# ---------------------------------------------------------------------------
# Optional pymupdf import
# ---------------------------------------------------------------------------

try:
    import fitz as _fitz  # type: ignore[import-untyped]

    _FITZ_AVAILABLE = True
except Exception:
    _fitz = None  # type: ignore[assignment]
    _FITZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Severity / Finding types
# ---------------------------------------------------------------------------


class PDFJSSeverity(str, Enum):
    """Severity of PDF JavaScript findings."""

    CRITICAL = "critical"
    """Auto-executing JavaScript or remote code launch detected."""

    HIGH = "high"
    """JavaScript action entry or dangerous named action detected."""

    MEDIUM = "medium"
    """Suspicious AcroForm event or embedded executable found."""

    LOW = "low"
    """Minor indicator; may be benign depending on context."""


@dataclass(frozen=True, slots=True)
class PDFJSFinding:
    """A single PDF JS scanner finding.

    Attributes:
        action_type: Category of detected action (e.g. ``open_action``).
        content:     The matched/excerpted raw content.
        location:    Human-readable location description.
        severity:    How severe this finding is.
        description: Human-readable explanation.
        confidence:  Detection confidence 0.0–1.0.
    """

    action_type: str
    content: str
    location: str
    severity: PDFJSSeverity
    description: str
    confidence: float = 0.85


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# /JS and /JavaScript dictionary keys
_RE_JS_KEY = re.compile(
    r"(?:/JS\s*(?:\(|<<|/)|/JavaScript\s*(?:\(|<<|/))",
    re.IGNORECASE,
)

# /OpenAction — auto-execute on document open
_RE_OPEN_ACTION = re.compile(
    r"/OpenAction\s*(?:<<|/|\[)",
    re.IGNORECASE,
)

# /AA (Additional Actions) trigger points
_RE_AA_ACTION = re.compile(
    r"/AA\s*<<",
    re.IGNORECASE,
)

# Dangerous named action types
_RE_LAUNCH = re.compile(r"/Launch\s*<<", re.IGNORECASE)
_RE_URI_ACTION = re.compile(r"/URI\s*\(", re.IGNORECASE)
_RE_SUBMIT_FORM = re.compile(r"/SubmitForm\s*<<", re.IGNORECASE)
_RE_GOTO_REMOTE = re.compile(r"/GoToR\s*<<", re.IGNORECASE)
_RE_IMPORT_DATA = re.compile(r"/ImportData\s*<<", re.IGNORECASE)

# AcroForm with JavaScript
_RE_ACROFORM = re.compile(r"/AcroForm\s*<<", re.IGNORECASE)

# Embedded files (may contain executables)
_RE_EMBEDDED_FILE = re.compile(r"/EmbeddedFile\b", re.IGNORECASE)
_RE_FILESPEC = re.compile(r"/F\s*\(([^)]+)\)", re.IGNORECASE)

# Incremental update — multiple %%EOF markers
_RE_EOF_MARKER = re.compile(r"%%EOF", re.IGNORECASE)
# Multiple startxref entries
_RE_STARTXREF = re.compile(r"startxref", re.IGNORECASE)
# Append-only cross-reference stream
_RE_XREF_STREAM = re.compile(r"/XRef\b", re.IGNORECASE)

# Executable file extensions in embedded filenames
_EXEC_EXTENSIONS = re.compile(
    r"\.(exe|dll|bat|cmd|com|ps1|vbs|vbe|js|jse|wsf|wsh|sh|bash|py|rb|pl|jar|msi|scr|pif)\b",
    re.IGNORECASE,
)

# Suspicious JavaScript function calls inside PDF JS streams
_SUSPICIOUS_JS_CALLS = [
    (re.compile(r"\beval\s*\(", re.IGNORECASE), "eval() call in embedded JS"),
    (re.compile(r"\bthis\s*\.\s*submitForm\s*\(", re.IGNORECASE), "form submission via JS"),
    (re.compile(r"\bapp\s*\.\s*launchURL\s*\(", re.IGNORECASE), "URL launch via JS"),
    (re.compile(r"\butil\s*\.\s*printd\b", re.IGNORECASE), "util.printd date manipulation"),
    (re.compile(r"\bthis\s*\.\s*exportDataObject\s*\(", re.IGNORECASE), "data export via JS"),
    (re.compile(r"\bapp\s*\.\s*openDoc\s*\(", re.IGNORECASE), "openDoc cross-document launch"),
    (re.compile(r"\bgetURL\s*\(", re.IGNORECASE), "getURL network call in JS"),
    (re.compile(r"\bXMLHttpRequest\b", re.IGNORECASE), "XMLHttpRequest in PDF JS"),
    (re.compile(r"\bnetwork\.send\b", re.IGNORECASE), "network.send API call"),
]


# ---------------------------------------------------------------------------
# Helpers — extract context window around a match
# ---------------------------------------------------------------------------


def _excerpt(text: str, match: re.Match, window: int = 80) -> str:
    s = max(0, match.start() - window)
    e = min(len(text), match.end() + window)
    return text[s:e].replace("\x00", "")


# ---------------------------------------------------------------------------
# Sub-detectors
# ---------------------------------------------------------------------------


def _detect_js_entries(text: str) -> list[PDFJSFinding]:
    findings: list[PDFJSFinding] = []
    for m in _RE_JS_KEY.finditer(text):
        findings.append(
            PDFJSFinding(
                action_type="js_entry",
                content=_excerpt(text, m),
                location=f"offset {m.start()}",
                severity=PDFJSSeverity.HIGH,
                description="PDF /JS or /JavaScript action entry detected — may execute on trigger.",
                confidence=0.90,
            )
        )
        # Only report first occurrence to avoid noise in large files
        break
    # Scan for suspicious JS call patterns
    for pattern, label in _SUSPICIOUS_JS_CALLS:
        for m in pattern.finditer(text):
            findings.append(
                PDFJSFinding(
                    action_type="js_call",
                    content=_excerpt(text, m, window=60),
                    location=f"offset {m.start()}",
                    severity=PDFJSSeverity.HIGH,
                    description=f"Suspicious JS call in PDF: {label}.",
                    confidence=0.88,
                )
            )
            break  # one per pattern type
    return findings


def _detect_open_action(text: str) -> list[PDFJSFinding]:
    findings: list[PDFJSFinding] = []
    for m in _RE_OPEN_ACTION.finditer(text):
        findings.append(
            PDFJSFinding(
                action_type="open_action",
                content=_excerpt(text, m),
                location=f"offset {m.start()}",
                severity=PDFJSSeverity.CRITICAL,
                description="/OpenAction detected — PDF will auto-execute code on document open.",
                confidence=0.92,
            )
        )
        break
    for m in _RE_AA_ACTION.finditer(text):
        findings.append(
            PDFJSFinding(
                action_type="additional_action",
                content=_excerpt(text, m),
                location=f"offset {m.start()}",
                severity=PDFJSSeverity.HIGH,
                description="/AA (Additional Actions) dictionary found — may trigger JS on events.",
                confidence=0.80,
            )
        )
        break
    return findings


def _detect_named_actions(text: str) -> list[PDFJSFinding]:
    findings: list[PDFJSFinding] = []
    named = [
        (_RE_LAUNCH, "launch_action", PDFJSSeverity.CRITICAL, "/Launch action — can execute arbitrary programs.", 0.95),
        (_RE_URI_ACTION, "uri_action", PDFJSSeverity.MEDIUM, "/URI action — opens external URL automatically.", 0.75),
        (_RE_SUBMIT_FORM, "submit_form", PDFJSSeverity.HIGH, "/SubmitForm action — exfiltrates form data.", 0.85),
        (_RE_GOTO_REMOTE, "goto_remote", PDFJSSeverity.MEDIUM, "/GoToR action — cross-document remote goto.", 0.70),
        (_RE_IMPORT_DATA, "import_data", PDFJSSeverity.HIGH, "/ImportData action — imports external data.", 0.85),
    ]
    for pattern, action_type, severity, description, confidence in named:
        for m in pattern.finditer(text):
            findings.append(
                PDFJSFinding(
                    action_type=action_type,
                    content=_excerpt(text, m),
                    location=f"offset {m.start()}",
                    severity=severity,
                    description=description,
                    confidence=confidence,
                )
            )
            break
    return findings


def _detect_acroform_js(text: str) -> list[PDFJSFinding]:
    findings: list[PDFJSFinding] = []
    has_acroform = bool(_RE_ACROFORM.search(text))
    has_js = bool(_RE_JS_KEY.search(text))
    if has_acroform and has_js:
        m = _RE_ACROFORM.search(text)
        assert m is not None
        findings.append(
            PDFJSFinding(
                action_type="acroform_js",
                content=_excerpt(text, m),
                location=f"offset {m.start()}",
                severity=PDFJSSeverity.HIGH,
                description="AcroForm with embedded JavaScript detected — may execute on field events.",
                confidence=0.88,
            )
        )
    return findings


def _detect_embedded_executables(text: str) -> list[PDFJSFinding]:
    findings: list[PDFJSFinding] = []
    has_embedded = bool(_RE_EMBEDDED_FILE.search(text))
    if not has_embedded:
        return findings
    for m in _RE_FILESPEC.finditer(text):
        filename = m.group(1)
        if _EXEC_EXTENSIONS.search(filename):
            findings.append(
                PDFJSFinding(
                    action_type="embedded_executable",
                    content=filename[:200],
                    location=f"offset {m.start()}",
                    severity=PDFJSSeverity.CRITICAL,
                    description=f"Embedded executable file detected: '{filename}'.",
                    confidence=0.92,
                )
            )
    if has_embedded and not findings:
        # /EmbeddedFile present but no exec extension — still note it
        m2 = _RE_EMBEDDED_FILE.search(text)
        assert m2 is not None
        findings.append(
            PDFJSFinding(
                action_type="embedded_file",
                content=_excerpt(text, m2),
                location=f"offset {m2.start()}",
                severity=PDFJSSeverity.MEDIUM,
                description="Embedded file stream found in PDF — verify content.",
                confidence=0.70,
            )
        )
    return findings


def _detect_incremental_update(text: str) -> list[PDFJSFinding]:
    """Detect incremental update injection (appended malicious objects).

    Legitimate PDFs may have one incremental update (signed documents).
    Two or more updates after the first EOF are suspicious.
    """
    findings: list[PDFJSFinding] = []
    eof_matches = list(_RE_EOF_MARKER.finditer(text))
    startxref_matches = list(_RE_STARTXREF.finditer(text))

    if len(eof_matches) > 1:
        findings.append(
            PDFJSFinding(
                action_type="incremental_update",
                content=f"Found {len(eof_matches)} %%EOF markers",
                location=f"offsets {[m.start() for m in eof_matches]}",
                severity=PDFJSSeverity.HIGH,
                description=(
                    f"PDF contains {len(eof_matches)} %%EOF markers — "
                    "possible incremental update injection after document end."
                ),
                confidence=0.80,
            )
        )
    if len(startxref_matches) > 2:
        findings.append(
            PDFJSFinding(
                action_type="multi_xref",
                content=f"Found {len(startxref_matches)} startxref entries",
                location=f"offsets {[m.start() for m in startxref_matches[:5]]}",
                severity=PDFJSSeverity.MEDIUM,
                description=(
                    f"PDF has {len(startxref_matches)} cross-reference tables — may indicate incremental update abuse."
                ),
                confidence=0.72,
            )
        )
    return findings


# ---------------------------------------------------------------------------
# Optional pymupdf deep scan
# ---------------------------------------------------------------------------


def _deep_scan_with_fitz(pdf_bytes: bytes) -> list[PDFJSFinding]:
    """Use pymupdf to extract and inspect JavaScript actions more precisely."""
    if not _FITZ_AVAILABLE or _fitz is None:
        return []
    findings: list[PDFJSFinding] = []
    try:
        doc = _fitz.open(stream=pdf_bytes, filetype="pdf")
        # Check document-level JavaScript
        js_actions = doc.get_js()  # type: ignore[attr-defined]
        if js_actions:
            findings.append(
                PDFJSFinding(
                    action_type="fitz_doc_js",
                    content=str(js_actions)[:200],
                    location="document-level JS (pymupdf)",
                    severity=PDFJSSeverity.HIGH,
                    description="pymupdf: document-level JavaScript found.",
                    confidence=0.95,
                )
            )
        doc.close()
    except Exception:
        pass
    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_pdf_js(pdf_content: str | bytes) -> list[PDFJSFinding]:
    """Scan PDF content for JavaScript and dangerous actions.

    Performs regex-based structural analysis always. If *pdf_content* is
    provided as ``bytes`` and pymupdf (fitz) is installed, also runs a
    deeper pymupdf parse.

    Detection coverage:
    - PDF /JS and /JavaScript action entries
    - OpenAction triggers (auto-execute on open)
    - Named actions: Launch, URI, SubmitForm, GoToR, ImportData
    - AcroForm with JavaScript field events
    - Embedded file streams with executable extensions
    - Incremental update injection (multiple %%EOF / startxref entries)
    - Suspicious JS calls: eval, launchURL, submitForm, getURL, etc.

    Args:
        pdf_content: Raw PDF bytes or latin-1 decoded string.

    Returns:
        List of :class:`PDFJSFinding` objects, possibly empty.
    """
    findings: list[PDFJSFinding] = []

    raw_bytes: bytes | None = None

    if isinstance(pdf_content, bytes):
        raw_bytes = pdf_content
        try:
            text = pdf_content.decode("latin-1", errors="replace")
        except Exception:
            return findings
    else:
        text = pdf_content

    if not text or len(text) < 10:
        return findings

    # Require PDF header — skip plain text quickly
    stripped = text.lstrip()
    if not (stripped.startswith("%PDF") or "%PDF-" in text[:1024]):
        return findings

    # Run all regex-based sub-detectors
    findings.extend(_detect_js_entries(text))
    findings.extend(_detect_open_action(text))
    findings.extend(_detect_named_actions(text))
    findings.extend(_detect_acroform_js(text))
    findings.extend(_detect_embedded_executables(text))
    findings.extend(_detect_incremental_update(text))

    # Optional deep scan via pymupdf
    if raw_bytes is not None:
        findings.extend(_deep_scan_with_fitz(raw_bytes))

    return findings
