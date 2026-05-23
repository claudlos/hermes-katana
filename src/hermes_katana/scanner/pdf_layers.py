"""
PDF layer analyzer for HermesKatana Scabbard.

Analyzes PDF documents to detect content in:
- Visible text layers
- Hidden/annotation layers
- Metadata (author, title, subject with suspicious content)
- Embedded JavaScript
- Form fields with default values

Catches steganographic payloads and hidden instructions in PDFs
that wouldn't appear in normal text extraction.

Speed: Microseconds to milliseconds depending on PDF size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PDFLayerSeverity",
    "PDFLayerFinding",
    "detect_pdf_layers",
]


class PDFLayerSeverity(str, Enum):
    """Severity of PDF layer findings."""

    CRITICAL = "critical"
    """Hidden malicious content detected in PDF."""

    HIGH = "high"
    """Suspicious hidden content in PDF layers."""

    MEDIUM = "medium"
    """Hidden content detected, needs review."""

    LOW = "low"
    """Minor hidden elements, likely benign."""


@dataclass(frozen=True, slots=True)
class PDFLayerFinding:
    """A single PDF layer analysis finding.

    Attributes:
        layer_type: Which PDF layer the content was found in.
        content: The suspicious content itself.
        location: Where in the PDF this was found.
        severity: How severe this finding is.
        description: Human-readable explanation.
        confidence: Detection confidence 0.0-1.0.
    """

    layer_type: str
    content: str
    location: str
    severity: PDFLayerSeverity
    description: str
    confidence: float = 0.85


# Suspicious keywords that may appear in hidden PDF content
_SUSPICIOUS_KEYWORDS = [
    "ignore",
    "override",
    "bypass",
    "instruction",
    "prompt",
    "inject",
    "hidden",
    "execute",
    "javascript",
    "eval",
    "function",
    "script",
    "admin",
    "debug",
    "system",
    "root",
    "sudo",
    "shell",
]


def _analyze_pdf_string_stream(data: str) -> list[PDFLayerFinding]:
    """Analyze raw PDF string stream for suspicious content patterns."""
    findings: list[PDFLayerFinding] = []

    # Look for suspicious keywords in the raw stream
    data_lower = data.lower()
    for keyword in _SUSPICIOUS_KEYWORDS:
        if keyword in data_lower:
            # Find context around the keyword
            idx = data_lower.find(keyword)
            context_start = max(0, idx - 50)
            context_end = min(len(data), idx + len(keyword) + 50)
            context = data[context_start:context_end]

            # Check if this is in an obviously benign context
            if any(
                benign in data_lower[max(0, idx - 200) : idx]
                for benign in ["author", "title", "subject", "keywords", "creator"]
            ):
                continue

            severity = PDFLayerSeverity.MEDIUM
            confidence = 0.70
            if keyword in ["ignore", "override", "inject", "bypass"]:
                severity = PDFLayerSeverity.HIGH
                confidence = 0.85

            findings.append(
                PDFLayerFinding(
                    layer_type="string_stream",
                    content=f"...{context}...",
                    location=f"keyword '{keyword}' in string stream",
                    severity=severity,
                    description=f"Suspicious keyword '{keyword}' found in PDF string stream.",
                    confidence=confidence,
                )
            )

    return findings


def _analyze_pdf_comments(pdf_text: str) -> list[PDFLayerFinding]:
    """Analyze PDF for comment-like patterns with suspicious content."""
    findings: list[PDFLayerFinding] = []

    # PDF comments are lines starting with %
    comment_lines = re.findall(r"%(.*?)$", pdf_text, re.MULTILINE)

    for comment in comment_lines:
        comment_stripped = comment.strip()
        if not comment_stripped:
            continue

        comment_lower = comment_stripped.lower()
        matches = [kw for kw in _SUSPICIOUS_KEYWORDS if kw in comment_lower]

        if matches:
            severity = PDFLayerSeverity.MEDIUM
            confidence = 0.75
            if any(kw in ["ignore", "override", "inject"] for kw in matches):
                severity = PDFLayerSeverity.HIGH
                confidence = 0.85

            findings.append(
                PDFLayerFinding(
                    layer_type="comment",
                    content=comment_stripped[:200],
                    location="PDF comment line",
                    severity=severity,
                    description=f"Suspicious PDF comment containing: {', '.join(matches)}",
                    confidence=confidence,
                )
            )

    return findings


def _analyze_pdf_javascript(pdf_text: str) -> list[PDFLayerFinding]:
    """Detect JavaScript embedded in PDF."""
    findings: list[PDFLayerFinding] = []

    # Look for JavaScript patterns
    js_patterns = [
        (r"\[/JavaScript\s*<<", "JavaScript dictionary in PDF"),
        (r"<<\s*/JS\s*\(", "JavaScript stream reference"),
        (r"eval\s*\(", "eval() in PDF JavaScript"),
        (r"this\.submitForm", "form submission in PDF JavaScript"),
        (r"getURL\s*\(", "URL fetch in PDF JavaScript"),
        (r"launchURL", "URL launch in PDF JavaScript"),
    ]

    for pattern, description in js_patterns:
        matches = re.findall(pattern, pdf_text, re.IGNORECASE)
        if matches:
            findings.append(
                PDFLayerFinding(
                    layer_type="javascript",
                    content=f"Found: {description}",
                    location="embedded JavaScript",
                    severity=PDFLayerSeverity.HIGH,
                    description=f"PDF contains JavaScript that may execute automatically: {description}.",
                    confidence=0.85,
                )
            )

    return findings


def _analyze_pdf_metadata(pdf_text: str) -> list[PDFLayerFinding]:
    """Analyze PDF metadata for suspicious content."""
    findings: list[PDFLayerFinding] = []

    # Extract metadata fields
    metadata_patterns = [
        r"/Author\s*\(([^)]+)\)",
        r"/Title\s*\(([^)]+)\)",
        r"/Subject\s*\(([^)]+)\)",
        r"/Keywords\s*\(([^)]+)\)",
        r"/Creator\s*\(([^)]+)\)",
        r"/Producer\s*\(([^)]+)\)",
    ]

    for pattern in metadata_patterns:
        matches = re.findall(pattern, pdf_text, re.IGNORECASE)
        for match in matches:
            match_lower = match.lower()
            suspicious = [kw for kw in _SUSPICIOUS_KEYWORDS if kw in match_lower]
            if suspicious:
                findings.append(
                    PDFLayerFinding(
                        layer_type="metadata",
                        content=match[:200],
                        location=f"PDF metadata field: {pattern[2 : pattern.index('(')]}",
                        severity=PDFLayerSeverity.MEDIUM,
                        description=f"PDF metadata contains suspicious keywords: {', '.join(suspicious)}",
                        confidence=0.75,
                    )
                )

    return findings


def _analyze_pdf_forms(pdf_text: str) -> list[PDFLayerFinding]:
    """Analyze PDF form fields for suspicious default values."""
    findings: list[PDFLayerFinding] = []

    # Find form field defaults
    form_patterns = [
        r"/V\s*\(([^)]+)\)",  # Field value
        r"/TU\s*\(([^)]+)\)",  # Tooltip/UserText
        r"/DV\s*\(([^)]+)\)",  # Default value
    ]

    for pattern in form_patterns:
        matches = re.findall(pattern, pdf_text)
        for match in matches:
            match_lower = match.lower()
            suspicious = [kw for kw in _SUSPICIOUS_KEYWORDS if kw in match_lower]
            if suspicious:
                findings.append(
                    PDFLayerFinding(
                        layer_type="form_field",
                        content=match[:200],
                        location="PDF form field default value",
                        severity=PDFLayerSeverity.MEDIUM,
                        description=f"PDF form field contains suspicious content: {', '.join(suspicious)}",
                        confidence=0.70,
                    )
                )

    return findings


def detect_pdf_layers(pdf_content: str) -> list[PDFLayerFinding]:
    """Analyze PDF content for hidden/malicious layers.

    Performs structural analysis of PDF content without requiring
    full PDF parsing. Detects:
    - Hidden content in string streams
    - Suspicious comments
    - Embedded JavaScript
    - Malicious metadata
    - Suspicious form field values

    Args:
        pdf_content: Raw PDF content (bytes decoded to string or bytes).

    Returns:
        List of PDFLayerFinding objects for each detected issue.
    """
    findings: list[PDFLayerFinding] = []

    # Handle bytes
    if isinstance(pdf_content, bytes):
        try:
            pdf_text = pdf_content.decode("latin-1", errors="replace")
        except Exception:
            return findings
    else:
        pdf_text = pdf_content

    if not pdf_text or len(pdf_text) < 10:
        return findings

    # Check if this is actually a PDF
    if not pdf_text.strip().startswith("%PDF") and "%!PS" not in pdf_text[:10]:
        # Not a PDF, return empty
        return findings

    # Analyze different layers
    findings.extend(_analyze_pdf_string_stream(pdf_text))
    findings.extend(_analyze_pdf_comments(pdf_text))
    findings.extend(_analyze_pdf_javascript(pdf_text))
    findings.extend(_analyze_pdf_metadata(pdf_text))
    findings.extend(_analyze_pdf_forms(pdf_text))

    return findings


def detect_pdf_layers_bytes(pdf_bytes: bytes) -> list[PDFLayerFinding]:
    """Analyze PDF from bytes.

    Convenience wrapper for binary PDF content.
    """
    return detect_pdf_layers(pdf_bytes.decode("latin-1", errors="ignore"))
