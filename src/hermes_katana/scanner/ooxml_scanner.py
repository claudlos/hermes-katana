"""
OOXML scanner for DOCX/XLSX/PPTX prompt-injection carriers.

Detects lightweight structural risks in Office Open XML packages:
- VBA macro projects
- external relationships
- hidden Word text
- prompt-injection text in document body, comments, notes, and metadata

The scanner is standard-library only and bounded so it is safe to run from the
aggregate text scanner against embedded OOXML data URIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import html
from io import BytesIO
import re
import zipfile

from .injection import detect_injection
from .multimodal import extract_data_uri_payloads

__all__ = [
    "OOXMLSeverity",
    "OOXMLFinding",
    "detect_ooxml_injection",
    "detect_ooxml_injection_bytes",
]


class OOXMLSeverity(str, Enum):
    """Severity of OOXML findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class OOXMLFinding:
    """A single OOXML scanner finding."""

    category: str
    content: str
    location: str
    severity: OOXMLSeverity
    description: str
    confidence: float = 0.85


_MAX_PARTS = 150
_MAX_PART_BYTES = 256 * 1024
_MAX_FINDINGS = 50

_OOXML_MIME_HINTS = (
    "openxmlformats-officedocument",
    "wordprocessingml",
    "spreadsheetml",
    "presentationml",
    "macroenabled",
    "application/vnd.ms-word",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/msword",
    "application/zip",
    "application/x-zip-compressed",
    "application/octet-stream",
)
_DANGEROUS_REL_TARGET_RE = re.compile(r"^(?:javascript:|data:|file:|https?://|//|\\\\)", re.IGNORECASE)
_HIGH_RISK_REL_TARGET_RE = re.compile(r"^(?:javascript:|data:|file:|//|\\\\)", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_PROMPT_MARKER_RE = re.compile(
    r"(?i)(?:"
    r"\b(?:system|developer|assistant|tool)\s*:"
    r"|<\s*/?\s*(?:system|developer|assistant|tool|instructions?)\s*>"
    r"|hidden\s+payload"
    r"|ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?|prompts?)"
    r"|reveal\s+(?:the\s+)?(?:system\s+)?prompt"
    r"|bypass\s+(?:safety|policy|rules?)"
    r")"
)

_TEXT_PART_EXACT = {
    "docprops/core.xml",
    "docprops/app.xml",
    "docprops/custom.xml",
    "word/document.xml",
    "word/comments.xml",
    "word/footnotes.xml",
    "word/endnotes.xml",
    "xl/sharedstrings.xml",
    "xl/comments.xml",
}
_TEXT_PART_PREFIXES = (
    "word/header",
    "word/footer",
    "word/comments",
    "ppt/slides/slide",
    "ppt/notesslides/notesslide",
    "ppt/comments",
    "xl/comments",
)


def detect_ooxml_injection(content: str | bytes) -> list[OOXMLFinding]:
    """Scan raw OOXML bytes or text containing embedded OOXML data URIs."""
    if isinstance(content, bytes):
        return detect_ooxml_injection_bytes(content)

    findings: list[OOXMLFinding] = []
    text = content or ""
    if not text:
        return findings

    for payload in extract_data_uri_payloads(text):
        decoded = payload.decoded_bytes
        if decoded is None:
            continue
        if not _data_uri_may_be_ooxml(payload.media_type, payload.header, decoded):
            continue
        findings.extend(detect_ooxml_injection_bytes(decoded))
        if len(findings) >= _MAX_FINDINGS:
            return findings[:_MAX_FINDINGS]

    stripped = text.lstrip()
    if stripped.startswith(("PK\x03\x04", "PK\x05\x06", "PK\x07\x08")):
        try:
            findings.extend(detect_ooxml_injection_bytes(stripped.encode("latin-1", errors="ignore")))
        except Exception:
            pass

    return findings[:_MAX_FINDINGS]


def detect_ooxml_injection_bytes(data: bytes) -> list[OOXMLFinding]:
    """Scan DOCX/XLSX/PPTX package bytes for hidden prompt-injection carriers."""
    findings: list[OOXMLFinding] = []
    if len(data) < 8:
        return findings

    buffer = BytesIO(data)
    if not zipfile.is_zipfile(buffer):
        return findings

    try:
        with zipfile.ZipFile(buffer) as zf:
            infos = zf.infolist()
            names = {info.filename.lower() for info in infos}
            if not _looks_like_ooxml(names):
                return findings

            for info in infos[:_MAX_PARTS]:
                lower_name = info.filename.lower()

                if lower_name.endswith("vbaproject.bin"):
                    findings.append(
                        OOXMLFinding(
                            category="macro_vba",
                            content=info.filename[:300],
                            location=info.filename,
                            severity=OOXMLSeverity.CRITICAL,
                            description="OOXML package contains a VBA macro project.",
                            confidence=0.95,
                        )
                    )

                if info.file_size > _MAX_PART_BYTES:
                    continue

                if lower_name.endswith(".rels"):
                    try:
                        rel_xml = zf.read(info).decode("utf-8", errors="ignore")
                    except Exception:
                        rel_xml = ""
                    findings.extend(_scan_relationships(rel_xml, info.filename))

                if _is_text_part(lower_name):
                    try:
                        xml_text = zf.read(info).decode("utf-8", errors="ignore")
                    except Exception:
                        xml_text = ""
                    findings.extend(_scan_text_part(xml_text, info.filename))

                if len(findings) >= _MAX_FINDINGS:
                    return findings[:_MAX_FINDINGS]
    except Exception:
        return findings

    return findings[:_MAX_FINDINGS]


def _data_uri_may_be_ooxml(media_type: str, header: str, decoded: bytes) -> bool:
    header_lower = f"{media_type};{header}".lower()
    return decoded.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")) or any(
        hint in header_lower for hint in _OOXML_MIME_HINTS
    )


def _looks_like_ooxml(names: set[str]) -> bool:
    if "[content_types].xml" not in names:
        return False
    return any(name.startswith(("word/", "xl/", "ppt/")) for name in names)


def _is_text_part(lower_name: str) -> bool:
    if lower_name in _TEXT_PART_EXACT:
        return True
    return any(lower_name.startswith(prefix) and lower_name.endswith(".xml") for prefix in _TEXT_PART_PREFIXES)


def _scan_relationships(xml_text: str, location: str) -> list[OOXMLFinding]:
    findings: list[OOXMLFinding] = []
    if not xml_text:
        return findings

    for match in re.finditer(r"<Relationship\b[^>]*>", xml_text, re.IGNORECASE):
        tag = match.group(0)
        target = _extract_xml_attr(tag, "Target")
        target_mode = _extract_xml_attr(tag, "TargetMode")
        if not target:
            continue
        target = html.unescape(target).strip()
        external_mode = target_mode.lower() == "external"
        dangerous_scheme = bool(_DANGEROUS_REL_TARGET_RE.search(target))
        if not external_mode and not dangerous_scheme:
            continue

        high_risk = bool(_HIGH_RISK_REL_TARGET_RE.search(target))
        findings.append(
            OOXMLFinding(
                category="external_relationship",
                content=target[:300],
                location=location,
                severity=OOXMLSeverity.HIGH if high_risk else OOXMLSeverity.MEDIUM,
                description="OOXML relationship points to an external resource.",
                confidence=0.9 if high_risk else 0.78,
            )
        )

    return findings


def _extract_xml_attr(tag: str, attr: str) -> str:
    pattern = re.compile(rf"""\b{re.escape(attr)}\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
    match = pattern.search(tag)
    if not match:
        return ""
    return match.group(1) if match.group(1) is not None else match.group(2)


def _scan_text_part(xml_text: str, location: str) -> list[OOXMLFinding]:
    findings: list[OOXMLFinding] = []
    if not xml_text:
        return findings

    lower_xml = xml_text.lower()
    text = _xml_to_text(xml_text)
    if not text:
        return findings

    category = "metadata_injection" if location.lower().startswith("docprops/") else "document_injection"
    for injection in detect_injection(text):
        findings.append(
            OOXMLFinding(
                category=category,
                content=injection.matched_text[:300],
                location=location,
                severity=_severity_for_confidence(injection.confidence),
                description=f"Prompt-injection text found in OOXML {location}.",
                confidence=injection.confidence,
            )
        )

    for match in _PROMPT_MARKER_RE.finditer(text):
        findings.append(
            OOXMLFinding(
                category=category,
                content=match.group(0)[:300],
                location=location,
                severity=OOXMLSeverity.HIGH,
                description=f"Prompt-control marker found in OOXML {location}.",
                confidence=0.86,
            )
        )
        break

    if "vanish" in lower_xml or 'w:val="ffffff"' in lower_xml or "w:val='ffffff'" in lower_xml:
        findings.append(
            OOXMLFinding(
                category="hidden_text",
                content=text[:300],
                location=location,
                severity=OOXMLSeverity.HIGH if findings else OOXMLSeverity.MEDIUM,
                description="OOXML part contains hidden or visually concealed text.",
                confidence=0.88 if findings else 0.72,
            )
        )

    return findings


def _xml_to_text(xml_text: str) -> str:
    text = _TAG_RE.sub(" ", xml_text)
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()


def _severity_for_confidence(confidence: float) -> OOXMLSeverity:
    if confidence >= 0.93:
        return OOXMLSeverity.CRITICAL
    if confidence >= 0.8:
        return OOXMLSeverity.HIGH
    if confidence >= 0.55:
        return OOXMLSeverity.MEDIUM
    return OOXMLSeverity.LOW
