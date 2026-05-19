"""
Structural meta-scanner for content-type-aware attack detection.

Routes input to specialised structural scanners (html_diff, pdf_layers,
markdown_audit) based on detected content type, and combines their results
with bloom-filter hits into a unified risk report suitable for future
Bonsai judge integration.

The scanner gracefully degrades when individual sub-scanners are not yet
installed — missing modules are logged once and silently skipped.

Usage::

    from hermes_katana.scanner.structural import detect_structural, StructuralReport

    report = detect_structural("<div style='display:none'>pwned</div>")
    print(report.content_type)   # "html"
    print(report.structural_score)  # 0.82
"""

from __future__ import annotations

__all__ = [
    "ContentType",
    "StructuralFlag",
    "StructuralReport",
    "detect_content_type",
    "detect_structural",
]

import functools
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content type detection
# ---------------------------------------------------------------------------


class ContentType(str, Enum):
    """Detected content type of the input text."""

    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    PLAIN = "plain"


# Heuristics — order matters (first match wins).
_HTML_RE = re.compile(
    r"(?:<\s*(?:html|head|body|div|span|script|style|img|a|p|table|form|iframe|link|meta)\b|<!DOCTYPE\s+html)",
    re.IGNORECASE,
)
_PDF_RE = re.compile(r"%PDF-\d+\.\d+")
_MARKDOWN_RE = re.compile(
    r"(?:^#{1,6}\s|\*\*[^*]+\*\*|^\s*[-*+]\s|\[.+?\]\(.+?\)|^```)",
    re.MULTILINE,
)


def detect_content_type(text: str) -> ContentType:
    """Detect the content type of *text* using lightweight heuristics.

    Args:
        text: The input text to classify.

    Returns:
        The detected :class:`ContentType`.
    """
    if not text or not text.strip():
        return ContentType.PLAIN

    # PDF magic bytes
    if _PDF_RE.search(text[:1024]):
        return ContentType.PDF

    # HTML tags
    if _HTML_RE.search(text[:4096]):
        return ContentType.HTML

    # Markdown — require ≥2 indicators to avoid false positives on plain text
    md_hits = len(_MARKDOWN_RE.findall(text[:4096]))
    if md_hits >= 2:
        return ContentType.MARKDOWN

    return ContentType.PLAIN


# ---------------------------------------------------------------------------
# Structural flag (individual finding)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StructuralFlag:
    """A single structural finding from a sub-scanner.

    Attributes:
        type:      Short tag (e.g. ``hidden_text``, ``layer_mismatch``).
        location:  Where in the document the finding was detected.
        excerpt:   Shortened excerpt of the suspicious content.
        severity:  One of ``critical``, ``high``, ``medium``, ``low``, ``info``.
    """

    type: str
    location: str
    excerpt: str
    severity: str = "medium"


# ---------------------------------------------------------------------------
# Unified risk report
# ---------------------------------------------------------------------------


@dataclass
class StructuralReport:
    """Unified risk report produced by the structural meta-scanner.

    Designed for future Bonsai judge integration — serialises cleanly to
    JSON via :meth:`to_dict` / :meth:`to_json`.

    Attributes:
        content_type:      Detected content type.
        flags:             List of structural findings.
        structural_score:  Aggregate risk score (0.0–1.0).
        pattern_matches:   Number of pattern-based hits from sub-scanners.
        bloom_hits:        Number of bloom-filter hits.
        metadata:          Free-form metadata dict.
    """

    content_type: str = ContentType.PLAIN.value
    flags: list[StructuralFlag] = field(default_factory=list)
    structural_score: float = 0.0
    pattern_matches: int = 0
    bloom_hits: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dict (JSON-ready)."""
        return {
            "content_type": self.content_type,
            "flags": [asdict(f) for f in self.flags],
            "structural_score": round(self.structural_score, 4),
            "pattern_matches": self.pattern_matches,
            "bloom_hits": self.bloom_hits,
        }

    def to_json(self) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Optional sub-scanner imports (graceful degradation)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _try_import(module_name: str):
    """Try to import a sub-scanner module, returning None on failure.

    Result is cached via lru_cache so importlib is only called once per name.
    """
    try:
        import importlib

        return importlib.import_module(f"hermes_katana.scanner.{module_name}")
    except ImportError:
        logger.debug("Structural sub-scanner '%s' not available — skipping", module_name)
        return None


# ---------------------------------------------------------------------------
# Sub-scanner routers
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHT = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "low": 0.2,
    "info": 0.05,
}


def _score_flags(flags: list[StructuralFlag]) -> float:
    """Compute an aggregate score from a list of flags.

    Uses a soft-max style aggregation so multiple medium findings can
    push the score above a single high finding.
    """
    if not flags:
        return 0.0
    total = 0.0
    for f in flags:
        w = _SEVERITY_WEIGHT.get(f.severity, 0.3)
        total += w
    # Normalise: first flag counts fully, each additional adds diminishing returns
    score = min(total / (1.0 + 0.3 * (len(flags) - 1)), 1.0)
    return round(score, 4)


@functools.lru_cache(maxsize=None)
def _load_detect_fn(module_name: str, *attr_names: str):
    """Load and cache a callable from a sub-scanner module.

    Tries each attribute name in order and returns the first one found.
    Returns None if the module cannot be imported or has no matching attribute.
    """
    mod = _try_import(module_name)
    if mod is None:
        return None
    for attr in attr_names:
        fn = getattr(mod, attr, None)
        if fn is not None:
            return fn
    logger.debug("Module '%s' has none of attributes %s", module_name, attr_names)
    return None


def _finding_severity(f: Any) -> str:
    sev = getattr(f, "severity", "medium")
    return sev if isinstance(sev, str) else getattr(sev, "value", "medium")


def _run_html_scanner(text: str, report: StructuralReport) -> None:
    """Route to the html_diff sub-scanner if available."""
    detect_fn = _load_detect_fn("html_diff", "detect_html_diff", "scan_html")
    if detect_fn is None:
        return
    try:
        for f in detect_fn(text):
            report.flags.append(
                StructuralFlag(
                    type=getattr(f, "type", getattr(f, "pattern_name", "html_issue")),
                    location=getattr(f, "location", str(getattr(f, "position", ""))),
                    excerpt=getattr(f, "hidden_text", getattr(f, "matched_text", ""))[:120],
                    severity=_finding_severity(f),
                )
            )
            report.pattern_matches += 1
    except Exception:
        logger.debug("html_diff scanner raised", exc_info=True)


def _run_pdf_scanner(text: str, report: StructuralReport) -> None:
    """Route to the pdf_layers sub-scanner if available."""
    detect_fn = _load_detect_fn("pdf_layers", "detect_pdf_layers", "scan_pdf")
    if detect_fn is None:
        return
    try:
        for f in detect_fn(text):
            report.flags.append(
                StructuralFlag(
                    type=getattr(f, "type", getattr(f, "pattern_name", "pdf_issue")),
                    location=getattr(f, "location", str(getattr(f, "position", ""))),
                    excerpt=getattr(f, "content", getattr(f, "matched_text", ""))[:120],
                    severity=_finding_severity(f),
                )
            )
            report.pattern_matches += 1
    except Exception:
        logger.debug("pdf_layers scanner raised", exc_info=True)


def _run_markdown_scanner(text: str, report: StructuralReport) -> None:
    """Route to the markdown_audit sub-scanner if available."""
    detect_fn = _load_detect_fn("markdown_audit", "detect_markdown_audit", "audit_markdown")
    if detect_fn is None:
        return
    try:
        for f in detect_fn(text):
            report.flags.append(
                StructuralFlag(
                    type=getattr(f, "type", getattr(f, "pattern_name", "markdown_issue")),
                    location=getattr(f, "location", str(getattr(f, "position", ""))),
                    excerpt=getattr(f, "content", getattr(f, "matched_text", ""))[:120],
                    severity=_finding_severity(f),
                )
            )
            report.pattern_matches += 1
    except Exception:
        logger.debug("markdown_audit scanner raised", exc_info=True)


def _run_bloom_filter(text: str, report: StructuralReport) -> None:
    """Query the bloom_filter sub-scanner if available."""
    scan_fn = _load_detect_fn("bloom_filter", "scan_bloom")
    if scan_fn is None:
        return
    try:
        result = scan_fn(text)
        # Accept int (hit count) or list of hits
        if isinstance(result, int):
            report.bloom_hits += result
        elif isinstance(result, (list, tuple)):
            report.bloom_hits += len(result)
            for hit in result:
                report.flags.append(
                    StructuralFlag(
                        type="bloom_hit",
                        location="bloom_filter",
                        excerpt=getattr(hit, "matched_text", str(hit))[:120],
                        severity="medium",
                    )
                )
    except Exception:
        logger.debug("bloom_filter scanner raised", exc_info=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Content-type → scanner router
_ROUTERS: dict[ContentType, Any] = {
    ContentType.HTML: _run_html_scanner,
    ContentType.PDF: _run_pdf_scanner,
    ContentType.MARKDOWN: _run_markdown_scanner,
    # PLAIN has no structural scanner — fast path
}


def detect_structural(text: str) -> StructuralReport:
    """Run structural analysis on *text* and return a unified risk report.

    1. Detects content type.
    2. Routes to the appropriate structural sub-scanner.
    3. Runs the bloom filter (all content types).
    4. Computes an aggregate structural score.

    Plain text takes the fast path — no sub-scanners are invoked.

    Args:
        text: The input text to analyse.

    Returns:
        A :class:`StructuralReport` with findings and score.
    """
    ct = detect_content_type(text)
    report = StructuralReport(content_type=ct.value)

    # Route to content-type-specific scanner
    router = _ROUTERS.get(ct)
    if router is not None:
        router(text, report)

    # Bloom filter runs on all content types (except empty)
    if text.strip():
        _run_bloom_filter(text, report)

    # Compute aggregate score
    report.structural_score = _score_flags(report.flags)

    return report
