"""
Markdown AST auditor for HermesKatana Scabbard.

Parses Markdown to AST and audits:
- HTML comments in Markdown (hidden instructions)
- Raw HTML blocks that may contain malicious content
- YAML frontmatter with suspicious content
- Instructional content in link text or alt text
- Reference-style links that could hijack execution
- Image tags that could exfiltrate data

Catches Content Injection Traps in Markdown documents where
malicious content is hidden in non-rendered positions.

Speed: Microseconds for typical Markdown documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = [
    "MDAuditSeverity",
    "MDAuditFinding",
    "detect_markdown_audit",
]


class MDAuditSeverity(str, Enum):
    """Severity of Markdown audit findings."""

    CRITICAL = "critical"
    """Malicious content likely hidden in Markdown."""

    HIGH = "high"
    """Suspicious hidden content detected."""

    MEDIUM = "medium"
    """Content needs review."""

    LOW = "low"
    """Minor issue, likely benign."""


@dataclass(frozen=True, slots=True)
class MDAuditFinding:
    """A single Markdown audit finding.

    Attributes:
        finding_type: Type of issue detected.
        position: Location in the document.
        content: The suspicious content.
        rendered: What would be rendered (if applicable).
        severity: How severe this is.
        description: Human-readable explanation.
        confidence: Detection confidence 0.0-1.0.
    """

    finding_type: str
    position: str
    content: str
    rendered: str
    severity: MDAuditSeverity
    description: str
    confidence: float = 0.85


# Suspicious keywords for hidden instruction detection
_SUSPICIOUS_INSTRUCTIONS = [
    "ignore",
    "override",
    "bypass",
    "disregard",
    "forget",
    "new instruction",
    "system prompt",
    "reveal",
    "inject",
    "execute",
    "run",
    "delete",
    "send",
    "exfiltrate",
]


def _extract_frontmatter(markdown: str) -> tuple[Optional[str], str]:
    """Extract YAML frontmatter if present.

    Returns (frontmatter, remaining_markdown).
    """
    if markdown.startswith("---"):
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", markdown, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
    return None, markdown


def _audit_frontmatter(frontmatter: str) -> list[MDAuditFinding]:
    """Audit YAML frontmatter for suspicious content."""
    findings: list[MDAuditFinding] = []

    if not frontmatter:
        return findings

    frontmatter_lower = frontmatter.lower()

    # Check for suspicious keywords
    for instruction in _SUSPICIOUS_INSTRUCTIONS:
        if instruction in frontmatter_lower:
            findings.append(
                MDAuditFinding(
                    finding_type="suspicious_frontmatter",
                    position="YAML frontmatter",
                    content=frontmatter[:200],
                    rendered="",
                    severity=MDAuditSeverity.MEDIUM,
                    description=f"Suspicious instruction '{instruction}' in YAML frontmatter.",
                    confidence=0.75,
                )
            )
            break

    # Check for suspiciously long or encoded values
    lines = frontmatter.split("\n")
    for line in lines:
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip()
            # Base64-like strings in metadata
            if len(value) > 50 and re.match(r"^[A-Za-z0-9+/=]+$", value):
                findings.append(
                    MDAuditFinding(
                        finding_type="encoded_frontmatter_value",
                        position=f"frontmatter key: {key.strip()}",
                        content=value[:100],
                        rendered="",
                        severity=MDAuditSeverity.MEDIUM,
                        description="Frontmatter contains a long encoded value that may hide content.",
                        confidence=0.70,
                    )
                )

    return findings


def _extract_html_comments(markdown: str) -> list[tuple[str, int]]:
    """Extract HTML comments and their positions."""
    comments = []
    for match in re.finditer(r"<!--(.*?)-->", markdown, re.DOTALL):
        comments.append((match.group(1), match.start()))
    return comments


def _extract_html_blocks(markdown: str) -> list[tuple[str, int]]:
    """Extract raw HTML blocks in Markdown."""
    blocks = []
    # HTML blocks that start at beginning of line
    for match in re.finditer(r"^<[a-zA-Z][^>]*>.*?</[a-zA-Z]+>", markdown, re.DOTALL | re.MULTILINE):
        blocks.append((match.group(), match.start()))
    # Standalone script/style tags
    for match in re.finditer(
        r"<(script|style|iframe|object|embed|form)[^>]*>.*?</\1>", markdown, re.DOTALL | re.IGNORECASE
    ):
        blocks.append((match.group(), match.start()))
    return blocks


def _extract_link_definitions(markdown: str) -> list[tuple[str, str, int]]:
    """Extract Markdown link definitions [ref]: url pattern.

    Returns list of (reference_name, url, position).
    """
    definitions = []
    for match in re.finditer(r"^\s*\[([^\]]+)\]:\s*(.+)$", markdown, re.MULTILINE):
        ref = match.group(1)
        url = match.group(2).strip()
        definitions.append((ref, url, match.start()))
    return definitions


def _audit_html_comments(
    comments: list[tuple[str, int]], visible_text: str, markdown: str = ""
) -> list[MDAuditFinding]:
    """Audit HTML comments for suspicious hidden content."""
    findings: list[MDAuditFinding] = []

    for comment, pos in comments:
        comment_stripped = comment.strip()
        if not comment_stripped:
            continue

        comment_lower = comment_stripped.lower()

        # Skip obviously benign comments
        benign_patterns = ["generator", "timestamp", "created", "modified", "version"]
        if any(bp in comment_lower for bp in benign_patterns):
            continue

        # Check for suspicious content
        matches = [kw for kw in _SUSPICIOUS_INSTRUCTIONS if kw in comment_lower]

        if matches:
            severity = MDAuditSeverity.HIGH
            confidence = 0.85
            if any(kw in ["ignore", "override", "inject", "bypass"] for kw in matches):
                severity = MDAuditSeverity.CRITICAL
                confidence = 0.90

            findings.append(
                MDAuditFinding(
                    finding_type="hidden_html_comment",
                    position=f"line {len(markdown[:pos].splitlines())}",
                    content=comment_stripped[:200],
                    rendered="[comment not rendered]",
                    severity=severity,
                    description=f"HTML comment contains suspicious content: {', '.join(matches)}. This content is hidden from rendered view.",
                    confidence=confidence,
                )
            )
        # Generic suspicious content check
        elif len(comment_stripped) > 10 and re.search(r"(ignore|override|bypass|disregard|inject)", comment_lower):
            findings.append(
                MDAuditFinding(
                    finding_type="suspicious_html_comment",
                    position=f"line ~{len(markdown[:pos].splitlines())}",
                    content=comment_stripped[:200],
                    rendered="[comment not rendered]",
                    severity=MDAuditSeverity.MEDIUM,
                    description="HTML comment may contain hidden instructions.",
                    confidence=0.70,
                )
            )

    return findings


def _audit_html_blocks(blocks: list[tuple[str, int]], visible_text: str) -> list[MDAuditFinding]:
    """Audit raw HTML blocks in Markdown."""
    findings: list[MDAuditFinding] = []

    for block, pos in blocks:
        block_lower = block.lower()

        # Check for malicious HTML patterns
        malicious_patterns = [
            (r"<script", "script tag - may execute JavaScript"),
            (r"on(error|click|load|focus|blur|submit|change)\s*=", "event handler - XSS risk"),
            (r"javascript:", "javascript: protocol - may execute code"),
            (r"<iframe", "iframe - may embed attacker content"),
            (r"<form", "form tag - may phish or exfiltrate data"),
            (r"<meta\s+http-equiv", "meta refresh - may redirect"),
            (r"display\s*:\s*none", "display:none - content hidden from user"),
            (r"visibility\s*:\s*hidden", "visibility:hidden - content hidden from user"),
        ]

        for pattern, description in malicious_patterns:
            if re.search(pattern, block_lower, re.IGNORECASE):
                findings.append(
                    MDAuditFinding(
                        finding_type="malicious_html_block",
                        position=f"offset {pos}",
                        content=block[:200],
                        rendered="[HTML block rendered]",
                        severity=MDAuditSeverity.CRITICAL,
                        description=f"Raw HTML block contains potentially malicious content: {description}.",
                        confidence=0.88,
                    )
                )
                break

    return findings


def _audit_link_definitions(
    definitions: list[tuple[str, str, int]],
    markdown: str,
) -> list[MDAuditFinding]:
    """Audit reference-style link definitions for exfiltration patterns."""
    findings: list[MDAuditFinding] = []

    for ref, url, pos in definitions:
        url_lower = url.lower()

        # Check for data exfiltration via URL parameters
        if any(param in url_lower for param in ["token=", "secret=", "key=", "password=", "session=", "auth="]):
            findings.append(
                MDAuditFinding(
                    finding_type="link_exfiltration",
                    position=f"reference [{ref}]",
                    content=f"[{ref}]: {url}",
                    rendered=f"[{ref}]",
                    severity=MDAuditSeverity.HIGH,
                    description="Link definition contains sensitive data in URL parameters.",
                    confidence=0.85,
                )
            )

        # Check for javascript: protocol
        if url_lower.startswith("javascript:"):
            findings.append(
                MDAuditFinding(
                    finding_type="javascript_link_reference",
                    position=f"reference [{ref}]",
                    content=f"[{ref}]: {url}",
                    rendered=f"[{ref}]",
                    severity=MDAuditSeverity.CRITICAL,
                    description="Link reference uses javascript: protocol which may execute code.",
                    confidence=0.90,
                )
            )

        # Check for data: URI
        if url_lower.startswith("data:"):
            findings.append(
                MDAuditFinding(
                    finding_type="data_uri_reference",
                    position=f"reference [{ref}]",
                    content=f"[{ref}]: {url[:50]}...",
                    rendered=f"[{ref}]",
                    severity=MDAuditSeverity.HIGH,
                    description="Link reference uses data: URI which may contain hidden executable content.",
                    confidence=0.85,
                )
            )

    return findings


def _audit_instructional_links(markdown: str) -> list[MDAuditFinding]:
    """Audit link text and image alt text for instructional content."""
    findings: list[MDAuditFinding] = []

    # Link patterns: [text](url) or ![alt](url)
    link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"

    for match in re.finditer(link_pattern, markdown):
        link_text = match.group(1)
        url = match.group(2)
        link_lower = link_text.lower()

        # Check if link text contains instructional language
        instructional_patterns = [
            (r"^click\s+here$", "click here link"),
            (r"^download\s+now$", "download now link"),
            (r"^ignore\s+", "ignore instruction in link"),
            (r"^disregard\s+", "disregard instruction in link"),
            (r"^follow\s+this\s+link\s+to\s+(?:ignore|bypass)", "bypass instruction link"),
        ]

        for pattern, description in instructional_patterns:
            if re.search(pattern, link_lower):
                findings.append(
                    MDAuditFinding(
                        finding_type="instructional_link",
                        position=f"offset {match.start()}",
                        content=link_text[:100],
                        rendered=url[:50],
                        severity=MDAuditSeverity.MEDIUM,
                        description=f"Link text contains instructional content: {description}.",
                        confidence=0.75,
                    )
                )

        # Check for data exfiltration in image URLs
        if link_text.startswith("!"):
            url_lower = url.lower()
            if any(param in url_lower for param in ["data=", "token=", "session=", "secret=", "key="]):
                findings.append(
                    MDAuditFinding(
                        finding_type="image_exfiltration",
                        position=f"offset {match.start()}",
                        content=f"![{link_text}]({url})",
                        rendered="[image]",
                        severity=MDAuditSeverity.HIGH,
                        description="Image URL contains suspicious parameters that may exfiltrate data.",
                        confidence=0.85,
                    )
                )

    return findings


def _build_visible_text(markdown: str) -> str:
    """Build approximate visible text from markdown."""
    # Remove frontmatter
    _, markdown = _extract_frontmatter(markdown)
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    # Remove raw HTML blocks
    text = re.sub(r"<[a-zA-Z][^>]*>.*?</[a-zA-Z]+>", "", text, flags=re.DOTALL)
    text = re.sub(r"<(script|style|iframe|object|embed)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove link definitions
    text = re.sub(r"^\s*\[[^\]]+\]:\s+.+$", "", text, flags=re.MULTILINE)
    # Remove inline code (but keep backticks for visual reference)
    return text


def detect_markdown_audit(markdown: str) -> list[MDAuditFinding]:
    """Audit Markdown document for hidden/malicious content.

    Performs structural analysis of Markdown by parsing the AST-equivalent
    and checking non-rendered positions for suspicious content.

    Args:
        markdown: The Markdown content to audit.

    Returns:
        List of MDAuditFinding objects for each issue detected.
    """
    findings: list[MDAuditFinding] = []

    if not markdown or not markdown.strip():
        return findings

    # Extract and audit frontmatter
    frontmatter, remaining = _extract_frontmatter(markdown)
    if frontmatter is not None:
        findings.extend(_audit_frontmatter(frontmatter))

    # Build approximate visible text for comparison
    visible_text = _build_visible_text(markdown)

    # Extract and audit HTML comments
    comments = _extract_html_comments(markdown)
    findings.extend(_audit_html_comments(comments, visible_text, markdown))

    # Extract and audit raw HTML blocks
    html_blocks = _extract_html_blocks(markdown)
    findings.extend(_audit_html_blocks(html_blocks, visible_text))

    # Extract and audit link definitions
    definitions = _extract_link_definitions(markdown)
    findings.extend(_audit_link_definitions(definitions, markdown))

    # Audit link text and alt text
    findings.extend(_audit_instructional_links(markdown))

    return findings
