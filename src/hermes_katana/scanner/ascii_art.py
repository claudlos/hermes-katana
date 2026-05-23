"""
ASCII Art Scanner for HermesKatana.

Detects payloads hidden in ASCII art — text arranged spatially to spell out
injection commands, or large art blocks used to pad/hide malicious instructions.

Detection approach:
1. ASCII art detection: consistent line lengths, high symbol ratio, structural chars
2. Horizontal text extraction: extract lines from art blocks
3. Vertical text extraction: read columns top-to-bottom
4. Diagonal text extraction: read chars along main diagonals
5. Figlet-style detection: known figlet character patterns
6. Grid-hidden detection: readable words formed by marked characters in a grid

Performance: sub-millisecond for typical inputs (<10KB). Patterns precompiled
at module load time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "AsciiArtCategory",
    "AsciiArtFinding",
    "detect_ascii_art",
]


class AsciiArtCategory(str, Enum):
    """Categories of ASCII art attacks."""

    VERTICAL_TEXT = "vertical_text"
    """Text hidden vertically in character grid columns."""

    DIAGONAL_TEXT = "diagonal_text"
    """Text hidden diagonally in character grid."""

    FIGLET_ENCODED = "figlet_encoded"
    """Large character art encoding hidden text (figlet/banner style)."""

    GRID_HIDDEN = "grid_hidden"
    """Readable words formed by highlighted/marked characters in a grid."""


@dataclass(frozen=True, slots=True)
class AsciiArtFinding:
    """A single ASCII art detection finding."""

    category: AsciiArtCategory
    """Category of the ASCII art attack."""

    extracted_text: str
    """The text extracted from the ASCII art that triggered detection."""

    severity: str
    """Severity label: 'high', 'medium', 'low'."""

    confidence: float
    """Confidence score from 0.0 (uncertain) to 1.0 (certain)."""

    description: str
    """Human-readable explanation."""


# ---------------------------------------------------------------------------
# Injection keywords to detect in extracted text
# ---------------------------------------------------------------------------

_INJECTION_KEYWORDS: frozenset[str] = frozenset(
    {
        "ignore",
        "forget",
        "override",
        "system",
        "prompt",
        "instructions",
        "admin",
        "root",
        "jailbreak",
        "bypass",
        "disable",
        "reveal",
        "output",
        "execute",
        "sudo",
    }
)

# Compiled regex for quick keyword detection
_RE_INJECTION_KEYWORD = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in sorted(_INJECTION_KEYWORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# ASCII art detection heuristics
# ---------------------------------------------------------------------------

# Characters that are "art-like" (symbols, box-drawing, etc.)
_RE_ART_CHARS = re.compile(r"^[|/\\_\-#*=+~^@%$&!?.:;\"'\[\]{}()<> \t]+$")

# Structural chars that appear at consistent column positions (box-drawing)
_RE_STRUCTURAL_LINE = re.compile(r"^\s*[|+\-+|]+\s*$")

# Figlet-style pattern: 3+ lines of mostly #, *, /, \, _, or space
_RE_FIGLET_LINE = re.compile(r"^[#*/\\_| ]+$")

# Threshold for "high symbol ratio" in a line
_SYMBOL_RATIO_THRESHOLD = 0.65

# Minimum line length to consider for ASCII art
_MIN_ART_LINE_LENGTH = 20

# Minimum number of lines with similar length to be considered art
_MIN_ART_LINES = 4

# Minimum column width for vertical/diagonal extraction
_MIN_GRID_COLS = 3


def _is_likely_ascii_art_line(line: str) -> bool:
    """Return True if line looks like part of an ASCII art block."""
    stripped = line.rstrip("\n\r")
    if len(stripped) < _MIN_ART_LINE_LENGTH:
        return False
    if _RE_STRUCTURAL_LINE.match(stripped):
        return True
    if _RE_FIGLET_LINE.match(stripped):
        return True
    # High ratio of non-alphanumeric symbols
    non_alnum = sum(1 for c in stripped if not c.isalnum() and c not in (" ", "\t"))
    ratio = non_alnum / len(stripped) if stripped else 0
    return ratio >= _SYMBOL_RATIO_THRESHOLD


def _detect_art_lines(text: str) -> list[tuple[int, str]]:
    """Detect lines that are likely part of ASCII art.

    Returns list of (line_index, line_content) for art-like lines.
    """
    lines = text.splitlines()
    art_lines: list[tuple[int, str]] = []

    for i, line in enumerate(lines):
        if _is_likely_ascii_art_line(line):
            art_lines.append((i, line))

    return art_lines


def _group_art_blocks(art_lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    """Group consecutive art lines into blocks."""
    if not art_lines:
        return []

    blocks: list[list[tuple[int, str]]] = []
    current_block: list[tuple[int, str]] = []

    for item in art_lines:
        if not current_block:
            current_block.append(item)
        elif item[0] == current_block[-1][0] + 1:
            current_block.append(item)
        else:
            blocks.append(current_block)
            current_block = [item]

    if current_block:
        blocks.append(current_block)

    return blocks


def _extract_horizontal(text: str, block: list[tuple[int, str]]) -> str:
    """Extract readable text from horizontal lines in art block."""
    parts: list[str] = []
    for _, line in block:
        # Strip art symbols but keep alphanumeric and spaces
        cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in line.rstrip("\n\r"))
        cleaned = " ".join(cleaned.split())
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts)


def _extract_vertical(grid: list[str]) -> str:
    """Extract text reading columns top-to-bottom."""
    if not grid:
        return ""
    max_cols = max(len(row) for row in grid)
    if max_cols < _MIN_GRID_COLS:
        return ""

    result: list[str] = []
    for col in range(max_cols):
        col_chars = []
        for row in grid:
            if col < len(row):
                c = row[col]
                if c.isalnum():
                    col_chars.append(c)
        if col_chars:
            result.append("".join(col_chars))

    return " ".join(result)


def _extract_main_diagonal(grid: list[str]) -> str:
    """Extract text reading the main diagonal (top-left to bottom-right)."""
    if not grid or len(grid) < 2:
        return ""
    max_cols = min(len(row) for row in grid)
    if max_cols < _MIN_GRID_COLS:
        return ""

    diag_chars = [grid[i][i] for i in range(min(len(grid), max_cols))]
    # Only keep alphanumeric sequences
    result = "".join(c for c in diag_chars if c.isalnum())
    return result


def _extract_anti_diagonal(grid: list[str]) -> str:
    """Extract text reading the anti-diagonal (top-right to bottom-left)."""
    if not grid or len(grid) < 2:
        return ""
    max_cols = max(len(row) for row in grid)
    if max_cols < _MIN_GRID_COLS:
        return ""

    size = min(len(grid), max_cols)
    diag_chars = []
    for i in range(size):
        row = grid[i]
        col_idx = max_cols - 1 - i
        if col_idx < len(row):
            c = row[col_idx]
            if c.isalnum():
                diag_chars.append(c)
    return "".join(diag_chars)


def _check_keywords(text: str) -> tuple[bool, list[str]]:
    """Check text for injection keywords. Returns (found, matched_keywords)."""
    if not text:
        return False, []
    matches = _RE_INJECTION_KEYWORD.findall(text)
    # Normalize to lowercase for deduplication
    unique = list(dict.fromkeys(m.lower() for m in matches))
    return len(unique) > 0, unique


def _severity_and_confidence(category: AsciiArtCategory, keyword_count: int) -> tuple[str, float]:
    """Determine severity and confidence based on category and keyword matches."""
    if keyword_count >= 3:
        severity = "high"
        confidence = 0.90
    elif keyword_count == 2:
        severity = "medium"
        confidence = 0.75
    else:
        severity = "low"
        confidence = 0.60

    # Category-specific adjustments
    if category == AsciiArtCategory.FIGLET_ENCODED:
        confidence = min(confidence + 0.05, 0.95)
    elif category == AsciiArtCategory.VERTICAL_TEXT:
        confidence = min(confidence + 0.03, 0.92)

    return severity, confidence


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------


def detect_ascii_art(text: str) -> list[AsciiArtFinding]:
    """
    Detect payloads hidden in ASCII art.

    Args:
        text: Raw prompt text to analyze.

    Returns:
        List of AsciiArtFinding for each detected threat.
    """
    findings: list[AsciiArtFinding] = []

    if not text or len(text) < 50:
        return findings

    # Step 1: Detect art-like lines
    art_lines = _detect_art_lines(text)
    if not art_lines:
        return findings

    # Step 2: Group into consecutive blocks
    blocks = _group_art_blocks(art_lines)

    for block in blocks:
        if len(block) < _MIN_ART_LINES:
            continue

        # Build grid from block
        grid = [line for _, line in block]

        # Check line length consistency (low variance = more likely art)
        lengths = [len(line.rstrip("\n\r")) for _, line in block]
        if len(lengths) >= _MIN_ART_LINES:
            avg_len = sum(lengths) / len(lengths)
            variance = sum((ln - avg_len) ** 2 for ln in lengths) / len(lengths)
            # Low variance (std < 5 chars) suggests deliberate art structure
            if variance > 100:  # std > 10
                continue

        # ----- Horizontal extraction -----
        horiz_text = _extract_horizontal(text, block)
        found, keywords = _check_keywords(horiz_text)
        if found:
            sev, conf = _severity_and_confidence(AsciiArtCategory.GRID_HIDDEN, len(keywords))
            findings.append(
                AsciiArtFinding(
                    category=AsciiArtCategory.GRID_HIDDEN,
                    extracted_text=horiz_text[:200],
                    severity=sev,
                    confidence=conf,
                    description=(
                        f"Horizontal extraction from ASCII art block ({len(block)} lines) "
                        f"contains injection keywords: {', '.join(keywords[:5])}"
                    ),
                )
            )

        # ----- Vertical text extraction -----
        vert_text = _extract_vertical(grid)
        found, keywords = _check_keywords(vert_text)
        if found:
            sev, conf = _severity_and_confidence(AsciiArtCategory.VERTICAL_TEXT, len(keywords))
            findings.append(
                AsciiArtFinding(
                    category=AsciiArtCategory.VERTICAL_TEXT,
                    extracted_text=vert_text[:200],
                    severity=sev,
                    confidence=conf,
                    description=(
                        f"Vertical text extraction from ASCII art ({len(block)} lines, "
                        f"~{max(len(row) for row in grid) if grid else 0} cols) "
                        f"contains injection keywords: {', '.join(keywords[:5])}"
                    ),
                )
            )

        # ----- Diagonal text extraction -----
        main_diag = _extract_main_diagonal(grid)
        anti_diag = _extract_anti_diagonal(grid)

        for diag_text, diag_name in [(main_diag, "main"), (anti_diag, "anti-diagonal")]:
            if len(diag_text) < 4:
                continue
            found, keywords = _check_keywords(diag_text)
            if found:
                sev, conf = _severity_and_confidence(AsciiArtCategory.DIAGONAL_TEXT, len(keywords))
                findings.append(
                    AsciiArtFinding(
                        category=AsciiArtCategory.DIAGONAL_TEXT,
                        extracted_text=diag_text[:200],
                        severity=sev,
                        confidence=conf,
                        description=(
                            f"{diag_name} diagonal extraction from ASCII art grid "
                            f"contains injection keywords: {', '.join(keywords[:5])}"
                        ),
                    )
                )

        # ----- Figlet-style detection -----
        # Check for figlet character patterns (lines of consistent art chars)
        figlet_lines = 0
        for _, line in block:
            if _RE_FIGLET_LINE.match(line.rstrip("\n\r")):
                figlet_lines += 1
        if figlet_lines >= 3 and figlet_lines >= len(block) * 0.5:
            # Extract text from figlet block
            figlet_text = _extract_horizontal(text, block)
            if figlet_text:
                found, keywords = _check_keywords(figlet_text)
                if found:
                    sev, conf = _severity_and_confidence(AsciiArtCategory.FIGLET_ENCODED, len(keywords))
                    findings.append(
                        AsciiArtFinding(
                            category=AsciiArtCategory.FIGLET_ENCODED,
                            extracted_text=figlet_text[:200],
                            severity=sev,
                            confidence=conf,
                            description=(
                                f"Figlet-style ASCII art ({figlet_lines}/{len(block)} art lines) "
                                f"contains injection keywords: {', '.join(keywords[:5])}"
                            ),
                        )
                    )

    # Deduplicate by extracted_text + category
    seen: set[tuple[str, AsciiArtCategory]] = set()
    unique_findings: list[AsciiArtFinding] = []
    for f in findings:
        key = (f.extracted_text[:50], f.category)
        if key not in seen:
            seen.add(key)
            unique_findings.append(f)

    return unique_findings
