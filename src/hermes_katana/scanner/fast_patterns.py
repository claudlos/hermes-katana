"""
Ultra-Fast Pattern Matching Engine using Aho-Corasick.

This module provides O(n) multi-pattern matching against 1000+ injection
patterns simultaneously using the Aho-Corasick algorithm.

Performance targets:
- <100 microseconds for typical input
- O(n) time complexity regardless of pattern count
- Unicode text handling

Usage:
    from hermes_katana.scanner.fast_patterns import detect_fast_patterns

    findings = detect_fast_patterns("Please ignore previous instructions...")
    for finding in findings:
        print(f"{finding.category}: {finding.matched_pattern} at {finding.position}")
"""

from __future__ import annotations

import ahocorasick
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "FastPatternCategory",
    "FastPatternFinding",
    "detect_fast_patterns",
]


class FastPatternCategory(str, Enum):
    """Categories of fast-pattern findings."""

    INJECTION_PHRASE = "injection_phrase"
    JAILBREAK_PHRASE = "jailbreak_phrase"
    EXFIL_PHRASE = "exfil_phrase"
    PERSONA_PHRASE = "persona_phrase"
    RESTRICTION_REMOVAL = "restriction_removal"
    SYSTEM_PROMPT = "system_prompt"


@dataclass(frozen=True, slots=True)
class FastPatternFinding:
    """A single pattern match finding.

    Attributes:
        category: The pattern category that matched.
        matched_pattern: The actual text that matched.
        position: Character position where match starts in input text.
        severity: Severity level of the finding.
        confidence: Confidence score from 0.0 to 1.0.
        description: Human-readable description of the finding.
    """

    category: FastPatternCategory
    matched_pattern: str
    position: int
    severity: str
    confidence: float
    description: str


# Severity and confidence mappings by category
_CATEGORY_SEVERITY: dict[FastPatternCategory, str] = {
    FastPatternCategory.INJECTION_PHRASE: "high",
    FastPatternCategory.JAILBREAK_PHRASE: "critical",
    FastPatternCategory.EXFIL_PHRASE: "high",
    FastPatternCategory.PERSONA_PHRASE: "medium",
    FastPatternCategory.RESTRICTION_REMOVAL: "high",
    FastPatternCategory.SYSTEM_PROMPT: "medium",
}

_CATEGORY_CONFIDENCE: dict[FastPatternCategory, float] = {
    FastPatternCategory.INJECTION_PHRASE: 0.85,
    FastPatternCategory.JAILBREAK_PHRASE: 0.95,
    FastPatternCategory.EXFIL_PHRASE: 0.80,
    FastPatternCategory.PERSONA_PHRASE: 0.70,
    FastPatternCategory.RESTRICTION_REMOVAL: 0.85,
    FastPatternCategory.SYSTEM_PROMPT: 0.75,
}

_CATEGORY_DESCRIPTIONS: dict[FastPatternCategory, str] = {
    FastPatternCategory.INJECTION_PHRASE: "Known injection phrase detected",
    FastPatternCategory.JAILBREAK_PHRASE: "Jailbreak pattern detected - attempts to bypass safety",
    FastPatternCategory.EXFIL_PHRASE: "Exfiltration pattern detected - attempts to extract system information",
    FastPatternCategory.PERSONA_PHRASE: "Persona manipulation pattern detected",
    FastPatternCategory.RESTRICTION_REMOVAL: "Restriction removal pattern detected",
    FastPatternCategory.SYSTEM_PROMPT: "System prompt reference detected - possible extraction attempt",
}


def _load_patterns() -> dict[FastPatternCategory, list[str]]:
    """Load patterns from the JSON data file."""
    data_path = Path(__file__).parent / "data" / "fast_patterns.json"
    with open(data_path, encoding="utf-8") as f:
        raw: dict[str, list[str]] = json.load(f)

    result: dict[FastPatternCategory, list[str]] = {}
    for category_str, patterns in raw.items():
        try:
            category = FastPatternCategory(category_str)
            result[category] = patterns
        except ValueError:
            continue
    return result


def _build_automaton() -> ahocorasick.Automaton:
    """Build the Aho-Corasick automaton from loaded patterns.

    Returns:
        Compiled automaton with (pattern_lower, category) payloads.
    """
    automaton = ahocorasick.Automaton(ahocorasick.STORE_ANY)

    patterns_by_category = _load_patterns()

    for category, patterns in patterns_by_category.items():
        for pattern in patterns:
            # Convert pattern to lowercase for case-insensitive matching
            pattern_lower = pattern.lower()
            # Add pattern to automaton with (pattern, category) as payload
            automaton.add_word(pattern_lower, (pattern_lower, category))

    # Extend with 145k corpus phrases - must be BEFORE make_automaton()
    _extend_with_corpus(automaton)

    automaton.make_automaton()

    return automaton


# ---------------------------------------------------------------------------
# 145k corpus patterns: extend automaton with derived phrases
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path(__file__).resolve().parents[3] / "training" / "scanner_data"


def _extend_with_corpus(automaton: ahocorasick.Automaton) -> int:
    """Load 145k-derived English phrases into the automaton.

    Maps phrases to categories based on keyword presence.
    Returns count of added patterns.
    """
    bloom_file = _CORPUS_DIR / "bloom_phrases_en.txt"
    if not bloom_file.exists():
        return 0

    _CATEGORY_KWS: dict[FastPatternCategory, list[str]] = {
        # Check INJECTION first since "ignore/disregard/forget" are the most
        # universal injection indicators and should win over broader matches.
        FastPatternCategory.INJECTION_PHRASE: ["ignore", "disregard", "forget", "new instructions", "from now on"],
        FastPatternCategory.JAILBREAK_PHRASE: [
            "jailbreak",
            "dan ",
            "do anything",
            "unrestricted",
            "uncensored",
            "godmode",
            "god mode",
            "evil mode",
        ],
        # "secret" and "system prompt" are more specific than standalone "instructions"
        FastPatternCategory.EXFIL_PHRASE: ["exfil", "secret", "system prompt", "reveal", "your prompt"],
        FastPatternCategory.RESTRICTION_REMOVAL: [
            "no filter",
            "no restriction",
            "no rule",
            "no safety",
            "no moral",
            "no ethic",
            "bypass",
            "override",
        ],
        FastPatternCategory.PERSONA_PHRASE: ["you are now", "act as", "pretend", "roleplay", "persona"],
    }

    count = 0
    try:
        for line in bloom_file.read_text().splitlines():
            phrase = line.strip().lower()
            if not phrase or len(phrase) < 6:
                continue

            # Pick the most specific category
            cat = FastPatternCategory.INJECTION_PHRASE  # default bucket
            for c, kws in _CATEGORY_KWS.items():
                if any(kw in phrase for kw in kws):
                    cat = c
                    break

            automaton.add_word(phrase, (phrase, cat))
            count += 1
    except Exception:
        pass

    return count


def detect_fast_patterns(text: str) -> list["FastPatternFinding"]:
    """Detect all fast patterns in the given text.

    Uses Aho-Corasick algorithm for O(n) multi-pattern matching.
    """
    if not text:
        return []

    norm = text.lower()
    findings: list["FastPatternFinding"] = []

    for end_idx, (pattern, category) in _AUTOMATON.iter(norm):
        start = end_idx - len(pattern) + 1
        findings.append(
            FastPatternFinding(
                category=FastPatternCategory(category),
                matched_pattern=pattern,
                position=start,
                severity="high" if category.value in ("jailbreak_phrase", "exfil_phrase") else "medium",
                confidence=0.9,
                description=f"Fast pattern match: {category.value}",
            )
        )

    findings.sort(key=lambda f: f.position)
    return findings


# Build automaton at module load time
_AUTOMATON = _build_automaton()
