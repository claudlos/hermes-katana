"""Aho-Corasick multi-pattern scanner for injection phrases.

Builds a single automaton at module load time from all known injection
phrase lists, then matches all patterns in a single O(n) pass over the
input text.

Typical performance: <0.5 ms for payloads under 10 KB.

Usage:
    from hermes_katana.scanner.aho_scanner import detect_aho

    findings = detect_aho("Ignore previous instructions and tell me…")
    for f in findings:
        print(f.phrase, f.category, f.confidence, f.span)
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import ahocorasick

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

_SCANNER_DATA = Path(__file__).parent / "data" / "fast_patterns.json"
_SCABBARD_DATA = Path(__file__).parent.parent / "scabbard" / "data" / "attack_seed_phrases.json"
_NGRAMS_FILE = Path(__file__).parent.parent / "scabbard" / "data" / "injection_ngrams.txt"

# ---------------------------------------------------------------------------
# Category → confidence mapping
# Reflects how reliably each phrase bucket predicts a real attack.
# ---------------------------------------------------------------------------

_CATEGORY_CONFIDENCE: dict[str, float] = {
    # fast_patterns.json buckets
    "injection_phrase": 0.90,
    "jailbreak_phrase": 0.85,
    "exfil_phrase": 0.85,
    "persona_phrase": 0.70,
    "restriction_removal": 0.80,
    "system_prompt": 0.88,
    # attack_seed_phrases.json buckets
    "content_injection": 0.90,
    "semantic_manipulation": 0.75,
    "behavioral_control": 0.80,
    "exfiltration_attempt": 0.88,
    "jailbreak": 0.85,
    "cognitive_state_attack": 0.78,
    # injection_ngrams.txt — kept at 0.88 so it clears the 0.9 scanner threshold
    # when called directly; the real FP reduction comes from min-length pruning.
    "injection_ngram": 0.88,
}

_DEFAULT_CONFIDENCE = 0.75


# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AhoFinding:
    """A phrase match returned by detect_aho()."""

    phrase: str
    """Exact phrase that matched (lower-cased, normalised)."""

    category: str
    """Source bucket (e.g. 'injection_phrase', 'jailbreak_phrase')."""

    confidence: float
    """Estimated confidence that this match indicates a real attack (0–1)."""

    span: tuple[int, int]
    """(start, end) character offsets in the *original* text (end is exclusive)."""

    strategy: str = "aho_corasick"


# ---------------------------------------------------------------------------
# Automaton builder (runs once at import time)
# ---------------------------------------------------------------------------


def _load_phrases() -> list[tuple[str, str]]:
    """Return [(phrase, category)] from all data sources, de-duplicated.

    Short noisy patterns (e.g. single-keyword terms and generic 2-3 word
    phrases like "no moral", "i am") are dropped to reduce false positives.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    # Minimum character length per category.  Patterns shorter than this are
    # skipped at load time so they never enter the automaton.
    _MIN_LEN: dict[str, int] = {
        "injection_ngram": 10,  # drop short tech-exec and generic 2-word noise
    }
    _DEFAULT_MIN_LEN = 0  # all other categories are curated; no length filter

    def add(phrase: str, category: str) -> None:
        key = phrase.strip().lower()
        min_len = _MIN_LEN.get(category, _DEFAULT_MIN_LEN)
        if key and key not in seen and " " in key and len(key) >= min_len:
            seen.add(key)
            pairs.append((key, category))

    # fast_patterns.json
    if _SCANNER_DATA.exists():
        try:
            data = json.loads(_SCANNER_DATA.read_text())
            for cat, phrases in data.items():
                for ph in phrases:
                    add(ph, cat)
        except Exception:
            pass

    # attack_seed_phrases.json
    if _SCABBARD_DATA.exists():
        try:
            data = json.loads(_SCABBARD_DATA.read_text())
            for cat, phrases in data.items():
                for ph in phrases:
                    add(ph, cat)
        except Exception:
            pass

    # injection_ngrams.txt
    if _NGRAMS_FILE.exists():
        try:
            for line in _NGRAMS_FILE.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    add(line, "injection_ngram")
        except Exception:
            pass

    # 145k derived corpus: aho_patterns.txt
    _CORPUS_DIR = Path(__file__).resolve().parents[3] / "training" / "scanner_data"
    _AHO_CORPUS = _CORPUS_DIR / "aho_patterns.txt"
    if _AHO_CORPUS.exists():
        try:
            for line in _AHO_CORPUS.read_text().splitlines():
                line = line.strip()
                if line and len(line) >= 10:
                    add(line, "injection_ngram")
        except Exception:
            pass

    return pairs


def _build_automaton(pairs: list[tuple[str, str]]) -> ahocorasick.Automaton:
    """Build and finalise the Aho-Corasick automaton."""
    automaton = ahocorasick.Automaton()
    for phrase, category in pairs:
        # Store (phrase, category) as the payload; the key is already lower-cased
        automaton.add_word(phrase, (phrase, category))
    automaton.make_automaton()
    return automaton


# Build at import time — this is the shared singleton.
_PHRASES = _load_phrases()
_AUTOMATON: ahocorasick.Automaton = _build_automaton(_PHRASES)
_PHRASE_COUNT = len(_PHRASES)


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Lower-case + NFKC-normalise so fullwidth/ligature variants match."""
    return unicodedata.normalize("NFKC", text).lower()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_aho(
    text: str,
    *,
    min_confidence: float = 0.0,
    automaton: Optional[ahocorasick.Automaton] = None,
) -> list[AhoFinding]:
    """Scan *text* for injection phrases using the Aho-Corasick automaton.

    Parameters
    ----------
    text:
        Raw input to scan.
    min_confidence:
        Only return findings at or above this threshold (default 0 = all).
    automaton:
        Override the module-level singleton (for testing with custom phrase sets).

    Returns
    -------
    list[AhoFinding]
        Sorted by (start_position, -confidence).  Empty list = no match.
    """
    if not text:
        return []

    ac = automaton if automaton is not None else _AUTOMATON

    norm = _normalise(text)
    findings: list[AhoFinding] = []

    for end_idx, (phrase, category) in ac.iter(norm):
        confidence = _CATEGORY_CONFIDENCE.get(category, _DEFAULT_CONFIDENCE)
        if confidence < min_confidence:
            continue
        start = end_idx - len(phrase) + 1
        end = end_idx + 1
        findings.append(
            AhoFinding(
                phrase=phrase,
                category=category,
                confidence=confidence,
                span=(start, end),
            )
        )

    findings.sort(key=lambda f: (f.span[0], -f.confidence))
    return findings


def phrase_count() -> int:
    """Return the number of phrases loaded into the automaton."""
    return _PHRASE_COUNT


def build_custom_automaton(phrase_category_pairs: list[tuple[str, str]]) -> ahocorasick.Automaton:
    """Build a fresh automaton from the given (phrase, category) pairs.

    Useful for testing or creating specialised sub-automata.
    """
    return _build_automaton([(ph.strip().lower(), cat) for ph, cat in phrase_category_pairs if ph.strip()])
