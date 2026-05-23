"""Shared helpers for running scanners against a corpus."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from hermes_katana.scanner.commands import detect_dangerous_command  # noqa: E402
from hermes_katana.scanner.content import scan_content  # noqa: E402
from hermes_katana.scanner.injection import detect_injection  # noqa: E402
from hermes_katana.scanner.secrets import scan_for_secrets  # noqa: E402
from hermes_katana.scanner.unicode import normalize_and_scan  # noqa: E402

# Structural Scabbard scanners
from hermes_katana.scanner.bloom_filter import scan_bloom  # noqa: E402
from hermes_katana.scanner.html_diff import scan_html  # noqa: E402
from hermes_katana.scanner.pdf_layers import detect_pdf_layers  # noqa: E402
from hermes_katana.scanner.markdown_audit import detect_markdown_audit  # noqa: E402
from hermes_katana.scanner.css_deobfuscator import detect_css_deobfuscate  # noqa: E402
from hermes_katana.scanner.structural import detect_structural  # noqa: E402

# Scanners being built by other agents — optional imports
try:
    from hermes_katana.scanner.decoder import decode_and_scan  # noqa: E402
except Exception:
    decode_and_scan = None

try:
    from hermes_katana.scanner.content_harm import scan_content_harm  # noqa: E402
except Exception:
    scan_content_harm = None

try:
    from hermes_katana.scanner.prompt_leak import detect_prompt_leak  # noqa: E402
except Exception:
    detect_prompt_leak = None

try:
    from hermes_katana.scanner.compositional import detect_compositional  # noqa: E402
except Exception:
    detect_compositional = None

try:
    from hermes_katana.scanner.multilingual import detect_multilingual  # noqa: E402
except Exception:
    detect_multilingual = None

try:
    from hermes_katana.scanner.ascii_art import detect_ascii_art  # noqa: E402
except Exception:
    detect_ascii_art = None

try:
    from hermes_katana.scanner.persona_detector import detect_persona_jailbreak  # noqa: E402
except Exception:
    detect_persona_jailbreak = None

# Unicode spoof detector
try:
    from hermes_katana.scanner.unicode_spoof import scan_unicode_spoof
except Exception:
    scan_unicode_spoof = None

# SVG sanitizer
try:
    from hermes_katana.scanner.svg_sanitizer import scan_svg
except Exception:
    scan_svg = None

# PDF JavaScript scanner
try:
    from hermes_katana.scanner.pdf_js_scanner import detect_pdf_js
except Exception:
    detect_pdf_js = None

# Multimodal scanner
try:
    from hermes_katana.scanner.multimodal import scan_data_uri
except Exception:
    scan_data_uri = None

# Behavioral scanner
try:
    from hermes_katana.scanner.behavioral import detect_behavioral
except Exception:
    detect_behavioral = None

# Stego scanner
try:
    from hermes_katana.scanner.stego_scanner import scan_stego
except Exception:
    scan_stego = None


# Stego scanner - scan_stego returns report with flags
def _stego_scan(text):
    if scan_stego is None:
        return []
    # scan_stego expects file bytes, for text input skip
    return []


# Behavioral scanner
try:
    from hermes_katana.scanner.behavioral import detect_behavioral
except Exception:
    detect_behavioral = None

# Stego scanner
try:
    from hermes_katana.scanner.stego_scanner import scan_stego
except Exception:
    scan_stego = None

# SVG sanitizer
try:
    from hermes_katana.scanner.svg_sanitizer import scan_svg
except Exception:
    scan_svg = None

# Unicode spoof
try:
    from hermes_katana.scanner.unicode_spoof import scan_unicode_spoof
except Exception:
    scan_unicode_spoof = None

# PDF JavaScript scanner
try:
    from hermes_katana.scanner.pdf_js_scanner import detect_pdf_js
except Exception:
    detect_pdf_js = None

# Multimodal scanner (data URIs)
try:
    from hermes_katana.scanner.multimodal import scan_data_uri
except Exception:
    scan_data_uri = None

# Semantic recall scanner
try:
    from hermes_katana.scanner.semantic_recall import detect_semantic  # noqa: E402
except Exception:
    detect_semantic = None

# Scabbard ML classifier
try:
    from hermes_katana.scabbard.scabbard import ScabbardClassifier  # noqa: E402
    from hermes_katana.scabbard.scabbard import ScabbardConfig  # noqa: E402
    from hermes_katana.scabbard.fusion import Decision  # noqa: E402

    _scabbard_classifier: ScabbardClassifier | None = None

    def _scabbard_scan(text):
        global _scabbard_classifier
        if _scabbard_classifier is None:
            _scabbard_classifier = ScabbardClassifier(ScabbardConfig(profile="minimal"))
        clf = _scabbard_classifier
        result = clf.classify(text)
        if result.decision != Decision.ALLOW:
            return [result]
        return []
except Exception:
    _scabbard_scan = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scanner registry
# ---------------------------------------------------------------------------

# Core scanners (always available)
SCANNER_NAMES_CORE = ("injection", "commands", "content", "unicode", "secrets")

# Structural scanners (always available)
SCANNER_NAMES_STRUCTURAL = (
    "bloom_filter",
    "html_diff",
    "pdf_layers",
    "markdown_audit",
    "css_deobfuscator",
    "structural",
)

# Optional scanners (may not be built yet)
SCANNER_NAMES_OPTIONAL = (
    "decoder",
    "content_harm",
    "prompt_leak",
    "compositional",
    "multilingual",
    "ascii_art",
    "persona_detector",
    "semantic_recall",
    "scabbard",
    "unicode_spoof",
    "behavioral",
    "svg_sanitizer",
    "pdf_js",
    "multimodal",
    "stego",
)


def _structural_findings(text: str) -> list:
    """Wrap detect_structural to return flags list for eval compatibility."""
    report = detect_structural(text)
    return report.flags if report.flags else []


def make_scanner_suite() -> dict[str, Any]:
    """Return a dict mapping scanner name -> callable(text) -> list[findings]."""
    suite: dict[str, Any] = {
        # Core scanners
        "injection": lambda text: detect_injection(text),
        "commands": lambda text: detect_dangerous_command(text),
        "content": lambda text: scan_content(text),
        "unicode": lambda text: normalize_and_scan(text)[1],
        "secrets": lambda text: scan_for_secrets(text),
        # Structural scanners
        "bloom_filter": lambda text: scan_bloom(text),
        "html_diff": lambda text: scan_html(text),
        "pdf_layers": lambda text: detect_pdf_layers(text),
        "markdown_audit": lambda text: detect_markdown_audit(text),
        "css_deobfuscator": lambda text: detect_css_deobfuscate(text),
        "structural": _structural_findings,
    }

    # Optional scanners — only add if the module was imported successfully
    if decode_and_scan is not None:
        suite["decoder"] = lambda text: decode_and_scan(text)
    if scan_content_harm is not None:
        suite["content_harm"] = lambda text: scan_content_harm(text)
    if detect_prompt_leak is not None:
        suite["prompt_leak"] = lambda text: detect_prompt_leak(text)
    if detect_compositional is not None:
        suite["compositional"] = lambda text: detect_compositional(text)
    if detect_multilingual is not None:
        suite["multilingual"] = lambda text: detect_multilingual(text)
    if detect_ascii_art is not None:
        suite["ascii_art"] = lambda text: detect_ascii_art(text)
    if detect_persona_jailbreak is not None:
        suite["persona_detector"] = lambda text: detect_persona_jailbreak(text)
    if detect_semantic is not None:
        suite["semantic_recall"] = lambda text: detect_semantic(text)
    if _scabbard_scan is not None:
        suite["scabbard"] = lambda text: _scabbard_scan(text)
    if scan_unicode_spoof is not None:
        suite["unicode_spoof"] = lambda text: scan_unicode_spoof(text)
    if detect_behavioral is not None:
        suite["behavioral"] = lambda text: detect_behavioral(text)
    if scan_svg is not None:
        suite["svg_sanitizer"] = lambda text: scan_svg(text)
    if detect_pdf_js is not None:
        suite["pdf_js"] = lambda text: detect_pdf_js(text)
    if scan_data_uri is not None:
        suite["multimodal"] = lambda text: scan_data_uri(text)
    if scan_stego is not None:
        suite["stego"] = lambda text: _stego_scan(text)

    return suite


# Combined name tuple for backwards compat
SCANNER_NAMES = SCANNER_NAMES_CORE + SCANNER_NAMES_STRUCTURAL


# ---------------------------------------------------------------------------
# Running scanners
# ---------------------------------------------------------------------------


def _scan_one(text: str, scanners: dict[str, Any]) -> dict[str, list]:
    """Run all scanners on a single text, returning {scanner_name: findings}."""
    results: dict[str, list] = {}
    for name, scan_fn in scanners.items():
        try:
            findings = scan_fn(text)
            results[name] = findings if findings else []
        except Exception:
            logger.debug("Scanner %s raised on text[:80]=%r", name, text[:80], exc_info=True)
            results[name] = []
    return results


def is_caught(per_scanner: dict[str, list]) -> bool:
    """Return True if any scanner produced findings."""
    return any(len(f) > 0 for f in per_scanner.values())


def run_scanners(corpus: list[dict], scanners: dict[str, Any]) -> tuple[int, int]:
    """Run scanners on corpus, return (caught_count, total_count)."""
    caught = 0
    total = len(corpus)
    for rec in corpus:
        text = rec.get("attack_text", "")
        per_scanner = _scan_one(text, scanners)
        if is_caught(per_scanner):
            caught += 1
    return caught, total


def run_scanners_detailed(corpus: list[dict], scanners: dict[str, Any]) -> dict[str, Any]:
    """Run scanners and return detailed breakdown.

    Returns dict with keys:
        total, caught, coverage,
        per_scanner: {name: {deny: int, flag: int}},
        per_category: {cat: {total: int, caught: int, coverage: float}},
        missed_samples: list of first 20 missed attack_text snippets
    """
    total = len(corpus)
    caught = 0
    per_scanner_deny: dict[str, int] = {n: 0 for n in scanners}
    per_category: dict[str, dict[str, int]] = {}
    missed: list[str] = []

    for rec in corpus:
        text = rec.get("attack_text", "")
        cat = rec.get("category", "unknown")

        if cat not in per_category:
            per_category[cat] = {"total": 0, "caught": 0}
        per_category[cat]["total"] += 1

        per_scanner_results = _scan_one(text, scanners)

        item_caught = is_caught(per_scanner_results)
        if item_caught:
            caught += 1
            per_category[cat]["caught"] += 1

        for sname, findings in per_scanner_results.items():
            if findings:
                per_scanner_deny[sname] += 1

        if not item_caught and len(missed) < 20:
            missed.append(text[:120])

    # Build per_category with coverage
    per_category_out = {}
    for cat, data in sorted(per_category.items(), key=lambda x: -x[1]["total"]):
        t, c = data["total"], data["caught"]
        per_category_out[cat] = {
            "total": t,
            "caught": c,
            "coverage": c / t if t > 0 else 0.0,
        }

    return {
        "total": total,
        "caught": caught,
        "coverage": caught / total if total > 0 else 0.0,
        "per_scanner": {n: {"deny": per_scanner_deny[n]} for n in scanners},
        "per_category": per_category_out,
        "missed_samples": missed,
    }


def run_scanners_on_benign(benign_texts: list[str], scanners: dict[str, Any]) -> tuple[int, int]:
    """Run scanners on benign texts, return (false_positives, total)."""
    fps = 0
    for text in benign_texts:
        per_scanner = _scan_one(text, scanners)
        if is_caught(per_scanner):
            fps += 1
    return fps, len(benign_texts)
