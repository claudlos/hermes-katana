"""Optional scanner imports and shared degraded-coverage metadata helpers."""

from __future__ import annotations

from importlib import import_module
import logging
from typing import Any

logger = logging.getLogger(__name__)

OPTIONAL_IMPORT_ERRORS: dict[str, str] = {}
RUNTIME_FAILURES_LOGGED: set[str] = set()


def _mark_optional_import_failure(name: str, exc: ImportError) -> None:
    """Record an optional scanner import failure without hiding the reason."""
    OPTIONAL_IMPORT_ERRORS[name] = f"{exc.__class__.__name__}: {exc}"
    logger.info("Optional scanner '%s' unavailable: %s", name, exc)


def attach_optional_import_metadata(result: Any) -> None:
    """Expose unavailable scanner modules in result metadata."""
    if OPTIONAL_IMPORT_ERRORS:
        result.metadata.setdefault("unavailable_scanners", dict(OPTIONAL_IMPORT_ERRORS))


def record_scanner_failure(result: Any, scanner_name: str, exc: Exception) -> None:
    """Record a runtime scanner failure without silently swallowing it."""
    failure = {"scanner": scanner_name, "error": f"{exc.__class__.__name__}: {exc}"}
    result.metadata.setdefault("scanner_failures", []).append(failure)
    if scanner_name not in RUNTIME_FAILURES_LOGGED:
        logger.warning(
            "Scanner '%s' failed during runtime; continuing with degraded coverage",
            scanner_name,
            exc_info=True,
        )
        RUNTIME_FAILURES_LOGGED.add(scanner_name)


def _load_optional(module_name: str, *symbols: str) -> dict[str, Any]:
    """Import optional scanner symbols, returning None placeholders on ImportError."""
    try:
        module = import_module(f".{module_name}", __package__)
    except ImportError as exc:
        _mark_optional_import_failure(module_name, exc)
        return {symbol: None for symbol in symbols}

    return {symbol: getattr(module, symbol) for symbol in symbols}


_decoder = _load_optional("decoder", "decode_and_scan")
decode_and_scan = _decoder["decode_and_scan"]

_content_harm = _load_optional("content_harm", "scan_content_harm")
scan_content_harm = _content_harm["scan_content_harm"]

_prompt_leak = _load_optional("prompt_leak", "detect_prompt_leak")
detect_prompt_leak = _prompt_leak["detect_prompt_leak"]

_compositional = _load_optional("compositional", "detect_compositional")
detect_compositional = _compositional["detect_compositional"]

_multilingual = _load_optional("multilingual", "detect_multilingual")
detect_multilingual = _multilingual["detect_multilingual"]

_ascii_art = _load_optional("ascii_art", "detect_ascii_art")
detect_ascii_art = _ascii_art["detect_ascii_art"]

_aho_scanner = _load_optional("aho_scanner", "AhoFinding", "detect_aho", "phrase_count")
AhoFinding = _aho_scanner["AhoFinding"]
detect_aho = _aho_scanner["detect_aho"]
phrase_count = _aho_scanner["phrase_count"]

_fast_patterns = _load_optional("fast_patterns", "FastPatternCategory", "FastPatternFinding", "detect_fast_patterns")
FastPatternCategory = _fast_patterns["FastPatternCategory"]
FastPatternFinding = _fast_patterns["FastPatternFinding"]
detect_fast_patterns = _fast_patterns["detect_fast_patterns"]

_persona_detector = _load_optional(
    "persona_detector",
    "PersonaFinding",
    "PersonaSeverity",
    "detect_persona_jailbreak",
    "persona_risk_score",
)
PersonaFinding = _persona_detector["PersonaFinding"]
PersonaSeverity = _persona_detector["PersonaSeverity"]
detect_persona_jailbreak = _persona_detector["detect_persona_jailbreak"]
persona_risk_score = _persona_detector["persona_risk_score"]

_semantic_recall = _load_optional(
    "semantic_recall",
    "SemanticCategory",
    "SemanticFinding",
    "SemanticSeverity",
    "detect_semantic",
    "scan_semantic",
)
SemanticCategory = _semantic_recall["SemanticCategory"]
SemanticFinding = _semantic_recall["SemanticFinding"]
SemanticSeverity = _semantic_recall["SemanticSeverity"]
detect_semantic = _semantic_recall["detect_semantic"]
scan_semantic = _semantic_recall["scan_semantic"]

_behavioral = _load_optional(
    "behavioral",
    "BehavioralCategory",
    "BehavioralFinding",
    "BehavioralSeverity",
    "detect_behavioral",
    "behavioral_risk_score",
)
BehavioralCategory = _behavioral["BehavioralCategory"]
BehavioralFinding = _behavioral["BehavioralFinding"]
BehavioralSeverity = _behavioral["BehavioralSeverity"]
detect_behavioral = _behavioral["detect_behavioral"]
behavioral_risk_score = _behavioral["behavioral_risk_score"]

_svg_sanitizer = _load_optional("svg_sanitizer", "SVGSanitizerFinding", "SVGSanitizerSeverity", "scan_svg")
SVGSanitizerFinding = _svg_sanitizer["SVGSanitizerFinding"]
SVGSanitizerSeverity = _svg_sanitizer["SVGSanitizerSeverity"]
scan_svg = _svg_sanitizer["scan_svg"]

_unicode_spoof = _load_optional("unicode_spoof", "SpoofFinding", "SpoofSeverity", "scan_unicode_spoof")
SpoofFinding = _unicode_spoof["SpoofFinding"]
SpoofSeverity = _unicode_spoof["SpoofSeverity"]
scan_unicode_spoof = _unicode_spoof["scan_unicode_spoof"]

_pdf_js = _load_optional("pdf_js_scanner", "PDFJSFinding", "PDFJSSeverity", "detect_pdf_js")
PDFJSFinding = _pdf_js["PDFJSFinding"]
PDFJSSeverity = _pdf_js["PDFJSSeverity"]
detect_pdf_js = _pdf_js["detect_pdf_js"]

_multimodal = _load_optional("multimodal", "MultimodalFinding", "MultimodalSeverity", "scan_data_uri")
MultimodalFinding = _multimodal["MultimodalFinding"]
MultimodalSeverity = _multimodal["MultimodalSeverity"]
scan_data_uri = _multimodal["scan_data_uri"]

_deberta = _load_optional(
    "deberta_classifier",
    "DeBERTaCategory",
    "DeBERTaFinding",
    "DeBERTaSeverity",
    "detect_deberta",
    "classify_deberta",
)
DeBERTaCategory = _deberta["DeBERTaCategory"]
DeBERTaFinding = _deberta["DeBERTaFinding"]
DeBERTaSeverity = _deberta["DeBERTaSeverity"]
detect_deberta = _deberta["detect_deberta"]
classify_deberta = _deberta["classify_deberta"]
