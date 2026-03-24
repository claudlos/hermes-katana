"""
Content scanner for HermesKatana (improved Tirith from hermes-aegis).

Scans LLM-generated content for:
- Homograph/confusable URL detection (30+ confusable chars)
- Code injection patterns in LLM responses
- Terminal escape / ANSI injection detection
- NEW: Markdown injection (image tags that exfiltrate data, link manipulation)
- NEW: HTML/SVG injection patterns
- Phishing and social engineering indicators

Design goals:
- Catch content-level attacks that bypass prompt-level detection
- Detect exfiltration attempts embedded in responses
- Flag unsafe content that could compromise the user's terminal or browser

Performance: <1ms for typical LLM responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ContentCategory(str, Enum):
    """Categories of content-level attacks."""

    HOMOGRAPH_URL = "homograph_url"
    """URL using confusable characters to impersonate a legitimate domain."""

    CODE_INJECTION = "code_injection"
    """Executable code patterns that could be dangerous if run."""

    TERMINAL_ESCAPE = "terminal_escape"
    """ANSI escape sequences that can manipulate terminal display."""

    MARKDOWN_INJECTION = "markdown_injection"
    """Markdown that exfiltrates data or manipulates display."""

    HTML_INJECTION = "html_injection"
    """HTML/SVG injection that could execute scripts or exfiltrate data."""

    PHISHING = "phishing"
    """Social engineering / phishing content patterns."""

    DATA_EXFILTRATION = "data_exfiltration"
    """Patterns that attempt to send data to external endpoints."""


class ContentSeverity(str, Enum):
    """Severity levels for content findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ContentFinding:
    """A content scanning finding.

    Attributes:
        pattern_name: Name of the pattern that matched.
        category: Type of content attack.
        severity: How dangerous this finding is.
        matched_text: The text that triggered detection.
        position: (start, end) character positions.
        description: Human-readable explanation.
        recommendation: Suggested action.
        confidence: Detection confidence 0.0-1.0.
    """

    pattern_name: str
    category: ContentCategory
    severity: ContentSeverity
    matched_text: str
    position: tuple[int, int]
    description: str
    recommendation: str = "Review and sanitize this content"
    confidence: float = 0.90


# ---------------------------------------------------------------------------
# Homograph / Confusable URL detection
# Expanded character map covering 30+ confusable characters
# ---------------------------------------------------------------------------

# Characters that look like ASCII but aren't (for URL spoofing)
_URL_CONFUSABLES: dict[str, str] = {
    # Cyrillic
    "\u0430": "a", "\u0435": "e", "\u043E": "o", "\u0440": "p",
    "\u0441": "c", "\u0443": "y", "\u0445": "x", "\u0456": "i",
    "\u0458": "j", "\u0455": "s", "\u04BB": "h", "\u04CF": "l",
    "\u051B": "q", "\u051D": "w",
    "\u0410": "A", "\u0412": "B", "\u0421": "C", "\u0415": "E",
    "\u041D": "H", "\u041A": "K", "\u041C": "M", "\u041E": "O",
    "\u0420": "P", "\u0422": "T", "\u0425": "X",
    # Greek
    "\u03BF": "o", "\u03B1": "a",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H",
    "\u0399": "I", "\u039A": "K", "\u039C": "M", "\u039D": "N",
    "\u039F": "O", "\u03A1": "P", "\u03A4": "T", "\u03A5": "Y",
    "\u03A7": "X",
    # Lookalike punctuation
    "\u2024": ".",  # ONE DOT LEADER
    "\u2219": ".",  # BULLET OPERATOR
    "\uFF0E": ".",  # FULLWIDTH FULL STOP
    "\u2215": "/",  # DIVISION SLASH
    "\u2044": "/",  # FRACTION SLASH
    "\uFF0F": "/",  # FULLWIDTH SOLIDUS
    "\u2236": ":",  # RATIO
    "\uFF1A": ":",  # FULLWIDTH COLON
    "\uFF0D": "-",  # FULLWIDTH HYPHEN
    "\u2010": "-",  # HYPHEN
    "\u2011": "-",  # NON-BREAKING HYPHEN
    "\u2212": "-",  # MINUS SIGN
}

_CONFUSABLE_SET = frozenset(_URL_CONFUSABLES.keys())

# URL pattern that may contain confusable characters
_URL_PATTERN = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"\')]{4,}",
    re.IGNORECASE,
)


def _check_homograph_urls(text: str) -> list[ContentFinding]:
    """Detect URLs containing confusable/homograph characters.

    These URLs look like legitimate domains but use characters from
    other scripts (Cyrillic, Greek) that visually resemble ASCII.
    Example: gооgle.com with Cyrillic 'о' instead of Latin 'o'.
    """
    findings: list[ContentFinding] = []

    for match in _URL_PATTERN.finditer(text):
        url = match.group()
        confusables_found: list[tuple[int, str, str]] = []

        for i, ch in enumerate(url):
            if ch in _CONFUSABLE_SET:
                confusables_found.append((i, ch, _URL_CONFUSABLES[ch]))

        if confusables_found:
            # Build the "real" URL
            real_chars = list(url)
            for i, _, replacement in confusables_found:
                real_chars[i] = replacement
            real_url = "".join(real_chars)

            findings.append(ContentFinding(
                pattern_name="homograph_url",
                category=ContentCategory.HOMOGRAPH_URL,
                severity=ContentSeverity.CRITICAL,
                matched_text=url,
                position=(match.start(), match.end()),
                description=(
                    f"URL contains {len(confusables_found)} confusable character(s) "
                    f"that make it look like '{real_url}'. "
                    f"Confusable chars at positions: "
                    f"{', '.join(f'{pos}: {repr(ch)}→{repl}' for pos, ch, repl in confusables_found)}"
                ),
                recommendation="Replace confusable characters with ASCII equivalents or reject URL",
                confidence=0.95,
            ))

    return findings


# ---------------------------------------------------------------------------
# Terminal escape / ANSI injection detection
# ---------------------------------------------------------------------------

_ANSI_PATTERNS: list[tuple[str, re.Pattern, str, ContentSeverity]] = [
    (
        "ansi_escape_csi",
        re.compile(r"\x1b\[[0-9;]*[A-Za-z]"),
        "ANSI CSI escape sequence - can manipulate terminal display, "
        "move cursor, clear screen, or change colors to hide text.",
        ContentSeverity.HIGH,
    ),
    (
        "ansi_escape_osc",
        re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"),
        "ANSI OSC escape sequence - can change terminal title, "
        "set clipboard content, or open URLs.",
        ContentSeverity.CRITICAL,
    ),
    (
        "ansi_escape_dcs",
        re.compile(r"\x1bP[^\x1b]*\x1b\\"),
        "ANSI DCS escape sequence - Device Control String, can send "
        "commands to the terminal emulator.",
        ContentSeverity.CRITICAL,
    ),
    (
        "raw_escape_byte",
        re.compile(r"\x1b[^[\]P]"),
        "Raw escape byte followed by unexpected character - "
        "potential terminal injection.",
        ContentSeverity.MEDIUM,
    ),
    (
        "carriage_return_overwrite",
        re.compile(r"\r[^\n]"),
        "Carriage return without newline - can overwrite displayed text "
        "to show different content than what's actually present.",
        ContentSeverity.HIGH,
    ),
    (
        "terminal_hyperlink",
        re.compile(r"\x1b\]8;;[^\x07\x1b]*(?:\x07|\x1b\\)"),
        "Terminal hyperlink escape - can make text clickable to "
        "arbitrary URLs. May be used for phishing.",
        ContentSeverity.HIGH,
    ),
]


def _check_terminal_escapes(text: str) -> list[ContentFinding]:
    """Detect ANSI/terminal escape sequence injection.

    LLM responses containing ANSI escapes can:
    - Overwrite displayed text to show false information
    - Set clipboard content to malicious commands
    - Open URLs in the user's browser
    - Change terminal title as social engineering
    """
    findings: list[ContentFinding] = []

    for name, pattern, description, severity in _ANSI_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(ContentFinding(
                pattern_name=name,
                category=ContentCategory.TERMINAL_ESCAPE,
                severity=severity,
                matched_text=repr(match.group()),
                position=(match.start(), match.end()),
                description=description,
                recommendation="Strip all ANSI escape sequences from output",
            ))

    return findings


# ---------------------------------------------------------------------------
# Code injection in LLM responses
# ---------------------------------------------------------------------------

_CODE_INJECTION_PATTERNS: list[tuple[str, re.Pattern, str, ContentSeverity]] = [
    (
        "shell_command_injection",
        re.compile(
            r"(?:^|\n)\s*(?:\$|#|>>>?)\s*(?:sudo|rm|chmod|curl\s+.*\|\s*(?:ba)?sh|wget\s+.*\|\s*sh|eval|exec)",
            re.MULTILINE,
        ),
        "Shell command with dangerous operations in LLM response. "
        "If auto-executed, this could compromise the system.",
        ContentSeverity.HIGH,
    ),
    (
        "python_exec_eval",
        re.compile(
            r"\b(?:exec|eval|compile)\s*\(\s*(?:input|request|os\.environ|__import__)",
            re.IGNORECASE,
        ),
        "Python exec/eval with dynamic input - arbitrary code execution.",
        ContentSeverity.HIGH,
    ),
    (
        "os_system_call",
        re.compile(
            r"\b(?:os\.system|os\.popen|subprocess\.(?:call|run|Popen)|commands\.getoutput)\s*\(",
            re.IGNORECASE,
        ),
        "System command execution in code - may be dangerous if the "
        "command includes user-controlled input.",
        ContentSeverity.MEDIUM,
    ),
    (
        "import_dangerous",
        re.compile(
            r"\b__import__\s*\(\s*['\"](?:os|subprocess|shutil|ctypes|socket)['\"]",
        ),
        "Dynamic import of dangerous module via __import__.",
        ContentSeverity.HIGH,
    ),
    (
        "pickle_deserialize",
        re.compile(r"\bpickle\.(?:loads?|Unpickler)\s*\("),
        "Pickle deserialization - can execute arbitrary code on untrusted data.",
        ContentSeverity.HIGH,
    ),
]


def _check_code_injection(text: str) -> list[ContentFinding]:
    """Detect dangerous code patterns in LLM responses.

    When LLM output is auto-executed (e.g., in coding assistants),
    injected code patterns can compromise the system.
    """
    findings: list[ContentFinding] = []

    for name, pattern, description, severity in _CODE_INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(ContentFinding(
                pattern_name=name,
                category=ContentCategory.CODE_INJECTION,
                severity=severity,
                matched_text=match.group(),
                position=(match.start(), match.end()),
                description=description,
                recommendation="Review code before execution. Do not auto-execute LLM output.",
            ))

    return findings


# ---------------------------------------------------------------------------
# Markdown injection
# ---------------------------------------------------------------------------

_MARKDOWN_PATTERNS: list[tuple[str, re.Pattern, str, ContentSeverity]] = [
    (
        "markdown_image_exfil",
        re.compile(
            r"!\[(?:[^\]]*)\]\(\s*https?://[^\s)]+(?:\?[^\s)]*(?:data|token|secret|key|password|session|cookie|auth)[^\s)]*)\s*\)",
            re.IGNORECASE,
        ),
        "Markdown image tag with suspicious query parameters - "
        "can exfiltrate data via image URL when rendered. "
        "The image loads automatically, sending data to the attacker's server.",
        ContentSeverity.CRITICAL,
    ),
    (
        "markdown_image_tracking",
        re.compile(
            r"!\[(?:[^\]]*)\]\(\s*https?://(?:(?!(?:github|gitlab|imgur|i\.stack))[^\s/)])+[^\s)]+\.(gif|png|jpg|jpeg|webp|svg)\?[^\s)]+\)",
            re.IGNORECASE,
        ),
        "Markdown image with query parameters from unknown domain - "
        "potential tracking pixel or data exfiltration.",
        ContentSeverity.HIGH,
    ),
    (
        "markdown_link_disguise",
        re.compile(
            r"\[(?:click\s+here|download|login|verify|confirm|update|secure)[^\]]*\]\(\s*https?://[^\s)]+\)",
            re.IGNORECASE,
        ),
        "Markdown link with social engineering text (click here, login, etc.) - "
        "potential phishing attempt.",
        ContentSeverity.MEDIUM,
    ),
    (
        "markdown_link_mismatch",
        re.compile(
            r"\[(?:https?://[^\]]+)\]\(\s*(https?://[^\s)]+)\)",
            re.IGNORECASE,
        ),
        "Markdown link where display URL differs from actual URL - "
        "classic phishing technique to disguise malicious links.",
        ContentSeverity.HIGH,
    ),
    (
        "markdown_html_injection",
        re.compile(
            r"(?:<script|<iframe|<object|<embed|<form|<input|<link\s+.*rel\s*=|<meta\s+.*http-equiv)",
            re.IGNORECASE,
        ),
        "HTML tags in markdown that can execute scripts, embed content, "
        "or manipulate the page when rendered.",
        ContentSeverity.CRITICAL,
    ),
    (
        "markdown_data_uri",
        re.compile(
            r"!\[[^\]]*\]\(\s*data:(?:text/html|application/javascript|text/javascript)[^)]+\)",
            re.IGNORECASE,
        ),
        "Markdown image with data: URI containing executable content. "
        "Can execute JavaScript when rendered.",
        ContentSeverity.CRITICAL,
    ),
    (
        "markdown_reference_hijack",
        re.compile(
            r"(?:^|\n)\s*\[[^\]]+\]:\s*(?:javascript:|data:text/html|vbscript:)",
            re.IGNORECASE | re.MULTILINE,
        ),
        "Markdown reference definition with dangerous protocol. "
        "Can inject JavaScript via reference-style links.",
        ContentSeverity.CRITICAL,
    ),
]


def _check_markdown_injection(text: str) -> list[ContentFinding]:
    """Detect markdown injection attacks.

    Markdown can be weaponized to:
    - Exfiltrate data via auto-loading image tags
    - Phish users with disguised links
    - Inject HTML/JavaScript when rendered
    - Track users with pixel tracking
    """
    findings: list[ContentFinding] = []

    for name, pattern, description, severity in _MARKDOWN_PATTERNS:
        for match in pattern.finditer(text):
            # For link mismatch, check if URLs actually differ
            if name == "markdown_link_mismatch":
                display_url = match.group().split("](")[0].lstrip("[")
                actual_url = match.group(1)
                if display_url.rstrip("/") == actual_url.rstrip("/"):
                    continue  # URLs match, not suspicious

            findings.append(ContentFinding(
                pattern_name=name,
                category=ContentCategory.MARKDOWN_INJECTION,
                severity=severity,
                matched_text=match.group()[:100] + ("..." if len(match.group()) > 100 else ""),
                position=(match.start(), match.end()),
                description=description,
                recommendation="Strip or sanitize markdown before rendering",
            ))

    return findings


# ---------------------------------------------------------------------------
# HTML / SVG injection
# ---------------------------------------------------------------------------

_HTML_PATTERNS: list[tuple[str, re.Pattern, str, ContentSeverity]] = [
    (
        "script_tag",
        re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
        "Script tag - direct JavaScript execution.",
        ContentSeverity.CRITICAL,
    ),
    (
        "event_handler",
        re.compile(
            r"<[^>]+\s+on(?:load|error|click|mouseover|focus|blur|submit|change|input|keyup|keydown|mouseenter|abort|animationend)\s*=",
            re.IGNORECASE,
        ),
        "HTML event handler attribute - JavaScript execution on user interaction.",
        ContentSeverity.CRITICAL,
    ),
    (
        "javascript_protocol",
        re.compile(
            r"(?:href|src|action|formaction|data|poster|background)\s*=\s*['\"]?\s*javascript:",
            re.IGNORECASE,
        ),
        "JavaScript protocol in URL attribute - code execution on navigation.",
        ContentSeverity.CRITICAL,
    ),
    (
        "svg_injection",
        re.compile(
            r"<svg[^>]*>.*?(?:<script|on\w+\s*=|<foreignObject|<use\s+.*href|animate.*attributeName)",
            re.IGNORECASE | re.DOTALL,
        ),
        "SVG with embedded script or animation - can execute JavaScript.",
        ContentSeverity.CRITICAL,
    ),
    (
        "iframe_injection",
        re.compile(r"<iframe[^>]*(?:src|srcdoc)\s*=", re.IGNORECASE),
        "Iframe injection - can load arbitrary external content.",
        ContentSeverity.HIGH,
    ),
    (
        "object_embed",
        re.compile(r"<(?:object|embed|applet)[^>]*(?:data|src|code)\s*=", re.IGNORECASE),
        "Object/embed/applet tag - can load and execute external content.",
        ContentSeverity.HIGH,
    ),
    (
        "form_injection",
        re.compile(r"<form[^>]*action\s*=\s*['\"]?https?://", re.IGNORECASE),
        "Form injection with external action URL - credential theft.",
        ContentSeverity.CRITICAL,
    ),
    (
        "meta_redirect",
        re.compile(r"<meta[^>]*http-equiv\s*=\s*['\"]?refresh['\"]?[^>]*url\s*=", re.IGNORECASE),
        "Meta refresh redirect - can redirect user to malicious site.",
        ContentSeverity.HIGH,
    ),
    (
        "base_tag",
        re.compile(r"<base[^>]*href\s*=", re.IGNORECASE),
        "Base tag injection - can redirect all relative URLs.",
        ContentSeverity.HIGH,
    ),
    (
        "style_injection",
        re.compile(
            r"<style[^>]*>.*?(?:expression\s*\(|behavior\s*:|url\s*\(['\"]?(?:javascript|data):)",
            re.IGNORECASE | re.DOTALL,
        ),
        "CSS injection with code execution (expression/behavior/javascript URI).",
        ContentSeverity.HIGH,
    ),
    (
        "css_import_exfil",
        re.compile(
            r"(?:@import\s+['\"]?https?://|background(?:-image)?\s*:\s*url\s*\(['\"]?https?://[^)]*\?)",
            re.IGNORECASE,
        ),
        "CSS import or background URL with query params - data exfiltration via CSS.",
        ContentSeverity.HIGH,
    ),
]


def _check_html_injection(text: str) -> list[ContentFinding]:
    """Detect HTML/SVG/CSS injection patterns.

    HTML injection in LLM responses can:
    - Execute JavaScript (XSS)
    - Redirect users to malicious sites
    - Steal credentials via form injection
    - Exfiltrate data via CSS
    """
    findings: list[ContentFinding] = []

    for name, pattern, description, severity in _HTML_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(ContentFinding(
                pattern_name=name,
                category=ContentCategory.HTML_INJECTION,
                severity=severity,
                matched_text=match.group()[:100] + ("..." if len(match.group()) > 100 else ""),
                position=(match.start(), match.end()),
                description=description,
                recommendation="Strip or escape HTML before rendering",
            ))

    return findings


# ---------------------------------------------------------------------------
# Data exfiltration patterns
# ---------------------------------------------------------------------------

_EXFIL_PATTERNS: list[tuple[str, re.Pattern, str, ContentSeverity]] = [
    (
        "webhook_exfil",
        re.compile(
            r"https?://(?:(?:hooks\.slack\.com|discord(?:app)?\.com/api/webhooks|webhook\.site|requestbin|pipedream|hookbin|beeceptor)[^\s<>\"']*)",
            re.IGNORECASE,
        ),
        "Webhook URL detected - commonly used for data exfiltration. "
        "Data sent to these endpoints can be captured by attackers.",
        ContentSeverity.HIGH,
    ),
    (
        "dns_exfil_pattern",
        re.compile(
            r"(?:dig|nslookup|host)\s+[A-Za-z0-9+/=-]+\.(?:[a-z0-9-]+\.)+[a-z]{2,}",
            re.IGNORECASE,
        ),
        "DNS query with encoded subdomain - DNS-based data exfiltration. "
        "Data is encoded in the subdomain of a DNS query.",
        ContentSeverity.HIGH,
    ),
    (
        "fetch_with_data",
        re.compile(
            r"(?:fetch|axios|requests?\.(?:get|post)|http\.(?:get|post)|curl|wget)\s*\([^)]*(?:\+|concat|encodeURI|btoa|JSON\.stringify)",
            re.IGNORECASE,
        ),
        "HTTP request with dynamic data construction - "
        "may exfiltrate sensitive information.",
        ContentSeverity.HIGH,
    ),
]


def _check_data_exfiltration(text: str) -> list[ContentFinding]:
    """Detect data exfiltration patterns in content."""
    findings: list[ContentFinding] = []

    for name, pattern, description, severity in _EXFIL_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(ContentFinding(
                pattern_name=name,
                category=ContentCategory.DATA_EXFILTRATION,
                severity=severity,
                matched_text=match.group()[:100] + ("..." if len(match.group()) > 100 else ""),
                position=(match.start(), match.end()),
                description=description,
                recommendation="Block external data transmission",
            ))

    return findings


# ---------------------------------------------------------------------------
# Phishing / social engineering patterns
# ---------------------------------------------------------------------------

_PHISHING_PATTERNS: list[tuple[str, re.Pattern, str, ContentSeverity]] = [
    (
        "urgency_credential",
        re.compile(
            r"(?:urgent|immediately|right\s+now|within\s+\d+\s+(?:hour|minute)|account\s+(?:will\s+be\s+)?(?:suspended|locked|terminated))"
            r"[^.]*(?:password|credential|login|sign\s+in|verify|confirm|update\s+(?:your|account))",
            re.IGNORECASE,
        ),
        "Urgency language combined with credential request - "
        "classic phishing technique.",
        ContentSeverity.HIGH,
    ),
    (
        "fake_auth_prompt",
        re.compile(
            r"(?:enter|provide|type|input|submit)\s+(?:your\s+)?(?:password|api\s+key|secret\s+key|access\s+token|credentials?|ssh\s+key|private\s+key)\s+(?:here|below|in\s+the\s+(?:box|field|form))",
            re.IGNORECASE,
        ),
        "Request to enter credentials in LLM conversation - "
        "credentials should never be shared in chat.",
        ContentSeverity.HIGH,
    ),
]


def _check_phishing(text: str) -> list[ContentFinding]:
    """Detect phishing and social engineering patterns."""
    findings: list[ContentFinding] = []

    for name, pattern, description, severity in _PHISHING_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(ContentFinding(
                pattern_name=name,
                category=ContentCategory.PHISHING,
                severity=severity,
                matched_text=match.group()[:100] + ("..." if len(match.group()) > 100 else ""),
                position=(match.start(), match.end()),
                description=description,
                recommendation="Do not follow credential requests in LLM responses",
            ))

    return findings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def scan_content(text: str) -> list[ContentFinding]:
    """Scan content for various attack patterns.

    Comprehensive content scanning including:
    - Homograph URL detection (confusable characters in URLs)
    - Code injection patterns in LLM responses
    - Terminal/ANSI escape injection
    - Markdown injection (data exfiltration, phishing)
    - HTML/SVG injection
    - Data exfiltration patterns
    - Phishing/social engineering

    Args:
        text: The content text to scan.

    Returns:
        List of ContentFinding objects, sorted by severity.

    Performance:
        <1ms for typical LLM responses.
        All patterns precompiled at module load time.

    Example:
        >>> findings = scan_content("Visit https://gооgle.com")  # Cyrillic 'о'
        >>> any(f.category == ContentCategory.HOMOGRAPH_URL for f in findings)
        True
    """
    if not text:
        return []

    findings: list[ContentFinding] = []

    findings.extend(_check_homograph_urls(text))
    findings.extend(_check_terminal_escapes(text))
    findings.extend(_check_code_injection(text))
    findings.extend(_check_markdown_injection(text))
    findings.extend(_check_html_injection(text))
    findings.extend(_check_data_exfiltration(text))
    findings.extend(_check_phishing(text))

    # Sort by severity
    severity_order = {
        ContentSeverity.CRITICAL: 0,
        ContentSeverity.HIGH: 1,
        ContentSeverity.MEDIUM: 2,
        ContentSeverity.LOW: 3,
        ContentSeverity.INFO: 4,
    }
    findings.sort(key=lambda f: severity_order.get(f.severity, 99))

    return findings


def content_risk_score(text: str) -> float:
    """Quick aggregate content risk score.

    Returns float from 0.0 (safe) to 1.0 (dangerous).
    """
    findings = scan_content(text)
    if not findings:
        return 0.0

    severity_scores = {
        ContentSeverity.CRITICAL: 0.9,
        ContentSeverity.HIGH: 0.7,
        ContentSeverity.MEDIUM: 0.4,
        ContentSeverity.LOW: 0.2,
        ContentSeverity.INFO: 0.1,
    }

    max_score = max(severity_scores.get(f.severity, 0.1) for f in findings)
    boost = min(len(findings) * 0.03, 0.1)

    return min(max_score + boost, 1.0)
