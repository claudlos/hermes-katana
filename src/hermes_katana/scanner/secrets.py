"""
Secret/credential scanner for HermesKatana.

Improved from hermes-aegis with new capabilities:
- 15+ regex patterns for known API key formats
- Multi-encoding detection (base64, hex, URL, reversed, ROT13, split)
- Exact value matching against vault contents
- NEW: Chunked/split secret detection across multiple strings
- NEW: Entropy-based detection for unknown secret formats (Shannon entropy)

Design goals:
- High recall: catch real secrets even when obfuscated
- Low false positives: specific patterns with length/format validation
- Fast: precompiled patterns, O(n) scanning per pattern

Performance: <1ms for typical text inputs (<10KB).
"""

from __future__ import annotations

import base64
import codecs
import math
import re
import urllib.parse
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SecretCategory(str, Enum):
    """Categories of detected secrets."""

    API_KEY = "api_key"
    """Known API key format (AWS, GitHub, OpenAI, etc.)."""

    TOKEN = "token"
    """Authentication token (JWT, OAuth, bearer)."""

    PASSWORD = "password"
    """Password or passphrase in plaintext."""

    PRIVATE_KEY = "private_key"
    """Cryptographic private key material."""

    CONNECTION_STRING = "connection_string"
    """Database or service connection string with credentials."""

    VAULT_MATCH = "vault_match"
    """Exact match against known vault values."""

    HIGH_ENTROPY = "high_entropy"
    """High-entropy string that may be an unknown secret format."""

    ENCODED_SECRET = "encoded_secret"
    """Secret detected in encoded form (base64, hex, ROT13, etc.)."""


class SecretSeverity(str, Enum):
    """Severity levels for secret findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """A single detected secret/credential.

    Attributes:
        pattern_name: Name of the pattern that detected this secret.
        category: Type of secret detected.
        severity: How dangerous the exposure is.
        matched_text: The detected secret (masked for safety).
        position: (start, end) character positions in the input.
        description: Human-readable explanation.
        encoding: How the secret was encoded (None = plaintext).
        confidence: Detection confidence 0.0-1.0.
    """

    pattern_name: str
    category: SecretCategory
    severity: SecretSeverity
    matched_text: str
    position: tuple[int, int]
    description: str
    encoding: Optional[str] = None
    confidence: float = 1.0


def _mask_secret(text: str, show_chars: int = 4) -> str:
    """Mask a secret value for safe logging/display.

    Shows first and last `show_chars` characters, masks the rest.
    """
    if len(text) <= show_chars * 2 + 2:
        return "*" * len(text)
    return text[:show_chars] + "*" * (len(text) - show_chars * 2) + text[-show_chars:]


# ---------------------------------------------------------------------------
# Known API key / token patterns (15+)
# Each: (name, pattern, category, severity, description)
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern, SecretCategory, SecretSeverity, str]] = []


def _sp(
    name: str,
    pattern: str,
    category: SecretCategory,
    severity: SecretSeverity,
    description: str,
    flags: int = 0,
) -> None:
    """Register a secret pattern."""
    _SECRET_PATTERNS.append((
        name,
        re.compile(pattern, flags),
        category,
        severity,
        description,
    ))


# AWS Access Key ID (starts with AKIA, 20 chars)
_sp(
    "aws_access_key",
    r"(?:^|[^A-Za-z0-9])(AKIA[0-9A-Z]{16})(?:[^A-Za-z0-9]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.CRITICAL,
    "AWS Access Key ID - provides access to AWS services.",
)

# AWS Secret Access Key (40 chars base64-ish)
_sp(
    "aws_secret_key",
    r"(?:aws[_\-]?secret[_\-]?(?:access)?[_\-]?key|secret[_\-]?key)\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
    SecretCategory.API_KEY,
    SecretSeverity.CRITICAL,
    "AWS Secret Access Key - full access to AWS account.",
    re.IGNORECASE,
)

# GitHub Personal Access Token (ghp_, gho_, ghu_, ghs_, ghr_)
_sp(
    "github_token",
    r"(?:^|[^A-Za-z0-9_])(gh[pousr]_[A-Za-z0-9_]{36,255})(?:[^A-Za-z0-9_]|$)",
    SecretCategory.TOKEN,
    SecretSeverity.CRITICAL,
    "GitHub Personal Access Token - access to GitHub repositories.",
)

# GitHub fine-grained token
_sp(
    "github_fine_grained",
    r"(?:^|[^A-Za-z0-9_])(github_pat_[A-Za-z0-9_]{22,255})(?:[^A-Za-z0-9_]|$)",
    SecretCategory.TOKEN,
    SecretSeverity.CRITICAL,
    "GitHub Fine-Grained Personal Access Token.",
)

# OpenAI API Key
_sp(
    "openai_key",
    r"(?:^|[^A-Za-z0-9_-])(sk-[A-Za-z0-9]{20,}(?:-[A-Za-z0-9]+)*)(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.CRITICAL,
    "OpenAI API Key - access to GPT/DALL-E services.",
)

# Anthropic API Key
_sp(
    "anthropic_key",
    r"(?:^|[^A-Za-z0-9_-])(sk-ant-[A-Za-z0-9\-]{20,})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.CRITICAL,
    "Anthropic API Key - access to Claude services.",
)

# Google Cloud API Key
_sp(
    "google_api_key",
    r"(?:^|[^A-Za-z0-9_-])(AIza[A-Za-z0-9\-_]{35})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.HIGH,
    "Google Cloud API Key.",
)

# Slack Bot Token
_sp(
    "slack_bot_token",
    r"(?:^|[^A-Za-z0-9_-])(xoxb-[0-9]{10,}-[0-9]{10,}-[A-Za-z0-9]{24,})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.TOKEN,
    SecretSeverity.HIGH,
    "Slack Bot Token - access to Slack workspace.",
)

# Slack User Token
_sp(
    "slack_user_token",
    r"(?:^|[^A-Za-z0-9_-])(xoxp-[0-9]{10,}-[0-9]{10,}-[A-Za-z0-9]{24,})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.TOKEN,
    SecretSeverity.HIGH,
    "Slack User Token - access to Slack as a user.",
)

# Stripe API Key (live or test)
_sp(
    "stripe_key",
    r"(?:^|[^A-Za-z0-9_-])(sk_live_[A-Za-z0-9]{24,})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.CRITICAL,
    "Stripe API Key - access to payment processing.",
)

# Twilio API Key
_sp(
    "twilio_key",
    r"(?:^|[^A-Za-z0-9])(SK[0-9a-fA-F]{32})(?:[^A-Za-z0-9]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.HIGH,
    "Twilio API Key - access to communications APIs.",
)

# SendGrid API Key
_sp(
    "sendgrid_key",
    r"(?:^|[^A-Za-z0-9._-])(SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43})(?:[^A-Za-z0-9._-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.HIGH,
    "SendGrid API Key - access to email services.",
)

# Mailgun API Key
_sp(
    "mailgun_key",
    r"(?:^|[^A-Za-z0-9_-])(key-[A-Za-z0-9]{32})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.HIGH,
    "Mailgun API Key - access to email services.",
)

# JWT Token
_sp(
    "jwt_token",
    r"(?:^|[^A-Za-z0-9._-])(eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})(?:[^A-Za-z0-9._-]|$)",
    SecretCategory.TOKEN,
    SecretSeverity.HIGH,
    "JWT Token - may contain authentication claims.",
)

# Generic Bearer Token
_sp(
    "bearer_token",
    r"(?:bearer|authorization)\s*[=:]\s*['\"]?(?:Bearer\s+)?([A-Za-z0-9\-_.~+/]{20,}={0,2})['\"]?",
    SecretCategory.TOKEN,
    SecretSeverity.HIGH,
    "Bearer/Authorization token.",
    re.IGNORECASE,
)

# Private Key blocks (RSA, EC, DSA, etc.)
_sp(
    "private_key",
    r"-----BEGIN\s+(?:RSA\s+|EC\s+|DSA\s+|OPENSSH\s+)?PRIVATE\s+KEY-----",
    SecretCategory.PRIVATE_KEY,
    SecretSeverity.CRITICAL,
    "Private key material - cryptographic key exposure.",
)

# Database connection strings
_sp(
    "database_url",
    r"(?:mongodb|postgres|postgresql|mysql|redis|amqp|mssql)://(?!localhost[:/]|127\.0\.0\.1[:/]|\[::1\][:/])[^\s'\"]{10,}",
    SecretCategory.CONNECTION_STRING,
    SecretSeverity.CRITICAL,
    "Database connection string with potential credentials.",
    re.IGNORECASE,
)

# Generic password in assignment
_sp(
    "password_assignment",
    r"(?:password|passwd|pwd|pass)\s*[=:]\s*['\"](?!placeholder|test(?:ing)?(?:pass|password|pwd|\d+)|password\d*|changeme|example|dummy|sample|foobar|P[@a]ss(?:w[o0]rd)?[!1]?|default|secret123|abc(?:def)?123|admin123|xxx+|12345678+|\*{3,})['\"]([^'\"\s]{8,})['\"]",
    SecretCategory.PASSWORD,
    SecretSeverity.HIGH,
    "Plaintext password assignment.",
    re.IGNORECASE,
)

# Heroku API Key
_sp(
    "heroku_key",
    r"(?:^|[^A-Za-z0-9_-])((?:heroku[_\-]?api[_\-]?key|HEROKU_API_KEY)\s*[=:]\s*[A-Fa-f0-9\-]{36,})(?:[^A-Za-z0-9_-]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.HIGH,
    "Heroku API Key.",
    re.IGNORECASE,
)

# Azure subscription key
_sp(
    "azure_key",
    r"(?:^|[^A-Za-z0-9])((?:Ocp-Apim-Subscription-Key|azure[_\-]?(?:api[_\-]?)?key)\s*[=:]\s*[A-Za-z0-9]{32,})(?:[^A-Za-z0-9]|$)",
    SecretCategory.API_KEY,
    SecretSeverity.HIGH,
    "Azure Subscription/API Key.",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shannon entropy calculation for unknown secret formats
# ---------------------------------------------------------------------------

def shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string.

    Higher entropy indicates more randomness, which is characteristic
    of secrets, tokens, and keys.

    Returns:
        Entropy in bits per character. Typical thresholds:
        - English text: ~3.5-4.5 bits
        - Hex strings: ~3.5-4.0 bits
        - Random base64: ~5.0-6.0 bits
        - Random alphanumeric: ~5.0-5.9 bits

    A threshold of 4.5 bits catches most secrets while avoiding
    English text false positives.
    """
    if not text:
        return 0.0

    freq: dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1

    length = len(text)
    entropy = 0.0
    for count in freq.values():
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
    return entropy


# Pattern for high-entropy candidate strings
_ENTROPY_CANDIDATE = re.compile(
    r"(?:^|[=:'\"\s])([A-Za-z0-9+/\-_]{16,}={0,3})(?:['\"\s,;]|$)"
)

# Minimum entropy threshold for flagging (bits per character)
_ENTROPY_THRESHOLD = 4.5

# Minimum length for entropy-based detection
_ENTROPY_MIN_LENGTH = 16


# ---------------------------------------------------------------------------
# Multi-encoding detection
# ---------------------------------------------------------------------------

def _decode_rot13(text: str) -> str:
    """Decode ROT13 encoded text."""
    return codecs.decode(text, "rot_13")


def _decode_reverse(text: str) -> str:
    """Reverse a string (to detect reversed secrets)."""
    return text[::-1]


def _try_decode_base64(text: str) -> Optional[str]:
    """Try to decode base64, return None on failure."""
    try:
        # Pad if needed
        padded = text + "=" * (4 - len(text) % 4) if len(text) % 4 else text
        decoded = base64.b64decode(padded).decode("utf-8", errors="strict")
        # Verify it's printable
        if all(c.isprintable() or c.isspace() for c in decoded):
            return decoded
    except Exception:
        pass
    return None


def _try_decode_hex(text: str) -> Optional[str]:
    """Try to decode hex, return None on failure."""
    clean = text.lstrip("0x").lstrip("0X")
    if len(clean) % 2 != 0:
        return None
    try:
        decoded = bytes.fromhex(clean).decode("utf-8", errors="strict")
        if all(c.isprintable() or c.isspace() for c in decoded):
            return decoded
    except Exception:
        pass
    return None


def _try_decode_url(text: str) -> Optional[str]:
    """Try to URL-decode, return None if no change."""
    try:
        decoded = urllib.parse.unquote(text)
        if decoded != text:
            return decoded
    except Exception:
        pass
    return None


def _scan_in_encoding(
    text: str,
    decoded: str,
    encoding_name: str,
    vault_values: Optional[set[str]],
) -> list[SecretFinding]:
    """Scan decoded content against secret patterns and vault values."""
    findings: list[SecretFinding] = []

    # Check against vault values
    if vault_values:
        for vault_val in vault_values:
            if vault_val and vault_val in decoded:
                findings.append(SecretFinding(
                    pattern_name=f"vault_match_{encoding_name}",
                    category=SecretCategory.VAULT_MATCH,
                    severity=SecretSeverity.CRITICAL,
                    matched_text=_mask_secret(vault_val),
                    position=(0, len(text)),
                    description=(
                        f"Vault secret found in {encoding_name}-encoded content."
                    ),
                    encoding=encoding_name,
                    confidence=1.0,
                ))

    # Check against known patterns
    for name, pattern, category, severity, desc in _SECRET_PATTERNS:
        for match in pattern.finditer(decoded):
            secret = match.group(1) if match.lastindex else match.group()
            findings.append(SecretFinding(
                pattern_name=f"{name}_{encoding_name}",
                category=SecretCategory.ENCODED_SECRET,
                severity=severity,
                matched_text=_mask_secret(secret),
                position=(0, len(text)),
                description=f"{desc} (found in {encoding_name}-encoded content)",
                encoding=encoding_name,
                confidence=0.90,
            ))

    return findings


# ---------------------------------------------------------------------------
# Chunked/split secret detection
# ---------------------------------------------------------------------------

def scan_for_secrets_chunked(
    chunks: list[str],
    vault_values: Optional[set[str]] = None,
) -> list[SecretFinding]:
    """Detect secrets that may be split across multiple strings/fields.

    Attackers sometimes split secrets across multiple fields to evade
    detection. This function concatenates chunks in various orders and
    scans the combinations.

    Args:
        chunks: List of text chunks to check individually and combined.
        vault_values: Optional set of known secret values to match against.

    Returns:
        List of SecretFinding objects from all chunks and combinations.

    Example:
        >>> findings = scan_for_secrets_chunked(["AKIA", "1234567890ABCDEF"])
        >>> any(f.pattern_name == "aws_access_key" for f in findings)
        True
    """
    findings: list[SecretFinding] = []

    # Scan each chunk individually
    for i, chunk in enumerate(chunks):
        chunk_findings = scan_for_secrets(chunk, vault_values)
        for f in chunk_findings:
            findings.append(SecretFinding(
                pattern_name=f.pattern_name,
                category=f.category,
                severity=f.severity,
                matched_text=f.matched_text,
                position=f.position,
                description=f"{f.description} (in chunk {i})",
                encoding=f.encoding,
                confidence=f.confidence,
            ))

    # Check concatenations of adjacent chunks (limit combinatorial explosion)
    if len(chunks) <= 10:
        for i in range(len(chunks)):
            for j in range(i + 1, min(i + 4, len(chunks) + 1)):
                combined = "".join(chunks[i:j])
                if combined and len(combined) > 8:
                    combo_findings = scan_for_secrets(combined, vault_values)
                    for f in combo_findings:
                        # Check if this wasn't found in individual chunks
                        is_new = not any(
                            ef.pattern_name == f.pattern_name
                            for ef in findings
                        )
                        if is_new:
                            findings.append(SecretFinding(
                                pattern_name=f.pattern_name,
                                category=f.category,
                                severity=f.severity,
                                matched_text=f.matched_text,
                                position=f.position,
                                description=(
                                    f"{f.description} (split across chunks {i}-{j-1})"
                                ),
                                encoding="chunked",
                                confidence=f.confidence * 0.9,  # Slightly lower confidence
                            ))

    # Check vault values across combined chunks
    if vault_values:
        full_text = "".join(chunks)
        for vault_val in vault_values:
            if vault_val and len(vault_val) >= 4 and vault_val in full_text:
                # Check if it spans chunk boundaries
                cumulative = 0
                for i, chunk in enumerate(chunks):
                    if vault_val in chunk:
                        break  # Found in single chunk, already detected
                    cumulative += len(chunk)
                else:
                    # Not in any single chunk - it's split
                    findings.append(SecretFinding(
                        pattern_name="vault_match_split",
                        category=SecretCategory.VAULT_MATCH,
                        severity=SecretSeverity.CRITICAL,
                        matched_text=_mask_secret(vault_val),
                        position=(0, len(full_text)),
                        description=(
                            "Vault secret found split across multiple chunks. "
                            "This is likely a deliberate evasion attempt."
                        ),
                        encoding="chunked",
                        confidence=1.0,
                    ))

    return findings


# ---------------------------------------------------------------------------
# Main scanning function
# ---------------------------------------------------------------------------

def scan_for_secrets(
    text: str,
    vault_values: Optional[set[str]] = None,
) -> list[SecretFinding]:
    """Scan text for secrets and credentials.

    Comprehensive detection using:
    1. Known API key/token format patterns (15+ services)
    2. Entropy-based detection for unknown formats
    3. Multi-encoding detection (base64, hex, URL, reversed, ROT13)
    4. Exact matching against vault values

    Args:
        text: The text to scan for secrets.
        vault_values: Optional set of known secret values from a vault.
            If provided, exact matches (including encoded) are flagged
            as CRITICAL with confidence 1.0.

    Returns:
        List of SecretFinding objects, sorted by severity then confidence.

    Performance:
        <1ms for typical inputs (<10KB).
        Patterns are precompiled at module load.

    Example:
        >>> findings = scan_for_secrets("My key is AKIAIOSFODNN7EXAMPLE")
        >>> len(findings) >= 1
        True
        >>> findings[0].category
        <SecretCategory.API_KEY: 'api_key'>
    """
    if not text:
        return []

    findings: list[SecretFinding] = []

    # --- 1. Pattern-based detection ---
    for name, pattern, category, severity, description in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            secret = match.group(1) if match.lastindex else match.group()
            findings.append(SecretFinding(
                pattern_name=name,
                category=category,
                severity=severity,
                matched_text=_mask_secret(secret),
                position=(match.start(), match.end()),
                description=description,
                confidence=0.95,
            ))

    # --- 2. Vault value matching ---
    if vault_values:
        for vault_val in vault_values:
            if not vault_val or len(vault_val) < 4:
                continue
            idx = text.find(vault_val)
            if idx >= 0:
                findings.append(SecretFinding(
                    pattern_name="vault_exact_match",
                    category=SecretCategory.VAULT_MATCH,
                    severity=SecretSeverity.CRITICAL,
                    matched_text=_mask_secret(vault_val),
                    position=(idx, idx + len(vault_val)),
                    description=(
                        "Exact match against a known vault secret value. "
                        "This secret is being leaked in plaintext."
                    ),
                    confidence=1.0,
                ))

    # --- 3. Entropy-based detection ---
    for match in _ENTROPY_CANDIDATE.finditer(text):
        candidate = match.group(1)
        if len(candidate) < _ENTROPY_MIN_LENGTH:
            continue

        # Skip if already caught by a known pattern
        already_found = any(
            f.position[0] <= match.start(1) < f.position[1]
            for f in findings
        )
        if already_found:
            continue

        entropy = shannon_entropy(candidate)
        if entropy >= _ENTROPY_THRESHOLD:
            # Higher entropy = higher confidence
            conf = min(0.5 + (entropy - _ENTROPY_THRESHOLD) * 0.2, 0.85)
            findings.append(SecretFinding(
                pattern_name="high_entropy",
                category=SecretCategory.HIGH_ENTROPY,
                severity=SecretSeverity.MEDIUM,
                matched_text=_mask_secret(candidate),
                position=(match.start(1), match.end(1)),
                description=(
                    f"High-entropy string detected (entropy: {entropy:.2f} bits). "
                    "This may be an API key, token, or secret in an unknown format."
                ),
                confidence=conf,
            ))

    # --- 4. Multi-encoding detection ---
    # Check base64 blobs
    base64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
    for match in base64_pattern.finditer(text):
        decoded = _try_decode_base64(match.group())
        if decoded and len(decoded) >= 8:
            encoded_findings = _scan_in_encoding(
                match.group(), decoded, "base64", vault_values
            )
            findings.extend(encoded_findings)

    # Check hex blobs
    hex_pattern = re.compile(r"(?:0[xX])?([0-9a-fA-F]{20,})")
    for match in hex_pattern.finditer(text):
        hex_str = match.group(1) if match.group(1) else match.group()
        decoded = _try_decode_hex(hex_str)
        if decoded and len(decoded) >= 8:
            encoded_findings = _scan_in_encoding(
                match.group(), decoded, "hex", vault_values
            )
            findings.extend(encoded_findings)

    # Check URL-encoded content
    url_pattern = re.compile(r"(?:%[0-9a-fA-F]{2}){3,}")
    for match in url_pattern.finditer(text):
        decoded = _try_decode_url(match.group())
        if decoded and len(decoded) >= 8:
            encoded_findings = _scan_in_encoding(
                match.group(), decoded, "url_encoded", vault_values
            )
            findings.extend(encoded_findings)

    # Check reversed text (for reversed secrets)
    if vault_values:
        reversed_text = _decode_reverse(text)
        for vault_val in vault_values:
            if vault_val and len(vault_val) >= 8 and vault_val in reversed_text:
                findings.append(SecretFinding(
                    pattern_name="vault_match_reversed",
                    category=SecretCategory.VAULT_MATCH,
                    severity=SecretSeverity.CRITICAL,
                    matched_text=_mask_secret(vault_val),
                    position=(0, len(text)),
                    description=(
                        "Vault secret found in reversed text. "
                        "This is a deliberate obfuscation attempt."
                    ),
                    encoding="reversed",
                    confidence=0.95,
                ))

    # Check ROT13
    if vault_values:
        rot13_text = _decode_rot13(text)
        for vault_val in vault_values:
            if vault_val and len(vault_val) >= 8 and vault_val in rot13_text:
                findings.append(SecretFinding(
                    pattern_name="vault_match_rot13",
                    category=SecretCategory.VAULT_MATCH,
                    severity=SecretSeverity.CRITICAL,
                    matched_text=_mask_secret(vault_val),
                    position=(0, len(text)),
                    description=(
                        "Vault secret found in ROT13-encoded text. "
                        "This is a deliberate obfuscation attempt."
                    ),
                    encoding="rot13",
                    confidence=0.95,
                ))

    # Deduplicate findings at same position
    seen_positions: set[tuple[int, int, str]] = set()
    deduped: list[SecretFinding] = []
    for f in findings:
        key = (f.position[0], f.position[1], f.pattern_name)
        if key not in seen_positions:
            seen_positions.add(key)
            deduped.append(f)

    # Sort: CRITICAL first, then by confidence
    severity_order = {
        SecretSeverity.CRITICAL: 0,
        SecretSeverity.HIGH: 1,
        SecretSeverity.MEDIUM: 2,
        SecretSeverity.LOW: 3,
    }
    deduped.sort(key=lambda f: (severity_order.get(f.severity, 99), -f.confidence))

    return deduped
