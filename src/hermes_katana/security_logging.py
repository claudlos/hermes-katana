"""Structured security logging helpers with conservative redaction."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping

SENSITIVE_KEY_MARKERS = (
    "secret",
    "token",
    "password",
    "authorization",
    "cookie",
    "credential",
    "vault",
    "session",
)
MAX_STRING_LEN = 120
MAX_COLLECTION_ITEMS = 20
REDACTED = "<redacted>"
MIN_EXACT_SECRET_LEN = 4

CREDENTIAL_TEXT_PATTERNS = (
    re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)\b((?:api[_-]?key|token|password|secret|credential)\s*[=:]\s*)['\"]?[^'\"\s,;]+"),
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.strip().lower()
    if lowered in {"args_keys", "sensitive_keys", "key_count", "key_names"}:
        return False
    if lowered == "key" or lowered.endswith("_key"):
        return True
    return any(marker in lowered for marker in SENSITIVE_KEY_MARKERS)


def redact_text_for_log(value: str, *, extra_values: Any = ()) -> str:
    """Redact credential-like strings and caller-supplied secret values."""
    redacted = value
    if isinstance(extra_values, str):
        candidates = [extra_values]
    else:
        try:
            candidates = [str(item) for item in extra_values if item]
        except TypeError:
            candidates = []

    for secret in sorted(set(candidates), key=len, reverse=True):
        if len(secret) >= MIN_EXACT_SECRET_LEN:
            redacted = redacted.replace(secret, REDACTED)

    for pattern in CREDENTIAL_TEXT_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(lambda match: match.group(1) + REDACTED, redacted)
        else:
            redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _sanitize_string(value: str) -> str:
    value = redact_text_for_log(value)
    compact = " ".join(value.split())
    if len(compact) <= MAX_STRING_LEN:
        return compact
    return compact[: MAX_STRING_LEN - 3] + "..."


def redact_for_log(value: Any, *, key: str | None = None) -> Any:
    """Return a log-safe representation of a value."""
    if key is not None and _looks_sensitive_key(key):
        return REDACTED

    if isinstance(value, Mapping):
        return {str(k): redact_for_log(v, key=str(k)) for k, v in list(value.items())[:MAX_COLLECTION_ITEMS]}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        redacted = [redact_for_log(item) for item in items[:MAX_COLLECTION_ITEMS]]
        if len(items) > MAX_COLLECTION_ITEMS:
            redacted.append(f"... ({len(items) - MAX_COLLECTION_ITEMS} more)")
        return redacted
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _sanitize_string(str(value))


def summarize_tool_call(
    tool_name: str,
    args: Mapping[str, Any] | None = None,
    *,
    task_id: str = "",
    call_id: str = "",
) -> dict[str, Any]:
    """Return a structured, redacted summary of a tool call."""
    args = args or {}
    keys = sorted(str(key) for key in args.keys())
    sensitive_keys = [key for key in keys if _looks_sensitive_key(key)]
    return {
        "tool_name": tool_name,
        "task_id": task_id or None,
        "call_id": call_id or None,
        "arg_count": len(keys),
        "args_keys": keys,
        "sensitive_keys": sensitive_keys,
        "arg_types": {str(k): type(v).__name__ for k, v in list(args.items())[:MAX_COLLECTION_ITEMS]},
    }


def log_security_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Any,
) -> None:
    """Emit a structured security event with redacted payload."""
    payload = {str(key): redact_for_log(value, key=str(key)) for key, value in fields.items()}
    logger.log(
        level,
        "security_event=%s payload=%s",
        event,
        json.dumps(payload, sort_keys=True),
        extra={
            "katana_event": event,
            "katana_payload": payload,
        },
    )
