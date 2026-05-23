"""Routing helpers for deciding when Scabbard should classify tool data.

Scabbard is trained to classify prompt/content-like natural language. Hermes tool
arguments are heterogeneous JSON: paths, URLs, numbers, enums, commands, and
actual prose can all appear next to each other. This module keeps that schema
awareness out of the classifier itself so security scanners can route each value
to the detector best suited for it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised on Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10 fallback for enum.StrEnum."""


class RouteMode(StrEnum):
    """Scabbard routing modes exposed by plugin/config."""

    OFF = "off"
    CONTENT_ONLY = "content_only"
    BALANCED = "balanced"
    MAX = "max"


class RouteKind(StrEnum):
    """Coarse kind assigned to a tool argument or output field."""

    NATURAL_LANGUAGE = "natural_language"
    STRUCTURAL = "structural"
    COMMAND = "command"
    URL = "url"
    URL_LIST = "url_list"
    PATH = "path"
    CONTROL = "control"
    ENUM = "enum"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    CONTAINER = "container"
    OUTPUT = "output"
    EMPTY = "empty"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScabbardRouteDecision:
    """Decision describing whether a value should be ML-classified."""

    scan: bool
    reason: str
    kind: RouteKind

    def to_dict(self, *, arg: str | None = None, path: str | None = None) -> dict[str, Any]:
        data: dict[str, Any] = {
            "scan": self.scan,
            "reason": self.reason,
            "kind": self.kind.value,
        }
        if arg is not None:
            data["arg"] = arg
        if path is not None:
            data["path"] = path
        return data


@dataclass(frozen=True, slots=True)
class RoutedText:
    """A text fragment selected from a tool output for Scabbard scanning."""

    path: str
    text: str
    decision: ScabbardRouteDecision


CONTENT_FIELD_NAMES = frozenset(
    {
        "body",
        "comment",
        "content",
        "description",
        "html",
        "instruction",
        "instructions",
        "markdown",
        "message",
        "messages",
        "note",
        "notes",
        "prompt",
        "question",
        "summary",
        "system_prompt",
        "text",
        "user_prompt",
    }
)

QUERY_FIELD_NAMES = frozenset({"query", "queries", "search", "snippet", "caption", "title"})

CONTROL_FIELD_NAMES = frozenset(
    {
        "action",
        "api_key",
        "aspect_ratio",
        "backend",
        "clear",
        "count",
        "device",
        "encoding",
        "format",
        "height",
        "id",
        "job_id",
        "limit",
        "max_tokens",
        "model",
        "name",
        "offset",
        "page",
        "page_size",
        "page_token",
        "profile",
        "provider",
        "ref",
        "repeat",
        "resolution",
        "schedule",
        "session_id",
        "target",
        "thread_id",
        "timeout",
        "timeout_s",
        "toolsets",
        "type",
        "width",
    }
)

PATH_FIELD_NAMES = frozenset(
    {
        "cwd",
        "dir",
        "directory",
        "file",
        "file_glob",
        "file_path",
        "filename",
        "folder",
        "output_path",
        "path",
        "paths",
        "root",
        "workdir",
    }
)

URL_FIELD_NAMES = frozenset({"url", "urls", "base_url", "image_url", "webhook_url"})
COMMAND_FIELD_NAMES = frozenset({"cmd", "command", "shell_command", "script"})

TOOL_ARG_ROUTES: dict[str, dict[str, RouteKind]] = {
    "read_file": {"path": RouteKind.PATH, "offset": RouteKind.CONTROL, "limit": RouteKind.CONTROL},
    "write_file": {"path": RouteKind.PATH, "content": RouteKind.NATURAL_LANGUAGE},
    "search_files": {
        "pattern": RouteKind.CONTROL,
        "path": RouteKind.PATH,
        "target": RouteKind.CONTROL,
        "file_glob": RouteKind.CONTROL,
        "limit": RouteKind.CONTROL,
        "offset": RouteKind.CONTROL,
        "output_mode": RouteKind.CONTROL,
        "context": RouteKind.CONTROL,
    },
    "terminal": {"command": RouteKind.COMMAND, "workdir": RouteKind.PATH, "timeout": RouteKind.CONTROL},
    "web_search": {"query": RouteKind.NATURAL_LANGUAGE, "limit": RouteKind.CONTROL},
    "web_extract": {"urls": RouteKind.URL_LIST},
    "browser_navigate": {"url": RouteKind.URL},
    "browser_type": {"text": RouteKind.NATURAL_LANGUAGE, "ref": RouteKind.CONTROL},
    "browser_console": {"expression": RouteKind.COMMAND, "clear": RouteKind.CONTROL},
    "image_generate": {"prompt": RouteKind.NATURAL_LANGUAGE, "aspect_ratio": RouteKind.CONTROL},
    "text_to_speech": {"text": RouteKind.NATURAL_LANGUAGE, "output_path": RouteKind.PATH},
    "cronjob": {"prompt": RouteKind.NATURAL_LANGUAGE, "script": RouteKind.PATH, "action": RouteKind.CONTROL},
    "delegate_task": {"goal": RouteKind.NATURAL_LANGUAGE, "context": RouteKind.NATURAL_LANGUAGE},
    "clarify": {"question": RouteKind.NATURAL_LANGUAGE, "choices": RouteKind.NATURAL_LANGUAGE},
    "send_message": {"message": RouteKind.NATURAL_LANGUAGE, "target": RouteKind.CONTROL},
}

_URL_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_PATH_RE = re.compile(r"(^~?/|^\.?\.?/|^[\w.-]+/[\w./-]+$|^[\w./-]+\.[A-Za-z0-9]{1,8}$)")
_ENUM_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,48}$")
_PROSE_RE = re.compile(r"\b(the|and|or|to|for|with|ignore|instruction|instructions|please|summarize|explain)\b", re.I)
_ADVERSARIAL_SIGNAL_RE = re.compile(
    r"\b("
    r"ignore|disregard|override|bypass|jailbreak|developer\s+mode|system\s+prompt|"
    r"previous\s+instructions?|new\s+instructions?|reveal|leak|exfiltrat(?:e|ion)|"
    r"secret|api\s*key|credential|password|tool\s*call|hidden\s+instruction|"
    r"act\s+as|you\s+are\s+now|do\s+not\s+(?:refuse|tell)|"
    r"base64|rot13|unicode|zero-width"
    r")\b",
    re.I,
)
_SHORT_EXFIL_SIGNAL_RE = re.compile(
    r"("
    r"~?/\.ssh|id_rsa|private\s+key|ssh\s+key|webhook|bearer\s+token|"
    r"auth(?:orization)?\s+header|session\s+cookie|env(?:ironment)?\s+vars?|"
    r"send\s+.{0,80}\b(?:secret|token|credential|password|cookie|webhook|ssh|key)\b|"
    r"upload\s+.{0,80}\b(?:secret|token|credential|password|cookie|ssh|key)\b"
    r")",
    re.I,
)
_REMOTE_ORIGINS = frozenset({"web", "remote", "tool_output", "retrieved", "untrusted", "network"})


def normalize_route_mode(mode: str | RouteMode | None) -> RouteMode:
    if isinstance(mode, RouteMode):
        return mode
    raw = (mode or RouteMode.BALANCED.value).strip().lower().replace("-", "_")
    try:
        return RouteMode(raw)
    except ValueError:
        return RouteMode.BALANCED


def _arg_key(arg_name: str) -> str:
    return (arg_name or "").strip().lower()


def _value_shape_kind(value: object) -> RouteKind:
    if value is None:
        return RouteKind.EMPTY
    if isinstance(value, bool):
        return RouteKind.BOOLEAN
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return RouteKind.NUMERIC
    if isinstance(value, (list, tuple, set, dict)):
        return RouteKind.CONTAINER
    text = str(value).strip()
    if not text:
        return RouteKind.EMPTY
    if _URL_RE.match(text):
        return RouteKind.URL
    if _PATH_RE.match(text):
        return RouteKind.PATH
    if _ENUM_RE.match(text):
        return RouteKind.ENUM
    return RouteKind.UNKNOWN


def _kind_for_arg(tool_name: str, arg_name: str, value: object) -> RouteKind:
    tool_routes = TOOL_ARG_ROUTES.get((tool_name or "").strip(), {})
    key = _arg_key(arg_name)
    if key in tool_routes:
        return tool_routes[key]
    if key in COMMAND_FIELD_NAMES:
        return RouteKind.COMMAND
    if key in URL_FIELD_NAMES:
        if isinstance(value, (list, tuple, set)):
            return RouteKind.URL_LIST
        return RouteKind.URL
    if key in PATH_FIELD_NAMES:
        return RouteKind.PATH
    if key in CONTROL_FIELD_NAMES:
        return RouteKind.CONTROL
    if key in CONTENT_FIELD_NAMES or key in QUERY_FIELD_NAMES:
        return RouteKind.NATURAL_LANGUAGE
    return _value_shape_kind(value)


def _is_content_only_field(arg_name: str) -> bool:
    key = _arg_key(arg_name)
    return key in CONTENT_FIELD_NAMES


def _looks_like_prose(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) >= 160:
        return True
    if " " in stripped and len(stripped) >= 32:
        return True
    return bool(_PROSE_RE.search(stripped))


def has_scabbard_adversarial_signal(text: str) -> bool:
    """Return True when text contains explicit prompt-injection/security cues.

    This is not a detector by itself. It is a precision guard for short,
    low-context tool fields where the ML classifier can otherwise overfit on
    benign product/project names.
    """

    value = text or ""
    return bool(_ADVERSARIAL_SIGNAL_RE.search(value) or _SHORT_EXFIL_SIGNAL_RE.search(value))


def should_scabbard_scan_arg(
    tool_name: str,
    arg_name: str,
    value: object,
    *,
    origin: str | None = None,
    mode: str | RouteMode = RouteMode.BALANCED,
) -> ScabbardRouteDecision:
    """Return whether Scabbard should classify a tool argument value."""

    route_mode = normalize_route_mode(mode)
    kind = _kind_for_arg(tool_name, arg_name, value)

    if route_mode == RouteMode.OFF:
        return ScabbardRouteDecision(False, "routing_disabled", kind)
    if kind == RouteKind.EMPTY:
        return ScabbardRouteDecision(False, "empty_value", kind)
    if route_mode == RouteMode.MAX:
        return ScabbardRouteDecision(
            kind not in {RouteKind.EMPTY, RouteKind.NUMERIC, RouteKind.BOOLEAN}, "max_mode", kind
        )
    if route_mode == RouteMode.CONTENT_ONLY:
        if _is_content_only_field(arg_name) and kind == RouteKind.NATURAL_LANGUAGE:
            return ScabbardRouteDecision(True, "content_only_field", kind)
        return ScabbardRouteDecision(False, "content_only_skip", kind)

    # Balanced mode.
    if kind == RouteKind.NATURAL_LANGUAGE:
        return ScabbardRouteDecision(True, "natural_language_field", kind)
    if kind in {
        RouteKind.BOOLEAN,
        RouteKind.COMMAND,
        RouteKind.CONTROL,
        RouteKind.ENUM,
        RouteKind.NUMERIC,
        RouteKind.PATH,
        RouteKind.STRUCTURAL,
        RouteKind.URL,
        RouteKind.URL_LIST,
    }:
        return ScabbardRouteDecision(False, f"{kind.value}_field", kind)
    if kind == RouteKind.CONTAINER:
        return ScabbardRouteDecision(False, "container_scanned_by_leaf_extraction", kind)

    text = str(value or "")
    if origin and origin.lower() in _REMOTE_ORIGINS and _looks_like_prose(text):
        return ScabbardRouteDecision(True, "remote_prose_fallback", RouteKind.NATURAL_LANGUAGE)
    if _looks_like_prose(text):
        return ScabbardRouteDecision(True, "prose_fallback", RouteKind.NATURAL_LANGUAGE)
    return ScabbardRouteDecision(False, "unknown_non_prose", kind)


def extract_scabbard_arg_texts(
    tool_name: str,
    arg_name: str,
    value: object,
    *,
    origin: str | None = None,
    mode: str | RouteMode = RouteMode.BALANCED,
) -> list[tuple[str, str, ScabbardRouteDecision]]:
    """Extract scan-worthy leaf text fragments from a tool argument."""

    decision = should_scabbard_scan_arg(tool_name, arg_name, value, origin=origin, mode=mode)
    if decision.scan:
        return [(arg_name, str(value), decision)]
    if not isinstance(value, (dict, list, tuple)):
        return []

    extracted: list[tuple[str, str, ScabbardRouteDecision]] = []

    def walk(obj: object, path: str) -> None:
        if isinstance(obj, dict):
            for key, leaf in obj.items():
                leaf_path = f"{path}.{key}" if path else str(key)
                leaf_decision = should_scabbard_scan_arg(tool_name, str(key), leaf, origin=origin, mode=mode)
                if leaf_decision.scan:
                    extracted.append((leaf_path, str(leaf), leaf_decision))
                elif isinstance(leaf, (dict, list, tuple)):
                    walk(leaf, leaf_path)
        elif isinstance(obj, (list, tuple)):
            for idx, leaf in enumerate(obj):
                leaf_path = f"{path}[{idx}]"
                if isinstance(leaf, (dict, list, tuple)):
                    walk(leaf, leaf_path)

    walk(value, arg_name)
    return extracted


def _parse_jsonish_output(output: object) -> object:
    if not isinstance(output, str):
        return output
    stripped = output.strip()
    if not stripped:
        return ""
    if stripped[0] not in "[{":
        return output
    try:
        return json.loads(stripped)
    except (TypeError, ValueError):
        return output


def extract_scabbard_output_texts(
    tool_name: str,
    output: object,
    *,
    mode: str | RouteMode = RouteMode.BALANCED,
    max_fragments: int = 16,
    max_chars_per_fragment: int = 4000,
) -> list[RoutedText]:
    """Extract content-like text fragments from a tool output."""

    route_mode = normalize_route_mode(mode)
    if route_mode == RouteMode.OFF or output is None:
        return []

    parsed = _parse_jsonish_output(output)
    results: list[RoutedText] = []

    def add(path: str, text: str, reason: str = "output_content_field") -> None:
        if not text.strip() or len(results) >= max_fragments:
            return
        if route_mode != RouteMode.MAX and not (_looks_like_prose(text) or has_scabbard_adversarial_signal(text)):
            return
        clipped = text[:max_chars_per_fragment]
        results.append(
            RoutedText(
                path=path,
                text=clipped,
                decision=ScabbardRouteDecision(True, reason, RouteKind.OUTPUT),
            )
        )

    def walk(obj: object, path: str) -> None:
        if len(results) >= max_fragments:
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                key_l = str(key).lower()
                if isinstance(value, str) and (key_l in CONTENT_FIELD_NAMES or key_l in QUERY_FIELD_NAMES):
                    add(child_path, value)
                elif isinstance(value, (dict, list, tuple)):
                    walk(value, child_path)
            return
        if isinstance(obj, (list, tuple)):
            for idx, value in enumerate(obj):
                walk(value, f"{path}[{idx}]")
            return
        if isinstance(obj, str):
            # Raw string output from retrieval/browser-like tools is usually content.
            if tool_name in {"web_extract", "web_search", "browser_snapshot", "browser_console", "read_file"}:
                if _looks_like_prose(obj) or route_mode == RouteMode.MAX:
                    add(path or "output", obj, "raw_output_prose")

    walk(parsed, "output")
    return results


__all__ = [
    "CONTENT_FIELD_NAMES",
    "QUERY_FIELD_NAMES",
    "RouteKind",
    "RouteMode",
    "RoutedText",
    "ScabbardRouteDecision",
    "TOOL_ARG_ROUTES",
    "extract_scabbard_arg_texts",
    "extract_scabbard_output_texts",
    "has_scabbard_adversarial_signal",
    "normalize_route_mode",
    "should_scabbard_scan_arg",
]
