"""Codec hooks — preserve taint across stdlib encoding/decoding boundaries.

The stdlib's ``base64``, ``codecs``, ``json``, ``urllib.parse``, and friends
are written in C (or pure Python but returning fresh ``str``/``bytes``).
Without hooks, they destroy taint metadata the moment a ``TaintedStr`` or
``TaintedBytes`` is passed through them.

This module monkey-patches the relevant stdlib functions with transparent
wrappers: each wrapper detects tainted inputs, calls the original stdlib
implementation on the unwrapped value, then re-wraps the result with the
merged source/reader metadata so taint flows through the transform.

Hooks installed by default
--------------------------
- ``base64``: b64encode, b64decode, encodebytes, decodebytes,
  urlsafe_b64encode, urlsafe_b64decode, standard_b64encode, standard_b64decode,
  b16encode, b16decode, b32encode, b32decode, b85encode, b85decode,
  a85encode, a85decode
- ``codecs``: encode, decode
- ``json``: dumps, loads
- ``urllib.parse``: quote, unquote, quote_plus, unquote_plus
- ``html``: escape, unescape

Opt-out
-------
Set ``HERMES_KATANA_CODEC_HOOKS=0`` in the environment, or call
:func:`uninstall_codec_hooks` to restore stdlib behaviour.
"""

from __future__ import annotations

import base64 as _base64
import codecs as _codecs
import html as _html
import json as _json
import logging
import os
import threading
import urllib.parse as _urlparse
from typing import Any, Callable

from hermes_katana.taint.value import (
    TaintedBytes,
    TaintedDict,
    TaintedList,
    TaintedStr,
    TaintedValue,
    collect_sources,
    unwrap,
)

logger = logging.getLogger(__name__)

__all__ = [
    "install_codec_hooks",
    "uninstall_codec_hooks",
    "codec_hooks_installed",
]


_HOOK_LOCK = threading.Lock()
_INSTALLED = False
# Saved originals for uninstall: list of (module, attr_name, original_fn)
_ORIGINALS: list[tuple[Any, str, Callable[..., Any]]] = []


def _has_taint(*values: Any) -> bool:
    """True if any positional value carries taint (shallow + recursive)."""
    for v in values:
        if isinstance(v, (TaintedStr, TaintedBytes, TaintedValue)):
            return True
        if isinstance(v, (list, tuple, dict)):
            if collect_sources(v):
                return True
    return False


def _collect_metadata_recursive(
    value: Any,
    all_sources: set,
    all_readers: set,
    deps: list,
) -> None:
    """Recursively collect sources, readers, and tainted dependencies."""
    if isinstance(value, (TaintedStr, TaintedBytes, TaintedValue)):
        all_sources.update(value.sources)
        all_readers.update(value.readers)
        deps.append(value)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_metadata_recursive(item, all_sources, all_readers, deps)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_metadata_recursive(key, all_sources, all_readers, deps)
            _collect_metadata_recursive(item, all_sources, all_readers, deps)


def _merge_metadata(*values: Any) -> tuple:
    """Collect union of sources, readers, and dep list from tainted inputs."""
    all_sources: set = set()
    all_readers: set = set()
    deps: list = []
    for v in values:
        _collect_metadata_recursive(v, all_sources, all_readers, deps)
    return frozenset(all_sources), frozenset(all_readers), tuple(deps)


def _wrap_result(
    raw_result: Any,
    sources: frozenset,
    readers: frozenset,
    deps: tuple,
) -> Any:
    """Wrap a raw result in the matching tainted subclass."""
    if not sources:
        return raw_result
    if isinstance(raw_result, bytes) and not isinstance(raw_result, TaintedBytes):
        return TaintedBytes(
            value=raw_result,
            sources=sources,
            readers=readers,
            dependencies=deps,
        )
    if isinstance(raw_result, str) and not isinstance(raw_result, TaintedStr):
        return TaintedStr(
            value=raw_result,
            sources=sources,
            readers=readers,
            dependencies=deps,
        )
    # Containers (e.g. from json.loads) get real TaintedDict/TaintedList
    # so attribute access and indexing still work.
    if isinstance(raw_result, dict) and not isinstance(raw_result, TaintedDict):
        return TaintedDict(
            value=raw_result,
            sources=sources,
            readers=readers,
            dependencies=deps,
        )
    if isinstance(raw_result, list) and not isinstance(raw_result, TaintedList):
        return TaintedList(
            value=raw_result,
            sources=sources,
            readers=readers,
            dependencies=deps,
        )
    if not isinstance(raw_result, TaintedValue):
        return TaintedValue(
            value=raw_result,
            sources=sources,
            readers=readers,
            dependencies=deps,
        )
    return raw_result


def _make_hook(original: Callable[..., Any], *, unwrap_args: bool = True) -> Callable[..., Any]:
    """Wrap *original* so tainted inputs produce tainted outputs.

    Parameters
    ----------
    original:
        The stdlib function being replaced.
    unwrap_args:
        When True, positional args are unwrapped before calling original.
        When False (rare), original handles tainted subclasses directly.
    """

    def hook(*args: Any, **kwargs: Any) -> Any:
        taint_inputs = (*args, *kwargs.values())
        if not _has_taint(*taint_inputs):
            return original(*args, **kwargs)
        sources, readers, deps = _merge_metadata(*taint_inputs)
        if unwrap_args:
            raw_args = tuple(unwrap(a) for a in args)
        else:
            raw_args = args
        raw_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
        raw_result = original(*raw_args, **raw_kwargs)
        return _wrap_result(raw_result, sources, readers, deps)

    hook.__wrapped__ = original  # type: ignore[attr-defined]
    hook.__name__ = getattr(original, "__name__", "hook")
    hook.__doc__ = (
        f"Taint-preserving wrapper around {original.__module__}.{hook.__name__}. "
        f"Installed by hermes_katana.taint.codecs."
    )
    return hook


def _patch(module: Any, attr: str) -> None:
    """Install a taint-aware wrapper for module.attr if not already hooked."""
    original = getattr(module, attr, None)
    if original is None:
        return
    if getattr(original, "__wrapped__", None) is not None:
        # Already a hook — skip
        return
    hook = _make_hook(original)
    setattr(module, attr, hook)
    _ORIGINALS.append((module, attr, original))


# ---------------------------------------------------------------------------
# json hooks — slightly different semantics
# ---------------------------------------------------------------------------


def _patch_json() -> None:
    """Install json.dumps and json.loads hooks.

    json.dumps(tainted_structure) → TaintedStr
    json.loads(tainted_str_or_bytes) → TaintedValue[dict/list/…]
    """
    original_dumps = _json.dumps
    original_loads = _json.loads

    def dumps_hook(obj: Any, *args: Any, **kwargs: Any) -> Any:
        sources, readers, deps = _merge_metadata(obj)
        if not sources and not readers:
            return original_dumps(obj, *args, **kwargs)
        raw = original_dumps(unwrap(obj), *args, **kwargs)
        return _wrap_result(raw, sources, readers, deps)

    def loads_hook(s: Any, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(s, (TaintedStr, TaintedBytes, TaintedValue)):
            return original_loads(s, *args, **kwargs)
        sources = collect_sources(s) or s.sources
        raw_s = unwrap(s)
        raw_result = original_loads(raw_s, *args, **kwargs)
        readers = frozenset(s.readers) if hasattr(s, "readers") else frozenset()
        return _wrap_result(raw_result, sources, readers, (s,))

    dumps_hook.__wrapped__ = original_dumps  # type: ignore[attr-defined]
    loads_hook.__wrapped__ = original_loads  # type: ignore[attr-defined]
    dumps_hook.__name__ = "dumps"
    loads_hook.__name__ = "loads"
    _json.dumps = dumps_hook  # type: ignore[assignment]
    _json.loads = loads_hook  # type: ignore[assignment]
    _ORIGINALS.append((_json, "dumps", original_dumps))
    _ORIGINALS.append((_json, "loads", original_loads))


# ---------------------------------------------------------------------------
# Public install / uninstall API
# ---------------------------------------------------------------------------


_BASE64_FUNCS = (
    "b64encode", "b64decode",
    "encodebytes", "decodebytes",
    "urlsafe_b64encode", "urlsafe_b64decode",
    "standard_b64encode", "standard_b64decode",
    "b16encode", "b16decode",
    "b32encode", "b32decode",
    "b85encode", "b85decode",
    "a85encode", "a85decode",
)

_CODECS_FUNCS = ("encode", "decode")

_URLPARSE_FUNCS = (
    "quote", "unquote", "quote_plus", "unquote_plus",
    "quote_from_bytes", "unquote_to_bytes",
)

_HTML_FUNCS = ("escape", "unescape")


def install_codec_hooks() -> None:
    """Monkey-patch stdlib codec functions to propagate taint.

    Idempotent — calling twice is a no-op. Disabled if
    ``HERMES_KATANA_CODEC_HOOKS`` env var is set to ``"0"`` or ``"false"``.
    """
    global _INSTALLED
    env = os.environ.get("HERMES_KATANA_CODEC_HOOKS", "").lower()
    if env in {"0", "false", "no", "off"}:
        logger.debug("Codec hooks disabled via HERMES_KATANA_CODEC_HOOKS env var")
        return

    with _HOOK_LOCK:
        if _INSTALLED:
            return
        for fn_name in _BASE64_FUNCS:
            _patch(_base64, fn_name)
        for fn_name in _CODECS_FUNCS:
            _patch(_codecs, fn_name)
        for fn_name in _URLPARSE_FUNCS:
            _patch(_urlparse, fn_name)
        for fn_name in _HTML_FUNCS:
            _patch(_html, fn_name)
        _patch_json()
        _INSTALLED = True
        logger.debug(
            "Installed %d codec hooks (base64, codecs, json, urllib.parse, html)",
            len(_ORIGINALS),
        )


def uninstall_codec_hooks() -> None:
    """Restore original stdlib codec functions. Idempotent."""
    global _INSTALLED
    with _HOOK_LOCK:
        if not _INSTALLED:
            return
        for module, attr, original in reversed(_ORIGINALS):
            setattr(module, attr, original)
        _ORIGINALS.clear()
        _INSTALLED = False
        logger.debug("Uninstalled codec hooks")


def codec_hooks_installed() -> bool:
    """Return True if codec hooks are currently active."""
    return _INSTALLED
