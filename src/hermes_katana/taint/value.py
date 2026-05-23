"""Tainted value wrappers with full provenance tracking.

Every Python value that enters the agent runtime through an external source
gets wrapped in a :class:`TaintedValue` (or one of its specialisations) so
that taint metadata — *sources*, *readers*, and *dependency chains* — propagate
automatically through string concatenation, slicing, formatting, and
collection operations.

Design inspired by CaMeL's ``CaMeLStr`` but generalised to arbitrary types
and extended with per-character tracking for strings.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Optional,
    SupportsIndex,
    TypeVar,
    Union,
    overload,
)

from hermes_katana.taint.labels import Reader, Source, TaintLabel, TrustLevel

_logger = logging.getLogger(__name__)

__all__ = [
    "TaintedValue",
    "CharTaint",
    "TaintedStr",
    "TaintedBytes",
    "TaintedList",
    "TaintedDict",
    "unwrap",
    "collect_sources",
    "taint_aware_json_dumps",
    "taint_aware_fstring",
]

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# ---------------------------------------------------------------------------
# Core wrapper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TaintedValue(Generic[T]):
    """Wraps an arbitrary Python value with taint provenance metadata.

    Parameters
    ----------
    value:
        The underlying (unwrapped) data.
    sources:
        Immutable set of :class:`Source` records describing *where* the data
        came from.
    readers:
        Immutable set of :class:`Reader` records that are permitted to consume
        this value.  An empty set means *no reader restriction* (public).
    dependencies:
        Tuple of upstream :class:`TaintedValue` instances that were combined
        to produce this value — enables full provenance-chain reconstruction.
    created_at:
        Unix timestamp of when this wrapper was created.
    """

    value: T
    sources: frozenset[Source] = field(default_factory=frozenset)
    readers: frozenset[Reader] = field(default_factory=frozenset)
    dependencies: tuple[TaintedValue[Any] | TaintedStr, ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.sources = _freeze_sources(self.sources)
        self.readers = _freeze_readers(self.readers)
        self.dependencies = tuple(self.dependencies or ())

    # -- Introspection --------------------------------------------------------

    @property
    def labels(self) -> frozenset[TaintLabel]:
        """Unique set of taint labels across all sources."""
        return frozenset(s.label for s in self.sources)

    def is_trusted(self) -> bool:
        """``True`` if *every* source is marked ``TRUSTED``."""
        if not self.sources:
            return False
        return all(s.trust_level is TrustLevel.TRUSTED for s in self.sources)

    def is_untrusted(self) -> bool:
        """``True`` if *any* source is marked ``UNTRUSTED``."""
        return any(s.trust_level is TrustLevel.UNTRUSTED for s in self.sources)

    def is_public(self) -> bool:
        """``True`` if no reader restrictions are set (anyone may consume)."""
        return len(self.readers) == 0

    def has_label(self, label: TaintLabel) -> bool:
        """Check whether any source carries the given *label*."""
        return any(s.label is label for s in self.sources)

    # -- Derivation -----------------------------------------------------------

    def derive(self, new_value: Any, *extra_deps: TaintedValue[Any] | TaintedStr) -> TaintedValue[Any]:
        """Create a new tainted value that inherits this value's metadata.

        The new value records ``self`` and *extra_deps* as dependencies so
        the provenance chain remains intact.
        """
        deps: list[TaintedValue[Any] | TaintedStr] = [self]
        for dep in extra_deps:
            deps.append(dep)
        return TaintedValue(
            value=new_value,
            sources=_merge_sources_from(self, *extra_deps),
            readers=_merge_readers_from(self, *extra_deps),
            dependencies=tuple(deps),
        )

    def merge_metadata(self, *others: TaintedValue[Any] | TaintedStr) -> TaintedValue[T]:
        """Return a copy of ``self`` whose metadata is the union of all inputs.

        The underlying *value* is unchanged — only metadata is merged.
        """
        deps: list[TaintedValue[Any] | TaintedStr] = list(self.dependencies)
        for other in others:
            deps.append(other)
        return TaintedValue(
            value=self.value,
            sources=_merge_sources_from(self, *others),
            readers=_merge_readers_from(self, *others),
            dependencies=tuple(deps),
            created_at=self.created_at,
        )

    # -- Convenience ----------------------------------------------------------

    def unwrap(self, audit: bool = True, reason: str = "") -> T:
        """Return the bare value, discarding taint (use with caution).

        Parameters
        ----------
        audit:
            If ``True`` (default), log a warning when taint is stripped.
        reason:
            Human-readable justification for stripping taint.
        """
        if audit and self.sources:
            labels = ", ".join(sorted(lbl.name for lbl in self.labels))
            _logger.warning(
                "Taint stripped via unwrap(): labels={%s}, reason=%r",
                labels,
                reason or "<no reason given>",
            )
        return self.value

    def __repr__(self) -> str:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        vr = repr(self.value)
        if len(vr) > 60:
            vr = vr[:57] + "..."
        return f"TaintedValue({vr}, labels={{{labels}}}, {'trusted' if self.is_trusted() else 'untrusted'})"

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaintedValue):
            return bool(self.value == other.value)
        return bool(self.value == other)

    def __hash__(self) -> int:
        try:
            return hash(self.value)
        except TypeError:
            return id(self)


# ---------------------------------------------------------------------------
# Character-level taint map used by TaintedStr
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CharTaint:
    """Per-character taint information for a string.

    Maps character indices → frozenset[Source].  Characters that share
    identical source sets are stored efficiently via run-length segments.
    """

    _map: dict[int, frozenset[Source]] = field(default_factory=dict)
    _default: frozenset[Source] = field(default_factory=frozenset)

    @classmethod
    def uniform(cls, length: int, sources: frozenset[Source]) -> CharTaint:
        """All characters share the same source set."""
        return cls(_default=sources)

    def get(self, index: int) -> frozenset[Source]:
        """Sources for the character at *index*."""
        return self._map.get(index, self._default)

    def set(self, index: int, sources: frozenset[Source]) -> None:
        """Override sources for a single character."""
        self._map[index] = sources

    def slice(self, start: int, stop: int, step: int = 1) -> CharTaint:
        """Return a new CharTaint for a substring slice."""
        new_map: dict[int, frozenset[Source]] = {}
        new_idx = 0
        for old_idx in range(start, stop, step):
            src = self.get(old_idx)
            if src != self._default:
                new_map[new_idx] = src
            new_idx += 1
        return CharTaint(_map=new_map, _default=self._default)

    def concat(self, other: CharTaint, my_len: int) -> CharTaint:
        """Concatenate two CharTaint maps (``self`` has *my_len* chars)."""
        new_default = self._default | other._default if self._default != other._default else self._default
        new_map: dict[int, frozenset[Source]] = {}
        # Copy self entries
        for idx, src in self._map.items():
            merged = src | other._default if new_default != src else src
            if merged != new_default:
                new_map[idx] = merged
        # Copy other entries, shifted
        for idx, src in other._map.items():
            merged = src | self._default if new_default != src else src
            new_idx = idx + my_len
            if merged != new_default:
                new_map[new_idx] = merged
        return CharTaint(_map=new_map, _default=new_default)

    def all_sources(self) -> frozenset[Source]:
        """Union of all character-level sources."""
        result = set(self._default)
        for src_set in self._map.values():
            result.update(src_set)
        return frozenset(result)


# ---------------------------------------------------------------------------
# TaintedStr — string subclass with character-level taint propagation
# ---------------------------------------------------------------------------


def _raw(s: object) -> str:
    """Extract the raw str from a TaintedStr or plain str."""
    if isinstance(s, TaintedStr):
        return str.__str__(s)
    if isinstance(s, TaintedValue):
        return str(s.value)
    return str(s)


_NO_READERS = frozenset({Reader("__katana_no_readers__", frozenset())})


def _freeze_sources(sources: Iterable[Source] | None) -> frozenset[Source]:
    """Copy source metadata into an immutable set."""
    return frozenset(sources or ())


def _freeze_readers(readers: Iterable[Reader] | None) -> frozenset[Reader]:
    """Copy reader metadata into an immutable set."""
    return frozenset(readers or ())


def _reader_intersection(left: frozenset[Reader], right: frozenset[Reader]) -> frozenset[Reader]:
    """Merge reader restrictions using CaMeL-style intersection semantics.

    Empty reader sets represent public/unrestricted input values. When both
    sides are restricted and disjoint, keep a private sentinel instead of
    returning an empty set, because an empty set already means public.
    """
    if not left:
        return right
    if not right:
        return left
    merged = left & right
    return merged if merged else _NO_READERS


def _merge_reader_sets(*sets_: frozenset[Reader]) -> frozenset[Reader]:
    merged: frozenset[Reader] = frozenset()
    for readers in sets_:
        merged = _reader_intersection(merged, readers)
    return merged


def _is_tainted_value(value: Any) -> bool:
    return isinstance(value, (TaintedStr, TaintedBytes, TaintedValue))


def _effective_sources(value: Any) -> frozenset[Source]:
    """Return all source metadata carried by a tainted value."""
    sources = set(getattr(value, "sources", frozenset()) or ())
    if isinstance(value, TaintedStr):
        sources.update(value.char_taint.all_sources())
    return frozenset(sources)


def _merge_sources_from(*values: Any) -> frozenset[Source]:
    result: set[Source] = set()
    for value in values:
        if _is_tainted_value(value):
            result.update(_effective_sources(value))
    return frozenset(result)


def _merge_readers_from(*values: Any) -> frozenset[Reader]:
    reader_sets = [
        _freeze_readers(getattr(value, "readers", frozenset())) for value in values if _is_tainted_value(value)
    ]
    return _merge_reader_sets(*reader_sets)


def _uniform_char_taint(length: int, *values: Any) -> CharTaint:
    return CharTaint.uniform(length, _merge_sources_from(*values))


class TaintedStr(str):
    """A tainted string with optional per-character source tracking.

    Subclasses ``str`` directly so that Python's C-level type checks accept
    this type as a ``str``.  Taint metadata is propagated through string
    operations via overridden dunder methods.

    Taint-preserving operations
    ---------------------------
    ``+``, ``__radd__``, ``__add__``, ``__mod__``, ``__rmod__``,
    ``format()``, ``join()``, ``split()``, ``strip()``, ``replace()``,
    ``upper()``, ``lower()``, ``__getitem__`` (slice/index),
    ``__format__``, ``__str__``, ``__repr__``.

    Known laundering vectors (taint IS lost)
    -----------------------------------------
    * **f-strings with surrounding literals**: ``f"prefix {tainted} suffix"``
      is compiled by CPython to a ``BUILD_STRING`` opcode that assembles
      parts via an internal C-level join, producing a plain ``str``.  Only a
      bare ``f"{tainted}"`` (single expression, no adjacent literals) calls
      ``__format__`` and preserves the type.  **Use** :func:`taint_aware_fstring`
      instead of f-string interpolation when taint must not be dropped.
    * **json.dumps()**: ``json.dumps`` traverses the value tree without
      calling Python-level string operators; use :func:`taint_aware_json_dumps`.
    * **bytes.decode()**: round-tripping through ``encode()``/``decode()``
      drops taint.  ``encode()`` emits a :mod:`warnings` warning; use
      :meth:`encode_tainted` to keep the metadata in a :class:`TaintedValue`.
    """

    def __new__(
        cls,
        value: str = "",
        sources: frozenset[Source] = frozenset(),
        readers: frozenset[Reader] = frozenset(),
        dependencies: tuple = (),
        created_at: Optional[float] = None,
        char_taint: Optional[CharTaint] = None,
    ) -> TaintedStr:
        val = _raw(value)
        return str.__new__(cls, val)

    def __init__(
        self,
        value: str = "",
        sources: frozenset[Source] = frozenset(),
        readers: frozenset[Reader] = frozenset(),
        dependencies: tuple = (),
        created_at: Optional[float] = None,
        char_taint: Optional[CharTaint] = None,
    ) -> None:
        # str is immutable — __new__ already set the string value.
        #
        # CPython's ``str()`` builtin on a str subclass invokes __init__ again
        # with default args. If we blindly assigned ``self.sources = sources``
        # (an empty frozenset), calling ``str(tainted)`` would destroy the
        # taint metadata in place. Guard against that: only overwrite
        # attributes when the caller actually passed non-default values OR
        # when attributes haven't been set yet.
        sources = _freeze_sources(sources)
        readers = _freeze_readers(readers)
        dependencies = tuple(dependencies or ())

        already_initialized = hasattr(self, "sources")
        if not already_initialized or sources:
            self.sources = sources
        if not already_initialized or readers:
            self.readers = readers
        if not already_initialized or dependencies:
            self.dependencies = dependencies
        if not already_initialized or created_at is not None:
            self.created_at = created_at or time.time()
        if not already_initialized or char_taint is not None:
            self.char_taint = char_taint or CharTaint.uniform(len(self), self.sources)

    # -- Backward compatibility: .value property ----------------------------

    @property
    def value(self) -> str:
        """Return the raw string value (for TaintedValue compatibility)."""
        return str.__str__(self)

    # -- Introspection (mirrors TaintedValue API) ---------------------------

    @property
    def labels(self) -> frozenset[TaintLabel]:
        """Unique set of taint labels across all sources."""
        return frozenset(s.label for s in self.sources)

    def is_trusted(self) -> bool:
        """``True`` if *every* source is marked ``TRUSTED``."""
        if not self.sources:
            return False
        return all(s.trust_level is TrustLevel.TRUSTED for s in self.sources)

    def is_untrusted(self) -> bool:
        """``True`` if *any* source is marked ``UNTRUSTED``."""
        return any(s.trust_level is TrustLevel.UNTRUSTED for s in self.sources)

    def is_public(self) -> bool:
        """``True`` if no reader restrictions are set."""
        return len(self.readers) == 0

    def has_label(self, label: TaintLabel) -> bool:
        """Check whether any source carries the given *label*."""
        return any(s.label is label for s in self.sources)

    # -- Derivation ---------------------------------------------------------

    def derive(self, new_value: Any, *extra_deps: Any) -> TaintedValue[Any]:
        """Create a new tainted value that inherits this string's metadata."""
        deps: list = [self]
        for dep in extra_deps:
            deps.append(dep)
        return TaintedValue(
            value=new_value,
            sources=_merge_sources_from(self, *extra_deps),
            readers=_merge_readers_from(self, *extra_deps),
            dependencies=tuple(deps),
        )

    def merge_metadata(self, *others: Any) -> TaintedStr:
        """Return a copy of ``self`` whose metadata is the union of all inputs."""
        deps: list = list(self.dependencies)
        for other in others:
            if isinstance(other, (TaintedStr, TaintedValue)):
                deps.append(other)
        return TaintedStr(
            value=str.__str__(self),
            sources=_merge_sources_from(self, *others),
            readers=_merge_readers_from(self, *others),
            dependencies=tuple(deps),
            created_at=self.created_at,
            char_taint=self.char_taint,
        )

    def unwrap(self, audit: bool = True, reason: str = "") -> str:
        """Return the bare string, discarding taint (use with caution)."""
        if audit and self.sources:
            labels = ", ".join(sorted(lbl.name for lbl in self.labels))
            _logger.warning(
                "Taint stripped via unwrap(): labels={%s}, reason=%r",
                labels,
                reason or "<no reason given>",
            )
        return str.__str__(self)

    # -- Python special methods ---------------------------------------------

    def __str__(self) -> TaintedStr:
        """Return self — TaintedStr IS a str so this is valid."""
        return self

    def __format__(self, format_spec: str) -> TaintedStr:
        """Format with taint propagation."""
        raw = str.__format__(self, format_spec)
        return TaintedStr(
            value=raw,
            sources=_effective_sources(self),
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), _effective_sources(self)),
        )

    def __bool__(self) -> bool:
        return str.__len__(self) > 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaintedValue) and not isinstance(other, TaintedStr):
            return bool(str.__str__(self) == other.value)
        # Use str's C-level comparison for str and TaintedStr
        return bool(str.__eq__(self, other))

    def __hash__(self) -> int:
        return str.__hash__(self)

    def __repr__(self) -> TaintedStr:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        vr = repr(str.__str__(self))
        if len(vr) > 60:
            vr = vr[:57] + "..."
        raw = f"TaintedStr({vr}, labels={{{labels}}})"
        return TaintedStr(
            value=raw,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def __rmod__(self, fmt: str) -> TaintedStr:
        """Handle ``'format' % tainted`` (printf-style with tainted arg)."""
        if isinstance(fmt, str) and not isinstance(fmt, TaintedStr):
            raw = str.__mod__(fmt, str.__str__(self))
            return TaintedStr(
                value=raw,
                sources=_effective_sources(self),
                readers=self.readers,
                dependencies=(self,),
                char_taint=CharTaint.uniform(len(raw), _effective_sources(self)),
            )
        return NotImplemented

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> TaintedBytes:
        """Encode to ``TaintedBytes``, preserving taint metadata.

        Historically this returned plain ``bytes`` and emitted a warning;
        attackers could exploit the gap by laundering tainted data through
        ``str.encode() → base64 → .decode()`` codec chains. Now the
        resulting bytes carry the same sources, readers, and dependency
        link forward.
        """
        raw = str.__str__(self).encode(encoding, errors)
        return TaintedBytes(
            value=raw,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def __mod__(self, args: Any) -> TaintedStr:
        """%-formatting with taint propagation."""
        unwrapped: Any
        if isinstance(args, tuple):
            unwrapped = tuple(_raw(a) if isinstance(a, (TaintedStr, TaintedValue)) else a for a in args)
        elif isinstance(args, (TaintedStr, TaintedValue)):
            unwrapped = _raw(args)
        else:
            unwrapped = args
        raw = str.__mod__(str.__str__(self), unwrapped)
        tainted_args = (
            [a for a in args if isinstance(a, (TaintedStr, TaintedValue))] if isinstance(args, tuple) else [args]
        )
        all_sources = _merge_sources_from(self, *tainted_args)
        return TaintedStr(
            value=raw,
            sources=all_sources,
            readers=_merge_readers_from(self, *tainted_args),
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), all_sources),
        )

    # -- Merge helpers ------------------------------------------------------

    def _merge_sources(self, *others: Any) -> frozenset[Source]:
        return _merge_sources_from(self, *others)

    def _merge_readers(self, *others: Any) -> frozenset[Reader]:
        return _merge_readers_from(self, *others)

    def _merge_deps(self, *others: Any) -> tuple:
        deps: list = [self]
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                deps.append(o)
        return tuple(deps)

    def _wrap_transformed(self, raw: str, *, char_taint: CharTaint | None = None) -> TaintedStr:
        sources = _effective_sources(self)
        if char_taint is None:
            char_taint = self.char_taint if len(raw) == len(self) else CharTaint.uniform(len(raw), sources)
        return TaintedStr(
            value=raw,
            sources=sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=char_taint,
        )

    # -- String operations that propagate taint -----------------------------

    def __add__(self, other: Union[str, TaintedStr]) -> TaintedStr:
        self_raw = str.__str__(self)
        if isinstance(other, TaintedStr):
            other_raw = str.__str__(other)
            new_val = self_raw + other_raw
            new_ct = self.char_taint.concat(other.char_taint, len(self_raw))
            return TaintedStr(
                value=new_val,
                sources=self._merge_sources(other),
                readers=self._merge_readers(other),
                dependencies=self._merge_deps(other),
                char_taint=new_ct,
            )
        # Plain str — inherit our taint
        new_val = self_raw + other
        plain_ct = CharTaint.uniform(len(other), frozenset())
        new_ct = self.char_taint.concat(plain_ct, len(self_raw))
        return TaintedStr(
            value=new_val,
            sources=_effective_sources(self),
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def __radd__(self, other: str) -> TaintedStr:
        """Handle ``plain_str + tainted`` with taint propagation."""
        if isinstance(other, str) and not isinstance(other, TaintedStr):
            self_raw = str.__str__(self)
            plain_ct = CharTaint.uniform(len(other), frozenset())
            new_ct = plain_ct.concat(self.char_taint, len(other))
            return TaintedStr(
                value=other + self_raw,
                sources=_effective_sources(self),
                readers=self.readers,
                dependencies=(self,),
                char_taint=new_ct,
            )
        return NotImplemented  # type: ignore[return-value]

    def __getitem__(self, key: Union[int, slice]) -> TaintedStr:  # type: ignore[override]
        self_raw = str.__str__(self)
        if isinstance(key, int):
            idx = key if key >= 0 else len(self_raw) + key
            char_sources = self.char_taint.get(idx)
            return TaintedStr(
                value=self_raw[key],
                sources=char_sources,
                readers=self.readers,
                dependencies=(self,),
                char_taint=CharTaint.uniform(1, char_sources),
            )
        # Slice
        raw = self_raw[key]
        start, stop, step = key.indices(len(self_raw))
        new_ct = self.char_taint.slice(start, stop, step)
        return TaintedStr(
            value=raw,
            sources=new_ct.all_sources(),
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def __len__(self) -> int:
        return str.__len__(self)

    def __iter__(self) -> Iterator[TaintedStr]:
        for i in range(len(self)):
            yield self[i]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, TaintedStr):
            return str.__contains__(self, str.__str__(item))
        if isinstance(item, str):
            return str.__contains__(self, item)
        return False

    # -- Common str methods that propagate taint ----------------------------

    def upper(self) -> TaintedStr:
        """Return uppercased copy with taint propagation."""
        return self._wrap_transformed(str.upper(str.__str__(self)))

    def lower(self) -> TaintedStr:
        """Return lowercased copy with taint propagation."""
        return self._wrap_transformed(str.lower(str.__str__(self)))

    def casefold(self) -> TaintedStr:
        return self._wrap_transformed(str.casefold(str.__str__(self)))

    def title(self) -> TaintedStr:
        return self._wrap_transformed(str.title(str.__str__(self)))

    def capitalize(self) -> TaintedStr:
        return self._wrap_transformed(str.capitalize(str.__str__(self)))

    def swapcase(self) -> TaintedStr:
        return self._wrap_transformed(str.swapcase(str.__str__(self)))

    def expandtabs(self, tabsize: SupportsIndex = 8) -> TaintedStr:  # type: ignore[override]
        return self._wrap_transformed(str.expandtabs(str.__str__(self), int(tabsize)))

    def strip(self, chars: Optional[str] = None) -> TaintedStr:
        """Return stripped copy with taint propagation."""
        self_raw = str.__str__(self)
        raw = self_raw.strip(chars)
        # Calculate offset directly from lstrip difference (avoids ambiguous index)
        start = len(self_raw) - len(self_raw.lstrip(chars)) if raw else 0
        stop = start + len(raw)
        new_ct = self.char_taint.slice(start, stop)
        return TaintedStr(
            value=raw,
            sources=new_ct.all_sources() if raw else self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def lstrip(self, chars: Optional[str] = None) -> TaintedStr:
        self_raw = str.__str__(self)
        raw = self_raw.lstrip(chars)
        start = len(self_raw) - len(raw)
        new_ct = self.char_taint.slice(start, len(self_raw))
        return TaintedStr(
            value=raw,
            sources=new_ct.all_sources() if raw else _effective_sources(self),
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def rstrip(self, chars: Optional[str] = None) -> TaintedStr:
        self_raw = str.__str__(self)
        raw = self_raw.rstrip(chars)
        new_ct = self.char_taint.slice(0, len(raw))
        return TaintedStr(
            value=raw,
            sources=new_ct.all_sources() if raw else _effective_sources(self),
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def split(self, sep: Optional[str] = None, maxsplit: SupportsIndex = -1) -> list[TaintedStr]:  # type: ignore[override]
        """Split with per-fragment taint propagation."""
        self_raw = str.__str__(self)
        parts = self_raw.split(sep, int(maxsplit))
        result: list[TaintedStr] = []
        cursor = 0
        for i, part in enumerate(parts):
            # For whitespace splitting, skip whitespace to find the part
            if sep is None:
                while cursor < len(self_raw) and self_raw[cursor] in " \t\n\r\x0b\x0c":
                    cursor += 1
            start = cursor
            stop = start + len(part)
            new_ct = self.char_taint.slice(start, stop)
            result.append(
                TaintedStr(
                    value=part,
                    sources=new_ct.all_sources() if part else self.sources,
                    readers=self.readers,
                    dependencies=(self,),
                    char_taint=new_ct,
                )
            )
            cursor = stop + (len(sep) if sep else 0)
        return result

    def rsplit(self, sep: Optional[str] = None, maxsplit: SupportsIndex = -1) -> list[TaintedStr]:  # type: ignore[override]
        raw_parts = str.__str__(self).rsplit(sep, int(maxsplit))
        sources = _effective_sources(self)
        return [
            TaintedStr(
                value=part,
                sources=sources,
                readers=self.readers,
                dependencies=(self,),
                char_taint=CharTaint.uniform(len(part), sources),
            )
            for part in raw_parts
        ]

    def splitlines(self, keepends: bool = False) -> list[TaintedStr]:  # type: ignore[override]
        raw_parts = str.__str__(self).splitlines(keepends)
        sources = _effective_sources(self)
        return [
            TaintedStr(
                value=part,
                sources=sources,
                readers=self.readers,
                dependencies=(self,),
                char_taint=CharTaint.uniform(len(part), sources),
            )
            for part in raw_parts
        ]

    def partition(self, sep: str) -> tuple[TaintedStr, TaintedStr, TaintedStr]:  # type: ignore[override]
        self_raw = str.__str__(self)
        idx = self_raw.find(sep)
        if idx < 0:
            return (
                self[:],
                TaintedStr("", sources=frozenset(), readers=self.readers),
                TaintedStr("", sources=frozenset(), readers=self.readers),
            )
        return (self[:idx], self[idx : idx + len(sep)], self[idx + len(sep) :])

    def rpartition(self, sep: str) -> tuple[TaintedStr, TaintedStr, TaintedStr]:  # type: ignore[override]
        self_raw = str.__str__(self)
        idx = self_raw.rfind(sep)
        if idx < 0:
            return (
                TaintedStr("", sources=frozenset(), readers=self.readers),
                TaintedStr("", sources=frozenset(), readers=self.readers),
                self[:],
            )
        return (self[:idx], self[idx : idx + len(sep)], self[idx + len(sep) :])

    def replace(self, old: str, new: str, count: SupportsIndex = -1) -> TaintedStr:  # type: ignore[override]
        """Replace with taint propagation."""
        self_raw = str.__str__(self)
        raw = self_raw.replace(old, new, int(count))
        sources = _effective_sources(self)
        return TaintedStr(
            value=raw,
            sources=sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), sources),
        )

    def join(self, iterable: Iterable[str]) -> TaintedStr:  # type: ignore[override]
        """Join with taint propagation across all elements."""
        self_raw = str.__str__(self)
        parts: list[str] = []
        all_deps: list = [self]
        tainted_items: list[Any] = [self]
        for item in iterable:
            if isinstance(item, TaintedStr):
                parts.append(str.__str__(item))
                all_deps.append(item)
                tainted_items.append(item)
            elif isinstance(item, TaintedValue):
                parts.append(str(item.value))
                all_deps.append(item)
                tainted_items.append(item)
            else:
                parts.append(item)
        raw = self_raw.join(parts)
        all_sources = _merge_sources_from(*tainted_items)
        return TaintedStr(
            value=raw,
            sources=all_sources,
            readers=_merge_readers_from(*tainted_items),
            dependencies=tuple(all_deps),
            char_taint=CharTaint.uniform(len(raw), all_sources),
        )

    def format(self, *args: Any, **kwargs: Any) -> TaintedStr:
        """Format with taint propagation through all arguments."""
        self_raw = str.__str__(self)
        unwrapped_args: list[Any] = []
        all_deps: list = [self]
        tainted_values: list[Any] = [self]
        for a in args:
            if isinstance(a, TaintedStr):
                unwrapped_args.append(str.__str__(a))
                all_deps.append(a)
                tainted_values.append(a)
            elif isinstance(a, TaintedValue):
                unwrapped_args.append(a.value)
                all_deps.append(a)
                tainted_values.append(a)
            else:
                unwrapped_args.append(a)
        unwrapped_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, TaintedStr):
                unwrapped_kwargs[k] = str.__str__(v)
                all_deps.append(v)
                tainted_values.append(v)
            elif isinstance(v, TaintedValue):
                unwrapped_kwargs[k] = v.value
                all_deps.append(v)
                tainted_values.append(v)
            else:
                unwrapped_kwargs[k] = v
        raw = self_raw.format(*unwrapped_args, **unwrapped_kwargs)
        all_sources = _merge_sources_from(*tainted_values)
        return TaintedStr(
            value=raw,
            sources=all_sources,
            readers=_merge_readers_from(*tainted_values),
            dependencies=tuple(all_deps),
            char_taint=CharTaint.uniform(len(raw), all_sources),
        )

    def ljust(self, width: SupportsIndex, fillchar: str = " ") -> TaintedStr:  # type: ignore[override]
        return self._wrap_transformed(str.ljust(str.__str__(self), int(width), fillchar))

    def rjust(self, width: SupportsIndex, fillchar: str = " ") -> TaintedStr:  # type: ignore[override]
        return self._wrap_transformed(str.rjust(str.__str__(self), int(width), fillchar))

    def center(self, width: SupportsIndex, fillchar: str = " ") -> TaintedStr:  # type: ignore[override]
        return self._wrap_transformed(str.center(str.__str__(self), int(width), fillchar))

    def zfill(self, width: SupportsIndex) -> TaintedStr:  # type: ignore[override]
        return self._wrap_transformed(str.zfill(str.__str__(self), int(width)))

    def translate(self, table: Any) -> TaintedStr:  # type: ignore[override]
        return self._wrap_transformed(str.translate(str.__str__(self), table))

    def removeprefix(self, prefix: str) -> TaintedStr:  # type: ignore[override]
        self_raw = str.__str__(self)
        if self_raw.startswith(prefix):
            return self[len(prefix) :]
        return self[:]

    def removesuffix(self, suffix: str) -> TaintedStr:  # type: ignore[override]
        self_raw = str.__str__(self)
        if suffix and self_raw.endswith(suffix):
            return self[: -len(suffix)]
        return self[:]

    def __mul__(self, n: SupportsIndex) -> TaintedStr:
        raw = str.__mul__(str.__str__(self), int(n))
        sources = _effective_sources(self)
        return TaintedStr(
            value=raw,
            sources=sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), sources),
        )

    def __rmul__(self, n: SupportsIndex) -> TaintedStr:
        return self.__mul__(n)

    def startswith(
        self,
        prefix: str | tuple[str, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> bool:
        """Check prefix, transparently unwrapping TaintedStr."""
        p = prefix
        self_raw = str.__str__(self)
        return self_raw.startswith(p, 0 if start is None else int(start), None if end is None else int(end))

    def endswith(
        self,
        suffix: str | tuple[str, ...],
        start: SupportsIndex | None = None,
        end: SupportsIndex | None = None,
    ) -> bool:
        """Check suffix, transparently unwrapping TaintedStr."""
        s = suffix
        self_raw = str.__str__(self)
        return self_raw.endswith(s, 0 if start is None else int(start), None if end is None else int(end))

    def encode_tainted(self, encoding: str = "utf-8", errors: str = "strict") -> TaintedValue[bytes]:
        """Encode to bytes, preserving taint metadata."""
        raw_bytes = str.__str__(self).encode(encoding, errors)
        return TaintedValue(
            value=raw_bytes,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )


# ---------------------------------------------------------------------------
# TaintedList — list with per-item taint
# ---------------------------------------------------------------------------


class TaintedList(TaintedValue[list[T]]):
    """A list where each item may carry independent taint metadata.

    The wrapper-level ``sources`` represent the taint of the *container
    itself* (e.g. "this list came from a web API").  Individual elements
    may carry additional taint via :class:`TaintedValue` wrapping.
    """

    __slots__ = ("_item_taint",)

    def __init__(
        self,
        value: list[T],
        sources: frozenset[Source] = frozenset(),
        readers: frozenset[Reader] = frozenset(),
        dependencies: tuple[TaintedValue[Any] | TaintedStr, ...] = (),
        created_at: Optional[float] = None,
        item_taint: Optional[dict[int, frozenset[Source]]] = None,
    ) -> None:
        super().__init__(
            value=list(value),
            sources=_freeze_sources(sources),
            readers=_freeze_readers(readers),
            dependencies=dependencies,
            created_at=created_at or time.time(),
        )
        self._item_taint: dict[int, frozenset[Source]] = {
            idx: _freeze_sources(srcs) for idx, srcs in (item_taint or {}).items()
        }

    def get_item_sources(self, index: int) -> frozenset[Source]:
        """Return the taint sources for a specific list index."""
        return self._item_taint.get(index, self.sources)

    def set_item_sources(self, index: int, sources: frozenset[Source]) -> None:
        """Override taint sources for a specific list index."""
        self._item_taint[index] = _freeze_sources(sources)

    # MutableSequence protocol ------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Any:
        raw = self.value[index]
        if isinstance(index, int):
            if isinstance(raw, (TaintedStr, TaintedValue)):
                return raw
            item_sources = self.get_item_sources(index if index >= 0 else len(self.value) + index)
            return TaintedValue(value=raw, sources=item_sources, readers=self.readers)
        return raw

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Any) -> None: ...

    def __setitem__(self, index: Union[int, slice], value: Any) -> None:
        self.value[index] = value  # type: ignore[index]
        if isinstance(index, int) and isinstance(value, (TaintedStr, TaintedValue)):
            self._item_taint[index] = _effective_sources(value)

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.value[index]

    def __len__(self) -> int:
        return len(self.value)

    def insert(self, index: int, value: T) -> None:
        self.value.insert(index, value)
        if isinstance(value, (TaintedStr, TaintedValue)):
            self._item_taint[index] = _effective_sources(value)

    def append_tainted(self, value: T, sources: frozenset[Source]) -> None:
        """Append a value with explicit taint sources."""
        idx = len(self.value)
        self.value.append(value)
        self._item_taint[idx] = _freeze_sources(sources)

    def all_sources(self) -> frozenset[Source]:
        """Union of container sources and all item-level sources."""
        result = set(self.sources)
        for src_set in self._item_taint.values():
            result.update(src_set)
        return frozenset(result)

    def __repr__(self) -> str:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        return f"TaintedList(len={len(self.value)}, labels={{{labels}}})"


# ---------------------------------------------------------------------------
# TaintedDict — dict with per-key taint
# ---------------------------------------------------------------------------


class TaintedDict(TaintedValue[dict[K, V]]):
    """A dict where each key-value pair may carry independent taint metadata.

    The wrapper-level ``sources`` represent the taint of the *container*
    (e.g. "this JSON object came from an MCP server").  Individual entries
    may be independently tainted.
    """

    __slots__ = ("_key_taint",)

    def __init__(
        self,
        value: dict[K, V],
        sources: frozenset[Source] = frozenset(),
        readers: frozenset[Reader] = frozenset(),
        dependencies: tuple[TaintedValue[Any] | TaintedStr, ...] = (),
        created_at: Optional[float] = None,
        key_taint: Optional[dict[K, frozenset[Source]]] = None,
    ) -> None:
        super().__init__(
            value=dict(value),
            sources=_freeze_sources(sources),
            readers=_freeze_readers(readers),
            dependencies=dependencies,
            created_at=created_at or time.time(),
        )
        self._key_taint: dict[K, frozenset[Source]] = {
            key: _freeze_sources(srcs) for key, srcs in (key_taint or {}).items()
        }

    def get_key_sources(self, key: K) -> frozenset[Source]:
        """Return the taint sources for a specific key."""
        return self._key_taint.get(key, self.sources)

    def set_key_sources(self, key: K, sources: frozenset[Source]) -> None:
        """Override taint sources for a specific key."""
        self._key_taint[key] = _freeze_sources(sources)

    # MutableMapping protocol -------------------------------------------------

    def __getitem__(self, key: K) -> Any:
        raw = self.value[key]
        if isinstance(raw, (TaintedStr, TaintedValue)):
            return raw
        key_sources = self.get_key_sources(key)
        return TaintedValue(value=raw, sources=key_sources, readers=self.readers)

    def __setitem__(self, key: K, value: V) -> None:
        self.value[key] = value
        if isinstance(value, (TaintedStr, TaintedValue)):
            self._key_taint[key] = _effective_sources(value)

    def __delitem__(self, key: K) -> None:
        del self.value[key]
        self._key_taint.pop(key, None)

    def __iter__(self) -> Iterator[K]:
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, key: object) -> bool:
        return key in self.value

    def all_sources(self) -> frozenset[Source]:
        """Union of container sources and all key-level sources."""
        result = set(self.sources)
        for src_set in self._key_taint.values():
            result.update(src_set)
        return frozenset(result)

    def __repr__(self) -> str:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        return f"TaintedDict(len={len(self.value)}, labels={{{labels}}})"


# ---------------------------------------------------------------------------
# TaintedBytes — bytes subclass that carries taint across codec boundaries
# ---------------------------------------------------------------------------


class TaintedBytes(bytes):
    """A tainted bytes object with source propagation.

    Subclasses ``bytes`` directly so that C-level type checks in stdlib
    codecs (``base64.b64encode``, ``codecs.encode``, ``zlib.compress``,
    etc.) accept it without stripping taint. Created primarily by
    :meth:`TaintedStr.encode` and by the codec-hook layer in
    :mod:`hermes_katana.taint.codecs`.

    Key operations (``+``, slicing, ``.decode()``) propagate sources
    forward so taint survives full round-trips like
    ``tainted_str → .encode() → b64encode → b64decode → .decode()``.
    """

    def __new__(
        cls,
        value: bytes = b"",
        sources: "frozenset[Source]" = frozenset(),
        readers: "frozenset[Reader]" = frozenset(),
        dependencies: tuple = (),
        created_at: Optional[float] = None,
    ) -> "TaintedBytes":
        if isinstance(value, TaintedBytes):
            raw = bytes(value)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
        elif isinstance(value, TaintedValue):
            raw = bytes(value.value)
        else:
            raw = bytes(value)
        return bytes.__new__(cls, raw)

    def __init__(
        self,
        value: bytes = b"",
        sources: "frozenset[Source]" = frozenset(),
        readers: "frozenset[Reader]" = frozenset(),
        dependencies: tuple = (),
        created_at: Optional[float] = None,
    ) -> None:
        # Same guard as TaintedStr: ``bytes(tainted_bytes)`` may re-invoke
        # __init__ on the same object with defaults, which would wipe taint.
        sources = _freeze_sources(sources)
        readers = _freeze_readers(readers)
        dependencies = tuple(dependencies or ())
        already_initialized = hasattr(self, "sources")
        if not already_initialized or sources:
            self.sources = sources
        if not already_initialized or readers:
            self.readers = readers
        if not already_initialized or dependencies:
            self.dependencies = dependencies
        if not already_initialized or created_at is not None:
            self.created_at = created_at or time.time()

    # -- Backward-compat / introspection (mirrors TaintedStr) --------------

    @property
    def value(self) -> bytes:
        """Return the raw bytes value."""
        return bytes(self)

    @property
    def labels(self) -> "frozenset[TaintLabel]":
        """Unique set of taint labels across all sources."""
        return frozenset(s.label for s in self.sources)

    def is_trusted(self) -> bool:
        if not self.sources:
            return False
        return all(s.trust_level is TrustLevel.TRUSTED for s in self.sources)

    def is_untrusted(self) -> bool:
        return any(s.trust_level is TrustLevel.UNTRUSTED for s in self.sources)

    def is_public(self) -> bool:
        return len(self.readers) == 0

    def has_label(self, label: "TaintLabel") -> bool:
        return any(s.label is label for s in self.sources)

    def unwrap(self, audit: bool = True, reason: str = "") -> bytes:
        """Return the bare bytes, discarding taint (use with caution)."""
        if audit and self.sources:
            labels = ", ".join(sorted(lbl.name for lbl in self.labels))
            _logger.warning(
                "Taint stripped via TaintedBytes.unwrap(): labels={%s}, reason=%r",
                labels,
                reason or "<no reason given>",
            )
        return bytes(self)

    def _merge_with(self, *others: Any) -> tuple:
        """Merge metadata with other tainted values. Returns (sources, readers, deps)."""
        deps: list = [self]
        for o in others:
            if isinstance(o, (TaintedStr, TaintedBytes, TaintedValue)):
                deps.append(o)
        return _merge_sources_from(self, *others), _merge_readers_from(self, *others), tuple(deps)

    # -- Propagating operations --------------------------------------------

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> TaintedStr:
        """Decode to ``TaintedStr``, preserving taint.

        This closes the evasion path where attackers chain
        ``.encode() → base64 → b64decode → .decode()`` to strip taint
        through codec round-trips.
        """
        raw = bytes(self).decode(encoding, errors)
        return TaintedStr(
            value=raw,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), self.sources),
        )

    def __add__(self, other: Any) -> "TaintedBytes":
        if isinstance(other, (bytes, bytearray, memoryview, TaintedBytes)):
            raw_other = bytes(other)
        else:
            return NotImplemented
        raw = bytes(self) + raw_other
        sources, readers, deps = self._merge_with(other)
        return TaintedBytes(
            value=raw,
            sources=sources,
            readers=readers,
            dependencies=deps,
        )

    def __radd__(self, other: Any) -> "TaintedBytes":
        if isinstance(other, (bytes, bytearray, memoryview)):
            raw = bytes(other) + bytes(self)
            return TaintedBytes(
                value=raw,
                sources=self.sources,
                readers=self.readers,
                dependencies=(self,),
            )
        return NotImplemented

    def __getitem__(self, key: Any) -> Any:
        result = bytes(self)[key]
        if isinstance(result, bytes):
            return TaintedBytes(
                value=result,
                sources=self.sources,
                readers=self.readers,
                dependencies=(self,),
            )
        # Single-byte int access returns plain int (no taint representation).
        return result

    def __mul__(self, n: SupportsIndex) -> "TaintedBytes":
        return TaintedBytes(
            value=bytes.__mul__(bytes(self), int(n)),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def __rmul__(self, n: SupportsIndex) -> "TaintedBytes":
        return self.__mul__(n)

    def __mod__(self, args: Any) -> "TaintedBytes":
        if isinstance(args, tuple):
            unwrapped = tuple(bytes(a) if isinstance(a, TaintedBytes) else a for a in args)
            tainted_args = [a for a in args if isinstance(a, (TaintedBytes, TaintedValue))]
        elif isinstance(args, TaintedBytes):
            unwrapped = bytes(args)
            tainted_args = [args]
        else:
            unwrapped = args
            tainted_args = []
        raw = bytes.__mod__(bytes(self), unwrapped)
        sources, readers, deps = self._merge_with(*tainted_args)
        return TaintedBytes(value=raw, sources=sources, readers=readers, dependencies=deps)

    def replace(self, old: bytes, new: bytes, count: SupportsIndex = -1) -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).replace(old, new, int(count)),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def split(self, sep: bytes | None = None, maxsplit: SupportsIndex = -1) -> list["TaintedBytes"]:  # type: ignore[override]
        return [
            TaintedBytes(value=part, sources=self.sources, readers=self.readers, dependencies=(self,))
            for part in bytes(self).split(sep, int(maxsplit))
        ]

    def join(self, iterable: Iterable[bytes]) -> "TaintedBytes":  # type: ignore[override]
        parts: list[bytes] = []
        tainted_values: list[Any] = [self]
        for item in iterable:
            if isinstance(item, (TaintedBytes, TaintedValue)):
                parts.append(bytes(item) if isinstance(item, TaintedBytes) else bytes(item.value))
                tainted_values.append(item)
            else:
                parts.append(bytes(item))
        raw = bytes(self).join(parts)
        sources, readers, deps = self._merge_with(*tainted_values[1:])
        return TaintedBytes(value=raw, sources=sources, readers=readers, dependencies=deps)

    def center(self, width: SupportsIndex, fillchar: bytes = b" ") -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).center(int(width), fillchar),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def ljust(self, width: SupportsIndex, fillchar: bytes = b" ") -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).ljust(int(width), fillchar),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def rjust(self, width: SupportsIndex, fillchar: bytes = b" ") -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).rjust(int(width), fillchar),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def zfill(self, width: SupportsIndex) -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).zfill(int(width)),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def translate(self, table: bytes | None, delete: bytes = b"") -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).translate(table, delete),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def hex(self, sep: str | bytes = "", bytes_per_sep: SupportsIndex = 1) -> TaintedStr:  # type: ignore[override]
        if sep:
            raw = bytes(self).hex(sep, int(bytes_per_sep))
        else:
            raw = bytes(self).hex()
        return TaintedStr(
            value=raw,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), self.sources),
        )

    def upper(self) -> "TaintedBytes":
        return TaintedBytes(value=bytes(self).upper(), sources=self.sources, readers=self.readers, dependencies=(self,))

    def lower(self) -> "TaintedBytes":
        return TaintedBytes(value=bytes(self).lower(), sources=self.sources, readers=self.readers, dependencies=(self,))

    def removeprefix(self, prefix: bytes) -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).removeprefix(prefix),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def removesuffix(self, suffix: bytes) -> "TaintedBytes":  # type: ignore[override]
        return TaintedBytes(
            value=bytes(self).removesuffix(suffix),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
        )

    def __repr__(self) -> str:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        raw_repr = bytes.__repr__(self)
        if len(raw_repr) > 60:
            raw_repr = raw_repr[:57] + "..."
        return f"TaintedBytes({raw_repr}, labels={{{labels}}})"


# ---------------------------------------------------------------------------
# Utility: unwrap any tainted value recursively
# ---------------------------------------------------------------------------


def unwrap(value: Any) -> Any:
    """Recursively strip all taint wrappers from *value*.

    Useful at system boundaries where you need the raw Python object but
    want to be explicit about discarding taint information.
    """
    # TaintedStr / TaintedBytes must come before str/bytes check (they ARE those)
    if isinstance(value, TaintedStr):
        return str.__str__(value)
    if isinstance(value, TaintedBytes):
        return bytes(value)
    if isinstance(value, TaintedDict):
        return {unwrap(k): unwrap(v) for k, v in value.value.items()}
    if isinstance(value, TaintedList):
        return [unwrap(item) for item in value.value]
    if isinstance(value, TaintedValue):
        return value.value
    if isinstance(value, dict):
        return {unwrap(k): unwrap(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cls = type(value)
        return cls(unwrap(item) for item in value)
    return value


def collect_sources(value: Any) -> frozenset[Source]:
    """Recursively collect all taint sources from a (possibly nested) value."""
    result: set[Source] = set()
    # TaintedStr / TaintedBytes must come before str/bytes check (they ARE those)
    if isinstance(value, TaintedStr):
        result.update(value.sources)
        result.update(value.char_taint.all_sources())
        return frozenset(result)
    if isinstance(value, TaintedBytes):
        result.update(value.sources)
        return frozenset(result)
    if isinstance(value, TaintedValue):
        result.update(value.sources)
        if isinstance(value, (TaintedDict, TaintedList)):
            result.update(value.all_sources())
            inner = value.value
            if isinstance(inner, dict):
                for k, v in inner.items():
                    result.update(collect_sources(k))
                    result.update(collect_sources(v))
            elif isinstance(inner, (list, tuple)):
                for item in inner:
                    result.update(collect_sources(item))
            return frozenset(result)
        # For non-container TaintedValues, recurse into the inner value
        return frozenset(result | collect_sources(value.value))
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(collect_sources(k))
            result.update(collect_sources(v))
    elif isinstance(value, (list, tuple)):
        for item in value:
            result.update(collect_sources(item))
    return frozenset(result)


def taint_aware_json_dumps(value: Any, **kwargs: Any) -> TaintedStr:
    """``json.dumps`` that preserves taint through serialisation.

    Collects all taint sources from the input value tree and attaches
    them to the resulting JSON string.
    """
    sources = collect_sources(value)
    raw = json.dumps(unwrap(value), **kwargs)
    return TaintedStr(
        value=raw,
        sources=sources,
        char_taint=CharTaint.uniform(len(raw), sources),
    )


def taint_aware_fstring(template: str, *args: Any, **kwargs: Any) -> TaintedStr:
    """Safe alternative to f-strings when tainted values are interpolated.

    CPython's ``BUILD_STRING`` opcode (used for f-strings with surrounding
    literal text) assembles string parts at the C level, bypassing
    Python-level ``__add__`` / ``__format__`` overrides and producing a plain
    ``str``.  This function preserves taint by collecting sources from all
    arguments **before** formatting and attaching them to the result.

    Usage::

        # Instead of:   f"Context: {web_data}"   (taint laundered!)
        # Use:
        result = taint_aware_fstring("Context: {}", web_data)

        # Keyword placeholders:
        result = taint_aware_fstring("Hello {name}!", name=user_value)

    Parameters
    ----------
    template:
        A ``str.format()``-style template string (``{}`` or ``{name}``
        placeholders, NOT f-string syntax).
    *args, **kwargs:
        Positional and keyword arguments to interpolate.

    Returns
    -------
    TaintedStr
        The formatted string with all input taint sources merged in.
    """
    tainted_values: list[Any] = []
    deps: list = []

    def _collect(val: Any) -> Any:
        if isinstance(val, TaintedStr):
            tainted_values.append(val)
            deps.append(val)
            return str.__str__(val)
        if isinstance(val, TaintedValue):
            tainted_values.append(val)
            deps.append(val)
            return val.value
        return val

    unwrapped_args = [_collect(a) for a in args]
    unwrapped_kwargs = {k: _collect(v) for k, v in kwargs.items()}
    raw = template.format(*unwrapped_args, **unwrapped_kwargs)
    sources = _merge_sources_from(*tainted_values)
    return TaintedStr(
        value=raw,
        sources=sources,
        readers=_merge_readers_from(*tainted_values),
        dependencies=tuple(deps),
        char_taint=CharTaint.uniform(len(raw), sources),
    )
