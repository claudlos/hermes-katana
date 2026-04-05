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
import warnings
from dataclasses import dataclass, field
from typing import (
    Any,
    Generic,
    Iterator,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
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
    "TaintedList",
    "TaintedDict",
    "unwrap",
    "collect_sources",
    "taint_aware_json_dumps",
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
    dependencies: tuple[TaintedValue[Any], ...] = field(default_factory=tuple)
    created_at: float = field(default_factory=time.time)

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

    def derive(self, new_value: Any, *extra_deps: TaintedValue[Any]) -> TaintedValue[Any]:
        """Create a new tainted value that inherits this value's metadata.

        The new value records ``self`` and *extra_deps* as dependencies so
        the provenance chain remains intact.
        """
        all_sources = set(self.sources)
        all_readers = set(self.readers)
        deps: list[TaintedValue[Any]] = [self]
        for dep in extra_deps:
            all_sources.update(dep.sources)
            all_readers.update(dep.readers)
            deps.append(dep)
        return TaintedValue(
            value=new_value,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
            dependencies=tuple(deps),
        )

    def merge_metadata(self, *others: TaintedValue[Any]) -> TaintedValue[T]:
        """Return a copy of ``self`` whose metadata is the union of all inputs.

        The underlying *value* is unchanged — only metadata is merged.
        """
        all_sources = set(self.sources)
        all_readers = set(self.readers)
        deps: list[TaintedValue[Any]] = list(self.dependencies)
        for other in others:
            all_sources.update(other.sources)
            all_readers.update(other.readers)
            deps.append(other)
        return TaintedValue(
            value=self.value,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
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
            return self.value == other.value
        return self.value == other

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


class TaintedStr(str):
    """A tainted string with optional per-character source tracking.

    Subclasses ``str`` directly so that Python's C-level type checks in
    ``__str__``, ``__repr__``, ``__format__``, ``json.dumps``, and f-strings
    all accept this type without stripping taint.

    Behaves like a normal ``str`` for common operations (``+``, slicing,
    ``format``, ``join``, ``split``, etc.) while automatically propagating
    taint metadata through every transformation.
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
        # str is immutable — __new__ already set the string value
        self.sources = sources
        self.readers = readers
        self.dependencies = dependencies
        self.created_at = created_at or time.time()
        self.char_taint = char_taint or CharTaint.uniform(len(self), sources)

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
        all_sources = set(self.sources)
        all_readers = set(self.readers)
        deps: list = [self]
        for dep in extra_deps:
            if isinstance(dep, (TaintedStr, TaintedValue)):
                all_sources.update(dep.sources)
                all_readers.update(dep.readers)
            deps.append(dep)
        return TaintedValue(
            value=new_value,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
            dependencies=tuple(deps),
        )

    def merge_metadata(self, *others: Any) -> TaintedStr:
        """Return a copy of ``self`` whose metadata is the union of all inputs."""
        all_sources = set(self.sources)
        all_readers = set(self.readers)
        deps: list = list(self.dependencies)
        for other in others:
            if isinstance(other, (TaintedStr, TaintedValue)):
                all_sources.update(other.sources)
                all_readers.update(other.readers)
                deps.append(other)
        return TaintedStr(
            value=str.__str__(self),
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
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
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), self.sources),
        )

    def __bool__(self) -> bool:
        return str.__len__(self) > 0

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TaintedValue) and not isinstance(other, TaintedStr):
            return str.__str__(self) == other.value
        # Use str's C-level comparison for str and TaintedStr
        return str.__eq__(self, other)

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
                sources=self.sources,
                readers=self.readers,
                dependencies=(self,),
                char_taint=CharTaint.uniform(len(raw), self.sources),
            )
        return NotImplemented

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        """Encode to bytes, warning that taint metadata is lost."""
        warnings.warn(
            "TaintedStr.encode() returns plain bytes — taint metadata is lost. "
            "Consider keeping the string as TaintedStr.",
            stacklevel=2,
        )
        return str.__str__(self).encode(encoding, errors)

    def __mod__(self, args: Any) -> TaintedStr:
        """%-formatting with taint propagation."""
        if isinstance(args, tuple):
            unwrapped = tuple(_raw(a) if isinstance(a, (TaintedStr, TaintedValue)) else a for a in args)
        elif isinstance(args, (TaintedStr, TaintedValue)):
            unwrapped = _raw(args)
        else:
            unwrapped = args
        raw = str.__mod__(str.__str__(self), unwrapped)
        all_sources = set(self.sources)
        if isinstance(args, tuple):
            for a in args:
                if isinstance(a, (TaintedStr, TaintedValue)):
                    all_sources.update(a.sources)
        elif isinstance(args, (TaintedStr, TaintedValue)):
            all_sources.update(args.sources)
        return TaintedStr(
            value=raw,
            sources=frozenset(all_sources),
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), frozenset(all_sources)),
        )

    # -- Merge helpers ------------------------------------------------------

    def _merge_sources(self, *others: Any) -> frozenset[Source]:
        result = set(self.sources)
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                result.update(o.sources)
        return frozenset(result)

    def _merge_readers(self, *others: Any) -> frozenset[Reader]:
        result = set(self.readers)
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                result.update(o.readers)
        return frozenset(result)

    def _merge_deps(self, *others: Any) -> tuple:
        deps: list = [self]
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                deps.append(o)
        return tuple(deps)

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
            sources=self.sources,
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
                sources=self.sources,
                readers=self.readers,
                dependencies=(self,),
                char_taint=new_ct,
            )
        return NotImplemented  # type: ignore[return-value]

    @overload
    def __getitem__(self, key: int) -> TaintedStr: ...
    @overload
    def __getitem__(self, key: slice) -> TaintedStr: ...

    def __getitem__(self, key: Union[int, slice]) -> TaintedStr:
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
        return TaintedStr(
            value=str.upper(self),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=self.char_taint,
        )

    def lower(self) -> TaintedStr:
        """Return lowercased copy with taint propagation."""
        return TaintedStr(
            value=str.lower(self),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=self.char_taint,
        )

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

    def split(self, sep: Optional[str] = None, maxsplit: int = -1) -> list[TaintedStr]:
        """Split with per-fragment taint propagation."""
        self_raw = str.__str__(self)
        parts = self_raw.split(sep, maxsplit)
        result: list[TaintedStr] = []
        cursor = 0
        for i, part in enumerate(parts):
            # For whitespace splitting, skip whitespace to find the part
            if sep is None:
                while cursor < len(self_raw) and self_raw[cursor] in ' \t\n\r\x0b\x0c':
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

    def replace(self, old: str, new: str, count: int = -1) -> TaintedStr:
        """Replace with taint propagation."""
        self_raw = str.__str__(self)
        raw = self_raw.replace(old, new, count)
        return TaintedStr(
            value=raw,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), self.sources),
        )

    def join(self, iterable: Sequence[Union[str, TaintedStr]]) -> TaintedStr:
        """Join with taint propagation across all elements."""
        self_raw = str.__str__(self)
        parts: list[str] = []
        all_sources: set[Source] = set(self.sources)
        all_readers: set[Reader] = set(self.readers)
        all_deps: list = [self]
        for item in iterable:
            if isinstance(item, TaintedStr):
                parts.append(str.__str__(item))
                all_sources.update(item.sources)
                all_readers.update(item.readers)
                all_deps.append(item)
            elif isinstance(item, TaintedValue):
                parts.append(str(item.value))
                all_sources.update(item.sources)
                all_readers.update(item.readers)
                all_deps.append(item)
            else:
                parts.append(item)
        raw = self_raw.join(parts)
        return TaintedStr(
            value=raw,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
            dependencies=tuple(all_deps),
            char_taint=CharTaint.uniform(len(raw), frozenset(all_sources)),
        )

    def format(self, *args: Any, **kwargs: Any) -> TaintedStr:
        """Format with taint propagation through all arguments."""
        self_raw = str.__str__(self)
        unwrapped_args: list[Any] = []
        all_sources: set[Source] = set(self.sources)
        all_readers: set[Reader] = set(self.readers)
        all_deps: list = [self]
        for a in args:
            if isinstance(a, TaintedStr):
                unwrapped_args.append(str.__str__(a))
                all_sources.update(a.sources)
                all_readers.update(a.readers)
                all_deps.append(a)
            elif isinstance(a, TaintedValue):
                unwrapped_args.append(a.value)
                all_sources.update(a.sources)
                all_readers.update(a.readers)
                all_deps.append(a)
            else:
                unwrapped_args.append(a)
        unwrapped_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, TaintedStr):
                unwrapped_kwargs[k] = str.__str__(v)
                all_sources.update(v.sources)
                all_readers.update(v.readers)
                all_deps.append(v)
            elif isinstance(v, TaintedValue):
                unwrapped_kwargs[k] = v.value
                all_sources.update(v.sources)
                all_readers.update(v.readers)
                all_deps.append(v)
            else:
                unwrapped_kwargs[k] = v
        raw = self_raw.format(*unwrapped_args, **unwrapped_kwargs)
        return TaintedStr(
            value=raw,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
            dependencies=tuple(all_deps),
            char_taint=CharTaint.uniform(len(raw), frozenset(all_sources)),
        )

    def startswith(self, prefix: Union[str, TaintedStr], start: int = 0, end: Optional[int] = None) -> bool:
        """Check prefix, transparently unwrapping TaintedStr."""
        p = str.__str__(prefix) if isinstance(prefix, TaintedStr) else prefix
        self_raw = str.__str__(self)
        return self_raw.startswith(p, start, end if end is not None else len(self_raw))

    def endswith(self, suffix: Union[str, TaintedStr], start: int = 0, end: Optional[int] = None) -> bool:
        """Check suffix, transparently unwrapping TaintedStr."""
        s = str.__str__(suffix) if isinstance(suffix, TaintedStr) else suffix
        self_raw = str.__str__(self)
        return self_raw.endswith(s, start, end if end is not None else len(self_raw))

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


class TaintedList(TaintedValue[list[T]], MutableSequence[T]):
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
        dependencies: tuple[TaintedValue[Any], ...] = (),
        created_at: Optional[float] = None,
        item_taint: Optional[dict[int, frozenset[Source]]] = None,
    ) -> None:
        super().__init__(
            value=list(value),
            sources=sources,
            readers=readers,
            dependencies=dependencies,
            created_at=created_at or time.time(),
        )
        self._item_taint: dict[int, frozenset[Source]] = item_taint or {}

    def get_item_sources(self, index: int) -> frozenset[Source]:
        """Return the taint sources for a specific list index."""
        return self._item_taint.get(index, self.sources)

    def set_item_sources(self, index: int, sources: frozenset[Source]) -> None:
        """Override taint sources for a specific list index."""
        self._item_taint[index] = sources

    # MutableSequence protocol ------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: Union[int, slice]) -> Union[T, list[T]]:
        raw = self.value[index]
        if isinstance(index, int):
            if isinstance(raw, TaintedValue):
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
        if isinstance(index, int) and isinstance(value, TaintedValue):
            self._item_taint[index] = value.sources

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self.value[index]

    def __len__(self) -> int:
        return len(self.value)

    def insert(self, index: int, value: T) -> None:
        self.value.insert(index, value)
        if isinstance(value, TaintedValue):
            self._item_taint[index] = value.sources

    def append_tainted(self, value: T, sources: frozenset[Source]) -> None:
        """Append a value with explicit taint sources."""
        idx = len(self.value)
        self.value.append(value)
        self._item_taint[idx] = sources

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


class TaintedDict(TaintedValue[dict[K, V]], MutableMapping[K, V]):
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
        dependencies: tuple[TaintedValue[Any], ...] = (),
        created_at: Optional[float] = None,
        key_taint: Optional[dict[K, frozenset[Source]]] = None,
    ) -> None:
        super().__init__(
            value=dict(value),
            sources=sources,
            readers=readers,
            dependencies=dependencies,
            created_at=created_at or time.time(),
        )
        self._key_taint: dict[K, frozenset[Source]] = key_taint or {}

    def get_key_sources(self, key: K) -> frozenset[Source]:
        """Return the taint sources for a specific key."""
        return self._key_taint.get(key, self.sources)

    def set_key_sources(self, key: K, sources: frozenset[Source]) -> None:
        """Override taint sources for a specific key."""
        self._key_taint[key] = sources

    # MutableMapping protocol -------------------------------------------------

    def __getitem__(self, key: K) -> V:
        raw = self.value[key]
        if isinstance(raw, TaintedValue):
            return raw
        key_sources = self.get_key_sources(key)
        return TaintedValue(value=raw, sources=key_sources, readers=self.readers)

    def __setitem__(self, key: K, value: V) -> None:
        self.value[key] = value
        if isinstance(value, TaintedValue):
            self._key_taint[key] = value.sources

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
# Utility: unwrap any tainted value recursively
# ---------------------------------------------------------------------------


def unwrap(value: Any) -> Any:
    """Recursively strip all taint wrappers from *value*.

    Useful at system boundaries where you need the raw Python object but
    want to be explicit about discarding taint information.
    """
    # TaintedStr must come before str check (it IS a str)
    if isinstance(value, TaintedStr):
        return str.__str__(value)
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
    # TaintedStr must come before str check (it IS a str)
    if isinstance(value, TaintedStr):
        result.update(value.sources)
        result.update(value.char_taint.all_sources())
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
