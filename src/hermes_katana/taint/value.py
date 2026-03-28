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

import time
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

    def unwrap(self) -> T:
        """Return the bare value, discarding taint (use with caution)."""
        return self.value

    def __repr__(self) -> str:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        trusted = "trusted" if self.is_trusted() else "untrusted"
        vr = repr(self.value)
        if len(vr) > 60:
            vr = vr[:57] + "..."
        return f"TaintedValue({vr}, labels={{{labels}}}, {trusted})"

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
# TaintedStr — string with character-level taint propagation
# ---------------------------------------------------------------------------


class TaintedStr(TaintedValue[str]):
    """A tainted string with optional per-character source tracking.

    Behaves like a normal ``str`` for common operations (``+``, slicing,
    ``format``, ``join``, ``split``, etc.) while automatically propagating
    taint metadata through every transformation.
    """

    __slots__ = ("char_taint",)

    def __init__(
        self,
        value: str,
        sources: frozenset[Source] = frozenset(),
        readers: frozenset[Reader] = frozenset(),
        dependencies: tuple[TaintedValue[Any], ...] = (),
        created_at: Optional[float] = None,
        char_taint: Optional[CharTaint] = None,
    ) -> None:
        super().__init__(
            value=value,
            sources=sources,
            readers=readers,
            dependencies=dependencies,
            created_at=created_at or time.time(),
        )
        self.char_taint = char_taint or CharTaint.uniform(len(value), sources)

    # -- String operations that propagate taint -------------------------------

    def _merge_sources(self, *others: TaintedStr | TaintedValue[str]) -> frozenset[Source]:
        """Union sources from self and others."""
        result = set(self.sources)
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                result.update(o.sources)
        return frozenset(result)

    def _merge_readers(self, *others: TaintedStr | TaintedValue[str]) -> frozenset[Reader]:
        result = set(self.readers)
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                result.update(o.readers)
        return frozenset(result)

    def _merge_deps(self, *others: TaintedStr | TaintedValue[str]) -> tuple[TaintedValue[Any], ...]:
        deps: list[TaintedValue[Any]] = [self]
        for o in others:
            if isinstance(o, (TaintedStr, TaintedValue)):
                deps.append(o)
        return tuple(deps)

    def __add__(self, other: Union[str, TaintedStr]) -> TaintedStr:
        if isinstance(other, TaintedStr):
            new_val = self.value + other.value
            new_ct = self.char_taint.concat(other.char_taint, len(self.value))
            return TaintedStr(
                value=new_val,
                sources=self._merge_sources(other),
                readers=self._merge_readers(other),
                dependencies=self._merge_deps(other),
                char_taint=new_ct,
            )
        # Plain str — inherit our taint
        new_val = self.value + other
        plain_ct = CharTaint.uniform(len(other), frozenset())
        new_ct = self.char_taint.concat(plain_ct, len(self.value))
        return TaintedStr(
            value=new_val,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def __radd__(self, other: str) -> TaintedStr:
        if isinstance(other, str) and not isinstance(other, TaintedStr):
            plain_ct = CharTaint.uniform(len(other), frozenset())
            new_ct = plain_ct.concat(self.char_taint, len(other))
            return TaintedStr(
                value=other + self.value,
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
        if isinstance(key, int):
            idx = key if key >= 0 else len(self.value) + key
            char_sources = self.char_taint.get(idx)
            return TaintedStr(
                value=self.value[key],
                sources=char_sources,
                readers=self.readers,
                dependencies=(self,),
                char_taint=CharTaint.uniform(1, char_sources),
            )
        # Slice
        raw = self.value[key]
        start, stop, step = key.indices(len(self.value))
        new_ct = self.char_taint.slice(start, stop, step)
        return TaintedStr(
            value=raw,
            sources=new_ct.all_sources(),
            readers=self.readers,
            dependencies=(self,),
            char_taint=new_ct,
        )

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self) -> Iterator[TaintedStr]:
        for i in range(len(self.value)):
            yield self[i]

    def __contains__(self, item: object) -> bool:
        if isinstance(item, TaintedStr):
            return item.value in self.value
        if isinstance(item, str):
            return item in self.value
        return False

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        labels = ", ".join(sorted(lbl.name for lbl in self.labels))
        vr = repr(self.value)
        if len(vr) > 60:
            vr = vr[:57] + "..."
        return f"TaintedStr({vr}, labels={{{labels}}})"

    # -- Common str methods that propagate taint ------------------------------

    def upper(self) -> TaintedStr:
        return TaintedStr(
            value=self.value.upper(),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=self.char_taint,
        )

    def lower(self) -> TaintedStr:
        return TaintedStr(
            value=self.value.lower(),
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=self.char_taint,
        )

    def strip(self, chars: Optional[str] = None) -> TaintedStr:
        raw = self.value.strip(chars)
        # Find the offset of the stripped result
        start = self.value.index(raw) if raw else 0
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
        """Split and propagate taint to each fragment."""
        parts = self.value.split(sep, maxsplit)
        result: list[TaintedStr] = []
        offset = 0
        for part in parts:
            start = self.value.index(part, offset)
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
            offset = stop + (len(sep) if sep else 1)
        return result

    def replace(self, old: str, new: str, count: int = -1) -> TaintedStr:
        """Replace substrings — result inherits taint from self."""
        raw = self.value.replace(old, new, count)
        return TaintedStr(
            value=raw,
            sources=self.sources,
            readers=self.readers,
            dependencies=(self,),
            char_taint=CharTaint.uniform(len(raw), self.sources),
        )

    def join(self, iterable: Sequence[Union[str, TaintedStr]]) -> TaintedStr:
        """Join strings, merging taint from all elements."""
        parts: list[str] = []
        all_sources: set[Source] = set(self.sources)
        all_readers: set[Reader] = set(self.readers)
        all_deps: list[TaintedValue[Any]] = [self]
        for item in iterable:
            if isinstance(item, TaintedStr):
                parts.append(item.value)
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
        raw = self.value.join(parts)
        return TaintedStr(
            value=raw,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
            dependencies=tuple(all_deps),
            char_taint=CharTaint.uniform(len(raw), frozenset(all_sources)),
        )

    def format(self, *args: Any, **kwargs: Any) -> TaintedStr:
        """``str.format()`` with taint propagation from arguments."""
        unwrapped_args: list[Any] = []
        all_sources: set[Source] = set(self.sources)
        all_readers: set[Reader] = set(self.readers)
        all_deps: list[TaintedValue[Any]] = [self]
        for a in args:
            if isinstance(a, TaintedValue):
                unwrapped_args.append(a.value)
                all_sources.update(a.sources)
                all_readers.update(a.readers)
                all_deps.append(a)
            else:
                unwrapped_args.append(a)
        unwrapped_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, TaintedValue):
                unwrapped_kwargs[k] = v.value
                all_sources.update(v.sources)
                all_readers.update(v.readers)
                all_deps.append(v)
            else:
                unwrapped_kwargs[k] = v
        raw = self.value.format(*unwrapped_args, **unwrapped_kwargs)
        return TaintedStr(
            value=raw,
            sources=frozenset(all_sources),
            readers=frozenset(all_readers),
            dependencies=tuple(all_deps),
            char_taint=CharTaint.uniform(len(raw), frozenset(all_sources)),
        )

    def startswith(self, prefix: Union[str, TaintedStr], start: int = 0, end: Optional[int] = None) -> bool:
        p = prefix.value if isinstance(prefix, TaintedStr) else prefix
        return self.value.startswith(p, start, end if end is not None else len(self.value))

    def endswith(self, suffix: Union[str, TaintedStr], start: int = 0, end: Optional[int] = None) -> bool:
        s = suffix.value if isinstance(suffix, TaintedStr) else suffix
        return self.value.endswith(s, start, end if end is not None else len(self.value))


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
        return self.value[index]

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
        return self.value[key]

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
    if isinstance(value, TaintedValue):
        result.update(value.sources)
        if isinstance(value, (TaintedDict, TaintedList)):
            result.update(value.all_sources())
            # Also recurse into the inner unwrapped collection
            # to catch any nested TaintedValue items
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
