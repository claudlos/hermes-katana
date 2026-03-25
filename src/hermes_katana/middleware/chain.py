"""
Middleware chain for HermesKatana tool-dispatch interception.

This module defines the core middleware abstraction used to intercept every
tool call in the Hermes agent runtime.  Middleware instances are composed
into an ordered :class:`MiddlewareChain` that evaluates pre-dispatch and
post-dispatch hooks with short-circuit semantics.

Architecture
------------
::

    ┌────────────┐   pre_dispatch()   ┌───────────────┐
    │  Incoming  │──────────────────▸│  Middleware 1  │─┐
    │  Tool Call │                    └───────────────┘ │
    └────────────┘                    ┌───────────────┐ │
                                      │  Middleware 2  │◂┘  if ALLOW
                                      └───────────────┘ │
                                      ┌───────────────┐ │
                                      │  Middleware N  │◂┘  if ALLOW
                                      └───────────────┘
                                              │
                                        ──── ALLOW ───▸  execute tool
                                        ──── DENY  ───▸  reject

After tool execution, ``post_dispatch()`` runs in *reverse* order so that
the outermost middleware can finalize audit entries with full context.

Thread safety: the chain is protected by a threading lock so middleware
can be added/removed while calls are in flight (hot-reload support).
"""

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dispatch decision
# ---------------------------------------------------------------------------


class DispatchDecision(str, enum.Enum):
    """Outcome of a middleware pre-dispatch evaluation.

    ALLOW    — the call may proceed to the next middleware (or execution).
    DENY     — the call is blocked; short-circuits the chain immediately.
    ESCALATE — the call requires human approval before proceeding.
    """

    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Call context — metadata bag threaded through the chain
# ---------------------------------------------------------------------------


class CallContext(BaseModel):
    """Metadata bag passed through every middleware in the chain.

    Carries information about the current tool call, accumulated decisions,
    and a free-form ``extras`` dict for inter-middleware communication.

    Attributes:
        call_id:        Unique identifier for this dispatch (auto-generated).
        tool_name:      The Hermes tool being invoked (e.g. ``terminal``).
        args:           The tool's keyword arguments.
        taint_context:  Taint metadata dict (same schema as PolicyEngine expects).
        decision:       Current accumulated decision (starts as ALLOW).
        deny_reasons:   List of reasons if any middleware issued DENY.
        escalate_reasons: List of reasons if any middleware issued ESCALATE.
        scan_results:   Scanner results attached by scan middleware.
        policy_result:  Policy evaluation result, if any.
        extras:         Free-form dict for inter-middleware data passing.
        timestamps:     Ordered list of (middleware_name, elapsed_ms) tuples.
        created_at:     Unix timestamp of context creation.
    """

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    taint_context: dict[str, Any] = Field(default_factory=dict)
    decision: DispatchDecision = DispatchDecision.ALLOW
    deny_reasons: list[str] = Field(default_factory=list)
    escalate_reasons: list[str] = Field(default_factory=list)
    scan_results: list[Any] = Field(default_factory=list)
    policy_result: Optional[Any] = None
    extras: dict[str, Any] = Field(default_factory=dict)
    timestamps: list[tuple[str, float]] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)

    # -- Tool output (populated after execution, before post_dispatch) ------
    tool_output: Optional[Any] = None
    tool_error: Optional[str] = None
    tool_duration_ms: float = 0.0

    model_config = {"arbitrary_types_allowed": True}

    def deny(self, reason: str) -> None:
        """Mark this context as DENY with a reason.

        Once denied, the decision cannot be downgraded back to ALLOW.
        """
        self.decision = DispatchDecision.DENY
        self.deny_reasons.append(reason)

    def escalate(self, reason: str) -> None:
        """Mark this context as ESCALATE with a reason.

        Only takes effect if the current decision is ALLOW (DENY overrides).
        """
        if self.decision != DispatchDecision.DENY:
            self.decision = DispatchDecision.ESCALATE
            self.escalate_reasons.append(reason)

    @property
    def is_denied(self) -> bool:
        """Whether the call has been denied by any middleware."""
        return self.decision == DispatchDecision.DENY

    @property
    def is_escalated(self) -> bool:
        """Whether the call requires human escalation."""
        return self.decision == DispatchDecision.ESCALATE

    @property
    def total_middleware_ms(self) -> float:
        """Total time spent in middleware (milliseconds)."""
        return sum(ms for _, ms in self.timestamps)


# ---------------------------------------------------------------------------
# Abstract middleware base class
# ---------------------------------------------------------------------------


class KatanaMiddleware(ABC):
    """Abstract base class for all Katana middleware.

    Subclasses must implement :meth:`pre_dispatch` and may optionally
    override :meth:`post_dispatch`.  Each middleware receives a mutable
    :class:`CallContext` and should update it in-place.

    Attributes:
        name:     Human-readable name for logging and metrics.
        enabled:  Toggle — disabled middleware is skipped by the chain.
        priority: Ordering hint (higher = runs first in pre-dispatch).
    """

    def __init__(self, name: str, *, enabled: bool = True, priority: int = 0) -> None:
        self.name = name
        self.enabled = enabled
        self.priority = priority

    @abstractmethod
    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Inspect / gate a tool call before execution.

        Implementations should:
        1. Examine ``ctx.tool_name``, ``ctx.args``, ``ctx.taint_context``.
        2. Update ``ctx`` (add scan results, taint info, deny reasons, etc.).
        3. Return ALLOW, DENY, or ESCALATE.

        If DENY is returned, the chain short-circuits and no further
        middleware or tool execution occurs.

        Args:
            ctx: The mutable call context.

        Returns:
            The dispatch decision for this middleware.
        """
        ...

    def post_dispatch(self, ctx: CallContext) -> None:
        """Process results after tool execution (or denial).

        Called in *reverse* chain order so the outermost middleware
        sees the fully-populated context.  The default implementation
        is a no-op.

        Args:
            ctx: The call context, now including ``tool_output`` / ``tool_error``.
        """

    def on_short_circuit(self, ctx: CallContext) -> None:
        """Observe a pre-dispatch short-circuit decision.

        Called when another middleware denies the call before this middleware
        would normally run in ``pre_dispatch``. The default implementation is
        a no-op; observer-style middleware such as audit can override it.

        Args:
            ctx: The denied call context.
        """

    def __repr__(self) -> str:
        state = "enabled" if self.enabled else "disabled"
        return f"<{self.__class__.__name__}({self.name!r}, {state}, pri={self.priority})>"


# ---------------------------------------------------------------------------
# Middleware chain
# ---------------------------------------------------------------------------


class MiddlewareChain:
    """Ordered collection of middleware with pre/post dispatch execution.

    The chain executes ``pre_dispatch()`` on each middleware in priority
    order (highest first).  If any middleware returns DENY, the chain
    short-circuits immediately.  After tool execution, ``post_dispatch()``
    runs in reverse order.

    Thread safety: all mutations and reads are protected by a reentrant lock.

    Usage::

        chain = MiddlewareChain()
        chain.add(taint_middleware)
        chain.add(scan_middleware)
        chain.add(policy_middleware)
        chain.add(audit_middleware)

        ctx = CallContext(tool_name="terminal", args={"command": "ls"})
        decision = chain.execute_pre(ctx)

        if decision == DispatchDecision.ALLOW:
            result = run_tool(ctx.tool_name, ctx.args)
            ctx.tool_output = result
            chain.execute_post(ctx)
    """

    def __init__(self) -> None:
        self._middleware: list[KatanaMiddleware] = []
        self._lock = threading.RLock()

    # -- Mutation -----------------------------------------------------------

    def add(self, mw: KatanaMiddleware) -> None:
        """Add a middleware to the chain.

        The chain is re-sorted by priority (descending) after each addition.
        If a middleware with the same name already exists, it is replaced.

        Args:
            mw: The middleware instance to add.
        """
        with self._lock:
            self._middleware = [m for m in self._middleware if m.name != mw.name]
            self._middleware.append(mw)
            self._middleware.sort(key=lambda m: m.priority, reverse=True)
        logger.debug("Added middleware %r to chain (total=%d)", mw.name, len(self._middleware))

    def remove(self, name: str) -> bool:
        """Remove a middleware by name.

        Args:
            name: The middleware name to remove.

        Returns:
            True if a middleware was found and removed.
        """
        with self._lock:
            before = len(self._middleware)
            self._middleware = [m for m in self._middleware if m.name != name]
            removed = len(self._middleware) < before
        if removed:
            logger.debug("Removed middleware %r from chain", name)
        return removed

    def clear(self) -> None:
        """Remove all middleware from the chain."""
        with self._lock:
            self._middleware.clear()

    # -- Queries ------------------------------------------------------------

    def list_middleware(self) -> list[KatanaMiddleware]:
        """Return a copy of all middleware, sorted by priority."""
        with self._lock:
            return list(self._middleware)

    def __len__(self) -> int:
        with self._lock:
            return len(self._middleware)

    def __bool__(self) -> bool:
        with self._lock:
            return bool(self._middleware)

    # -- Execution ----------------------------------------------------------

    def execute_pre(self, ctx: CallContext) -> DispatchDecision:
        """Run pre-dispatch hooks on all middleware.

        Executes each enabled middleware's ``pre_dispatch()`` in priority
        order.  Short-circuits on the first DENY.  ESCALATE is sticky
        but does not short-circuit (later middleware may upgrade to DENY).

        Args:
            ctx: The mutable call context.

        Returns:
            The final accumulated decision (ALLOW, DENY, or ESCALATE).
        """
        with self._lock:
            chain = list(self._middleware)

        for mw in chain:
            if not mw.enabled:
                continue

            start = time.monotonic()
            try:
                decision = mw.pre_dispatch(ctx)
            except Exception:
                logger.exception("Middleware %r raised during pre_dispatch", mw.name)
                # Fail-safe: deny on middleware error
                ctx.deny(f"Middleware '{mw.name}' raised an exception (fail-safe DENY)")
                decision = DispatchDecision.DENY

            elapsed_ms = (time.monotonic() - start) * 1000
            ctx.timestamps.append((mw.name, elapsed_ms))

            if decision == DispatchDecision.DENY:
                logger.info(
                    "Middleware %r denied call to %r: %s",
                    mw.name,
                    ctx.tool_name,
                    ctx.deny_reasons[-1] if ctx.deny_reasons else "no reason",
                )
                ctx.decision = DispatchDecision.DENY
                ctx.extras["short_circuit_middleware"] = mw.name
                self._notify_short_circuit(ctx, chain)
                return DispatchDecision.DENY

            if decision == DispatchDecision.ESCALATE:
                ctx.decision = DispatchDecision.ESCALATE

        return ctx.decision

    def _notify_short_circuit(
        self,
        ctx: CallContext,
        chain: list[KatanaMiddleware],
    ) -> None:
        """Notify middleware that a pre-dispatch denial has short-circuited."""
        ctx.extras["short_circuit_notified"] = True

        for mw in reversed(chain):
            if not mw.enabled:
                continue

            start = time.monotonic()
            try:
                mw.on_short_circuit(ctx)
            except Exception:
                logger.exception("Middleware %r raised during on_short_circuit", mw.name)

            elapsed_ms = (time.monotonic() - start) * 1000
            ctx.timestamps.append((f"{mw.name}:short", elapsed_ms))

    def execute_post(self, ctx: CallContext) -> None:
        """Run post-dispatch hooks on all middleware in reverse order.

        Called after tool execution (or denial).  Each middleware sees
        the fully-populated context including ``tool_output`` / ``tool_error``.

        Args:
            ctx: The call context with execution results.
        """
        with self._lock:
            chain = list(reversed(self._middleware))

        for mw in chain:
            if not mw.enabled:
                continue

            start = time.monotonic()
            try:
                mw.post_dispatch(ctx)
            except Exception:
                logger.exception("Middleware %r raised during post_dispatch", mw.name)

            elapsed_ms = (time.monotonic() - start) * 1000
            ctx.timestamps.append((f"{mw.name}:post", elapsed_ms))

    def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        taint_context: dict[str, Any] | None = None,
        *,
        tool_executor: Any | None = None,
    ) -> CallContext:
        """Full lifecycle: pre-dispatch → execute → post-dispatch.

        Convenience method that creates a :class:`CallContext`, runs the
        full chain, optionally executes the tool, and returns the final
        context.

        Args:
            tool_name:      The tool being invoked.
            args:           Tool arguments.
            taint_context:  Taint metadata (optional).
            tool_executor:  Optional callable ``(tool_name, args) -> result``.
                            If None, no tool execution occurs.

        Returns:
            The fully-populated :class:`CallContext`.
        """
        ctx = CallContext(
            tool_name=tool_name,
            args=args,
            taint_context=taint_context or {},
        )

        # Pre-dispatch
        decision = self.execute_pre(ctx)

        # Execute tool if allowed
        if decision == DispatchDecision.ALLOW and tool_executor is not None:
            start = time.monotonic()
            try:
                ctx.tool_output = tool_executor(tool_name, args)
            except Exception as exc:
                ctx.tool_error = str(exc)
                logger.error("Tool %r raised: %s", tool_name, exc)
            ctx.tool_duration_ms = (time.monotonic() - start) * 1000

        # Post-dispatch
        self.execute_post(ctx)

        return ctx

    def __repr__(self) -> str:
        with self._lock:
            names = [m.name for m in self._middleware]
        return f"MiddlewareChain({names})"
