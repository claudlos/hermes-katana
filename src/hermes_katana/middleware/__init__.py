"""
HermesKatana Middleware — pluggable middleware chain for Hermes tool dispatch.

The middleware module provides an ordered chain of interceptors that wrap
every Hermes tool call.  Each middleware can inspect / modify the call
context, block dangerous calls, or log audit events.

Core types
----------
- :class:`KatanaMiddleware`  — abstract base class for all middleware
- :class:`MiddlewareChain`   — ordered executor with short-circuit on DENY
- :class:`CallContext`       — metadata bag passed through the chain
- :class:`DispatchDecision`  — ALLOW / DENY / ESCALATE

Quick start::

    from hermes_katana.middleware import MiddlewareChain, CallContext

    chain = MiddlewareChain()
    chain.add(my_taint_middleware)
    chain.add(my_scan_middleware)

    ctx = CallContext(tool_name="terminal", args={"command": "ls"})
    decision = chain.execute_pre(ctx)
"""

from hermes_katana.middleware.chain import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
    MiddlewareChain,
)

__all__ = [
    "KatanaMiddleware",
    "MiddlewareChain",
    "CallContext",
    "DispatchDecision",
]
