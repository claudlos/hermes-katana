"""Shared timeout and concurrency helpers for optional remote judges."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar

REMOTE_JUDGE_TIMEOUT_ENV = "HERMES_KATANA_REMOTE_JUDGE_TIMEOUT"
REMOTE_JUDGE_MAX_CONCURRENCY_ENV = "HERMES_KATANA_REMOTE_JUDGE_MAX_CONCURRENCY"
DEFAULT_REMOTE_JUDGE_TIMEOUT = 0.5
DEFAULT_REMOTE_JUDGE_MAX_CONCURRENCY = 2

_T = TypeVar("_T")


def _read_positive_float_env(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0.0 else default


def _read_positive_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


REMOTE_JUDGE_MAX_CONCURRENCY = _read_positive_int_env(
    REMOTE_JUDGE_MAX_CONCURRENCY_ENV,
    DEFAULT_REMOTE_JUDGE_MAX_CONCURRENCY,
)
_REMOTE_JUDGE_EXECUTOR = ThreadPoolExecutor(
    max_workers=REMOTE_JUDGE_MAX_CONCURRENCY,
    thread_name_prefix="katana-remote-judge",
)
_REMOTE_JUDGE_GATE = threading.BoundedSemaphore(REMOTE_JUDGE_MAX_CONCURRENCY)


def validate_remote_judge_timeout(timeout: float | None) -> float:
    """Resolve and validate a remote-judge timeout budget."""
    resolved = (
        _read_positive_float_env(REMOTE_JUDGE_TIMEOUT_ENV, DEFAULT_REMOTE_JUDGE_TIMEOUT)
        if timeout is None
        else float(timeout)
    )
    if resolved <= 0.0:
        raise ValueError("remote judge timeout must be greater than 0")
    return resolved


async def run_in_limited_executor(
    operation_name: str,
    call: Callable[[], _T],
    *,
    timeout: float | None,
) -> _T:
    """Run a blocking remote-judge operation with bounded concurrency."""
    timeout_budget = validate_remote_judge_timeout(timeout)
    loop = asyncio.get_running_loop()
    acquired = await loop.run_in_executor(
        None,
        lambda: _REMOTE_JUDGE_GATE.acquire(timeout=timeout_budget),
    )
    if not acquired:
        raise TimeoutError(f"{operation_name} concurrency gate saturated after {timeout_budget:.3f}s")

    def _guarded_call() -> _T:
        try:
            return call()
        finally:
            _REMOTE_JUDGE_GATE.release()

    future = loop.run_in_executor(_REMOTE_JUDGE_EXECUTOR, _guarded_call)
    try:
        return await asyncio.wait_for(future, timeout=timeout_budget)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{operation_name} timed out after {timeout_budget:.3f}s") from exc
