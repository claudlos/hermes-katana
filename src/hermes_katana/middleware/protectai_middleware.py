"""KatanaProtectAIMiddleware — ProtectAI binary gate middleware (priority=88).

Sits between the primary Scabbard middleware (pri=90) and the secondary
Scabbard middleware (pri=85) in the dispatch chain.

Behaviour:
- INJECTION with confidence > block_threshold  → DENY
- INJECTION with confidence > flag_threshold   → ESCALATE (flag for review)
- SAFE with confidence > safe_passthrough      → ALLOW (skip remaining checks)
- Otherwise                                    → ALLOW (let downstream decide)

The gate stores its result in ``ctx.extras["protectai_result"]`` so downstream
middleware can read it without re-running the model.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from hermes_katana.middleware.chain import (
    CallContext,
    DispatchDecision,
    KatanaMiddleware,
)

logger = logging.getLogger(__name__)

# Default thresholds
_DEFAULT_BLOCK_THRESHOLD = 0.92
_DEFAULT_FLAG_THRESHOLD = 0.70
_DEFAULT_SAFE_PASSTHROUGH = 0.95


class KatanaProtectAIMiddleware(KatanaMiddleware):
    """ProtectAI binary injection gate middleware.

    Runs the ProtectAI DeBERTa model on every tool call's string arguments
    before Tier 2 (Scabbard classifier) runs.

    Args:
        gate:             Optional :class:`ProtectAIGate` instance.
                          If not provided, one is lazily constructed.
        block_threshold:  INJECTION confidence above which to DENY outright.
        flag_threshold:   INJECTION confidence above which to ESCALATE.
        safe_passthrough: SAFE confidence above which to ALLOW fast-path.
        enabled:          Toggle the middleware on/off.
    """

    def __init__(
        self,
        gate: Optional[Any] = None,
        *,
        block_threshold: float = _DEFAULT_BLOCK_THRESHOLD,
        flag_threshold: float = _DEFAULT_FLAG_THRESHOLD,
        safe_passthrough: float = _DEFAULT_SAFE_PASSTHROUGH,
        enabled: bool = True,
    ) -> None:
        super().__init__(name="katana.protectai", enabled=enabled, priority=88)
        self._gate = gate
        self.block_threshold = block_threshold
        self.flag_threshold = flag_threshold
        self.safe_passthrough = safe_passthrough

    @property
    def gate(self) -> Any:
        """Lazy-load the ProtectAI gate to avoid import cost at wire time."""
        if self._gate is None:
            from hermes_katana.scanner.protectai_gate import ProtectAIGate

            self._gate = ProtectAIGate()
        return self._gate

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(ctx: CallContext) -> str:
        """Extract a single string to scan from the call context.

        Scans all top-level string args and joins them.  Limits the combined
        text to 4 000 characters to avoid unnecessary model overhead.
        """
        parts: list[str] = []
        for val in ctx.args.values():
            if isinstance(val, str):
                parts.append(val)
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, str):
                        parts.append(item)
        combined = " ".join(parts)
        return combined[:4000]

    # ------------------------------------------------------------------
    # Pre-dispatch
    # ------------------------------------------------------------------

    def pre_dispatch(self, ctx: CallContext) -> DispatchDecision:
        """Run the ProtectAI gate and gate the call.

        Stores the :class:`ProtectAIResult` in ``ctx.extras["protectai_result"]``
        regardless of decision so downstream middleware can read it.
        """
        text = self._extract_text(ctx)
        if not text.strip():
            return DispatchDecision.ALLOW

        try:
            result = self.gate.scan(text)
        except Exception:  # noqa: BLE001
            logger.warning("KatanaProtectAIMiddleware: gate.scan() raised", exc_info=True)
            return DispatchDecision.ALLOW

        ctx.extras["protectai_result"] = result

        if not result.model_available:
            # Gate is in stub mode — don't make decisions without a real model
            return DispatchDecision.ALLOW

        if result.is_injection:
            if result.confidence >= self.block_threshold:
                ctx.deny(
                    f"ProtectAI gate: INJECTION detected (confidence={result.confidence:.3f} "
                    f">= block_threshold={self.block_threshold})"
                )
                return DispatchDecision.DENY

            if result.confidence >= self.flag_threshold:
                ctx.escalate(
                    f"ProtectAI gate: possible INJECTION (confidence={result.confidence:.3f} "
                    f">= flag_threshold={self.flag_threshold})"
                )
                return DispatchDecision.ESCALATE

        elif result.confidence >= self.safe_passthrough:
            # Very confident it is safe — mark fast-path so downstream
            # middleware can optionally skip expensive checks.
            ctx.extras["protectai_safe_passthrough"] = True

        return DispatchDecision.ALLOW
