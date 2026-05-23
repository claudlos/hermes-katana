"""Budget ledger — first-class cost tokens for the research agent.

Unconstrained ReAct loops consume budget unpredictably (AI Scientist v2
post-mortem explicitly flagged this). Our agent plans against explicit
budgets and hard-stops when any bucket empties.

Buckets tracked:

  claude_max    — Claude Max rolling-window utilization (0-1.0 fraction).
                  Observed but not directly chargeable — we estimate from
                  elapsed wall-time × concurrent CLI workers + a scaling
                  factor; real number comes from `ccusage` / user report.
  openai_usd    — dollar balance against a user-set cap.
  anthropic_usd — dollar balance against a user-set cap (for direct
                  Anthropic API use when NOT using Max).
  gpu_thermal   — peak package temperature (°C) across the last poll
                  window. Hard stop beyond a safety ceiling.
  disk_mb       — results/ directory size, hard-cap to catch runaway
                  shard-run writes.

Agent semantics:
  - `charge(bucket, amount)` decrements a bucket (or increments for
    cumulative buckets like disk_mb).
  - `query(bucket)` returns current state.
  - `exhausted()` returns a list of exhausted buckets (for step-level check).
  - `snapshot()` emits a dict for BudgetTick event.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Bucket:
    name: str
    limit: float  # hard cap (interpreted by bucket)
    used: float = 0.0  # cumulative for usage buckets
    # Optional live-observation hooks for non-chargeable buckets (thermals).
    live_value: float | None = None


class BudgetLedger:
    def __init__(
        self,
        *,
        claude_max_limit: float = 0.60,  # stop at 60% of 5h window
        openai_usd_limit: float = 10.0,
        anthropic_usd_limit: float = 0.0,  # 0 = no API spend allowed
        gpu_thermal_ceiling_c: float = 85.0,
        disk_mb_limit: float = 20_000.0,  # 20 GB soft cap on results/
    ):
        self.buckets: dict[str, Bucket] = {
            "claude_max": Bucket("claude_max", claude_max_limit),
            "openai_usd": Bucket("openai_usd", openai_usd_limit),
            "anthropic_usd": Bucket("anthropic_usd", anthropic_usd_limit),
            "gpu_thermal": Bucket("gpu_thermal", gpu_thermal_ceiling_c),
            "disk_mb": Bucket("disk_mb", disk_mb_limit),
        }
        self._last_update = time.time()

    # ---------------------------------------------------------------- writes
    def charge(self, bucket_name: str, amount: float) -> None:
        b = self.buckets[bucket_name]
        b.used += amount
        self._last_update = time.time()

    def observe_live(self, bucket_name: str, value: float) -> None:
        """Set a live reading (for non-chargeable, non-cumulative buckets)."""
        self.buckets[bucket_name].live_value = value
        self._last_update = time.time()

    def set_limit(self, bucket_name: str, new_limit: float) -> None:
        self.buckets[bucket_name].limit = new_limit

    # ---------------------------------------------------------------- reads
    def query(self, bucket_name: str) -> dict:
        b = self.buckets[bucket_name]
        remaining = max(0.0, b.limit - b.used)
        return {
            "limit": b.limit,
            "used": b.used,
            "live": b.live_value,
            "remaining": remaining,
            "fraction_used": (b.used / b.limit) if b.limit else 0.0,
        }

    def exhausted(self) -> list[str]:
        out: list[str] = []
        for name, b in self.buckets.items():
            if name == "gpu_thermal":
                # Thermal is live, not cumulative: exhausted if live > ceiling.
                if b.live_value is not None and b.live_value > b.limit:
                    out.append(name)
                continue
            if b.used >= b.limit:
                out.append(name)
        return out

    def snapshot(self) -> dict:
        return {name: self.query(name) for name in self.buckets}

    # ---------------------------------------------------------------- cost estimators
    @staticmethod
    def estimate_openai_batch_cost(
        n_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        input_usd_per_m: float = 0.20,
        output_usd_per_m: float = 0.80,
    ) -> float:
        """gpt-4.1-mini batch pricing as default. Override for other models.

        Batch API is 50% of standard pricing; defaults reflect that.
        """
        m = 1_000_000.0
        return (
            n_requests * avg_input_tokens / m * input_usd_per_m + n_requests * avg_output_tokens / m * output_usd_per_m
        )

    @staticmethod
    def estimate_claude_max_usage(
        n_cli_workers: int,
        wall_seconds: float,
        sec_per_full_cycle: float = 18_000.0,  # 5h rolling window
    ) -> float:
        """Very rough: assume N concurrent workers burn ~ N * wall / window.

        User can (and should) override with real numbers from `ccusage` or
        the claude UI telemetry.
        """
        return n_cli_workers * wall_seconds / sec_per_full_cycle
