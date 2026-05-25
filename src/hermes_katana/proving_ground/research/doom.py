"""Doom-loop detector.

Documented failure mode in every published agent post-mortem (SWE-Agent,
Devin, OpenHands): the agent gets stuck re-attempting variations of the
same action against the same observation, burning tokens without progress.

Detection strategy: fingerprint the last N Actions (tool + salient args)
and count near-duplicates. If > threshold within a window, fire.

When the detector fires, the ResearchKernel enters a "drop a gear" mode:
- Switch to read-only actions for K steps
- Re-query the hypothesis DAG / event stream to replan
- Escalate to human if it fires twice within the same campaign

This detector is INTENTIONALLY cheap — it runs on every event, not on a
separate pass. False positives (e.g., legitimately re-running a failed
job) are okay because drop-a-gear isn't destructive; it just forces the
agent to re-plan before continuing.
"""

from __future__ import annotations

import hashlib
from collections import Counter, deque
from dataclasses import dataclass, field


def _fingerprint(tool: str, args: dict, salient_keys: tuple[str, ...] = ()) -> str:
    """Canonical hash of a tool call."""
    salient = {k: args.get(k) for k in salient_keys} if salient_keys else args
    payload = f"{tool}::{sorted(salient.items())}" if isinstance(salient, dict) else f"{tool}::{salient}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class DoomLoopDetector:
    """Fingerprints recent actions; fires when N duplicates in window W."""

    window: int = 20
    threshold: int = 5
    # Tools where args are large / noisy — hash only the keys that matter
    # for equivalence. Everything else uses the full args dict.
    salient_args: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "schedule_run": ("agent_id", "shard_id", "channel"),
            "query_results": ("run_id",),
            "fleet.launch": ("spec",),
        }
    )
    _recent: deque = field(default_factory=deque)

    def observe(self, tool: str, args: dict) -> None:
        fp = _fingerprint(tool, args, self.salient_args.get(tool, ()))
        self._recent.append(fp)
        while len(self._recent) > self.window:
            self._recent.popleft()

    def fires(self) -> tuple[bool, dict]:
        if not self._recent:
            return False, {}
        counts = Counter(self._recent)
        top, cnt = counts.most_common(1)[0]
        if cnt >= self.threshold:
            return True, {
                "window": self.window,
                "threshold": self.threshold,
                "fingerprint": top,
                "count": cnt,
                "recent": list(self._recent),
            }
        return False, {}

    def reset(self) -> None:
        self._recent.clear()
