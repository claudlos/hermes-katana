"""Bonsai 4B Judge — optional LLM-based security judgment for ambiguous cases.

Bonsai doesn't scan content directly. It reviews pre-digested risk reports
from the fast deterministic scanner stack and makes BLOCK/ALLOW/QUARANTINE
decisions. Only called when the risk score is ambiguous (0.3-0.7), which
is ~1-2% of all content in normal use.

The module degrades gracefully: if Bonsai is unavailable, the scanner
verdict stands as-is. No hard dependency on the model server.

Architecture reference: research/catch-everything-brainstorm/papers/
  hermes-scabbard-brainstorm-v2.md (Idea 2)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

from hermes_katana.scanner.judge_runtime import (
    run_in_limited_executor,
    validate_remote_judge_timeout,
)
from hermes_katana.security_logging import log_security_event

logger = logging.getLogger(__name__)

# Default Bonsai endpoint (local inference server)
BONSAI_URL = "http://localhost:8080/v1/chat/completions"
BONSAI_MODEL = "bonsai-4b"

# System prompt for Bonsai judge
_SYSTEM_PROMPT = """\
You are a security judge. You receive a structured risk report from \
deterministic scanners that flagged suspicious content. Your job is to \
decide whether the flagged content is actually malicious or a false positive.

Respond with EXACTLY one JSON object (no markdown, no explanation outside JSON):
{"decision": "block"|"allow"|"quarantine", "reasoning": "<1-2 sentences>", "confidence": 0.0-1.0}

Guidelines:
- "block": clear malicious intent (prompt injection, jailbreak, exfiltration)
- "allow": false positive or benign content that triggered scanner heuristics
- "quarantine": ambiguous — flag for human review
- confidence: how sure you are (0.0 = guessing, 1.0 = certain)
"""


@dataclass(frozen=True, slots=True)
class BonsaiJudgment:
    """Result from the Bonsai 4B judge."""

    decision: str  # "block", "allow", "quarantine"
    reasoning: str
    confidence: float
    model_available: bool


# Scabbard for when model is unavailable
_UNAVAILABLE = BonsaiJudgment(
    decision="quarantine",
    reasoning="Bonsai model unavailable — falling back to scanner verdict",
    confidence=0.0,
    model_available=False,
)


def _build_prompt(risk_report: dict[str, Any]) -> str:
    """Build a compact user prompt from the risk report."""
    return f"Risk report:\n```json\n{json.dumps(risk_report, indent=2)}\n```\nDecision?"


def _parse_response(text: str) -> dict[str, Any]:
    """Parse Bonsai's JSON response, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return cast(dict[str, Any], json.loads(text))


def _call_bonsai(
    risk_report: dict[str, Any],
    *,
    url: str = BONSAI_URL,
    model: str = BONSAI_MODEL,
    timeout: float | None = None,
) -> BonsaiJudgment:
    """Make a synchronous HTTP call to the Bonsai inference server."""
    timeout = validate_remote_judge_timeout(timeout)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(risk_report)},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
    }

    req = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except (URLError, OSError, TimeoutError) as exc:
        logger.debug("Bonsai call failed: %s", exc)
        log_security_event(
            logger,
            logging.WARNING,
            "remote_judge_unavailable",
            judge_name="bonsai",
            endpoint=url,
            model=model,
            timeout_seconds=timeout,
            error_type=exc.__class__.__name__,
            reason=str(exc) or exc.__class__.__name__,
        )
        return _UNAVAILABLE
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.debug("Bonsai response decode error: %s", exc)
        log_security_event(
            logger,
            logging.WARNING,
            "remote_judge_response_invalid",
            judge_name="bonsai",
            endpoint=url,
            model=model,
            timeout_seconds=timeout,
            error_type=exc.__class__.__name__,
            reason=str(exc) or exc.__class__.__name__,
        )
        return BonsaiJudgment(
            decision="quarantine",
            reasoning=f"Failed to decode Bonsai response: {exc}",
            confidence=0.0,
            model_available=True,
        )

    # Extract the assistant message
    try:
        content = body["choices"][0]["message"]["content"]
        parsed = _parse_response(content)
        decision = parsed.get("decision", "quarantine")
        if decision not in ("block", "allow", "quarantine"):
            decision = "quarantine"
        return BonsaiJudgment(
            decision=decision,
            reasoning=parsed.get("reasoning", ""),
            confidence=float(parsed.get("confidence", 0.5)),
            model_available=True,
        )
    except (KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
        logger.debug("Bonsai response parse error: %s", exc)
        log_security_event(
            logger,
            logging.WARNING,
            "remote_judge_response_invalid",
            judge_name="bonsai",
            endpoint=url,
            model=model,
            timeout_seconds=timeout,
            error_type=exc.__class__.__name__,
            reason=str(exc) or exc.__class__.__name__,
        )
        return BonsaiJudgment(
            decision="quarantine",
            reasoning=f"Failed to parse Bonsai response: {exc}",
            confidence=0.0,
            model_available=True,
        )


def judge_with_bonsai_sync(
    risk_report: dict[str, Any],
    *,
    timeout: float | None = None,
    url: str = BONSAI_URL,
    model: str = BONSAI_MODEL,
) -> BonsaiJudgment:
    """Synchronous Bonsai judgment.

    Args:
        risk_report: Pre-digested risk summary from fast scanner stack.
        timeout: HTTP timeout in seconds (default: 0.5s).
        url: Bonsai inference server URL.
        model: Model name to request.

    Returns:
        BonsaiJudgment with decision, reasoning, confidence, and availability.
    """
    return _call_bonsai(risk_report, url=url, model=model, timeout=timeout)


async def judge_with_bonsai(
    risk_report: dict[str, Any],
    *,
    timeout: float | None = None,
    url: str = BONSAI_URL,
    model: str = BONSAI_MODEL,
) -> BonsaiJudgment:
    """Async Bonsai judgment — runs sync call in a thread executor.

    Args:
        risk_report: Pre-digested risk summary from fast scanner stack.
        timeout: HTTP timeout in seconds (default: 0.5s).
        url: Bonsai inference server URL.
        model: Model name to request.

    Returns:
        BonsaiJudgment with decision, reasoning, confidence, and availability.
    """
    try:
        return await run_in_limited_executor(
            "bonsai",
            lambda: _call_bonsai(risk_report, url=url, model=model, timeout=timeout),
            timeout=timeout,
        )
    except TimeoutError as exc:
        timeout_budget = validate_remote_judge_timeout(timeout)
        log_security_event(
            logger,
            logging.WARNING,
            "remote_judge_timeout",
            judge_name="bonsai",
            endpoint=url,
            model=model,
            timeout_seconds=timeout_budget,
            error_type=exc.__class__.__name__,
            reason=str(exc),
        )
        return _UNAVAILABLE
