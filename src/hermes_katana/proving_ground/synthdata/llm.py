"""Thin LLM wrapper used by all Simula steps.

Single choice-point for which provider/model runs each role (teacher,
critic_a, critic_b). Supports:

  anthropic     — Anthropic SDK, billed against ANTHROPIC_API_KEY
  claude_cli    — shells out to the `claude` CLI, billed against
                  Claude Max / the logged-in account. No API key used.
                  Slower per call (~5-15 s overhead) but zero API spend.
  codex_cli     — shells out to the `codex` CLI, billed against the
                  logged-in ChatGPT/Codex subscription. Same benefits
                  as claude_cli for OpenAI-family models.
  hermes_cli    — shells out to `hermes chat`, billed against whatever
                  inference provider Hermes is configured to use
                  (MiniMax direct, Nous, OpenRouter, etc.). Useful when
                  you have a Hermes subscription / pool credential and
                  want to avoid Codex/Claude quota.
  openai        — OpenAI chat completions
  openrouter    — OpenAI SDK pointed at OpenRouter
  nous          — OpenAI SDK pointed at Nous Portal

No litellm — we want zero extra deps so the pipeline runs on a Colab
notebook without wrestling with installs.
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Any

# Provider keys that downstream CLI drivers genuinely need. Everything else
# matching the sensitive-name pattern is dropped from the subprocess env so
# unrelated ambient credentials (AWS_*, GITHUB_TOKEN, GPG_PASSPHRASE, etc.)
# don't leak into the spawned LLM CLI.
_ALLOWED_PROVIDER_SECRET_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "MINIMAX_API_KEY",
    "NOUS_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "XIAOMI_API_KEY",
}
_SENSITIVE_ENV_NAME_RE = re.compile(
    r"(^|_)(?:API_KEY|ACCESS_KEY|KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE|AUTH)(_|$)",
    re.IGNORECASE,
)


def _scrubbed_subprocess_env(*, drop: tuple[str, ...] = ()) -> dict[str, str]:
    """Build a subprocess env that drops unrelated ambient credentials.

    Keeps PATH / locale / and the explicit provider keys above; drops anything
    else whose name matches the sensitive-name pattern. Additional keys to
    remove can be passed via ``drop`` (e.g. ANTHROPIC_API_KEY for the Claude
    Max driver, where the env var would route the CLI to API auth and bypass
    the OAuth subscription).
    """
    env = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if upper in drop:
            continue
        if upper in _ALLOWED_PROVIDER_SECRET_ENV_KEYS:
            env[k] = v
            continue
        if _SENSITIVE_ENV_NAME_RE.search(upper):
            continue
        env[k] = v
    return env


@dataclass
class LLMConfig:
    model: str
    max_tokens: int = 2000
    temperature: float = 0.7
    provider: str = "anthropic"  # anthropic | openrouter | nous | openai
    # Populated from env if empty:
    api_key: str = ""
    base_url: str = ""


class LLMClient:
    """Blocking, retrying LLM client. Not async — Simula stages are
    trivially parallelized at the pool-level in `run.py` via
    multiprocessing if we need throughput."""

    SDK_TIMEOUT_SEC = 60.0

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._sdk: Any = None
        if cfg.provider == "anthropic":
            from anthropic import Anthropic

            self._sdk = Anthropic(
                api_key=cfg.api_key or os.environ.get("ANTHROPIC_API_KEY"),
                timeout=self.SDK_TIMEOUT_SEC,
            )
        elif cfg.provider in ("claude_cli", "codex_cli", "hermes_cli"):
            # No SDK; we shell out per call in _raw_complete.
            self._sdk = None
        elif cfg.provider in ("openrouter", "nous", "openai"):
            import openai

            base_url = cfg.base_url or _default_base_url(cfg.provider)
            api_key = cfg.api_key or _default_api_key(cfg.provider)
            self._sdk = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=self.SDK_TIMEOUT_SEC)
        else:
            raise ValueError(f"unknown provider: {cfg.provider}")

    def complete(
        self,
        system: str,
        user: str,
        *,
        stop_sequences: list[str] | None = None,
        expect_json: bool = False,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> str:
        """Return the assistant message as a string.

        If `expect_json`, assert the response parses as JSON and retry
        with a firmer instruction on failure.
        """
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                text = self._raw_complete(system, user, stop_sequences=stop_sequences)
                if expect_json:
                    # Validate via the same fence-tolerant parser callers use.
                    parse_json_block(text)
                return text
            except Exception as e:  # broad by design — includes JSONDecodeError + HTTPError
                last_err = e
                wait = backoff_base**attempt + random.random()
                time.sleep(wait)
                if expect_json:
                    user = user + "\n\nIMPORTANT: return ONLY valid JSON, no prose or markdown fences."
        raise RuntimeError(f"LLM complete failed after {max_retries} attempts: {last_err}")

    def _raw_complete(self, system: str, user: str, *, stop_sequences: list[str] | None) -> str:
        if self.cfg.provider == "claude_cli":
            return _claude_cli_complete(self.cfg, system, user)
        if self.cfg.provider == "codex_cli":
            return _codex_cli_complete(self.cfg, system, user)
        if self.cfg.provider == "hermes_cli":
            return _hermes_cli_complete(self.cfg, system, user)
        if self.cfg.provider == "anthropic":
            resp = self._sdk.messages.create(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
                stop_sequences=stop_sequences or [],
            )
            # Anthropic returns a list of content blocks; concat text blocks
            parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "".join(parts).strip()
        # OpenAI-style
        resp = self._sdk.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stop=stop_sequences or None,
        )
        return (resp.choices[0].message.content or "").strip()


def _claude_cli_complete(cfg: LLMConfig, system: str, user: str) -> str:
    """Run one completion via the `claude` CLI (Claude Max auth).

    Uses --print mode with --output-format=json. Model is passed via
    `--model` (cfg.model — alias like 'opus' or full 'claude-opus-4-7').
    No workspace state, no hooks, no CLAUDE.md — we pass the system
    prompt with `--system-prompt` and the user message via stdin.

    IMPORTANT: ANTHROPIC_API_KEY is removed from the subprocess env so
    the CLI is forced to use the OAuth/Max session. If the env var is
    set, Claude Code prefers API auth and bills against the key (which
    can be $0-balance), bypassing Max entirely.
    """
    env = _scrubbed_subprocess_env(drop=("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN"))
    cmd = [
        "claude",
        "-p",  # print-and-exit mode
        "--model",
        cfg.model,
        "--system-prompt",
        system,
        "--output-format",
        "json",
        "--no-session-persistence",
    ]
    proc = subprocess.run(
        cmd,
        input=user,
        text=True,
        capture_output=True,
        timeout=600,
        env=env,
    )
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"claude CLI failed rc={proc.returncode}: {proc.stderr[:500]}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"claude CLI non-JSON output: {proc.stdout[:500]}") from e
    if payload.get("is_error"):
        raise RuntimeError(f"claude CLI API error: {payload.get('result', '')[:500]}")
    return str(payload.get("result", "")).strip()


def _codex_cli_complete(cfg: LLMConfig, system: str, user: str) -> str:
    """Run one completion via the `codex` CLI (OpenAI Codex subscription).

    codex exec has no `--system-prompt` flag, so we prepend the system
    instructions inline with a clear separator. We read the single
    assistant response via --output-last-message to a tmpfile (cleanest
    path; alternative is stream-json on stdout which is noisier).
    """
    import tempfile

    combined = f"System:\n{system}\n\nUser:\n{user}"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        outpath = tf.name
    cmd = [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--model",
        cfg.model,
        "-s",
        "read-only",
        "--output-last-message",
        outpath,
        "-",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=combined,
            text=True,
            capture_output=True,
            timeout=600,
            env=_scrubbed_subprocess_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"codex CLI failed rc={proc.returncode}: {proc.stderr[:500]}")
        with open(outpath) as f:
            result = f.read().strip()
        return result
    finally:
        try:
            os.unlink(outpath)
        except OSError:
            pass


def _hermes_cli_complete(cfg: LLMConfig, system: str, user: str) -> str:
    """Run one completion via the `hermes chat` CLI.

    cfg.model is the model name (e.g. "MiniMax-M2.7"); the upstream
    provider is encoded in cfg.base_url (re-purposed here as the Hermes
    `--provider` value, e.g. "minimax", "nous", "openrouter"). Default
    is "minimax".

    Hermes has no `--system-prompt` flag in non-interactive mode, so we
    prepend the system instructions inline.

    --max-turns 1 disables tool-use loops; --yolo skips approvals;
    -Q (quiet) suppresses banner/spinner/tool-preview noise so the
    final stdout is the assistant text we want.
    """
    hermes_provider = cfg.base_url or "minimax"
    combined = f"System:\n{system}\n\nUser:\n{user}"
    cmd = [
        "hermes",
        "chat",
        "-q",
        combined,
        "-Q",
        "--model",
        cfg.model,
        "--provider",
        hermes_provider,
        "--max-turns",
        "1",
        "--yolo",
        "--ignore-rules",
        "--ignore-user-config",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=600,
        env=_scrubbed_subprocess_env(),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"hermes CLI failed rc={proc.returncode}: {proc.stderr[:500]}")
    out = proc.stdout
    # Quiet mode prepends one "session_id:" line; strip it cleanly.
    cleaned: list[str] = []
    for line in out.splitlines():
        if line.startswith("session_id:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _default_base_url(provider: str) -> str:
    return {
        "openrouter": "https://openrouter.ai/api/v1",
        "nous": "https://inference-api.nousresearch.com/v1",
        "openai": "https://api.openai.com/v1",
    }[provider]


def _default_api_key(provider: str) -> str:
    return {
        "openrouter": os.environ.get("OPENROUTER_API_KEY", ""),
        "nous": os.environ.get("NOUS_API_KEY", os.environ.get("NOUS_PORTAL_API_KEY", "")),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
    }[provider]


def parse_json_block(text: str) -> Any:
    """Best-effort JSON parse, handling code fences the model may add."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # peel fence
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        # chop trailing fence if present
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
    return json.loads(stripped.strip())
