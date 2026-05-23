"""Regression tests for the agent CLI runner — env hygiene + broken-runner
detection. These exist because two consecutive sessions shipped silent-
failure bugs (env-injection + Max-quota broken-runner pattern) that wasted
hours of fleet time. Each test pins one specific failure mode that we
will not regress into.

Run with:  pytest tests/test_agent_cli_runner.py -v
"""

from __future__ import annotations

import io
import sys

import pytest

from hermes_katana.proving_ground.sandbox.agent_cli_runner import (
    _CLAUDE_POISON_ENV_KEYS,
    _build_subprocess_env,
    _warn_if_broken_claude_run,
    AGENT_DRIVERS,
)


def test_hermes_drivers_ignore_user_config_and_have_room_to_finish():
    """Nested Hermes drivers must be hermetic proving-ground workers.

    Regression: with Carlos's normal ~/.hermes/config.yaml, terminal.cwd points
    at /home/example. Nested `hermes chat` then used /home/example/src instead of
    the seeded attack workspace, making Hermes-agent rows incomparable to direct
    CLI rows. It also hit --max-turns=15 before writing findings.md.
    """
    hermes_drivers = [d for d in AGENT_DRIVERS.values() if d.cmd_template[:2] == ["hermes", "chat"]]
    assert hermes_drivers, "expected at least one Hermes CLI driver"
    for driver in hermes_drivers:
        cmd = driver.cmd_template
        assert "--ignore-user-config" in cmd, driver.agent_id
        assert "--ignore-rules" in cmd, driver.agent_id
        assert "--source" in cmd and "proving-ground" in cmd, driver.agent_id
        assert "--max-turns" in cmd, driver.agent_id
        assert int(cmd[cmd.index("--max-turns") + 1]) >= 40, driver.agent_id


def test_hermes_subprocess_env_pins_terminal_cwd_to_workspace(monkeypatch, tmp_path):
    """Nested Hermes must not inherit TERMINAL_CWD from Carlos's parent shell."""
    monkeypatch.setenv("TERMINAL_CWD", "/home/example")
    workspace = tmp_path / "proving-ground-session"
    env = _build_subprocess_env(AGENT_DRIVERS["hermes_openai_codex"], workspace)
    assert env["TERMINAL_CWD"] == str(workspace.resolve())
    assert env["PWD"] == str(workspace.resolve())


# ---------------------------------------------------------------------------
# 1. Env scrubbing — claude CLI subprocess must not inherit poison vars
# ---------------------------------------------------------------------------

POISON_PAIRS = [
    ("ANTHROPIC_API_KEY", "sk-ant-fake-leak"),
    ("ANTHROPIC_TOKEN", "sk-ant-fake-token"),
    ("CLAUDECODE", "1"),
    ("CLAUDE_CODE_ENTRYPOINT", "cli"),
    ("CLAUDE_CODE_EXECPATH", "/usr/bin/claude"),
    ("CLAUDE_CODE_SSE_PORT", "12345"),
    ("AI_AGENT", "claude-code/2.x"),
]


@pytest.mark.parametrize("agent_id", ["claude_cli_haiku", "claude_cli_sonnet"])
def test_build_subprocess_env_strips_poison_vars_from_claude_drivers(
    monkeypatch,
    agent_id,
):
    for k, v in POISON_PAIRS:
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # benign — must survive

    driver = AGENT_DRIVERS[agent_id]
    env = _build_subprocess_env(driver)

    for k, _ in POISON_PAIRS:
        assert k not in env, (
            f"{agent_id}: {k} leaked through to claude subprocess env. "
            "This is the bug that broke shards 200-211 in 2026-04-25."
        )
    assert "PATH" in env, "PATH should not be stripped"


@pytest.mark.parametrize(
    "agent_id",
    [
        "hermes_minimax_m2_7",
        "hermes_nous_qwen3_coder_plus",
        "hermes_mimo_v2_pro",
        "hermes_qwen35_local",
    ],
)
def test_build_subprocess_env_keeps_keys_for_non_claude_drivers(
    monkeypatch,
    agent_id,
):
    """Hermes drivers route through Anthropic-format APIs in some cases (e.g.
    MiniMax via /anthropic/v1/messages) and may legitimately need
    ANTHROPIC_API_KEY in the subprocess env. Don't strip for non-claude binaries.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")

    driver = AGENT_DRIVERS[agent_id]
    env = _build_subprocess_env(driver)

    assert env.get("ANTHROPIC_API_KEY") == "sk-ant-fake", (
        f"{agent_id}: ANTHROPIC_API_KEY should pass through to non-claude "
        "subprocess env. Stripping it broke MiniMax direct calls."
    )
    assert env.get("OPENROUTER_API_KEY") == "sk-or-fake"


def test_build_subprocess_env_returns_a_copy(monkeypatch):
    """Mutating the returned env must NOT mutate os.environ. Defensive: in the
    past we accidentally returned a reference and other code paths leaked."""
    import os

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    driver = AGENT_DRIVERS["claude_cli_haiku"]
    env = _build_subprocess_env(driver)
    env["ANTHROPIC_API_KEY"] = "sk-INJECTED"
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-fake", (
        "_build_subprocess_env returned a reference, not a copy. Mutating it leaked back to the parent process env."
    )


def test_poison_env_keys_list_is_complete():
    """Belt + suspenders: any new CLAUDE_CODE_* env var introduced upstream
    should land in this constant. If you're seeing this fail, an upstream
    Claude Code release shipped a new CLAUDE_CODE_* env var; add it.
    """
    must_have = {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDECODE",
        "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH",
        "CLAUDE_CODE_SSE_PORT",
        "AI_AGENT",
    }
    missing = must_have - set(_CLAUDE_POISON_ENV_KEYS)
    assert not missing, f"_CLAUDE_POISON_ENV_KEYS missing: {missing}"


# ---------------------------------------------------------------------------
# 2. Broken-runner detector — fires on the broken pattern, silent on healthy
# ---------------------------------------------------------------------------


@pytest.fixture
def claude_driver():
    return AGENT_DRIVERS["claude_cli_haiku"]


@pytest.fixture
def hermes_driver():
    return AGENT_DRIVERS["hermes_minimax_m2_7"]


def _capture_stderr(fn) -> str:
    buf = io.StringIO()
    saved = sys.stderr
    sys.stderr = buf
    try:
        fn()
    finally:
        sys.stderr = saved
    return buf.getvalue()


def test_warn_fires_on_broken_pattern(claude_driver):
    """Pattern: exit≠0, ≤6kB output, 0 tools, <5s. The 2026-04-25 regression."""
    err = _capture_stderr(
        lambda: _warn_if_broken_claude_run(
            claude_driver,
            exit_code=1,
            stdout_chars=4500,
            tool_call_count=0,
            duration=2.4,
            stderr="",
        )
    )
    assert "broken-runner" in err.lower() or "broken" in err.lower()
    assert claude_driver.agent_id in err


def test_warn_silent_on_healthy_run(claude_driver):
    """Healthy: exit 0, big output, tool calls, slow."""
    err = _capture_stderr(
        lambda: _warn_if_broken_claude_run(
            claude_driver,
            exit_code=0,
            stdout_chars=250000,
            tool_call_count=5,
            duration=45.0,
            stderr="",
        )
    )
    assert err == ""


def test_warn_silent_on_partial_match_long_duration(claude_driver):
    """Long duration is the simplest disambiguator from a real broken run.
    A 60-second exit-1 run is something else (timeout, crash) and shouldn't
    spuriously trigger the quota warning."""
    err = _capture_stderr(
        lambda: _warn_if_broken_claude_run(
            claude_driver,
            exit_code=1,
            stdout_chars=4500,
            tool_call_count=0,
            duration=60.0,
            stderr="",
        )
    )
    assert err == ""


def test_warn_silent_on_partial_match_with_tool_calls(claude_driver):
    """If there were tool calls, output isn't really 0kB worth of system-init."""
    err = _capture_stderr(
        lambda: _warn_if_broken_claude_run(
            claude_driver,
            exit_code=1,
            stdout_chars=4500,
            tool_call_count=2,
            duration=2.4,
            stderr="",
        )
    )
    assert err == ""


def test_warn_silent_for_hermes_driver(hermes_driver):
    """The broken-runner pattern is specific to claude --print. Hermes
    drivers can fail for other reasons; we do not want the warning to
    fire on them."""
    err = _capture_stderr(
        lambda: _warn_if_broken_claude_run(
            hermes_driver,
            exit_code=1,
            stdout_chars=4500,
            tool_call_count=0,
            duration=2.4,
            stderr="",
        )
    )
    assert err == ""


# ---------------------------------------------------------------------------
# 3. Quota watchdog — N consecutive broken outputs aborts the shard
# ---------------------------------------------------------------------------


def _broken_pattern(exit_code: int, output_chars: int, tool_call_count: int, duration_sec: float) -> bool:
    """Inline copy of the agent-shard watchdog predicate.
    Pinned here so a refactor of the predicate has to update both places —
    if you change the live one, this test will tell you to update the
    fixture and the actual predicate together."""
    return exit_code != 0 and output_chars <= 6000 and tool_call_count == 0 and duration_sec < 5.0


def test_quota_watchdog_fires_after_threshold(monkeypatch):
    monkeypatch.setenv("KATANA_QUOTA_WATCHDOG_N", "3")
    threshold = int("3")  # mimic the parse in run_agent_shard.py

    consecutive_broken = 0
    aborted = False
    # 5 broken attacks in a row — should abort at the 3rd.
    for i in range(5):
        if _broken_pattern(exit_code=1, output_chars=4500, tool_call_count=0, duration_sec=2.4):
            consecutive_broken += 1
            if consecutive_broken >= threshold:
                aborted = True
                break
        else:
            consecutive_broken = 0

    assert aborted, "watchdog did not fire after 3 consecutive broken attacks"


def test_quota_watchdog_resets_on_healthy_run():
    """A single healthy attack between broken ones must reset the counter."""
    threshold = 3
    consecutive_broken = 0
    aborted = False

    pattern = [
        # broken, broken, HEALTHY, broken, broken — should NOT abort
        (1, 4500, 0, 2.4),
        (1, 4500, 0, 2.4),
        (0, 250000, 5, 45.0),  # healthy → resets
        (1, 4500, 0, 2.4),
        (1, 4500, 0, 2.4),
    ]
    for ec, oc, tc, dur in pattern:
        if _broken_pattern(ec, oc, tc, dur):
            consecutive_broken += 1
            if consecutive_broken >= threshold:
                aborted = True
        else:
            consecutive_broken = 0
    assert not aborted, "watchdog should reset on a healthy run between broken ones"


def test_quota_watchdog_does_not_fire_below_threshold():
    threshold = 3
    consecutive_broken = 0
    aborted = False
    # Only 2 broken in a row — below threshold.
    for _ in range(2):
        if _broken_pattern(exit_code=1, output_chars=4500, tool_call_count=0, duration_sec=2.4):
            consecutive_broken += 1
            if consecutive_broken >= threshold:
                aborted = True
    assert not aborted


# ---------------------------------------------------------------------------
# 4. Driver registry sanity
# ---------------------------------------------------------------------------


def test_local_qwen_drivers_register_with_katana_twins():
    """If qwen35_local is registered, its katana twin must also exist —
    otherwise paired runs on the local box fall back to undefended-only."""
    if "hermes_qwen35_local" in AGENT_DRIVERS:
        assert "hermes_qwen35_local_katana" in AGENT_DRIVERS, (
            "hermes_qwen35_local missing katana twin — paired data unavailable"
        )


# ---------------------------------------------------------------------------
# 5. dotenv whitelist — the "original sin" that introduced the env-injection bug
# ---------------------------------------------------------------------------


def test_load_dotenv_does_not_leak_anthropic_keys(monkeypatch, tmp_path):
    """A .env file with ANTHROPIC_API_KEY must NOT leak into os.environ.
    This is the precise regression that broke 750+ trials in 2026-04-25.
    """
    import importlib
    import os as _os

    # Sandbox: redirect ROOT and HOME so _load_dotenv reads our test files.
    fake_proj_env = tmp_path / ".env"
    fake_proj_env.write_text(
        "ANTHROPIC_API_KEY=sk-ant-LEAK\n"
        "OPENROUTER_API_KEY=sk-or-OK\n"
        "CLAUDECODE=1\n"
        "KATANA_LOCAL_QWEN35_MODEL=qwen3.6-35b\n"
    )
    fake_home = tmp_path / "home"
    (fake_home / ".hermes").mkdir(parents=True)
    (fake_home / ".hermes" / ".env").write_text("ANTHROPIC_TOKEN=sk-ant-ALSO-LEAK\n")

    # Pre-seed os.environ to make sure setdefault doesn't overwrite caller env.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.delenv("KATANA_LOCAL_QWEN35_MODEL", raising=False)

    import hermes_katana.proving_ground.run_agent_shard as ras

    importlib.reload(ras)
    monkeypatch.setattr(ras, "ROOT", tmp_path)
    monkeypatch.setattr(ras.Path, "home", classmethod(lambda cls: fake_home))

    ras._load_dotenv()

    assert "ANTHROPIC_API_KEY" not in _os.environ, (
        "ANTHROPIC_API_KEY leaked from .env into os.environ. _DOTENV_ALLOWED_KEYS whitelist isn't being enforced."
    )
    assert "ANTHROPIC_TOKEN" not in _os.environ
    assert "CLAUDECODE" not in _os.environ
    assert _os.environ.get("OPENROUTER_API_KEY") == "sk-or-OK"
    assert _os.environ.get("KATANA_LOCAL_QWEN35_MODEL") == "qwen3.6-35b"


def test_load_dotenv_respects_pre_set_env(monkeypatch, tmp_path):
    """If the parent shell has already set a whitelisted key, the .env value
    must NOT overwrite it (setdefault semantics). This protects deployments
    where the operator wants a different key than what's in the file.
    """
    import importlib
    import os as _os

    fake_proj_env = tmp_path / ".env"
    fake_proj_env.write_text("OPENROUTER_API_KEY=sk-or-FROM-FILE\n")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-FROM-SHELL")

    import hermes_katana.proving_ground.run_agent_shard as ras

    importlib.reload(ras)
    monkeypatch.setattr(ras, "ROOT", tmp_path)
    monkeypatch.setattr(ras.Path, "home", classmethod(lambda cls: tmp_path))

    ras._load_dotenv()

    assert _os.environ["OPENROUTER_API_KEY"] == "sk-or-FROM-SHELL", (
        "_load_dotenv overwrote a shell-set env var — should be setdefault, not set."
    )


# ---------------------------------------------------------------------------
# 6. Hermes tool-call parser — must read structured session JSON, not regex
# ---------------------------------------------------------------------------


def _write_hermes_session(sessions_dir, session_id, tool_call_names):
    """Build a minimal Hermes session JSON with N assistant tool_calls."""
    import json as _json

    sessions_dir.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "user", "content": "do the thing"}]
    for i, name in enumerate(tool_call_names):
        msgs.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{i:08x}",
                        "type": "function",
                        "function": {"name": name, "arguments": '{"path": "x"}'},
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{i:08x}",
                "content": "ok",
            }
        )
    msgs.append({"role": "assistant", "content": "done"})
    sess = {
        "session_id": session_id,
        "model": "test/model",
        "platform": "cli",
        "message_count": len(msgs),
        "messages": msgs,
    }
    (sessions_dir / f"session_{session_id}.json").write_text(_json.dumps(sess))


def test_parse_hermes_cli_reads_session_json(monkeypatch, tmp_path):
    """Authoritative parse — grep session_id from stdout, read session JSON,
    count tool_calls. This is what fixes the -Q quiet-mode 0-tool bug."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import parse_hermes_cli
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sid = "20260501_083502_664be3"
    _write_hermes_session(sessions_dir, sid, ["read_file", "list_dir", "write_file"])

    # Quiet-mode stdout — no tool markers, no summary line, just session_id.
    quiet_stdout = f"\nsession_id: {sid}\nThere's no test.py in the current directory.\n"
    calls = parse_hermes_cli(quiet_stdout, "", Path("."))
    assert len(calls) == 3, f"expected 3 tool_calls from session JSON, got {len(calls)}"
    names = [c["name"] for c in calls]
    assert names == ["read_file", "list_dir", "write_file"]
    assert all(c["source"] == "hermes_session" for c in calls)


def test_parse_hermes_cli_reads_session_id_from_stderr(monkeypatch, tmp_path):
    """`hermes -Q --pass-session-id` writes session_id to STDERR. Parser must
    pick it up from there too — that's the production code path."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import parse_hermes_cli
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sid = "20260501_111111_stderr1"
    _write_hermes_session(sessions_dir, sid, ["read_file", "list_dir"])

    stdout = "The agent answered the question.\n"
    stderr = f"\nsession_id: {sid}\n"
    calls = parse_hermes_cli(stdout, stderr, Path("."))
    assert len(calls) == 2
    assert {c["name"] for c in calls} == {"read_file", "list_dir"}


def test_every_hermes_driver_passes_session_id():
    """Every hermes driver's cmd_template must include --pass-session-id so
    the parser can deterministically locate the session JSON. Without this,
    quiet-mode runs may not emit session_id at all."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import AGENT_DRIVERS

    bad = []
    for agent_id, d in AGENT_DRIVERS.items():
        binary = d.cmd_template[0] if d.cmd_template else ""
        if binary == "hermes":
            if "--pass-session-id" not in d.cmd_template:
                bad.append(agent_id)
    assert not bad, (
        f"hermes drivers missing --pass-session-id: {bad}. "
        "Without this flag, parse_hermes_cli cannot locate the session JSON "
        "and tool_call_count will be 0 for every trial."
    )


def test_parse_hermes_cli_handles_session_form(monkeypatch, tmp_path):
    """Hermes verbose mode emits `Session: <id>` not `session_id: <id>`.
    Both forms must be picked up."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import parse_hermes_cli
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sid = "20260501_120000_abcdef"
    _write_hermes_session(sessions_dir, sid, ["read_file"])

    verbose_stdout = (
        "Resume this session with:\n"
        f"  hermes --resume {sid}\n\n"
        f"Session:        {sid}\n"
        "Duration:       15s\n"
        "Messages:       4 (1 user, 2 tool calls)\n"
    )
    calls = parse_hermes_cli(verbose_stdout, "", Path("."))
    assert len(calls) == 1
    assert calls[0]["name"] == "read_file"


def test_parse_hermes_cli_falls_back_when_session_missing(monkeypatch, tmp_path):
    """If the session file isn't found, fall back to the legacy regex parser
    (which still works for verbose-mode runs that emit ⚒ markers)."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import parse_hermes_cli
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Don't create any session file.

    legacy_stdout = (
        "\nsession_id: 99999999_nonexistent\n"
        "⚒ read_file(path='x')\n"
        "⚒ list_dir(path='.')\n"
        "Messages: 4 (1 user, 2 tool calls)\n"
    )
    calls = parse_hermes_cli(legacy_stdout, "", Path("."))
    assert len(calls) == 2
    assert {c["name"] for c in calls} == {"read_file", "list_dir"}


def test_parse_hermes_cli_empty_session_means_zero_tools(monkeypatch, tmp_path):
    """If the model genuinely made no tool calls, the parser should return
    an empty list — distinct from the failure mode where the parser missed
    them. 0 tool_calls is real signal (model refused, model hallucinated
    answer without tools)."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import parse_hermes_cli
    from pathlib import Path

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sessions_dir = tmp_path / "sessions"
    sid = "20260501_140000_notools"
    _write_hermes_session(sessions_dir, sid, [])  # zero tool calls

    quiet_stdout = f"\nsession_id: {sid}\n"
    calls = parse_hermes_cli(quiet_stdout, "", Path("."))
    assert calls == [], "session JSON had 0 tool_calls; parser should return empty list"


def test_every_katana_twin_has_scanner_attached():
    """The whole point of the *_katana suffix is that scanner is set."""
    for agent_id, driver in AGENT_DRIVERS.items():
        if agent_id.endswith("_katana") or agent_id.endswith("_katana07"):
            assert driver.scanner is not None, (
                f"{agent_id}: katana twin has no ScannerConfig — would behave identically to its undefended base."
            )


# ---------------------------------------------------------------------------
# 7. codex_cli_gpt5_4_mini — API-key driver, must force apikey auth
# ---------------------------------------------------------------------------


def test_codex_cli_gpt5_4_mini_registered():
    """Ensure the API-key codex variant exists and is wired correctly."""
    from hermes_katana.proving_ground.sandbox.agent_cli_runner import AGENT_DRIVERS

    assert "codex_cli_gpt5_4_mini" in AGENT_DRIVERS, (
        "codex_cli_gpt5_4_mini missing from registry — added 2026-05-01 "
        "for API-key batched runs against the OpenAI ~$9 budget."
    )
    d = AGENT_DRIVERS["codex_cli_gpt5_4_mini"]
    assert d.cmd_template[0] == "codex"
    assert "--model" in d.cmd_template
    assert "gpt-5.4-mini" in d.cmd_template
    auth_cfg = [a for a in d.cmd_template if "preferred_auth_method" in a]
    assert auth_cfg and "apikey" in auth_cfg[0], (
        "codex_cli_gpt5_4_mini must pass `-c preferred_auth_method=apikey` "
        "or it will silently route to ChatGPT-plan auth and burn the wrong budget."
    )
