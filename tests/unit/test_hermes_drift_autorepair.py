"""Tests for Hermes drift auto-repair helpers."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "hermes_drift_autorepair.py"


def _load_autorepair_script():
    spec = importlib.util.spec_from_file_location("hermes_drift_autorepair", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, text=True, capture_output=True)


def test_restore_hermes_checkout_removes_patch_mutations_and_backups(tmp_dir):
    script = _load_autorepair_script()
    checkout = tmp_dir / "hermes-agent"
    checkout.mkdir()
    subprocess.run(["git", "init", str(checkout)], check=True, text=True, capture_output=True)
    _git(checkout, "config", "user.name", "Test User")
    _git(checkout, "config", "user.email", "test@example.com")
    tracked = checkout / "model_tools.py"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(checkout, "add", "model_tools.py")
    _git(checkout, "commit", "-m", "initial")

    tracked.write_text("patched\n", encoding="utf-8")
    backup = checkout / "model_tools.py.katana-backup"
    backup.write_text("clean\n", encoding="utf-8")

    script._restore_hermes_checkout(checkout)

    assert tracked.read_text(encoding="utf-8") == "clean\n"
    assert not backup.exists()


def test_open_or_update_pr_fetches_branch_before_force_with_lease(monkeypatch, tmp_dir):
    script = _load_autorepair_script()
    calls = []

    def fake_git(*args: str, check: bool = False):
        calls.append(("git", args, check))

    def fake_run(command):
        calls.append(("run", tuple(command), False))

        class Result:
            stdout = "123\n"

        return Result()

    monkeypatch.setattr(script, "_git", fake_git)
    monkeypatch.setattr(script, "_run", fake_run)
    monkeypatch.setattr(script, "_run_visible_checked", lambda command: calls.append(("visible", tuple(command), True)))

    script._open_or_update_pr("automation/hermes-current-snapshot-abc123", "title", tmp_dir / "body.md", "master")

    fetch_index = next(i for i, call in enumerate(calls) if call[0] == "git" and call[1][:2] == ("fetch", "origin"))
    push_index = next(
        i for i, call in enumerate(calls) if call[0] == "git" and call[1][:2] == ("push", "--force-with-lease")
    )
    assert fetch_index < push_index


def test_main_skips_snapshot_pr_when_drift_passes(monkeypatch, tmp_dir, capsys):
    script = _load_autorepair_script()
    checkout = tmp_dir / "hermes-agent"
    checkout.mkdir()

    monkeypatch.setattr(script, "_hermes_commit", lambda checkout: ("a" * 40, "a" * 12))
    monkeypatch.setattr(script, "_check_drift", lambda checkout: (0, "OK"))
    monkeypatch.setattr(script, "_can_open_pr", lambda default_branch: True)

    def fail_refresh(checkout):
        raise AssertionError("snapshot refresh should not run when drift passes")

    monkeypatch.setattr(script, "_refresh_snapshot", fail_refresh)

    result = script.main(["--checkout", str(checkout), "--default-branch", "master", "--pr"])

    assert result == 0
    assert "Skipping snapshot PR: drift check passed." in capsys.readouterr().out
