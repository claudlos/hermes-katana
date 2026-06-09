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
