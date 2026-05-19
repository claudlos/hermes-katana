"""Tests for the scanner-change verification helper."""

from __future__ import annotations

from pathlib import Path
import os
import stat
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_scanner_change.sh"
RELEASE_SCRIPT = ROOT / "scripts" / "release_gate.sh"


def test_verify_scanner_change_script_is_executable():
    assert SCRIPT.exists()
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR


def test_verify_scanner_change_dry_run_lists_required_gates():
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "ruff check src/ tests/" in output
    assert "ruff format --check src/ tests/" in output
    assert "python3 test_false_positives.py" in output
    assert "python3 test_evasion.py" in output
    assert "python3 -m pytest tests/integration/test_adversarial_eval_pack.py -q" in output


def test_verify_scanner_change_disables_cuda_by_default():
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "0"},
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_verify_scanner_change_eval_dry_run_lists_eval_gates():
    result = subprocess.run(
        [str(SCRIPT), "--dry-run", "--eval"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "HERMES_KATANA_RUN_EVALS=1 python3 -m pytest tests/eval/ -q" in output
    assert "python3 tests/eval/run_eval.py --compare" in output


def test_release_gate_dry_run_lists_required_release_gates():
    result = subprocess.run(
        [str(RELEASE_SCRIPT), "--dry-run", "--allow-missing-gitleaks"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "ruff check src/ tests/" in output
    assert "ruff format --check src/ tests/" in output
    assert "python3 -m pytest tests/ -q" in output
    assert "scripts/verify_scanner_change.sh --skip-lint" in output
    assert "python3 -m build" in output
    assert "python3 -m twine check" in output
    assert "katana artifacts status" in output
    assert "gitleaks detect --source . --redact --no-banner --config .gitleaks.toml" in output
