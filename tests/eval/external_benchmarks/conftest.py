"""Fixtures for JailbreakBench external benchmark tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[3])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from tests.eval.scanner_runner import make_scanner_suite  # noqa: E402
from tests.eval.external_benchmarks.loader import (  # noqa: E402
    load_full_jbb_corpus,
    load_jbb_artifacts,
    load_jbb_behaviors,
    load_jbb_benign_prompts,
)


JBB_ENV = "HERMES_KATANA_RUN_JBB"


def jbb_enabled() -> bool:
    """Check if JBB benchmarks are enabled via env var."""
    return os.getenv(JBB_ENV) == "1"


@pytest.fixture(scope="session", autouse=True)
def eval_runtime_setup():
    """Override parent eval gate — JBB uses its own env var."""
    if not jbb_enabled():
        pytest.skip(f"JBB benchmarks disabled. Set {JBB_ENV}=1 to run.")


@pytest.fixture(scope="session")
def jbb_scanner_suite():
    """Scanner suite for JBB evaluation."""
    return make_scanner_suite()


@pytest.fixture(scope="session")
def jbb_artifact_corpus():
    """JBB artifact prompts (PAIR/GCG/JBC jailbreaks)."""
    corpus = load_jbb_artifacts()
    if not corpus:
        pytest.skip("No JBB artifacts loaded (jailbreakbench not installed?)")
    return corpus


@pytest.fixture(scope="session")
def jbb_behavior_corpus():
    """JBB behavior goal descriptions."""
    corpus = load_jbb_behaviors()
    if not corpus:
        pytest.skip("No JBB behaviors loaded")
    return corpus


@pytest.fixture(scope="session")
def jbb_full_corpus():
    """Full JBB corpus: artifacts + behaviors."""
    corpus = load_full_jbb_corpus()
    if not corpus:
        pytest.skip("No JBB corpus loaded")
    return corpus


@pytest.fixture(scope="session")
def jbb_benign_prompts():
    """Benign prompts for FP testing."""
    return load_jbb_benign_prompts()
