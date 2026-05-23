"""Session-scoped fixtures for the evaluation harness."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.eval._control import (
    configured_eval_corpus_path,
    configured_max_records,
    eval_execution_enabled,
    eval_execution_skip_reason,
    load_jsonl_records,
)

# Ensure project is importable
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
_src = str(Path(_root) / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from tests.eval.scanner_runner import make_scanner_suite  # noqa: E402

CORPUS_PATH = configured_eval_corpus_path(Path(_root))


def _warmup_scanners():
    """Warm up heavy scanners (semantic recall loads ML model) before eval starts."""
    from hermes_katana.scanner import scan_input

    try:
        scan_input("warmup query to load ML model", check_content_harm=True)
    except Exception:
        pass  # Model loading failed is non-critical


@pytest.fixture(scope="session", autouse=True)
def eval_runtime_setup():
    """Require explicit opt-in for heavy eval execution and optionally warm scanners."""
    if not eval_execution_enabled():
        pytest.skip(eval_execution_skip_reason())
    if os.getenv("HERMES_KATANA_EVAL_WARMUP") == "1":
        _warmup_scanners()


def _load_corpus() -> list[dict]:
    """Load the full corpus from disk."""
    return load_jsonl_records(CORPUS_PATH, max_records=configured_max_records())


@pytest.fixture(scope="session")
def full_corpus():
    """Full evaluation corpus (all labels). Skips if file missing."""
    if not CORPUS_PATH.exists():
        pytest.skip(f"Corpus not found: {CORPUS_PATH}")
    return _load_corpus()


@pytest.fixture(scope="session")
def injection_corpus(full_corpus):
    """Only records with clean_label == 'injection'."""
    corpus = [r for r in full_corpus if r.get("clean_label") == "injection"]
    if not corpus:
        pytest.skip("No injection-labeled records in corpus")
    return corpus


@pytest.fixture(scope="session")
def content_harm_corpus(full_corpus):
    """Only records with clean_label == 'content_harm'."""
    corpus = [r for r in full_corpus if r.get("clean_label") == "content_harm"]
    if not corpus:
        pytest.skip("No content_harm-labeled records in corpus")
    return corpus


@pytest.fixture(scope="session")
def system_prompt_leak_corpus(full_corpus):
    """Only records with clean_label == 'system_prompt_leak'."""
    corpus = [r for r in full_corpus if r.get("clean_label") == "system_prompt_leak"]
    if not corpus:
        pytest.skip("No system_prompt_leak-labeled records in corpus")
    return corpus


@pytest.fixture(scope="session")
def meta_discussion_corpus(full_corpus):
    """Only records with clean_label == 'meta_discussion'."""
    corpus = [r for r in full_corpus if r.get("clean_label") == "meta_discussion"]
    if not corpus:
        pytest.skip("No meta_discussion-labeled records in corpus")
    return corpus


@pytest.fixture(scope="session")
def full_corpus_all(full_corpus):
    """All 5091 records across all labels (injection, content_harm, system_prompt_leak, meta_discussion)."""
    return full_corpus


@pytest.fixture(scope="session")
def scanner_suite():
    """Dict of scanner_name -> callable."""
    return make_scanner_suite()


BENIGN_CORPUS_PATH = Path(_root) / "research" / "wild-attacks-2026-04-05" / "benign_corpus_extended.txt"
BENIGN_CORPUS_FALLBACK = Path(_root) / "research" / "wild-attacks-2026-04-05" / "benign_corpus.txt"


@pytest.fixture(scope="session")
def benign_corpus():
    """Load extended benign corpus for false positive testing."""
    path = BENIGN_CORPUS_PATH if BENIGN_CORPUS_PATH.exists() else BENIGN_CORPUS_FALLBACK
    if not path.exists():
        pytest.skip(f"Benign corpus not found: {path}")
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                texts.append(line)
    return texts
