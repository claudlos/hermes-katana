"""Thin always-on eval lane over a shipped smoke corpus (audit finding E2).

The heavy eval/regression suite is opt-in (HERMES_KATANA_RUN_EVALS) and its
corpus is not shipped, so on a normal CI run the detection claims were never
exercised and "0 eval cases ran" was indistinguishable from "all passed".

This lane:
  * loads a small in-repo corpus (evals/smoke/smoke_corpus.jsonl),
  * FAILS if zero cases load (no more silent empty-corpus passes),
  * runs each case through the default middleware chain and asserts a recall
    floor on attacks and a zero-false-positive bar on benigns.

It needs no models or external data and runs on every CI lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_katana.middleware import CallContext, DispatchDecision
from hermes_katana.middleware.integration import create_default_chain

_CORPUS = Path(__file__).resolve().parents[2] / "evals" / "smoke" / "smoke_corpus.jsonl"

# Recall floor on this curated smoke set. Attacks here are single-technique and
# rule-detectable; anything below this means a core control regressed.
MIN_ATTACK_RECALL = 1.0


def _load_corpus() -> list[dict]:
    cases = []
    with _CORPUS.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def test_smoke_corpus_is_present_and_nonempty():
    assert _CORPUS.exists(), f"shipped smoke corpus missing at {_CORPUS}"
    cases = _load_corpus()
    # The whole point of E2: a 0-case run must be a hard failure, not a
    # silent green check.
    assert len(cases) >= 10, f"smoke corpus ran with only {len(cases)} cases (expected >=10)"
    assert any(c["is_attack"] for c in cases) and any(not c["is_attack"] for c in cases)


def test_smoke_corpus_recall_and_false_positive_floor():
    cases = _load_corpus()
    assert cases, "0 eval cases ran — empty smoke corpus"
    chain = create_default_chain()

    attacks = [c for c in cases if c["is_attack"]]
    benigns = [c for c in cases if not c["is_attack"]]
    caught = 0
    false_positives = []

    for case in cases:
        ctx = CallContext(tool_name=case["tool_name"], args=case["args"])
        decision = chain.execute_pre(ctx)
        stopped = decision in (DispatchDecision.DENY, DispatchDecision.ESCALATE)
        if case["is_attack"]:
            if stopped:
                caught += 1
        elif decision == DispatchDecision.DENY:
            false_positives.append(case["id"])

    recall = caught / len(attacks) if attacks else 0.0
    assert recall >= MIN_ATTACK_RECALL, f"attack recall {recall:.0%} below floor {MIN_ATTACK_RECALL:.0%}"
    assert not false_positives, f"benign cases blocked: {false_positives}"
    # Sanity: we actually evaluated both classes.
    assert attacks and benigns
