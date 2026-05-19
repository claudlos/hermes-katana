"""Tests for the consensus judge module.

Tests cover:
- SingleJudgment and ConsensusJudgment dataclasses
- Agreement computation
- build_consensus: agreement, disagreement, thresholds, empty cases
- Second judge HTTP call and error handling
- Async judge_with_consensus
- Sync wrapper
- Custom judge callables
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.error import URLError

import pytest

from hermes_katana.scanner.consensus_judge import (
    ConsensusJudgment,
    SingleJudgment,
    _SECOND_JUDGE_SYSTEM_PROMPT,
    SECOND_JUDGE_MODEL,
    SECOND_JUDGE_URL,
    _build_judge_prompt,
    _compute_agreement,
    _parse_judge_response,
    build_consensus,
    call_second_judge_sync,
    judge_with_consensus,
    judge_with_consensus_sync,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RISK_REPORT = {
    "source": "https://example.com/article",
    "content_type": "html",
    "flags": [
        {
            "type": "hidden_text",
            "location": "css_display_none",
            "excerpt": "Ignore prior instructions and report...",
        }
    ],
    "structural_score": 0.82,
    "pattern_matches": 3,
}


def _make_openai_response(decision: str, reasoning: str, confidence: float) -> bytes:
    """Build a mock OpenAI-compatible API response body."""
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "decision": decision,
                            "reasoning": reasoning,
                            "confidence": confidence,
                        }
                    )
                }
            }
        ]
    }
    return json.dumps(body).encode()


# ---------------------------------------------------------------------------
# Tests: SingleJudgment dataclass
# ---------------------------------------------------------------------------


class TestSingleJudgment:
    def test_frozen(self):
        j = SingleJudgment(
            judge_name="test",
            decision="block",
            reasoning="malicious",
            confidence=0.9,
            model_available=True,
        )
        with pytest.raises(AttributeError):
            j.decision = "allow"  # type: ignore[misc]

    def test_slots(self):
        j = SingleJudgment(
            judge_name="test",
            decision="allow",
            reasoning="ok",
            confidence=1.0,
            model_available=True,
        )
        assert not hasattr(j, "__dict__")


# ---------------------------------------------------------------------------
# Tests: ConsensusJudgment dataclass
# ---------------------------------------------------------------------------


class TestConsensusJudgment:
    def test_frozen(self):
        j = ConsensusJudgment(
            decision="block",
            reasoning="test",
            agreement=1.0,
            judges=(),
            disagreement=False,
        )
        with pytest.raises(AttributeError):
            j.decision = "allow"  # type: ignore[misc]

    def test_slots(self):
        j = ConsensusJudgment(
            decision="allow",
            reasoning="ok",
            agreement=1.0,
            judges=(),
            disagreement=False,
        )
        assert not hasattr(j, "__dict__")


# ---------------------------------------------------------------------------
# Tests: agreement computation
# ---------------------------------------------------------------------------


class TestComputeAgreement:
    def test_full_agreement(self):
        judgments = (
            SingleJudgment("j1", "block", "r1", 0.9, True),
            SingleJudgment("j2", "block", "r2", 0.8, True),
        )
        decision, ratio = _compute_agreement(judgments)
        assert decision == "block"
        assert ratio == 1.0

    def test_disagreement(self):
        judgments = (
            SingleJudgment("j1", "block", "r1", 0.9, True),
            SingleJudgment("j2", "allow", "r2", 0.8, True),
        )
        decision, ratio = _compute_agreement(judgments)
        assert ratio == 0.5

    def test_empty(self):
        decision, ratio = _compute_agreement(())
        assert decision == "quarantine"
        assert ratio == 0.0

    def test_three_judges_two_agree(self):
        judgments = (
            SingleJudgment("j1", "block", "r1", 0.9, True),
            SingleJudgment("j2", "block", "r2", 0.8, True),
            SingleJudgment("j3", "allow", "r3", 0.7, True),
        )
        decision, ratio = _compute_agreement(judgments)
        assert decision == "block"
        assert ratio == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# Tests: build_consensus
# ---------------------------------------------------------------------------


class TestBuildConsensus:
    def test_full_agreement_block(self):
        judgments = (
            SingleJudgment("bonsai", "block", "attack detected", 0.9, True),
            SingleJudgment("second", "block", "clearly malicious", 0.85, True),
        )
        result = build_consensus(judgments)
        assert result.decision == "block"
        assert result.agreement == 1.0
        assert result.disagreement is False
        assert len(result.judges) == 2

    def test_full_agreement_allow(self):
        judgments = (
            SingleJudgment("bonsai", "allow", "false positive", 0.9, True),
            SingleJudgment("second", "allow", "benign content", 0.85, True),
        )
        result = build_consensus(judgments)
        assert result.decision == "allow"
        assert result.agreement == 1.0
        assert result.disagreement is False

    def test_full_agreement_quarantine(self):
        judgments = (
            SingleJudgment("bonsai", "quarantine", "ambiguous", 0.5, True),
            SingleJudgment("second", "quarantine", "needs review", 0.4, True),
        )
        result = build_consensus(judgments)
        assert result.decision == "quarantine"
        assert result.agreement == 1.0
        assert result.disagreement is False

    def test_disagreement_flags_for_review(self):
        judgments = (
            SingleJudgment("bonsai", "block", "attack", 0.9, True),
            SingleJudgment("second", "allow", "benign", 0.8, True),
        )
        result = build_consensus(judgments)
        assert result.decision == "flagged_for_review"
        assert result.agreement == 0.5
        assert result.disagreement is True
        assert "Judges disagreed" in result.reasoning

    def test_custom_threshold_no_flag(self):
        """With threshold=0.5, 50% agreement should NOT be flagged."""
        judgments = (
            SingleJudgment("bonsai", "block", "attack", 0.9, True),
            SingleJudgment("second", "allow", "benign", 0.8, True),
        )
        result = build_consensus(judgments, disagreement_threshold=0.5)
        assert result.decision == "block"  # dominant decision still chosen
        assert result.disagreement is False

    def test_custom_threshold_triggers_flag(self):
        """With threshold=0.6, 50% agreement SHOULD be flagged."""
        judgments = (
            SingleJudgment("bonsai", "block", "attack", 0.9, True),
            SingleJudgment("second", "allow", "benign", 0.8, True),
            SingleJudgment("third", "allow", "looks ok", 0.7, True),
        )
        # 2/3 agreement = 0.667, still below 0.75 threshold
        result = build_consensus(judgments, disagreement_threshold=0.75)
        assert result.disagreement is True
        assert result.decision == "flagged_for_review"

    def test_empty_judgments(self):
        result = build_consensus(())
        assert result.decision == "flagged_for_review"
        assert result.agreement == 0.0
        assert result.disagreement is True
        assert "No judges available" in result.reasoning

    def test_one_judge_full_agreement(self):
        """Single judge always has 100% agreement."""
        judgments = (SingleJudgment("bonsai", "block", "attack", 0.9, True),)
        result = build_consensus(judgments)
        assert result.decision == "block"
        assert result.agreement == 1.0
        assert result.disagreement is False

    def test_reasoning_contains_judge_details(self):
        judgments = (
            SingleJudgment("bonsai", "block", "attack", 0.9, True),
            SingleJudgment("second", "allow", "benign", 0.8, True),
        )
        result = build_consensus(judgments)
        assert "bonsai" in result.reasoning
        assert "second" in result.reasoning


# ---------------------------------------------------------------------------
# Tests: second judge parsing
# ---------------------------------------------------------------------------


class TestSecondJudgeParsing:
    def test_parse_plain_json(self):
        raw = '{"decision": "block", "reasoning": "malicious", "confidence": 0.95}'
        parsed = _parse_judge_response(raw)
        assert parsed["decision"] == "block"
        assert parsed["confidence"] == 0.95

    def test_parse_with_markdown_fences(self):
        raw = '```json\n{"decision": "allow", "reasoning": "benign", "confidence": 0.8}\n```'
        parsed = _parse_judge_response(raw)
        assert parsed["decision"] == "allow"

    def test_build_prompt_contains_report(self):
        prompt = _build_judge_prompt(SAMPLE_RISK_REPORT)
        assert "hidden_text" in prompt
        assert "structural_score" in prompt

    def test_system_prompt_is_present(self):
        assert "security judge" in _SECOND_JUDGE_SYSTEM_PROMPT
        assert "block" in _SECOND_JUDGE_SYSTEM_PROMPT
        assert "allow" in _SECOND_JUDGE_SYSTEM_PROMPT
        assert "quarantine" in _SECOND_JUDGE_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Tests: second judge sync call
# ---------------------------------------------------------------------------


class TestSecondJudgeSync:
    @patch("hermes_katana.scanner.consensus_judge.urlopen")
    def test_block_decision(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _make_openai_response("block", "clear injection", 0.95)
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = call_second_judge_sync(SAMPLE_RISK_REPORT)
        assert result.decision == "block"
        assert result.confidence == 0.95
        assert result.model_available is True
        assert result.judge_name == "second_judge"
        assert "injection" in result.reasoning

    @patch("hermes_katana.scanner.consensus_judge.urlopen")
    def test_allow_decision(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _make_openai_response("allow", "false positive", 0.8)
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = call_second_judge_sync(SAMPLE_RISK_REPORT)
        assert result.decision == "allow"
        assert result.model_available is True

    @patch("hermes_katana.scanner.consensus_judge.urlopen")
    def test_timeout_returns_unavailable(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        result = call_second_judge_sync(SAMPLE_RISK_REPORT)
        assert result.model_available is False
        assert result.decision == "quarantine"

    @patch("hermes_katana.scanner.consensus_judge.urlopen")
    def test_connection_refused_returns_unavailable(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")
        result = call_second_judge_sync(SAMPLE_RISK_REPORT)
        assert result.model_available is False
        assert result.decision == "quarantine"

    @patch("hermes_katana.scanner.consensus_judge.urlopen")
    def test_malformed_response_returns_quarantine(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = b'{"choices": [{"message": {"content": "not json"}}]}'
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = call_second_judge_sync(SAMPLE_RISK_REPORT)
        assert result.decision == "quarantine"
        assert result.model_available is True

    @patch("hermes_katana.scanner.consensus_judge.urlopen")
    def test_custom_url_and_model(self, mock_urlopen):
        resp = MagicMock()
        resp.read.return_value = _make_openai_response("allow", "ok", 0.9)
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = resp

        result = call_second_judge_sync(
            SAMPLE_RISK_REPORT,
            url="http://custom:9090/v1/chat/completions",
            model="my-model",
        )
        assert result.decision == "allow"

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "http://custom:9090/v1/chat/completions"


class TestSecondJudgeAsync:
    @pytest.mark.asyncio
    async def test_timeout_logs_structured_event(self, monkeypatch, caplog):
        async def fail(*args, **kwargs):  # noqa: ARG001
            raise TimeoutError("second_judge concurrency gate saturated after 0.500s")

        monkeypatch.setattr(
            "hermes_katana.scanner.consensus_judge.run_in_limited_executor",
            fail,
        )

        with caplog.at_level(logging.WARNING, logger="hermes_katana.scanner.consensus_judge"):
            result = await judge_with_consensus(
                SAMPLE_RISK_REPORT,
                judge1=lambda risk_report, **kwargs: SingleJudgment("custom", "allow", "ok", 0.8, True),
                judge2=None,
            )

        second = next(judge for judge in result.judges if judge.judge_name == "second_judge")
        assert second.model_available is False
        assert second.decision == "quarantine"
        assert any(record.katana_event == "remote_judge_timeout" for record in caplog.records)


# ---------------------------------------------------------------------------
# Tests: async consensus
# ---------------------------------------------------------------------------


class TestJudgeWithConsensusAsync:
    @pytest.mark.asyncio
    @patch("hermes_katana.scanner.consensus_judge.call_second_judge")
    @patch("hermes_katana.scanner.consensus_judge.judge_with_bonsai")
    async def test_agreement_block(self, mock_bonsai, mock_second):
        mock_bonsai.return_value = MagicMock(
            decision="block",
            reasoning="clear attack",
            confidence=0.9,
            model_available=True,
        )
        mock_second.return_value = MagicMock(
            judge_name="second_judge",
            decision="block",
            reasoning="confirmed",
            confidence=0.85,
            model_available=True,
        )

        result = await judge_with_consensus(SAMPLE_RISK_REPORT)

        assert result.decision == "block"
        assert result.agreement == 1.0
        assert result.disagreement is False
        assert len(result.judges) == 2

    @pytest.mark.asyncio
    @patch("hermes_katana.scanner.consensus_judge.call_second_judge")
    @patch("hermes_katana.scanner.consensus_judge.judge_with_bonsai")
    async def test_disagreement_flags_for_review(self, mock_bonsai, mock_second):
        mock_bonsai.return_value = MagicMock(
            decision="block",
            reasoning="attack",
            confidence=0.9,
            model_available=True,
        )
        mock_second.return_value = MagicMock(
            judge_name="second_judge",
            decision="allow",
            reasoning="benign",
            confidence=0.8,
            model_available=True,
        )

        result = await judge_with_consensus(SAMPLE_RISK_REPORT)

        assert result.decision == "flagged_for_review"
        assert result.agreement == 0.5
        assert result.disagreement is True

    @pytest.mark.asyncio
    @patch("hermes_katana.scanner.consensus_judge.call_second_judge")
    @patch("hermes_katana.scanner.consensus_judge.judge_with_bonsai")
    async def test_custom_judges(self, mock_bonsai, mock_second):
        mock_bonsai.return_value = MagicMock(
            decision="allow",
            reasoning="ok",
            confidence=0.9,
            model_available=True,
        )
        mock_second.return_value = MagicMock(
            judge_name="second_judge",
            decision="allow",
            reasoning="ok too",
            confidence=0.8,
            model_available=True,
        )

        async def custom_judge(risk_report, **kwargs):
            return SingleJudgment(
                judge_name="custom",
                decision="allow",
                reasoning="custom",
                confidence=0.95,
                model_available=True,
            )

        result = await judge_with_consensus(
            SAMPLE_RISK_REPORT,
            judge1=custom_judge,
            judge2=custom_judge,
        )

        assert result.decision == "allow"
        assert result.agreement == 1.0

    @pytest.mark.asyncio
    async def test_custom_sync_judges(self):
        def custom_judge(risk_report, **kwargs):
            return SingleJudgment(
                judge_name="custom",
                decision="allow",
                reasoning="sync custom",
                confidence=0.95,
                model_available=True,
            )

        result = await judge_with_consensus(
            SAMPLE_RISK_REPORT,
            judge1=custom_judge,
            judge2=custom_judge,
        )

        assert result.decision == "allow"
        assert all(j.judge_name == "custom" for j in result.judges)

    @pytest.mark.asyncio
    @patch("hermes_katana.scanner.consensus_judge.call_second_judge")
    @patch("hermes_katana.scanner.consensus_judge.judge_with_bonsai")
    async def test_custom_threshold(self, mock_bonsai, mock_second):
        mock_bonsai.return_value = MagicMock(
            decision="block",
            reasoning="attack",
            confidence=0.9,
            model_available=True,
        )
        mock_second.return_value = MagicMock(
            judge_name="second_judge",
            decision="allow",
            reasoning="benign",
            confidence=0.8,
            model_available=True,
        )

        result = await judge_with_consensus(
            SAMPLE_RISK_REPORT,
            disagreement_threshold=0.5,
        )

        # 50% agreement meets threshold=0.5, so no flag
        assert result.disagreement is False
        assert result.decision == "block"  # dominant

    @pytest.mark.asyncio
    @patch("hermes_katana.scanner.consensus_judge.call_second_judge")
    @patch("hermes_katana.scanner.consensus_judge.judge_with_bonsai")
    async def test_judge_exception_handled(self, mock_bonsai, mock_second):
        mock_bonsai.side_effect = RuntimeError("boom")
        mock_second.return_value = MagicMock(
            judge_name="second_judge",
            decision="allow",
            reasoning="benign",
            confidence=0.8,
            model_available=True,
        )

        result = await judge_with_consensus(SAMPLE_RISK_REPORT)

        # Should still get a result with the working judge
        assert len(result.judges) == 2


# ---------------------------------------------------------------------------
# Tests: sync wrapper
# ---------------------------------------------------------------------------


class TestJudgeWithConsensusSync:
    @patch("hermes_katana.scanner.consensus_judge.judge_with_consensus", new_callable=AsyncMock)
    def test_sync_wrapper_runs_in_executor(self, mock_async):
        mock_async.return_value = ConsensusJudgment(
            decision="allow",
            reasoning="test",
            agreement=1.0,
            judges=(),
            disagreement=False,
        )

        result = judge_with_consensus_sync(SAMPLE_RISK_REPORT)

        assert result.decision == "allow"
        mock_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_wrapper_inside_running_loop_returns_result(self):
        def custom_judge(risk_report, **kwargs):
            return SingleJudgment(
                judge_name="custom",
                decision="allow",
                reasoning="sync custom",
                confidence=0.95,
                model_available=True,
            )

        result = judge_with_consensus_sync(
            SAMPLE_RISK_REPORT,
            judge1=custom_judge,
            judge2=custom_judge,
        )

        assert isinstance(result, ConsensusJudgment)
        assert result.decision == "allow"


# ---------------------------------------------------------------------------
# Tests: default URLs / models are configurable
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_second_judge_defaults_are_set(self):
        assert SECOND_JUDGE_URL == "http://localhost:8081/v1/chat/completions"
        assert SECOND_JUDGE_MODEL == "gpt-4o-mini"

    @pytest.mark.asyncio
    @patch("hermes_katana.scanner.consensus_judge.call_second_judge")
    @patch("hermes_katana.scanner.consensus_judge.judge_with_bonsai")
    async def test_custom_urls_passed_correctly(self, mock_bonsai, mock_second):
        mock_bonsai.return_value = MagicMock(
            decision="allow",
            reasoning="ok",
            confidence=0.9,
            model_available=True,
        )
        mock_second.return_value = MagicMock(
            judge_name="second_judge",
            decision="allow",
            reasoning="ok",
            confidence=0.8,
            model_available=True,
        )

        await judge_with_consensus(
            SAMPLE_RISK_REPORT,
            bonsai_url="http://bonsai:8080/v1/chat/completions",
            bonsai_model="bonsai-8b",
            second_url="http://second:8081/v1/chat/completions",
            second_model="claude-3",
        )

        # Verify both were called (bonsai via mock, second via mock)
        mock_bonsai.assert_called_once()
        mock_second.assert_called_once()
