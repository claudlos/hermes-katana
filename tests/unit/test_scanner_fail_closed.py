"""Regression tests for scanner fail-closed behavior."""

from __future__ import annotations

from hermes_katana.ml_artifacts import UnsafeArtifactError
from hermes_katana.scanner import ScanVerdict, scan_input


def test_optional_scanner_runtime_failure_blocks_in_high_security(monkeypatch):
    import hermes_katana.scanner as scanner

    def broken_semantic(_text: str, topk: int = 5):
        raise RuntimeError("semantic backend unavailable")

    monkeypatch.setattr(scanner, "detect_semantic", broken_semantic)

    result = scan_input(
        "ordinary text that depends on optional high-security coverage",
        check_injection=False,
        check_secrets=False,
        check_unicode=False,
        check_content_harm=False,
        check_prompt_leak=False,
        security_level="high",
    )

    assert result.verdict == ScanVerdict.BLOCK
    assert result.risk_score >= 0.7
    assert result.metadata["scanner_failures"][0]["scanner"] == "semantic_recall"


def test_unexpected_keyerror_blocks_in_high_security(monkeypatch):
    import hermes_katana.scanner as scanner

    def broken_semantic(_text: str, topk: int = 5):
        raise KeyError("unexpected_state")

    monkeypatch.setattr(scanner, "detect_semantic", broken_semantic)

    result = scan_input(
        "ordinary text that depends on optional high-security coverage",
        check_injection=False,
        check_secrets=False,
        check_unicode=False,
        check_content_harm=False,
        check_prompt_leak=False,
        security_level="high",
    )

    assert result.verdict == ScanVerdict.BLOCK
    assert result.risk_score >= 0.7
    assert result.metadata["scanner_failure_action"] == "block"


def test_untrusted_ml_artifact_failure_blocks_in_high_security(monkeypatch):
    import hermes_katana.scanner as scanner

    def unsafe_semantic(_text: str, topk: int = 5):
        raise UnsafeArtifactError("untrusted semantic projector")

    monkeypatch.setattr(scanner, "detect_semantic", unsafe_semantic)

    result = scan_input(
        "ordinary text that depends on optional high-security coverage",
        check_injection=False,
        check_secrets=False,
        check_unicode=False,
        check_content_harm=False,
        check_prompt_leak=False,
        security_level="high",
    )

    assert result.verdict == ScanVerdict.BLOCK
    assert result.risk_score >= 0.7
    assert result.metadata["scanner_failure_action"] == "block"
    assert result.metadata["scanner_failures"][0]["scanner"] == "semantic_recall"


def test_known_semantic_index_lock_failure_records_without_blocking(monkeypatch):
    import hermes_katana.scanner as scanner

    def locked_semantic(_text: str, topk: int = 5):
        raise RuntimeError("Can't lock read-write collection")

    monkeypatch.setattr(scanner, "detect_semantic", locked_semantic)

    result = scan_input(
        "ordinary text while the optional semantic index is locked elsewhere",
        check_injection=False,
        check_secrets=False,
        check_unicode=False,
        check_content_harm=False,
        check_prompt_leak=False,
        security_level="high",
    )

    assert result.verdict == ScanVerdict.ALLOW
    assert result.metadata["scanner_failure_action"] == "record"
    assert result.metadata["scanner_failures"][0]["scanner"] == "semantic_recall"


def test_known_semantic_zvec_api_mismatch_records_without_blocking(monkeypatch):
    import hermes_katana.scanner as scanner

    def mismatched_zvec_semantic(_text: str, topk: int = 5):
        raise AttributeError("module 'zvec' has no attribute 'open'")

    monkeypatch.setattr(scanner, "detect_semantic", mismatched_zvec_semantic)

    result = scan_input(
        "ordinary text while optional semantic recall has an incompatible zvec package",
        check_injection=False,
        check_secrets=False,
        check_unicode=False,
        check_content_harm=False,
        check_prompt_leak=False,
        security_level="high",
    )

    assert result.verdict == ScanVerdict.ALLOW
    assert result.metadata["scanner_failure_action"] == "record"
    assert result.metadata["scanner_failures"][0]["scanner"] == "semantic_recall"


def test_optional_scanner_runtime_failure_warns_in_medium_security(monkeypatch):
    import hermes_katana.scanner as scanner

    def broken_behavioral(_text: str):
        raise RuntimeError("behavioral backend unavailable")

    monkeypatch.setattr(scanner, "detect_behavioral", broken_behavioral)

    result = scan_input(
        "ordinary text that depends on optional medium-security coverage",
        check_injection=False,
        check_secrets=False,
        check_unicode=False,
        check_content_harm=False,
        check_prompt_leak=False,
        security_level="medium",
    )

    assert result.verdict == ScanVerdict.WARN
    assert result.risk_score >= 0.3
    assert result.metadata["scanner_failures"][0]["scanner"] == "behavioral"
