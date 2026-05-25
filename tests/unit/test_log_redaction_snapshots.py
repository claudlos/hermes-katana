"""Snapshot-style tests for high-risk runtime log redaction."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

from hermes_katana.proxy.addon import KatanaAddon
from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.security_logging import REDACTED, log_security_event, redact_for_log, redact_text_for_log

from tests.test_proxy_addon import MockFlow


def test_security_logging_redacts_credential_text_snapshot(caplog) -> None:
    logger = logging.getLogger("tests.security_redaction")
    caplog.set_level(logging.INFO, logger=logger.name)

    payload = {
        "setup_message": "Authorization: Bearer sk-phase2-redaction-token123",
        "vault_path": "/tmp/hermes/vault.json",
        "migration": "password=hunter2",
        "runner_env": {"OPENAI_API_KEY": "sk-phase2-env-token123"},
    }
    log_security_event(logger, logging.INFO, "phase2_redaction_snapshot", **payload)

    record = caplog.records[-1]
    assert "sk-phase2-redaction-token123" not in record.message
    assert "sk-phase2-env-token123" not in record.message
    assert "hunter2" not in record.message
    assert record.katana_payload == {
        "migration": f"password={REDACTED}",
        "runner_env": {"OPENAI_API_KEY": REDACTED},
        "setup_message": f"Authorization: Bearer {REDACTED}",
        "vault_path": REDACTED,
    }


def test_redact_for_log_redacts_nested_sensitive_keys_snapshot() -> None:
    value = {
        "provider_token": "sk-phase2-provider-token123",
        "safe_field": "visible",
        "nested": {"authorization": "Bearer sk-phase2-auth-token123"},
    }

    assert redact_for_log(value) == {
        "provider_token": REDACTED,
        "safe_field": "visible",
        "nested": {"authorization": REDACTED},
    }


def test_redact_text_for_log_redacts_current_vault_values_snapshot() -> None:
    secret = "plain-runtime-secret-value"

    assert redact_text_for_log(f"scanner summary mentioned {secret}", extra_values={secret}) == (
        f"scanner summary mentioned {REDACTED}"
    )


def test_proxy_scan_logs_audit_and_block_messages_redact_vault_values(caplog) -> None:
    secret = "plain-runtime-secret-value"
    token = "sk-phase2-proxy-token123"
    summary = f"detected leaked {secret} with Authorization: Bearer {token}"
    audit = MagicMock()
    vault = MagicMock()
    vault._get_all_values.return_value = {"OPENAI_API_KEY": secret}
    addon = KatanaAddon(config=ProxyConfig(inject_credentials=False), vault=vault, audit=audit)
    flow = MockFlow(host="example.com", body=b"body")
    blocked = {"verdict": "block", "risk_score": 100, "is_blocked": True, "finding_count": 1, "summary": summary}

    caplog.set_level(logging.WARNING, logger="hermes_katana.proxy.addon")
    with patch.object(addon, "_scan_text", return_value=blocked):
        addon.request(flow)

    audit_entry = audit.log.call_args.args[0]
    response_body = flow.response.get_content().decode("utf-8")
    serialized_entry = json.dumps(audit_entry.__dict__, sort_keys=True, default=str)

    assert secret not in caplog.text
    assert token not in caplog.text
    assert secret not in serialized_entry
    assert token not in serialized_entry
    assert secret not in response_body
    assert token not in response_body
    assert REDACTED in serialized_entry
    assert REDACTED in response_body
