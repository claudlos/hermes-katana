"""Property tests for proxy credential injection boundaries."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from hermes_katana.proxy.addon import KatanaAddon
from hermes_katana.proxy.config import ProxyConfig
from hermes_katana.proxy.injector import PROVIDER_REGISTRY, get_provider_for_domain, inject_credentials_with_metadata


REGISTERED_DOMAINS = tuple(domain for provider in PROVIDER_REGISTRY for domain in provider.domains)
UNKNOWN_DOMAINS = ("example.com", "api.openai.com.evil.test", "localhost", "not-a-provider.invalid")


def _make_flow(host: str, headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(
            host=host,
            url=f"https://{host}/v1/chat",
            headers=dict(headers or {}),
            query={},
            get_content=lambda: b"",
        ),
        response=None,
        client_conn=SimpleNamespace(peername=("127.0.0.1", 54321)),
    )


def _vault_for(provider_key: str | None, value: str | None) -> MagicMock:
    vault = MagicMock()
    vault.get.side_effect = lambda key: value if provider_key is not None and key == provider_key else None
    vault._get_all_values.return_value = {provider_key: value} if provider_key and value else {}
    return vault


@settings(max_examples=80)
@given(
    host=st.sampled_from(REGISTERED_DOMAINS + UNKNOWN_DOMAINS),
    existing_header=st.sampled_from([None, "", "  ", "Bearer already-present"]),
    vault_has_secret=st.booleans(),
    secret_suffix=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=48, max_codepoint=122),
        min_size=8,
        max_size=24,
    ),
)
def test_injector_injects_only_for_provider_blank_header_and_present_secret(
    host: str,
    existing_header: str | None,
    vault_has_secret: bool,
    secret_suffix: str,
) -> None:
    provider = get_provider_for_domain(host)
    headers: dict[str, str] = {}
    if provider is not None and existing_header is not None:
        headers[provider.header_field] = existing_header
    flow = _make_flow(host, headers=headers)
    secret = f"sk-prop-{secret_suffix}" if vault_has_secret else None
    vault = _vault_for(provider.key_name if provider is not None else None, secret)

    result = inject_credentials_with_metadata(flow, vault)

    expected = provider is not None and vault_has_secret and (existing_header is None or not existing_header.strip())
    assert (result is not None) is expected
    if expected:
        assert result is not None
        assert result.provider_name == provider.name
        assert flow.request.headers[provider.header_field] == result.header_value
        assert secret in result.header_value
    elif provider is not None and existing_header and existing_header.strip():
        assert flow.request.headers[provider.header_field] == existing_header


@settings(max_examples=len(REGISTERED_DOMAINS))
@given(host=st.sampled_from(REGISTERED_DOMAINS))
def test_proxy_refuses_injection_when_tls_verification_is_disabled(host: str) -> None:
    provider = get_provider_for_domain(host)
    assert provider is not None
    addon = KatanaAddon(
        config=ProxyConfig(inject_credentials=True, tls_verify=False),
        vault=_vault_for(provider.key_name, "sk-prop-disabledtls123"),
        audit=None,
    )
    flow = _make_flow(host)

    with patch("hermes_katana.proxy.addon.inject_credentials_with_metadata") as inject:
        addon.request(flow)

    inject.assert_not_called()
    assert flow.response is not None
    assert flow.response.status_code == 502
    assert addon.get_stats().get("requests_blocked_insecure_tls", 0) == 1


@settings(max_examples=len(UNKNOWN_DOMAINS))
@given(host=st.sampled_from(UNKNOWN_DOMAINS))
def test_unknown_domains_never_receive_vault_credentials(host: str) -> None:
    addon = KatanaAddon(
        config=ProxyConfig(inject_credentials=True),
        vault=_vault_for("OPENAI_API_KEY", "sk-prop-unknown123"),
        audit=None,
    )
    flow = _make_flow(host)
    clean = {"verdict": "pass", "risk_score": 0, "is_blocked": False, "finding_count": 0, "summary": ""}

    with patch.object(addon, "_scan_text", return_value=clean):
        addon.request(flow)

    assert "Authorization" not in flow.request.headers
    assert addon.get_stats().get("credentials_injected", 0) == 0
