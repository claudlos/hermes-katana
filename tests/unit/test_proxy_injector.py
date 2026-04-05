"""Tests for HermesKatana proxy credential injector."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_katana.proxy.injector import (
    PROVIDER_REGISTRY,
    get_provider_for_domain,
    inject_credentials,
    list_providers,
)


# ======================================================================
# Provider Registry
# ======================================================================


class TestProviderRegistry:
    def test_registry_has_12_plus_providers(self):
        assert len(PROVIDER_REGISTRY) >= 12

    def test_all_providers_have_required_fields(self):
        for p in PROVIDER_REGISTRY:
            assert p.name, "Provider missing name"
            assert p.domains, f"{p.name} has no domains"
            assert p.key_name, f"{p.name} has no key_name"
            assert p.header_field, f"{p.name} has no header_field"

    def test_provider_is_frozen(self):
        p = PROVIDER_REGISTRY[0]
        with pytest.raises(AttributeError):
            p.name = "hacked"

    def test_openai_provider(self):
        p = get_provider_for_domain("api.openai.com")
        assert p is not None
        assert p.name == "OpenAI"
        assert p.key_name == "OPENAI_API_KEY"

    def test_anthropic_provider(self):
        p = get_provider_for_domain("api.anthropic.com")
        assert p is not None
        assert p.name == "Anthropic"
        assert p.header_field == "x-api-key"
        assert p.auth_scheme == ""  # No Bearer prefix

    def test_google_provider_multiple_domains(self):
        p1 = get_provider_for_domain("generativelanguage.googleapis.com")
        p2 = get_provider_for_domain("aiplatform.googleapis.com")
        assert p1 is not None
        assert p2 is not None
        assert p1.name == "Google"
        assert p1 is p2  # Same provider object


# ======================================================================
# get_provider_for_domain
# ======================================================================


class TestGetProviderForDomain:
    def test_known_domain(self):
        assert get_provider_for_domain("api.openai.com") is not None

    def test_unknown_domain(self):
        assert get_provider_for_domain("example.com") is None

    def test_case_insensitive(self):
        p = get_provider_for_domain("API.OPENAI.COM")
        assert p is not None
        assert p.name == "OpenAI"

    def test_all_registered_domains_resolve(self):
        for provider in PROVIDER_REGISTRY:
            for domain in provider.domains:
                result = get_provider_for_domain(domain)
                assert result is not None, f"Domain {domain} not in index"
                assert result.name == provider.name


# ======================================================================
# inject_credentials
# ======================================================================


def _make_flow(host: str, headers: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(
            host=host,
            headers=dict(headers or {}),
        ),
    )


def _make_vault(secrets: dict[str, str]) -> MagicMock:
    vault = MagicMock()
    vault.get.side_effect = lambda k: secrets.get(k)
    return vault


class TestInjectCredentials:
    def test_injects_openai_bearer(self):
        flow = _make_flow("api.openai.com")
        vault = _make_vault({"OPENAI_API_KEY": "sk-test123"})
        result = inject_credentials(flow, vault)
        assert result == "OpenAI"
        assert "Authorization" in flow.request.headers
        assert "sk-test123" in flow.request.headers["Authorization"]

    def test_anthropic_no_bearer_prefix(self):
        flow = _make_flow("api.anthropic.com")
        vault = _make_vault({"ANTHROPIC_API_KEY": "sk-ant-test"})
        result = inject_credentials(flow, vault)
        assert result == "Anthropic"
        # Anthropic uses x-api-key with no Bearer prefix
        assert flow.request.headers.get("x-api-key") == "sk-ant-test"

    def test_google_custom_header(self):
        flow = _make_flow("generativelanguage.googleapis.com")
        vault = _make_vault({"GOOGLE_API_KEY": "AIza-test"})
        result = inject_credentials(flow, vault)
        assert result == "Google"
        assert flow.request.headers.get("x-goog-api-key") == "AIza-test"

    def test_skips_existing_header(self):
        flow = _make_flow(
            "api.openai.com",
            headers={"Authorization": "Bearer existing-key"},
        )
        vault = _make_vault({"OPENAI_API_KEY": "sk-new"})
        result = inject_credentials(flow, vault)
        assert result is None  # Should not inject
        assert flow.request.headers["Authorization"] == "Bearer existing-key"

    def test_empty_existing_header_gets_injected(self):
        flow = _make_flow(
            "api.openai.com",
            headers={"Authorization": "  "},
        )
        vault = _make_vault({"OPENAI_API_KEY": "sk-injected"})
        result = inject_credentials(flow, vault)
        assert result == "OpenAI"

    def test_unknown_domain_returns_none(self):
        flow = _make_flow("random-api.com")
        vault = _make_vault({})
        result = inject_credentials(flow, vault)
        assert result is None

    def test_missing_vault_key_returns_none(self):
        flow = _make_flow("api.openai.com")
        vault = _make_vault({})  # No keys
        result = inject_credentials(flow, vault)
        assert result is None

    def test_flow_missing_host_attribute(self):
        flow = SimpleNamespace(request=SimpleNamespace())
        vault = _make_vault({})
        result = inject_credentials(flow, vault)
        assert result is None

    def test_vault_get_raises_returns_none(self):
        flow = _make_flow("api.openai.com")
        vault = MagicMock()
        vault.get.side_effect = Exception("vault locked")
        result = inject_credentials(flow, vault)
        assert result is None


# ======================================================================
# list_providers
# ======================================================================


class TestListProviders:
    def test_returns_list_of_dicts(self):
        providers = list_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 12
        for p in providers:
            assert "name" in p
            assert "domains" in p
            assert "key_name" in p
            assert "header_field" in p
            assert "auth_scheme" in p

    def test_all_providers_represented(self):
        names = {p["name"] for p in list_providers()}
        expected = {
            "OpenAI",
            "Anthropic",
            "Google",
            "Groq",
            "Together",
            "OpenRouter",
            "DeepSeek",
            "Mistral",
            "Cohere",
            "Replicate",
            "HuggingFace",
        }
        assert expected.issubset(names)
