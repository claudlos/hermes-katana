"""Tests for hermes_katana.proxy.injector — credential injection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from hermes_katana.proxy.injector import (
    PROVIDER_REGISTRY,
    Provider,
    get_provider_for_domain,
    inject_credentials,
    list_providers,
    _build_domain_index,
)


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockHeaders(dict):
    pass


class MockRequest:
    def __init__(self, host, headers=None):
        self.host = host
        self.headers = MockHeaders(headers or {})


class MockFlow:
    def __init__(self, host, headers=None):
        self.request = MockRequest(host, headers)


# ---------------------------------------------------------------------------
# Provider registry tests
# ---------------------------------------------------------------------------

class TestProviderRegistry:
    def test_registry_has_12_plus_providers(self):
        assert len(PROVIDER_REGISTRY) >= 12

    def test_all_providers_have_required_fields(self):
        for p in PROVIDER_REGISTRY:
            assert p.name
            assert p.domains
            assert p.key_name
            assert p.header_field

    def test_provider_is_frozen(self):
        p = PROVIDER_REGISTRY[0]
        with pytest.raises(AttributeError):
            p.name = "changed"


class TestGetProviderForDomain:
    def test_openai(self):
        p = get_provider_for_domain("api.openai.com")
        assert p is not None
        assert p.name == "OpenAI"

    def test_anthropic(self):
        p = get_provider_for_domain("api.anthropic.com")
        assert p is not None
        assert p.name == "Anthropic"

    def test_google_generative(self):
        p = get_provider_for_domain("generativelanguage.googleapis.com")
        assert p is not None
        assert p.name == "Google"

    def test_google_aiplatform(self):
        p = get_provider_for_domain("aiplatform.googleapis.com")
        assert p is not None
        assert p.name == "Google"

    def test_groq(self):
        p = get_provider_for_domain("api.groq.com")
        assert p is not None
        assert p.name == "Groq"

    def test_together_xyz(self):
        p = get_provider_for_domain("api.together.xyz")
        assert p is not None
        assert p.name == "Together"

    def test_together_ai(self):
        p = get_provider_for_domain("api.together.ai")
        assert p is not None
        assert p.name == "Together"

    def test_openrouter(self):
        p = get_provider_for_domain("openrouter.ai")
        assert p is not None
        assert p.name == "OpenRouter"

    def test_deepseek(self):
        p = get_provider_for_domain("api.deepseek.com")
        assert p is not None
        assert p.name == "DeepSeek"

    def test_mistral(self):
        p = get_provider_for_domain("api.mistral.ai")
        assert p is not None
        assert p.name == "Mistral"

    def test_cohere(self):
        p = get_provider_for_domain("api.cohere.ai")
        assert p is not None
        assert p.name == "Cohere"

    def test_replicate(self):
        p = get_provider_for_domain("api.replicate.com")
        assert p is not None
        assert p.name == "Replicate"

    def test_huggingface(self):
        p = get_provider_for_domain("api-inference.huggingface.co")
        assert p is not None
        assert p.name == "HuggingFace"

    def test_unknown_domain(self):
        p = get_provider_for_domain("unknown.example.com")
        assert p is None

    def test_case_insensitive(self):
        p = get_provider_for_domain("API.OPENAI.COM")
        assert p is not None
        assert p.name == "OpenAI"

    def test_vercel(self):
        p = get_provider_for_domain("api.vercel.ai")
        assert p is not None
        assert p.name == "Vercel"


class TestInjectCredentials:
    def test_inject_openai(self):
        vault = MagicMock()
        vault.get.return_value = "sk-test-key"
        flow = MockFlow("api.openai.com")
        result = inject_credentials(flow, vault)
        assert result == "OpenAI"
        assert "Authorization" in flow.request.headers

    def test_inject_anthropic_uses_x_api_key(self):
        vault = MagicMock()
        vault.get.return_value = "sk-ant-test"
        flow = MockFlow("api.anthropic.com")
        result = inject_credentials(flow, vault)
        assert result == "Anthropic"
        assert "x-api-key" in flow.request.headers

    def test_skip_if_header_already_set(self):
        vault = MagicMock()
        vault.get.return_value = "sk-test"
        flow = MockFlow("api.openai.com", headers={"Authorization": "Bearer existing"})
        result = inject_credentials(flow, vault)
        assert result is None

    def test_unknown_domain_returns_none(self):
        vault = MagicMock()
        flow = MockFlow("unknown.example.com")
        result = inject_credentials(flow, vault)
        assert result is None

    def test_vault_key_missing(self):
        vault = MagicMock()
        vault.get.side_effect = KeyError("not found")
        flow = MockFlow("api.openai.com")
        result = inject_credentials(flow, vault)
        assert result is None

    def test_flow_missing_host(self):
        flow = MagicMock(spec=[])
        vault = MagicMock()
        result = inject_credentials(flow, vault)
        assert result is None

    def test_vault_returns_empty_string(self):
        vault = MagicMock()
        vault.get.return_value = ""
        flow = MockFlow("api.openai.com")
        result = inject_credentials(flow, vault)
        assert result is None

    def test_google_no_bearer_prefix(self):
        vault = MagicMock()
        vault.get.return_value = "google-key-123"
        flow = MockFlow("generativelanguage.googleapis.com")
        result = inject_credentials(flow, vault)
        assert result == "Google"
        val = flow.request.headers.get("x-goog-api-key", "")
        # Google uses empty auth_scheme, so raw key
        assert "google-key-123" in val


class TestListProviders:
    def test_returns_list(self):
        providers = list_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 12

    def test_provider_dict_fields(self):
        providers = list_providers()
        for p in providers:
            assert "name" in p
            assert "domains" in p
            assert "key_name" in p
            assert "header_field" in p
            assert "auth_scheme" in p

    def test_openai_in_list(self):
        providers = list_providers()
        names = [p["name"] for p in providers]
        assert "OpenAI" in names


class TestBuildDomainIndex:
    def test_builds_index(self):
        idx = _build_domain_index()
        assert isinstance(idx, dict)
        assert "api.openai.com" in idx

    def test_all_domains_indexed(self):
        idx = _build_domain_index()
        for provider in PROVIDER_REGISTRY:
            for domain in provider.domains:
                assert domain.lower() in idx
