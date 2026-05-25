"""
API key injection for LLM provider traffic.

Maps LLM provider domains to their vault key names and HTTP header
fields, enabling transparent credential injection into proxied requests.

Supports 12+ providers including OpenAI, Anthropic, Google, Groq,
Together, OpenRouter, Vercel, DeepSeek, Mistral, Cohere, Replicate,
and HuggingFace.
"""

from __future__ import annotations

__all__ = [
    "InjectedCredential",
    "Provider",
    "PROVIDER_REGISTRY",
    "get_provider_for_domain",
    "inject_credentials",
    "inject_credentials_with_metadata",
    "list_providers",
]


import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_katana.vault.store import Vault

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Provider:
    """An LLM API provider definition.

    Attributes:
        name: Human-readable provider name.
        domains: List of domains associated with this provider.
        key_name: Vault key name for the API credential.
        header_field: HTTP header to inject the credential into.
        auth_scheme: Authorization scheme prefix (e.g., 'Bearer').
            Empty string means the raw key is injected without prefix.
    """

    name: str
    domains: list[str] = field(default_factory=list)
    key_name: str = ""
    header_field: str = "Authorization"
    auth_scheme: str = "Bearer"


@dataclass(frozen=True, slots=True)
class InjectedCredential:
    """Metadata for a credential injection performed on a request."""

    provider_name: str
    header_field: str
    header_value: str
    secret_value: str


# ---------------------------------------------------------------------------
# Provider Registry
# ---------------------------------------------------------------------------

PROVIDER_REGISTRY: list[Provider] = [
    Provider(
        name="OpenAI",
        domains=["api.openai.com"],
        key_name="OPENAI_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="Anthropic",
        domains=["api.anthropic.com"],
        key_name="ANTHROPIC_API_KEY",
        header_field="x-api-key",
        auth_scheme="",
    ),
    Provider(
        name="Google",
        domains=[
            "generativelanguage.googleapis.com",
            "aiplatform.googleapis.com",
        ],
        key_name="GOOGLE_API_KEY",
        header_field="x-goog-api-key",
        auth_scheme="",
    ),
    Provider(
        name="Groq",
        domains=["api.groq.com"],
        key_name="GROQ_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="Together",
        domains=["api.together.xyz", "api.together.ai"],
        key_name="TOGETHER_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="OpenRouter",
        domains=["openrouter.ai"],
        key_name="OPENROUTER_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="Vercel",
        domains=["api.vercel.ai", "sdk.vercel.ai"],
        key_name="VERCEL_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="DeepSeek",
        domains=["api.deepseek.com"],
        key_name="DEEPSEEK_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="Mistral",
        domains=["api.mistral.ai"],
        key_name="MISTRAL_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="Cohere",
        domains=["api.cohere.ai", "api.cohere.com"],
        key_name="COHERE_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="Replicate",
        domains=["api.replicate.com"],
        key_name="REPLICATE_API_TOKEN",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
    Provider(
        name="HuggingFace",
        domains=[
            "api-inference.huggingface.co",
            "huggingface.co",
        ],
        key_name="HUGGINGFACE_API_KEY",
        header_field="Authorization",
        auth_scheme="Bearer",
    ),
]


def _build_domain_index() -> dict[str, Provider]:
    """Build a domain -> Provider lookup index."""
    index: dict[str, Provider] = {}
    for provider in PROVIDER_REGISTRY:
        for domain in provider.domains:
            index[domain.lower()] = provider
    return index


# Pre-built domain index for O(1) lookups
_DOMAIN_INDEX: dict[str, Provider] = _build_domain_index()


def get_provider_for_domain(domain: str) -> Optional[Provider]:
    """Look up the LLM provider for a given domain.

    Args:
        domain: The request hostname (e.g., 'api.openai.com').

    Returns:
        The matching Provider, or None if the domain is not recognized.
    """
    return _DOMAIN_INDEX.get(domain.lower())


def inject_credentials(
    flow: Any,
    vault: "Vault",
) -> Optional[str]:
    """Inject API credentials from the vault and return the provider name."""
    result = inject_credentials_with_metadata(flow, vault)
    return result.provider_name if result is not None else None


def inject_credentials_with_metadata(
    flow: Any,
    vault: "Vault",
) -> Optional[InjectedCredential]:
    """Inject API credentials from the vault into an HTTP flow.

    Looks up the flow's target domain in the provider registry. If a match
    is found and the vault contains the corresponding API key, injects it
    into the request headers.

    Args:
        flow: A mitmproxy HTTP flow object (or any object with
            ``request.host`` and ``request.headers`` attributes).
        vault: The Vault instance to retrieve credentials from.

    Returns:
        Injection metadata if credentials were injected, None otherwise.

    Note:
        This function does NOT overwrite existing authorization headers
        unless they are empty. This prevents accidentally replacing
        user-provided credentials.
    """
    try:
        host = flow.request.host.lower()
    except AttributeError:
        logger.warning("Flow object missing request.host attribute")
        return None

    provider = get_provider_for_domain(host)
    if provider is None:
        return None

    # Skip if the header already has a value
    existing = flow.request.headers.get(provider.header_field, "")
    if existing.strip():
        logger.debug(
            "Skipping credential injection for %s: header '%s' already set",
            provider.name,
            provider.header_field,
        )
        return None

    # Retrieve key from vault
    try:
        api_key = vault.get(provider.key_name)
    except Exception:
        logger.debug(
            "No vault key '%s' for provider %s",
            provider.key_name,
            provider.name,
        )
        return None

    if not api_key:
        return None

    # Build the header value
    if provider.auth_scheme:
        header_value = f"{provider.auth_scheme} {api_key}"
    else:
        header_value = api_key

    flow.request.headers[provider.header_field] = header_value
    logger.info(
        "Injected credentials for %s via header '%s'",
        provider.name,
        provider.header_field,
    )
    return InjectedCredential(
        provider_name=provider.name,
        header_field=provider.header_field,
        header_value=header_value,
        secret_value=api_key,
    )


def list_providers() -> list[dict[str, Any]]:
    """Return a summary of all registered providers.

    Returns:
        List of dicts with provider info (name, domains, key_name, header).
    """
    return [
        {
            "name": p.name,
            "domains": list(p.domains),
            "key_name": p.key_name,
            "header_field": p.header_field,
            "auth_scheme": p.auth_scheme,
        }
        for p in PROVIDER_REGISTRY
    ]
