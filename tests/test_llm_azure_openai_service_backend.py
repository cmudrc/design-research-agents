from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from design_research_agents.llm._backends._providers import _azure_openai_service as azure_service


def test__azure_openai_service_backend_resolves_env_and_creates_client(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = azure_service.AzureOpenAIServiceBackend(
        name="azure-openai",
        default_model="gpt-4o-mini",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_key=None,
        azure_endpoint_env="AZURE_OPENAI_ENDPOINT",
        azure_endpoint=None,
        api_version_env="AZURE_OPENAI_API_VERSION",
        api_version=None,
        config_hash="cfg",
    )
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    created: list[dict[str, object]] = []

    class _AzureClient:
        def __init__(self, **kwargs: object) -> None:
            created.append(dict(kwargs))

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AzureOpenAI=_AzureClient))

    client = backend._create_client()

    assert isinstance(client, _AzureClient)
    assert backend.kind == "azure_openai_service"
    assert created == [
        {
            "api_key": "env-key",
            "azure_endpoint": "https://example-resource.openai.azure.com",
            "api_version": "2025-01-01-preview",
        }
    ]


def test__azure_openai_service_backend_missing_settings_and_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = azure_service.AzureOpenAIServiceBackend(
        name="azure-openai",
        default_model="gpt-4o-mini",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_key="local-key",
        azure_endpoint_env="AZURE_OPENAI_ENDPOINT",
        azure_endpoint=None,
        api_version_env="AZURE_OPENAI_API_VERSION",
        api_version=None,
        config_hash="cfg",
    )
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)

    with pytest.raises(RuntimeError, match="AZURE_OPENAI_ENDPOINT is not set"):
        backend._resolve_azure_endpoint()
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_VERSION is not set"):
        backend._resolve_api_version()

    backend = azure_service.AzureOpenAIServiceBackend(
        name="azure-openai",
        default_model="gpt-4o-mini",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_key="local-key",
        azure_endpoint_env="AZURE_OPENAI_ENDPOINT",
        azure_endpoint="https://example-resource.openai.azure.com",
        api_version_env="AZURE_OPENAI_API_VERSION",
        api_version="2025-01-01-preview",
        config_hash="cfg",
    )

    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "openai":
            raise ImportError("missing openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="openai"):
        backend._create_client()
