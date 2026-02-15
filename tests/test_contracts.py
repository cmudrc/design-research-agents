from __future__ import annotations

from pathlib import Path

import pytest

from design_research_agents.llm.backends.echo_test import EchoTestBackend
from design_research_agents.llm.backends.factory import build_backend, build_backends
from design_research_agents.llm.backends.llama_cpp import LlamaCppBackend
from design_research_agents.llm.config import (
    EchoTestConfig,
    LlamaCppConfig,
    LLMConfig,
    OpenAICompatibleHTTPConfig,
    OpenAIServiceConfig,
    TransformersLocalConfig,
    backend_config_hash,
    load_config,
)


def _write_config(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8")


def test_load_config_rejects_duplicate_backend_names(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.yaml"
    _write_config(
        config_path,
        "\n".join(
            [
                "backends:",
                "  - name: same",
                "    kind: echo_test",
                "  - name: same",
                "    kind: echo_test",
            ]
        ),
    )

    with pytest.raises(ValueError, match="Duplicate backend name"):
        load_config(str(config_path))


def test_load_config_requires_openai_compatible_base_url(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.yaml"
    _write_config(
        config_path,
        "\n".join(
            [
                "backends:",
                "  - name: compat",
                "    kind: openai_compatible_http",
                "    default_model: local-model",
            ]
        ),
    )

    with pytest.raises(ValueError, match="base_url"):
        load_config(str(config_path))


def test_load_config_parses_capability_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "llm.yaml"
    _write_config(
        config_path,
        "\n".join(
            [
                "backends:",
                "  - name: compat",
                "    kind: openai_compatible_http",
                "    base_url: http://localhost:8000/v1",
                "    default_model: my-model",
                "    capabilities:",
                "      streaming: true",
                "      tool_calling: true",
                "      json_mode: false",
            ]
        ),
    )

    config = load_config(str(config_path))
    backend = config.backends[0]
    built = build_backend(backend)

    caps = built.capabilities()
    assert caps.streaming is True
    assert caps.tool_calling == "native"
    assert caps.json_mode == "none"


def test_build_backends_covers_all_supported_config_types() -> None:
    config = LLMConfig(
        backends=(
            EchoTestConfig(name="echo", kind="echo_test", model="echo-v1"),
            OpenAIServiceConfig(
                name="openai",
                kind="openai_service",
                default_model="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
            ),
            OpenAICompatibleHTTPConfig(
                name="compat",
                kind="openai_compatible_http",
                default_model="local-model",
                base_url="http://localhost:8000/v1",
            ),
            TransformersLocalConfig(
                name="hf",
                kind="transformers_local",
                model_id="distilgpt2",
                default_model="distilgpt2",
            ),
            LlamaCppConfig(
                name="llama",
                kind="llama_cpp",
                model_path="/tmp/model.gguf",
                default_model="local-model",
                api_model="local-model",
            ),
        )
    )

    backends = build_backends(config.backends)

    assert [backend.name for backend in backends] == ["echo", "openai", "compat", "hf", "llama"]
    assert isinstance(backends[0], EchoTestBackend)
    assert isinstance(backends[-1], LlamaCppBackend)


def test_backend_config_hash_is_stable() -> None:
    config = EchoTestConfig(name="echo", kind="echo_test", model="echo-v1")

    first = backend_config_hash(config)
    second = backend_config_hash(config)

    assert first == second
    assert len(first) == 12


def test_llama_cpp_server_command_contains_expected_args() -> None:
    backend = build_backend(
        LlamaCppConfig(
            name="llama",
            kind="llama_cpp",
            model_path="/tmp/model.gguf",
            default_model="local-model",
            api_model="local-model",
            host="0.0.0.0",
            port=9000,
            hf_model_repo_id="repo/id",
            extra_server_args=("--n-gpu-layers", "35"),
        )
    )

    assert isinstance(backend, LlamaCppBackend)
    command = backend._backend._build_command()

    assert "llama_cpp.server" in command
    assert "--model" in command
    assert "/tmp/model.gguf" in command
    assert "--model_alias" in command
    assert "local-model" in command
    assert "--hf_model_repo_id" in command
    assert "repo/id" in command
    assert "--n-gpu-layers" in command
