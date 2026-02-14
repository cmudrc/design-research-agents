import pytest

from design_research_agents import complete
from design_research_agents.llm import (
    complete as llm_complete,
)
from design_research_agents.llm import (
    configure_llama_cpp_server,
    configure_openai,
    parse_backend,
    shutdown_llama_cpp_server,
)
from design_research_agents.llm.backends.llama_cpp_server import (
    create_backend as create_llama_cpp_backend,
)


def test_local_backend_completion() -> None:
    # Local backend is deterministic and should echo a normalized prompt.
    result = complete("Hello from tests")
    assert result.startswith("[local-echo]")
    assert "Hello from tests" in result


def test_unknown_backend_raises_value_error() -> None:
    # Unknown backend names should fail fast with a clear validation error.
    with pytest.raises(ValueError):
        llm_complete("hello", backend="does-not-exist")


def test_backend_name_parsing() -> None:
    # Parsing normalizes backend names as plain strings.
    assert parse_backend("openai") == "openai"
    assert parse_backend(" LOCAL ") == "local"


def test_openai_backend_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenAI backend should require credentials when no explicit key is provided.
    configure_openai(
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_key=None,
        base_url=None,
        require_api_key=True,
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        llm_complete("hello", backend="openai")


def test_llama_backend_requires_configuration() -> None:
    # The managed llama server must be configured before it can be used.
    shutdown_llama_cpp_server()
    with pytest.raises(RuntimeError):
        llm_complete("hello", backend="llama-cpp-server")


def test_configure_llama_backend_replaces_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Capture create/close events to verify lifecycle transitions.
    events: list[tuple[str, str]] = []

    class FakeLlamaBackend:
        # Minimal stand-in to track shutdown behavior.
        def __init__(self, model: str) -> None:
            self.model = model

        def close(self) -> None:
            events.append(("close", self.model))

        def complete(self, prompt: str) -> str:
            return prompt

    def fake_factory(model: str, **_: object) -> FakeLlamaBackend:
        events.append(("create", model))
        return FakeLlamaBackend(model)

    # Swap real backend construction with a deterministic fake.
    monkeypatch.setattr("design_research_agents.llm.create_llama_cpp_server_backend", fake_factory)

    configure_llama_cpp_server(model="/tmp/first.gguf")
    configure_llama_cpp_server(model="/tmp/second.gguf")
    shutdown_llama_cpp_server()

    assert events == [
        ("create", "/tmp/first.gguf"),
        ("close", "/tmp/first.gguf"),
        ("create", "/tmp/second.gguf"),
        ("close", "/tmp/second.gguf"),
    ]


def test_llama_backend_model_source_validation() -> None:
    # The upstream-aligned API always requires a non-empty model argument.
    with pytest.raises(ValueError):
        create_llama_cpp_backend(model="")

    backend = create_llama_cpp_backend(
        model="tinyllama.Q4_K_M.gguf",
        hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    )
    assert backend.model == "tinyllama.Q4_K_M.gguf"
    assert backend.hf_model_repo_id == "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"


def test_configure_llama_backend_accepts_hf_args(monkeypatch: pytest.MonkeyPatch) -> None:
    # The public configure API should pass Hugging Face args through to backend creation.
    captured: dict[str, object] = {}

    class FakeLlamaBackend:
        def close(self) -> None:
            return

        def complete(self, prompt: str) -> str:
            return prompt

    def fake_factory(**kwargs: object) -> FakeLlamaBackend:
        captured.update(kwargs)
        return FakeLlamaBackend()

    monkeypatch.setattr("design_research_agents.llm.create_llama_cpp_server_backend", fake_factory)

    configure_llama_cpp_server(
        model="tinyllama.Q4_K_M.gguf",
        hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    )
    shutdown_llama_cpp_server()

    assert captured["model"] == "tinyllama.Q4_K_M.gguf"
    assert captured["hf_model_repo_id"] == "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"


def test_configure_openai_updates_default_call_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Configure once and verify OpenAI calls reuse those defaults.
    captured: dict[str, object] = {}

    def fake_openai_complete(prompt: str, **kwargs: object) -> str:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr("design_research_agents.llm.openai_complete", fake_openai_complete)

    configure_openai(
        model="gpt-test",
        api_key_env="ALT_OPENAI_API_KEY",
        api_key="explicit-key",
        base_url="http://localhost:9000/v1",
        require_api_key=False,
    )

    assert llm_complete("hello", backend="openai") == "ok"
    assert captured["prompt"] == "hello"
    assert captured["model"] == "gpt-test"
    assert captured["api_key_env"] == "ALT_OPENAI_API_KEY"
    assert captured["api_key"] == "explicit-key"
    assert captured["base_url"] == "http://localhost:9000/v1"
    assert captured["require_api_key"] is False

    # Reset to project defaults so other tests and examples remain predictable.
    configure_openai(
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_key=None,
        base_url=None,
        require_api_key=True,
    )
