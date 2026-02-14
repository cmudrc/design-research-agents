"""Core LLM backend behavior tests.

This suite validates backend parsing, configuration precedence, compatibility
fallbacks, and llama-cpp lifecycle/readiness handling.
"""

import sys
import types
from urllib.error import HTTPError

import pytest

from design_research_agents import complete
from design_research_agents.llm import (
    BaseLLMClient,
    configure_llama_cpp_server,
    configure_openai,
    parse_backend,
    resolve_default_model,
    shutdown_llama_cpp_server,
)
from design_research_agents.llm import (
    complete as llm_complete,
)
from design_research_agents.llm.backends.llama_cpp_server import (
    LlamaCppServerBackend,
)
from design_research_agents.llm.backends.llama_cpp_server import (
    create_backend as create_llama_cpp_backend,
)
from design_research_agents.llm.backends.openai import OpenAIBackend


def test_echo_test_backend_completion() -> None:
    # Echo-test backend is deterministic and should echo a normalized prompt.
    result = complete("Hello from tests", backend="echo-test")
    assert result.startswith("[echo-test]")
    assert "Hello from tests" in result


def test_unknown_backend_raises_value_error() -> None:
    # Unknown backend names should fail fast with a clear validation error.
    with pytest.raises(ValueError):
        llm_complete("hello", backend="does-not-exist")


def test_backend_name_parsing() -> None:
    # Parsing normalizes backend names as plain strings.
    assert parse_backend("openai") == "openai"
    assert parse_backend(" echo-test ") == "echo-test"
    with pytest.raises(ValueError):
        parse_backend(" LOCAL ")


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


def test_openai_backend_uses_chat_fallback_for_compatible_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAI-compatible local servers may not implement /v1/responses.
    captured: dict[str, object] = {}

    class FakeNotFoundError(Exception):
        status_code = 404

    class FakeResponses:
        def create(self, *, model: str, input: str) -> object:
            captured["responses_model"] = model
            captured["responses_input"] = input
            raise FakeNotFoundError()

    class FakeChatCompletions:
        def create(self, *, model: str, messages: list[dict[str, str]]) -> object:
            captured["chat_model"] = model
            captured["chat_messages"] = messages
            message = types.SimpleNamespace(content="hello from chat fallback")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = FakeResponses()
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    backend = OpenAIBackend(
        model="local-model",
        api_key="not-needed",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
    )

    result = backend.complete("hello")
    assert result == "hello from chat fallback"
    assert captured["responses_model"] == "local-model"
    assert captured["chat_model"] == "local-model"


def test_openai_backend_uses_chat_fallback_when_not_found_has_url_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAI SDK errors can carry URL objects instead of plain strings.
    captured: dict[str, object] = {}

    class FakeUrl:
        def __str__(self) -> str:
            return "http://127.0.0.1:8001/v1/responses"

    class FakeNotFoundError(Exception):
        status_code = 404

        def __init__(self) -> None:
            self.request = types.SimpleNamespace(url=FakeUrl())

    class FakeResponses:
        def create(self, *, model: str, input: str) -> object:
            captured["responses_model"] = model
            captured["responses_input"] = input
            raise FakeNotFoundError()

    class FakeChatCompletions:
        def create(self, *, model: str, messages: list[dict[str, str]]) -> object:
            captured["chat_model"] = model
            captured["chat_messages"] = messages
            message = types.SimpleNamespace(content="hello from object-url fallback")
            choice = types.SimpleNamespace(message=message)
            return types.SimpleNamespace(choices=[choice])

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = FakeResponses()
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    backend = OpenAIBackend(
        model="local-model",
        api_key="not-needed",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
    )

    result = backend.complete("hello")
    assert result == "hello from object-url fallback"
    assert captured["responses_model"] == "local-model"
    assert captured["chat_model"] == "local-model"


def test_openai_backend_skips_chat_fallback_for_unrelated_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 404s unrelated to /responses should surface instead of masking with chat fallback.
    class FakeNotFoundError(Exception):
        status_code = 404

        def __str__(self) -> str:
            return "model not found"

    class FakeResponses:
        def create(self, *, model: str, input: str) -> object:
            del model, input
            raise FakeNotFoundError()

    class FakeChatCompletions:
        def create(self, *, model: str, messages: list[dict[str, str]]) -> object:
            del model, messages
            raise AssertionError("chat fallback should not be used for unrelated 404 errors")

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.responses = FakeResponses()
            self.chat = types.SimpleNamespace(completions=FakeChatCompletions())

    fake_openai_module = types.ModuleType("openai")
    fake_openai_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_openai_module)

    backend = OpenAIBackend(
        model="local-model",
        api_key="not-needed",
        base_url="http://127.0.0.1:8001/v1",
        require_api_key=False,
    )

    with pytest.raises(FakeNotFoundError):
        backend.complete("hello")


def test_llama_backend_requires_configuration() -> None:
    # The managed llama server must be configured before it can be used.
    shutdown_llama_cpp_server()
    with pytest.raises(RuntimeError):
        llm_complete("hello", backend="llama-cpp-server")


def test_default_backend_is_llama_cpp_server() -> None:
    # The package default backend now routes through the managed llama-cpp server.
    configure_llama_cpp_server(model="/tmp/default-backend-check.gguf")
    shutdown_llama_cpp_server()
    with pytest.raises(RuntimeError):
        complete("hello")


def test_resolve_default_model_uses_llama_api_model() -> None:
    configure_llama_cpp_server(model="/tmp/default-model-check.gguf", api_model="llama-api-model")
    assert resolve_default_model() == "llama-api-model"
    shutdown_llama_cpp_server()


def test_resolve_default_model_uses_openai_model() -> None:
    configure_openai(
        model="gpt-default-model-check",
        api_key_env="OPENAI_API_KEY",
        api_key="test-key",
        base_url="http://localhost:9000/v1",
        require_api_key=False,
    )
    assert resolve_default_model() == "gpt-default-model-check"


def test_base_llm_client_default_model_respects_backend_override() -> None:
    configure_openai(
        model="gpt-openai-default",
        api_key_env="OPENAI_API_KEY",
        api_key="test-key",
        base_url="http://localhost:9000/v1",
        require_api_key=False,
    )
    llm_client = BaseLLMClient(backend="echo-test")
    assert llm_client.default_model() == "echo-test-model"


def test_configure_llama_backend_replaces_existing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Capture create/close events to verify lifecycle transitions.
    events: list[tuple[str, str]] = []

    class FakeLlamaBackend:
        # Simple stand-in to track shutdown behavior.
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


def test_configure_llama_backend_accepts_hf_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_llama_backend_builds_server_module_command() -> None:
    # The backend should launch the packaged llama-cpp server module entry point.
    backend = create_llama_cpp_backend(
        model="/tmp/model.gguf",
        python_executable="/usr/bin/python3",
    )

    command = backend._build_command()
    assert command[0] == "/usr/bin/python3"
    assert command[1:3] == ["-m", "llama_cpp.server"]
    assert "--model_alias" in command
    assert "local-model" in command
    assert "--model" in command
    assert "/tmp/model.gguf" in command


def test_llama_backend_requires_server_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Missing llama-cpp dependency should raise before subprocess launch.
    backend = create_llama_cpp_backend(model="/tmp/model.gguf")

    monkeypatch.setattr(
        "design_research_agents.llm.backends.llama_cpp_server.find_spec",
        lambda _: None,
    )

    with pytest.raises(RuntimeError, match=r"pip install -e '\.\[local\]'"):
        backend.start()


def test_llama_backend_requires_huggingface_hub_for_hf_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HF model repo support requires huggingface-hub to be available.
    backend = create_llama_cpp_backend(
        model="tinyllama.Q4_K_M.gguf",
        hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    )

    def _fake_find_spec(name: str) -> object | None:
        if name == "llama_cpp.server":
            return object()
        if name == "huggingface_hub":
            return None
        return object()

    monkeypatch.setattr(
        "design_research_agents.llm.backends.llama_cpp_server.find_spec",
        _fake_find_spec,
    )

    with pytest.raises(RuntimeError, match="huggingface-hub is required"):
        backend.start()


def test_llama_backend_resolves_hf_quantized_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Short GGUF names should map to a unique file in the HF repository.
    backend = create_llama_cpp_backend(
        model="tinyllama.Q4_K_M.gguf",
        hf_model_repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    )

    def _fake_list_repo_files(repo_id: str) -> list[str]:
        assert repo_id == "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
        return [
            "README.md",
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        ]

    fake_huggingface_hub = types.ModuleType("huggingface_hub")
    fake_huggingface_hub.list_repo_files = _fake_list_repo_files
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    backend._resolve_hf_model_name()
    assert backend.model == "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"


def test_configure_openai_updates_default_call_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    configure_llama_cpp_server(model="/tmp/reset-default-backend.gguf")
    shutdown_llama_cpp_server()


def test_configure_openai_sets_active_backend_for_default_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # OpenAI configuration should make backend-free calls route to OpenAI.
    captured: dict[str, object] = {}

    def fake_openai_complete(prompt: str, **kwargs: object) -> str:
        captured["prompt"] = prompt
        captured.update(kwargs)
        return "ok-default-openai"

    monkeypatch.setattr("design_research_agents.llm.openai_complete", fake_openai_complete)

    configure_openai(
        model="gpt-default-route",
        api_key_env="ALT_OPENAI_API_KEY",
        api_key="explicit-key",
        base_url="http://localhost:9000/v1",
        require_api_key=False,
    )

    assert llm_complete("hello without backend") == "ok-default-openai"
    assert captured["prompt"] == "hello without backend"
    assert captured["model"] == "gpt-default-route"

    # Reset to project defaults and active llama backend for deterministic follow-up tests.
    configure_openai(
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        api_key=None,
        base_url=None,
        require_api_key=True,
    )
    configure_llama_cpp_server(model="/tmp/reset-default-backend.gguf")
    shutdown_llama_cpp_server()


def test_llama_backend_strict_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strict mode is the default: runtime failures should propagate.
    backend = create_llama_cpp_backend(model="/tmp/missing.gguf")

    def _raise_start(self: LlamaCppServerBackend) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(LlamaCppServerBackend, "start", _raise_start)

    with pytest.raises(RuntimeError):
        backend.complete("prompt")


def test_llama_backend_readiness_accepts_auth_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Auth failures indicate the HTTP server is up and reachable.
    backend = create_llama_cpp_backend(model="/tmp/model.gguf")

    class FakeRunningProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return

        def wait(self, timeout: float | None = None) -> None:
            del timeout
            return

        def kill(self) -> None:
            return

    backend._process = FakeRunningProcess()

    def _raise_unauthorized(url: str, timeout: float) -> object:
        del timeout
        raise HTTPError(url=url, code=401, msg="Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(
        "design_research_agents.llm.backends.llama_cpp_server.urlopen",
        _raise_unauthorized,
    )

    backend._wait_until_ready()
    backend.close()


def test_llama_backend_readiness_retries_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Missing /models should keep waiting and eventually time out.
    backend = create_llama_cpp_backend(
        model="/tmp/model.gguf",
        startup_timeout_seconds=0.1,
        poll_interval_seconds=0.0,
    )

    class FakeRunningProcess:
        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return

        def wait(self, timeout: float | None = None) -> None:
            del timeout
            return

        def kill(self) -> None:
            return

    backend._process = FakeRunningProcess()

    monotonic_ticks = iter((0.0, 0.0, 1.0))
    monkeypatch.setattr(
        "design_research_agents.llm.backends.llama_cpp_server.time.monotonic",
        lambda: next(monotonic_ticks),
    )
    monkeypatch.setattr(
        "design_research_agents.llm.backends.llama_cpp_server.time.sleep",
        lambda _: None,
    )

    def _raise_not_found(url: str, timeout: float) -> object:
        del timeout
        raise HTTPError(url=url, code=404, msg="Not Found", hdrs=None, fp=None)

    monkeypatch.setattr(
        "design_research_agents.llm.backends.llama_cpp_server.urlopen",
        _raise_not_found,
    )

    with pytest.raises(RuntimeError, match="Timed out waiting for llama-cpp server readiness"):
        backend._wait_until_ready()
    backend.close()
