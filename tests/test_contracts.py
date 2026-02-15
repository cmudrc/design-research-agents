from __future__ import annotations

import pytest

from design_research_agents.llm import LlamaCppServerLLMClient
from design_research_agents.llm.backends.providers import llama_cpp_server


def test_llama_cpp_server_command_contains_expected_args() -> None:
    backend = llama_cpp_server.create_backend(
        model="/tmp/model.gguf",
        hf_model_repo_id="repo/id",
        api_model="local-model",
        host="0.0.0.0",
        port=9000,
        extra_server_args=("--n-gpu-layers", "35"),
    )

    command = backend._build_command()

    assert "llama_cpp.server" in command
    assert "--model" in command
    assert "/tmp/model.gguf" in command
    assert "--model_alias" in command
    assert "local-model" in command
    assert "--hf_model_repo_id" in command
    assert "repo/id" in command
    assert "--n-gpu-layers" in command


def test_llama_cpp_wait_until_ready_retries_after_timeout(
    monkeypatch,
) -> None:
    backend = llama_cpp_server.LlamaCppServerBackend(
        model="/tmp/model.gguf",
        startup_timeout_seconds=1.0,
        poll_interval_seconds=0.0,
    )

    class _AliveProcess:
        def __init__(self) -> None:
            self._terminated = False

        def poll(self) -> int | None:
            return 0 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def kill(self) -> None:
            self._terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self._terminated = True
            return 0

    backend._process = _AliveProcess()  # type: ignore[assignment]
    attempts = {"count": 0}

    class _ReadyResponse:
        status = 200

        def __enter__(self) -> _ReadyResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

    def _flaky_probe(url: str, timeout: float) -> _ReadyResponse:
        del url, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("timed out")
        return _ReadyResponse()

    monkeypatch.setattr(llama_cpp_server, "urlopen", _flaky_probe)
    backend._wait_until_ready()

    assert attempts["count"] == 2


def test_llama_cpp_wait_until_ready_timeout_error_becomes_runtime_error(
    monkeypatch,
) -> None:
    backend = llama_cpp_server.LlamaCppServerBackend(
        model="/tmp/model.gguf",
        startup_timeout_seconds=0.01,
        poll_interval_seconds=0.0,
    )

    class _AliveProcess:
        def __init__(self) -> None:
            self._terminated = False

        def poll(self) -> int | None:
            return 0 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def kill(self) -> None:
            self._terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self._terminated = True
            return 0

    backend._process = _AliveProcess()  # type: ignore[assignment]

    def _timeout_probe(url: str, timeout: float) -> object:
        del url, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(llama_cpp_server, "urlopen", _timeout_probe)
    monkeypatch.setattr(llama_cpp_server.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Timed out waiting for llama-cpp server readiness"):
        backend._wait_until_ready()


def test_llama_cpp_client_constructor_propagates_settings() -> None:
    client = LlamaCppServerLLMClient(
        name="custom-llama",
        model="/tmp/model.gguf",
        hf_model_repo_id="repo/id",
        api_model="custom-model",
        host="0.0.0.0",
        port=9100,
        context_window=2048,
        extra_server_args=("--n-gpu-layers", "35"),
        model_patterns=("custom-model",),
    )
    try:
        assert client.default_model() == "custom-model"
        assert client._backend.name == "custom-llama"
        assert client._backend.model_patterns == ("custom-model",)

        command = client._llama_server._build_command()
        assert "--n_ctx" in command
        assert "2048" in command
        assert "--n-gpu-layers" in command
        assert "35" in command
    finally:
        client.close()
