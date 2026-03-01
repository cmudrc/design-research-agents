from __future__ import annotations

import subprocess
from urllib.error import HTTPError

import pytest

from design_research_agents._contracts._llm import (
    LLMInvalidRequestError,
    LLMProviderError,
    LLMRateLimitError,
)
from design_research_agents.llm._backends._providers import (
    _ollama_local,
    _ollama_server,
)
from design_research_agents.llm.clients import OllamaLLMClient
from tests._llm_openai_backends_test_helpers import request


class _ResponseContext:
    def __init__(self, *, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


def test__ollama_local_backend_payload_and_chat_url() -> None:
    backend = _ollama_local.OllamaLocalBackend(
        name="ollama",
        base_url="http://127.0.0.1:11434",
        default_model="qwen2.5:1.5b-instruct",
        request_timeout_seconds=12.0,
        config_hash="cfg",
    )
    assert backend._chat_url == "http://127.0.0.1:11434/api/chat"

    payload = backend._build_payload(
        request(
            temperature=0.4,
            max_tokens=33,
            provider_options={"options": {"num_ctx": 8192}, "keep_alive": "5m"},
        ),
        stream=False,
    )
    assert payload["stream"] is False
    assert payload["keep_alive"] == "5m"
    options = payload["options"]
    assert isinstance(options, dict)
    assert options["temperature"] == 0.4
    assert options["num_predict"] == 33
    assert options["num_ctx"] == 8192


def test__ollama_local_backend_generate_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ollama_local.OllamaLocalBackend(
        name="ollama",
        base_url="http://127.0.0.1:11434",
        default_model="qwen2.5:1.5b-instruct",
        request_timeout_seconds=10.0,
        config_hash="cfg",
    )
    monkeypatch.setattr(
        _ollama_local,
        "_post_json_with_retry",
        lambda *args, **kwargs: {
            "model": "qwen2.5:1.5b-instruct",
            "message": {
                "content": "  done  ",
                "tool_calls": [{"id": "c1", "function": {"name": "calculator", "arguments": "{}"}}],
            },
            "done_reason": "stop",
            "prompt_eval_count": 2,
            "eval_count": 3,
        },
    )
    response = backend._generate(request())
    assert response.text == "done"
    assert response.tool_calls[0].name == "calculator"
    assert response.usage is not None
    assert response.usage.total_tokens == 5

    stream_lines = [
        b'{"message":{"content":"he"},"done":false}\n',
        b'{"message":{"tool_calls":[{"function":{"name":"calculator","arguments":{"x":1}}}]},"done":false}\n',
        b'{"done":true,"prompt_eval_count":1,"eval_count":2}\n',
    ]
    monkeypatch.setattr(
        _ollama_local,
        "_post_stream_with_retry",
        lambda *args, **kwargs: _ResponseContext(lines=stream_lines),
    )
    deltas = list(backend._stream(request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.name == "calculator"
    assert deltas[2].usage_delta is not None
    assert deltas[2].usage_delta.total_tokens == 3


def test_ollama_retry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [LLMRateLimitError("rate"), {"ok": True}]
    sleeps: list[float] = []

    def _flaky_post_json(*_args: object, **_kwargs: object):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(_ollama_local, "_post_json", _flaky_post_json)
    monkeypatch.setattr(_ollama_local.time, "sleep", lambda seconds: sleeps.append(seconds))
    parsed = _ollama_local._post_json_with_retry(
        "http://unit",
        {"a": 1},
        timeout_seconds=1.0,
        max_retries=2,
    )
    assert parsed == {"ok": True}
    assert sleeps == [0.5]

    monkeypatch.setattr(
        _ollama_local,
        "_post_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMInvalidRequestError("bad")),
    )
    with pytest.raises(LLMInvalidRequestError):
        _ollama_local._post_json_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=2,
        )


def test_ollama_http_error_and_parsers() -> None:
    http_error = HTTPError(
        url="http://unit",
        code=429,
        msg="rate",
        hdrs=None,
        fp=None,
    )
    mapped = _ollama_local._http_error(http_error)
    assert mapped.__class__.__name__ == "LLMRateLimitError"

    parsed = _ollama_local._parse_ollama_usage({"prompt_eval_count": 1, "eval_count": 2})
    assert parsed is not None
    assert parsed.total_tokens == 3

    assert _ollama_local._coerce_int(2.9) == 2
    assert _ollama_local._coerce_int(True) is None
    assert _ollama_local._extract_tool_call_deltas("bad") == []


def test_ollama_server_backend_dependency_and_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _ollama_server.OllamaServerBackend(
        host="0.0.0.0",
        port=11435,
        ollama_executable="ollama",
    )
    command = backend._build_command()
    assert command == ["ollama", "serve"]
    assert backend._serve_env()["OLLAMA_HOST"] == "0.0.0.0:11435"

    monkeypatch.setattr(_ollama_server.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="was not found in PATH"):
        backend._ensure_server_dependency()


def test_ollama_server_wait_until_ready_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ollama_server.OllamaServerBackend(
        startup_timeout_seconds=1.0,
        poll_interval_seconds=0.001,
    )

    class _AliveProcess:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
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

    def _flaky_probe(_url: str, timeout: float) -> _ReadyResponse:
        del timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("timed out")
        return _ReadyResponse()

    monkeypatch.setattr(_ollama_server, "urlopen", _flaky_probe)
    backend._wait_until_ready()
    assert attempts["count"] == 2

    timeout_backend = _ollama_server.OllamaServerBackend(
        startup_timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )
    timeout_backend._process = _AliveProcess()  # type: ignore[assignment]
    monkeypatch.setattr(
        _ollama_server,
        "urlopen",
        lambda _url, timeout: (_ for _ in ()).throw(TimeoutError("t")),
    )
    monkeypatch.setattr(_ollama_server.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="Timed out waiting for Ollama server readiness"):
        timeout_backend._wait_until_ready()


def test_ollama_server_pull_and_close_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _ollama_server.OllamaServerBackend(
        auto_pull_model=True,
        default_model="qwen2.5:1.5b-instruct",
    )
    monkeypatch.setattr(
        _ollama_server.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr="",
        ),
    )
    backend._pull_model("qwen2.5:1.5b-instruct")

    monkeypatch.setattr(
        _ollama_server.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="boom",
        ),
    )
    with pytest.raises(RuntimeError, match="Failed to pull Ollama model"):
        backend._pull_model("qwen2.5:1.5b-instruct")

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(_ollama_server.subprocess, "run", _raise_timeout)
    with pytest.raises(RuntimeError, match="Timed out pulling Ollama model"):
        backend._pull_model("qwen2.5:1.5b-instruct")

    class _StubbornProcess:
        def __init__(self) -> None:
            self.killed = False
            self.wait_calls = 0

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="ollama", timeout=5)
            return 0

    process = _StubbornProcess()
    backend._process = process  # type: ignore[assignment]
    backend.close()
    assert process.killed is True
    assert backend._process is None


def test_ollama_client_constructor_and_modes() -> None:
    with OllamaLLMClient(
        name="ollama-managed",
        default_model="qwen2.5:1.5b-instruct",
        manage_server=True,
        auto_pull_model=False,
        model_patterns=("qwen2.5:*",),
    ) as managed_client:
        assert managed_client.default_model() == "qwen2.5:1.5b-instruct"
        assert managed_client._backend.name == "ollama-managed"
        assert managed_client._backend.model_patterns == ("qwen2.5:*",)
        assert managed_client._ollama_server is not None

    assert managed_client._ollama_server is not None
    assert managed_client._ollama_server._process is None

    connect_client = OllamaLLMClient(
        default_model="custom-model",
        host="127.0.0.1",
        port=12500,
        manage_server=False,
    )
    with connect_client as entered_client:
        assert entered_client is connect_client
        assert connect_client._ollama_server is None
        assert connect_client._backend.base_url == "http://127.0.0.1:12500"


def test__ollama_local_backend_requires_valid_timeout_and_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        _ollama_local.OllamaLocalBackend(
            name="x",
            base_url=" ",
            default_model="m",
            request_timeout_seconds=10.0,
            config_hash="cfg",
        )
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        _ollama_local.OllamaLocalBackend(
            name="x",
            base_url="http://127.0.0.1:11434",
            default_model="m",
            request_timeout_seconds=0.0,
            config_hash="cfg",
        )


def test_ollama_retry_stream_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _ollama_local,
        "_post_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMInvalidRequestError("bad")),
    )
    with pytest.raises(LLMInvalidRequestError):
        _ollama_local._post_stream_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=2,
        )

    monkeypatch.setattr(
        _ollama_local,
        "_post_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMProviderError("offline")),
    )
    with pytest.raises(LLMProviderError):
        _ollama_local._post_stream_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=0,
        )
