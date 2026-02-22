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
    _vllm_local,
    _vllm_server,
)
from design_research_agents.llm.clients import VllmServerLLMClient
from tests._llm_openai_backends_test_helpers import request


class _ResponseContext:
    def __init__(self, *, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


def test__vllm_local_backend_payload_and_chat_url() -> None:
    backend = _vllm_local.VllmLocalBackend(
        name="vllm",
        base_url="https://host/api",
        default_model="demo-model",
        request_timeout_seconds=12.0,
        config_hash="cfg",
    )
    assert backend._chat_url == "https://host/api/v1/chat/completions"

    payload = backend._build_payload(
        request(
            temperature=0.3,
            max_tokens=64,
            provider_options={"seed": 123},
        )
    )
    assert payload["temperature"] == 0.3
    assert payload["max_tokens"] == 64
    assert payload["seed"] == 123


def test__vllm_local_backend_generate_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _vllm_local.VllmLocalBackend(
        name="vllm",
        base_url="http://127.0.0.1:8002/v1",
        default_model="demo-model",
        request_timeout_seconds=10.0,
        config_hash="cfg",
    )
    monkeypatch.setattr(
        _vllm_local,
        "_post_json_with_retry",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "message": {
                        "content": "  hi  ",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {"name": "calculator", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"total_tokens": 3},
        },
    )

    response = backend._generate(request())
    assert response.text == "hi"
    assert response.tool_calls[0].name == "calculator"
    assert response.usage is not None

    lines = [
        b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
        b"\n",
        (
            b'data: {"choices":[{"delta":{"tool_calls":[{"id":"c1","function":{"name":"calc",'
            b'"arguments":"{\\"x\\":1}"}}]}}],"usage":{"prompt_tokens":1}}\n'
        ),
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]
    monkeypatch.setattr(
        _vllm_local,
        "_post_stream_with_retry",
        lambda *args, **kwargs: _ResponseContext(lines=lines),
    )
    deltas = list(backend._stream(request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.call_id == "c1"
    assert deltas[2].usage_delta is not None


def test__vllm_local_retry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [LLMRateLimitError("rate"), {"ok": True}]
    sleeps: list[float] = []

    def _flaky_post_json(*_args: object, **_kwargs: object):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(_vllm_local, "_post_json", _flaky_post_json)
    monkeypatch.setattr(_vllm_local.time, "sleep", lambda seconds: sleeps.append(seconds))
    parsed = _vllm_local._post_json_with_retry(
        "http://unit",
        {"a": 1},
        timeout_seconds=1.0,
        max_retries=2,
    )
    assert parsed == {"ok": True}
    assert sleeps == [0.5]

    monkeypatch.setattr(
        _vllm_local,
        "_post_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMInvalidRequestError("bad")),
    )
    with pytest.raises(LLMInvalidRequestError):
        _vllm_local._post_json_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=2,
        )


def test_vllm_http_error_and_response_parser() -> None:
    http_error = HTTPError(
        url="http://unit",
        code=401,
        msg="unauthorized",
        hdrs=None,
        fp=None,
    )
    mapped = _vllm_local._http_error(http_error)
    assert mapped.__class__.__name__ == "LLMAuthError"

    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        _vllm_local._parse_completion_response({}, request(), provider="vllm")

    assert _vllm_local._extract_tool_call_deltas("bad") == []


def test_vllm_server_backend_command_and_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _vllm_server.VllmServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_model="qwen2.5-1.5b-instruct",
        host="0.0.0.0",
        port=9002,
        extra_server_args=("--dtype", "auto"),
    )
    command = backend._build_command()
    assert "vllm.entrypoints.openai.api_server" in command
    assert "--served-model-name" in command
    assert "qwen2.5-1.5b-instruct" in command
    assert "--dtype" in command

    monkeypatch.setattr(_vllm_server, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="vLLM dependency is missing"):
        backend._ensure_server_dependency()


def test_vllm_server_wait_until_ready_and_timeout_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _vllm_server.VllmServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_model="qwen2.5-1.5b-instruct",
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

    monkeypatch.setattr(_vllm_server, "urlopen", _flaky_probe)
    backend._wait_until_ready()
    assert attempts["count"] == 2

    timeout_backend = _vllm_server.VllmServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_model="qwen2.5-1.5b-instruct",
        startup_timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )
    timeout_backend._process = _AliveProcess()  # type: ignore[assignment]

    def _timeout_probe(_url: str, timeout: float) -> object:
        del timeout
        raise TimeoutError("t")

    monkeypatch.setattr(_vllm_server, "urlopen", _timeout_probe)
    monkeypatch.setattr(_vllm_server.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="Timed out waiting for vLLM server readiness"):
        timeout_backend._wait_until_ready()


def test_vllm_server_close_forces_kill_when_terminate_stalls() -> None:
    backend = _vllm_server.VllmServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_model="qwen2.5-1.5b-instruct",
    )

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
                raise subprocess.TimeoutExpired(cmd="vllm", timeout=5)
            return 0

    process = _StubbornProcess()
    backend._process = process  # type: ignore[assignment]
    backend.close()
    assert process.killed is True
    assert backend._process is None


def test_vllm_client_constructor_and_modes() -> None:
    managed_client = VllmServerLLMClient(
        name="vllm-managed",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        api_model="qwen2.5-1.5b-instruct",
        manage_server=True,
        model_patterns=("qwen2.5-*",),
    )
    try:
        assert managed_client.default_model() == "qwen2.5-1.5b-instruct"
        assert managed_client._backend.name == "vllm-managed"
        assert managed_client._backend.model_patterns == ("qwen2.5-*",)
        assert managed_client._vllm_server is not None
    finally:
        managed_client.close()

    connect_client = VllmServerLLMClient(
        manage_server=False,
        base_url="http://127.0.0.1:9002/v1",
        api_model="custom-model",
    )
    assert connect_client._vllm_server is None
    assert connect_client._backend.base_url == "http://127.0.0.1:9002/v1"

    with pytest.raises(ValueError, match="base_url cannot be provided"):
        VllmServerLLMClient(
            manage_server=True,
            base_url="http://127.0.0.1:9002/v1",
        )


def test__vllm_local_backend_requires_valid_timeout_and_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        _vllm_local.VllmLocalBackend(
            name="x",
            base_url=" ",
            default_model="m",
            request_timeout_seconds=10.0,
            config_hash="cfg",
        )
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        _vllm_local.VllmLocalBackend(
            name="x",
            base_url="http://127.0.0.1:8002/v1",
            default_model="m",
            request_timeout_seconds=0.0,
            config_hash="cfg",
        )


def test_vllm_retry_stream_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _vllm_local,
        "_post_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMInvalidRequestError("bad")),
    )
    with pytest.raises(LLMInvalidRequestError):
        _vllm_local._post_stream_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=2,
        )

    monkeypatch.setattr(
        _vllm_local,
        "_post_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMProviderError("offline")),
    )
    with pytest.raises(LLMProviderError):
        _vllm_local._post_stream_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=0,
        )
