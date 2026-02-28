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
    _sglang_local,
    _sglang_server,
)
from design_research_agents.llm.clients import SGLangServerLLMClient
from tests._llm_openai_backends_test_helpers import request


class _ResponseContext:
    def __init__(self, *, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


def test__sglang_local_backend_payload_and_chat_url() -> None:
    backend = _sglang_local.SglangLocalBackend(
        name="sglang",
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


def test__sglang_local_backend_generate_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _sglang_local.SglangLocalBackend(
        name="sglang",
        base_url="http://127.0.0.1:30000/v1",
        default_model="demo-model",
        request_timeout_seconds=10.0,
        config_hash="cfg",
    )
    monkeypatch.setattr(
        _sglang_local,
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
        _sglang_local,
        "_post_stream_with_retry",
        lambda *args, **kwargs: _ResponseContext(lines=lines),
    )
    deltas = list(backend._stream(request()))
    assert deltas[0].text_delta == "he"
    assert deltas[1].tool_call_delta is not None
    assert deltas[1].tool_call_delta.call_id == "c1"
    assert deltas[2].usage_delta is not None


def test__sglang_local_retry_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = [LLMRateLimitError("rate"), {"ok": True}]
    sleeps: list[float] = []

    def _flaky_post_json(*_args: object, **_kwargs: object):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(_sglang_local, "_post_json", _flaky_post_json)
    monkeypatch.setattr(_sglang_local.time, "sleep", lambda seconds: sleeps.append(seconds))
    parsed = _sglang_local._post_json_with_retry(
        "http://unit",
        {"a": 1},
        timeout_seconds=1.0,
        max_retries=2,
    )
    assert parsed == {"ok": True}
    assert sleeps == [0.5]

    monkeypatch.setattr(
        _sglang_local,
        "_post_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMInvalidRequestError("bad")),
    )
    with pytest.raises(LLMInvalidRequestError):
        _sglang_local._post_json_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=2,
        )


def test_sglang_http_error_and_response_parser() -> None:
    http_error = HTTPError(
        url="http://unit",
        code=401,
        msg="unauthorized",
        hdrs=None,
        fp=None,
    )
    mapped = _sglang_local._http_error(http_error)
    assert mapped.__class__.__name__ == "LLMAuthError"

    with pytest.raises(LLMInvalidRequestError, match="no choices"):
        _sglang_local._parse_completion_response({}, request(), provider="sglang")

    assert _sglang_local._extract_tool_call_deltas("bad") == []


def test_sglang_server_backend_command_and_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _sglang_server.SglangServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        host="0.0.0.0",
        port=9003,
        extra_server_args=("--tp-size", "1"),
    )
    command = backend._build_command()
    assert "sglang.launch_server" in command
    assert "--model-path" in command
    assert "Qwen/Qwen2.5-1.5B-Instruct" in command
    assert "--tp-size" in command

    monkeypatch.setattr(_sglang_server, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="SGLang dependency is missing"):
        backend._ensure_server_dependency()


def test_sglang_server_wait_until_ready_and_timeout_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _sglang_server.SglangServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
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

    monkeypatch.setattr(_sglang_server, "urlopen", _flaky_probe)
    backend._wait_until_ready()
    assert attempts["count"] == 2

    timeout_backend = _sglang_server.SglangServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        startup_timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )
    timeout_backend._process = _AliveProcess()  # type: ignore[assignment]

    def _timeout_probe(_url: str, timeout: float) -> object:
        del timeout
        raise TimeoutError("t")

    monkeypatch.setattr(_sglang_server, "urlopen", _timeout_probe)
    monkeypatch.setattr(_sglang_server.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError, match="Timed out waiting for SGLang server readiness"):
        timeout_backend._wait_until_ready()


def test_sglang_server_close_forces_kill_when_terminate_stalls() -> None:
    backend = _sglang_server.SglangServerBackend(
        model="Qwen/Qwen2.5-1.5B-Instruct",
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
                raise subprocess.TimeoutExpired(cmd="sglang", timeout=5)
            return 0

    process = _StubbornProcess()
    backend._process = process  # type: ignore[assignment]
    backend.close()
    assert process.killed is True
    assert backend._process is None


def test_sglang_client_constructor_and_modes() -> None:
    with SGLangServerLLMClient(
        name="sglang-managed",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        manage_server=True,
        model_patterns=("Qwen/*",),
    ) as managed_client:
        assert managed_client.default_model() == "Qwen/Qwen2.5-1.5B-Instruct"
        assert managed_client._backend.name == "sglang-managed"
        assert managed_client._backend.model_patterns == ("Qwen/*",)
        assert managed_client._sglang_server is not None

    assert managed_client._sglang_server is not None
    assert managed_client._sglang_server._process is None

    connect_client = SGLangServerLLMClient(
        manage_server=False,
        base_url="http://127.0.0.1:39000/v1",
        model="custom-model",
    )
    with connect_client as entered_client:
        assert entered_client is connect_client
        assert connect_client._sglang_server is None
        assert connect_client._backend.base_url == "http://127.0.0.1:39000/v1"

    with pytest.raises(ValueError, match="base_url cannot be provided"):
        SGLangServerLLMClient(
            manage_server=True,
            base_url="http://127.0.0.1:39000/v1",
        )


def test__sglang_local_backend_requires_valid_timeout_and_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        _sglang_local.SglangLocalBackend(
            name="x",
            base_url=" ",
            default_model="m",
            request_timeout_seconds=10.0,
            config_hash="cfg",
        )
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        _sglang_local.SglangLocalBackend(
            name="x",
            base_url="http://127.0.0.1:30000/v1",
            default_model="m",
            request_timeout_seconds=0.0,
            config_hash="cfg",
        )


def test_sglang_retry_stream_non_retryable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _sglang_local,
        "_post_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMInvalidRequestError("bad")),
    )
    with pytest.raises(LLMInvalidRequestError):
        _sglang_local._post_stream_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=2,
        )

    monkeypatch.setattr(
        _sglang_local,
        "_post_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMProviderError("offline")),
    )
    with pytest.raises(LLMProviderError):
        _sglang_local._post_stream_with_retry(
            "http://unit",
            {},
            timeout_seconds=1.0,
            max_retries=0,
        )
