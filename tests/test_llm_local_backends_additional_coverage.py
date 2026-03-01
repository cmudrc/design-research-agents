from __future__ import annotations

import io
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from design_research_agents._contracts._llm import LLMInvalidRequestError, LLMProviderError
from design_research_agents.llm._backends._providers import (
    _ollama_local,
    _ollama_server,
    _sglang_local,
    _sglang_server,
    _vllm_local,
    _vllm_server,
)
from tests._llm_openai_backends_test_helpers import request


class _JsonResponse:
    def __init__(self, *, text: str, status: int = 200) -> None:
        self._text = text
        self.status = status

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        del exc_type, exc, tb
        return False

    def read(self) -> bytes:
        return self._text.encode("utf-8")


class _StreamResponse:
    def __init__(self, *, lines: list[bytes]) -> None:
        self._lines = lines

    def __iter__(self):
        return iter(self._lines)


class _ManagedServerStub:
    def __init__(self) -> None:
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1


@pytest.mark.parametrize(
    ("module", "backend_cls", "default_model"),
    [
        (_vllm_local, _vllm_local.VllmLocalBackend, "api-model"),
        (_sglang_local, _sglang_local.SglangLocalBackend, "startup-model"),
    ],
)
def test_openai_like_local_backends_cover_additional_branches(
    module: object,
    backend_cls: type[object],
    default_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = _ManagedServerStub()
    backend = backend_cls(
        name="local",
        base_url="http://127.0.0.1:9999/v1/",
        default_model=default_model,
        request_timeout_seconds=10.0,
        config_hash="cfg",
        managed_server=managed,
    )

    capabilities = backend.capabilities()
    assert capabilities.streaming is True
    assert capabilities.json_mode == "native"
    assert backend.healthcheck().ok is True
    assert backend._chat_url.endswith("/chat/completions")

    trailing = backend_cls(
        name="local",
        base_url="http://127.0.0.1:9999/",
        default_model=default_model,
        request_timeout_seconds=10.0,
        config_hash="cfg",
    )
    assert trailing._chat_url == "http://127.0.0.1:9999/v1/chat/completions"

    lines = [
        b"data: not-json\n",
        b"\n",
        b"data: []\n",
        b"\n",
        b'data: {"choices":[]}\n',
        b"\n",
        b'data: {"choices":[{"delta":"bad"}]}\n',
        b"\n",
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]
    monkeypatch.setattr(
        module,
        "_post_stream_with_retry",
        lambda *args, **kwargs: _StreamResponse(lines=lines),
    )
    deltas = list(backend._stream(request(model=default_model)))
    assert managed.start_calls == 1
    assert [delta.text_delta for delta in deltas if delta.text_delta] == ["ok"]

    trailing_event = list(module._iter_sse_events([b'data: {"x":1}\n']))
    assert trailing_event == ['{"x":1}']


@pytest.mark.parametrize(
    ("module", "error_message"),
    [
        (_vllm_local, "vLLM response must be a JSON object."),
        (_sglang_local, "SGLang response must be a JSON object."),
    ],
)
def test_openai_like_local_transport_helpers_cover_error_paths(
    module: object,
    error_message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: _JsonResponse(text="[]"))
    with pytest.raises(LLMInvalidRequestError, match=error_message):
        module._post_json("http://unit", {"x": 1}, timeout_seconds=1.0)

    monkeypatch.setattr(
        module,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(LLMProviderError):
        module._post_json("http://unit", {"x": 1}, timeout_seconds=1.0)

    http_exc = HTTPError(
        url="http://unit",
        code=429,
        msg="rate",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"slow down"}}'),
    )
    monkeypatch.setattr(
        module,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(http_exc),
    )
    with pytest.raises(Exception):  # noqa: B017
        module._post_stream("http://unit", {"x": 1}, timeout_seconds=1.0)

    monkeypatch.setattr(
        module,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("down")),
    )
    with pytest.raises(LLMProviderError):
        module._post_stream("http://unit", {"x": 1}, timeout_seconds=1.0)

    monkeypatch.setattr(module, "_post_json", lambda *_args, **_kwargs: {"ok": True})
    assert module._post_json_with_retry(
        "http://unit",
        {"x": 1},
        timeout_seconds=1.0,
        max_retries=-1,
    ) == {"ok": True}

    stream_obj = object()
    monkeypatch.setattr(module, "_post_stream", lambda *_args, **_kwargs: stream_obj)
    assert (
        module._post_stream_with_retry(
            "http://unit",
            {"x": 1},
            timeout_seconds=1.0,
            max_retries=-1,
        )
        is stream_obj
    )

    parsed = module._parse_completion_response(
        {"choices": [{"message": "bad"}]},
        request(model="demo-model"),
        provider="provider",
    )
    assert parsed.text == ""

    messages = module._format_messages(
        [
            SimpleNamespace(role="user", content="hello", name="alice", tool_call_id="t1"),
            SimpleNamespace(role=None, content="skip"),
        ]
    )
    assert messages[0]["name"] == "alice"
    assert messages[0]["tool_call_id"] == "t1"
    assert len(messages) == 1

    mapped = module._http_error(
        HTTPError(
            url="http://unit",
            code=400,
            msg="bad",
            hdrs=None,
            fp=io.BytesIO(b'{"error":{"message":"payload message"}}'),
        )
    )
    assert "payload message" in str(mapped)

    tool_deltas = module._extract_tool_call_deltas(["skip", {"id": 1, "function": "bad"}])
    assert len(tool_deltas) == 1
    assert tool_deltas[0].call_id == "1"


@pytest.mark.parametrize(
    ("factory", "create_fn", "module_name"),
    [
        (
            lambda **kwargs: _vllm_server.VllmServerBackend(model="m", api_model="a", **kwargs),
            lambda: _vllm_server.create_backend("m", api_model="a"),
            "vllm",
        ),
        (
            lambda **kwargs: _sglang_server.SglangServerBackend(model="m", **kwargs),
            lambda: _sglang_server.create_backend("m"),
            "sglang",
        ),
    ],
)
def test_openai_like_server_backends_cover_start_and_wait_paths(
    factory: object,
    create_fn: object,
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = factory()
    assert backend.is_running() is False
    assert create_fn().is_running() is False

    class _AliveProcess:
        def __init__(self) -> None:
            self.terminated = False

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    already_running = factory()
    already_running._process = _AliveProcess()  # type: ignore[assignment]
    already_running.start()

    started = {"ready": False}
    spawned = _AliveProcess()
    fresh = factory()
    monkeypatch.setattr(fresh, "_ensure_server_dependency", lambda: None)
    monkeypatch.setattr(
        "design_research_agents.llm._backends._providers._vllm_server.subprocess.Popen"
        if module_name == "vllm"
        else "design_research_agents.llm._backends._providers._sglang_server.subprocess.Popen",
        lambda *args, **kwargs: spawned,
    )
    monkeypatch.setattr(fresh, "_wait_until_ready", lambda: started.update({"ready": True}))
    fresh.start()
    assert started["ready"] is True
    assert fresh.is_running() is True
    fresh.close()

    not_initialized = factory()
    not_initialized._process = None
    with pytest.raises(RuntimeError, match="not initialized"):
        not_initialized._wait_until_ready()

    exited = factory(startup_timeout_seconds=0.1, poll_interval_seconds=0.001)
    exited._process = SimpleNamespace(poll=lambda: 1)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        exited._wait_until_ready()

    auth_ready = factory(startup_timeout_seconds=0.1, poll_interval_seconds=0.001)
    auth_ready._process = _AliveProcess()  # type: ignore[assignment]
    target_module = (
        "design_research_agents.llm._backends._providers._vllm_server"
        if module_name == "vllm"
        else "design_research_agents.llm._backends._providers._sglang_server"
    )
    monkeypatch.setattr(
        f"{target_module}.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HTTPError(url="http://unit", code=401, msg="auth", hdrs=None, fp=None)
        ),
    )
    auth_ready._wait_until_ready()


def test__ollama_local_additional_transport_and_parser_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = _ManagedServerStub()
    backend = _ollama_local.OllamaLocalBackend(
        name="ollama",
        base_url="http://127.0.0.1:11434/",
        default_model="demo",
        request_timeout_seconds=10.0,
        config_hash="cfg",
        managed_server=managed,
    )
    assert backend.capabilities().streaming is True
    assert backend.capabilities().json_mode == "native"
    assert backend.healthcheck().ok is True
    assert backend._chat_url == "http://127.0.0.1:11434/api/chat"

    lines = [
        b"\n",
        b"data: not-json\n",
        b'data: {"message":"bad","done":false}\n',
        b'{"message":{"content":"ok"},"done":true}\n',
    ]
    monkeypatch.setattr(
        _ollama_local,
        "_post_stream_with_retry",
        lambda *args, **kwargs: _StreamResponse(lines=lines),
    )
    deltas = list(backend._stream(request(model="demo")))
    assert managed.start_calls == 1
    assert [delta.text_delta for delta in deltas if delta.text_delta] == ["ok"]

    monkeypatch.setattr(
        _ollama_local,
        "urlopen",
        lambda *_args, **_kwargs: _JsonResponse(text="[]"),
    )
    with pytest.raises(LLMInvalidRequestError, match="Ollama response must be a JSON object"):
        _ollama_local._post_json("http://unit", {"x": 1}, timeout_seconds=1.0)

    monkeypatch.setattr(
        _ollama_local,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(LLMProviderError):
        _ollama_local._post_stream("http://unit", {"x": 1}, timeout_seconds=1.0)

    parsed = _ollama_local._parse_completion_response(
        {"message": "bad"},
        request(model="demo"),
        provider="ollama",
    )
    assert parsed.text == ""
    assert _ollama_local._parse_ollama_usage({}) is None

    messages = _ollama_local._format_messages(
        [
            SimpleNamespace(role="user", content="hello", name="alice"),
            SimpleNamespace(role="assistant", content=None),
        ]
    )
    assert messages[0]["name"] == "alice"
    assert len(messages) == 1

    deltas = _ollama_local._extract_tool_call_deltas([None, {"function": {"name": "calc", "arguments": {"x": 1}}}])
    assert deltas[0].call_id == "call_2"
    assert deltas[0].arguments_json_delta is not None


def test__ollama_server_additional_start_and_validation_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="host must not be empty"):
        _ollama_server.OllamaServerBackend(host=" ")
    with pytest.raises(ValueError, match="port must be > 0"):
        _ollama_server.OllamaServerBackend(port=0)
    with pytest.raises(ValueError, match="ollama_executable must not be empty"):
        _ollama_server.OllamaServerBackend(ollama_executable=" ")

    backend = _ollama_server.OllamaServerBackend(
        auto_pull_model=True,
        default_model="qwen2.5:1.5b-instruct",
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
    backend.start()

    launched = {"pulled": False}
    ready_backend = _ollama_server.OllamaServerBackend(
        auto_pull_model=True,
        default_model="qwen2.5:1.5b-instruct",
    )
    monkeypatch.setattr(ready_backend, "_ensure_server_dependency", lambda: None)
    monkeypatch.setattr(ready_backend, "_wait_until_ready", lambda: None)
    monkeypatch.setattr(
        ready_backend,
        "_pull_model",
        lambda _model: launched.update({"pulled": True}),
    )
    monkeypatch.setattr(
        _ollama_server.subprocess,
        "Popen",
        lambda *args, **kwargs: _AliveProcess(),
    )
    ready_backend.start()
    assert launched["pulled"] is True
    ready_backend.close()

    not_initialized = _ollama_server.OllamaServerBackend()
    with pytest.raises(RuntimeError, match="not initialized"):
        not_initialized._wait_until_ready()

    exited = _ollama_server.OllamaServerBackend(
        startup_timeout_seconds=0.1,
        poll_interval_seconds=0.001,
    )
    exited._process = SimpleNamespace(poll=lambda: 1)  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="exited before becoming ready"):
        exited._wait_until_ready()

    auth_ready = _ollama_server.OllamaServerBackend(
        startup_timeout_seconds=0.1,
        poll_interval_seconds=0.001,
    )
    auth_ready._process = _AliveProcess()  # type: ignore[assignment]
    monkeypatch.setattr(
        _ollama_server,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            HTTPError(url="http://unit", code=401, msg="auth", hdrs=None, fp=None)
        ),
    )
    auth_ready._wait_until_ready()
