from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator, Sequence
from types import ModuleType, SimpleNamespace

import pytest

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMInvalidRequestError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents.llm._backends import _base as base_module
from design_research_agents.llm._backends._base import (
    BaseLLMBackend,
    _matches_model_pattern,
    _parse_tool_calls,
)
from design_research_agents.llm._backends._providers._echo_test import EchoTestBackend
from design_research_agents.llm._backends._providers._html import HTMLBackend
from design_research_agents.llm._backends._providers._llama_cpp import LlamaCppBackend
from design_research_agents.llm._backends._providers._mlx_local import MlxLocalBackend


def _request(
    *,
    model: str | None = "stub-default",
    tools: Sequence[ToolSpec] = (),
    response_schema: dict[str, object] | None = None,
    messages: Sequence[LLMMessage] | None = None,
) -> LLMRequest:
    resolved_messages = list(messages) if messages is not None else [LLMMessage(role="user", content="hello")]
    return LLMRequest(
        messages=resolved_messages,
        model=model,
        tools=tools,
        response_schema=response_schema,
    )


def _tool(name: str = "calculator") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Test tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
    )


class _BestEffortBackend(BaseLLMBackend):
    def __init__(self) -> None:
        super().__init__(
            name="stub",
            kind="stub",
            default_model="stub-default",
            config_hash="hash",
            model_patterns=("stub-*",),
        )
        self.generated: list[LLMRequest] = []

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=True,
            tool_calling="best_effort",
            json_mode="prompt+validate",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True, message="ok")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        self.generated.append(request)
        return LLMResponse(
            text='{"backend":"ok"}',
            model=request.model,
            provider=self.name,
            raw={"existing": True},
        )

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        yield LLMDelta(text_delta=str(request.model))


class _LlamaBackendStub:
    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:8080/v1"
        self.start_calls = 0

    def start(self) -> None:
        self.start_calls += 1


def test_echo_backend_supports_health_generate_and_stream() -> None:
    backend = EchoTestBackend(name="echo", model="echo-model", config_hash="cfg")
    assert backend.capabilities().streaming is True
    assert backend.healthcheck().ok is True

    response = backend.generate(_request(model="echo-model", messages=[]))
    assert "Hello from design-research-agents." in response.text

    streamed = list(backend.stream(_request(model="echo-model", messages=[])))
    assert len(streamed) == 1
    assert streamed[0].text_delta == response.text


def test_html_backend_supports_health_generate_and_stream() -> None:
    backend = HTMLBackend(name="html", model="html-standin-v1", config_hash="cfg")
    assert backend.capabilities().streaming is True
    assert backend.healthcheck().ok is True

    response = backend.generate(
        _request(
            model="html-standin-v1",
            messages=[LLMMessage(role="assistant", content=""), LLMMessage(role="system", content="fallback")],
        )
    )
    assert response.text == (
        "<html>\n  <body>\n    <h1>Mock Model Response</h1>\n    <p>fallback</p>\n  </body>\n</html>"
    )

    streamed = list(
        backend.stream(
            _request(
                model="html-standin-v1",
                messages=[LLMMessage(role="user", content="Prototype a repair guide")],
            )
        )
    )
    assert len(streamed) == 1
    assert "Prototype a repair guide" in (streamed[0].text_delta or "")


def test_llama_cpp_backend_delegates_to_server_and_http_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llama_server = _LlamaBackendStub()
    backend = LlamaCppBackend(
        name="llama",
        llama_backend=llama_server,  # type: ignore[arg-type]
        default_model="llama-model",
        config_hash="cfg",
    )
    monkeypatch.setattr(
        backend._http_backend,
        "generate",
        lambda request: LLMResponse(text="llama-ok", model=request.model, provider="llama"),
    )

    assert backend.capabilities().tool_calling == "best_effort"
    assert backend.capabilities().json_mode == "native"
    assert backend.healthcheck().ok is True

    request = _request(model="llama-model")
    response = backend.generate(request)
    assert response.text == "llama-ok"
    assert llama_server.start_calls == 1

    streamed = list(backend.stream(request))
    assert [chunk.text_delta for chunk in streamed] == ["llama-ok"]
    assert llama_server.start_calls == 2


def test_mlx_local_backend_generate_stream_and_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("mlx_lm")
    load_calls: list[str] = []

    class _Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, object]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            del tokenize, add_generation_prompt
            return f"chat:{messages[0]['content']}"

    def _load(model_id: str) -> tuple[object, object]:
        load_calls.append(model_id)
        return object(), _Tokenizer()

    def _generate(
        model: object,
        tokenizer: object,
        prompt: str,
        *,
        max_tokens: int,
        temp: float,
        stream: bool = False,
    ) -> str | Iterator[str]:
        del model, tokenizer
        if stream:
            assert max_tokens == 256
            assert temp == 0.7
            return iter(["a", "", "b"])
        assert prompt.startswith("chat:")
        return "mlx-ok"

    fake_module.load = _load
    fake_module.generate = _generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    backend = MlxLocalBackend(
        name="mlx",
        model_id="mlx-model",
        default_model="mlx-model",
        quantization="4bit",
        config_hash="cfg",
    )
    response = backend.generate(_request(model="mlx-model"))
    assert response.text == "mlx-ok"
    streamed = list(backend.stream(_request(model="mlx-model")))
    assert [delta.text_delta for delta in streamed] == ["a", "b"]
    assert load_calls == ["mlx-model"]


def test_mlx_local_backend_stream_handles_string_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MlxLocalBackend(
        name="mlx",
        model_id="mlx-model",
        default_model="mlx-model",
        quantization="4bit",
        config_hash="cfg",
    )
    monkeypatch.setattr(backend, "_ensure_model", lambda: (object(), object()))
    monkeypatch.setattr(
        "design_research_agents.llm._backends._providers._mlx_local._mlx_supports_streaming",
        lambda: True,
    )
    monkeypatch.setattr(
        "design_research_agents.llm._backends._providers._mlx_local._format_prompt",
        lambda request, tokenizer: "prompt",
    )
    monkeypatch.setattr(
        "design_research_agents.llm._backends._providers._mlx_local._mlx_generate",
        lambda *args, **kwargs: "single-chunk",
    )
    assert [delta.text_delta for delta in backend.stream(_request(model="mlx-model"))] == ["single-chunk"]

    monkeypatch.setattr(
        "design_research_agents.llm._backends._providers._mlx_local._mlx_generate",
        lambda *args, **kwargs: "",
    )
    assert list(backend.stream(_request(model="mlx-model"))) == []


def test_mlx_local_backend_ensure_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = MlxLocalBackend(
        name="mlx",
        model_id="mlx-model",
        default_model="mlx-model",
        quantization="4bit",
        config_hash="cfg",
    )
    real_import = builtins.__import__

    def _import_missing(name: str, *args: object, **kwargs: object):
        if name == "mlx_lm":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_missing)
    with pytest.raises(RuntimeError, match="mlx-lm"):
        backend._ensure_model()


def test_mlx_local_backend_rejects_unexpected_load_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("mlx_lm")
    fake_module.load = lambda _model_id: "bad-payload"
    fake_module.generate = lambda *args, **kwargs: "unused"
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    backend = MlxLocalBackend(
        name="mlx",
        model_id="mlx-model",
        default_model="mlx-model",
        quantization="4bit",
        config_hash="cfg",
    )
    with pytest.raises(RuntimeError, match="unexpected result"):
        backend._ensure_model()


def test_base_backend_structured_generation_paths_and_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _BestEffortBackend()
    tool = _tool()

    def _fake_generate_json(
        *,
        generate_fn,
        request: LLMRequest,
        schema: dict[str, object] | None,
        max_retries: int,
        extra_instructions: str | None,
    ) -> SimpleNamespace:
        del max_retries, extra_instructions
        response = generate_fn(request)
        if schema is not None and schema.get("required") == ["tool_calls"]:
            parsed = {"tool_calls": [{"name": "calculator", "arguments": {"value": 2}}]}
        else:
            parsed = {"answer": 42}
        return SimpleNamespace(response=response, parsed=parsed, attempts=0)

    monkeypatch.setattr(base_module, "generate_json", _fake_generate_json)

    json_response = backend.generate(_request(model=None, response_schema={"type": "object"}))
    assert json_response.text == '{"answer": 42}'
    assert backend.generated[-1].model == "stub-default"

    tool_response = backend.generate(_request(model="stub-default", tools=(tool,)))
    assert tool_response.tool_calls[0].call_id == "call_1"
    assert tool_response.tool_calls[0].name == "calculator"
    assert tool_response.raw is not None
    assert tool_response.raw["structured_output"]["attempts"] == 1

    with pytest.raises(LLMInvalidRequestError, match="explicit model id"):
        backend.generate(_request(model="*"))
    with pytest.raises(LLMInvalidRequestError, match="does not support model"):
        backend.generate(_request(model="other-model"))


def test_base_backend_tool_call_parser_edge_cases() -> None:
    tool = _tool()
    tools = (tool,)

    with pytest.raises(ValueError, match="not a list"):
        _parse_tool_calls("invalid", tools)
    with pytest.raises(ValueError, match="must be an object"):
        _parse_tool_calls([1], tools)
    with pytest.raises(ValueError, match="must be an object"):
        _parse_tool_calls([{"name": "calculator", "arguments": "bad"}], tools)

    assert _matches_model_pattern("prefix-model-suffix", "prefix*suffix") is True
