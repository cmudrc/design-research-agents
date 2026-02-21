from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from design_research_agents.contracts.llm import LLMMessage, LLMRequest
from design_research_agents.llm.backends.providers.transformers_local import (
    TransformersLocalBackend,
    _move_to_device,
)

pytestmark = pytest.mark.contract


class _FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values
        self.shape = (1, len(values))

    def to(self, _device: object) -> _FakeTensor:
        return self


class _FakeTokenizer:
    def __init__(self) -> None:
        self.last_prompt = ""

    def apply_chat_template(self, messages: object, **_kwargs: object) -> str:
        return f"prompt:{messages}"

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, _FakeTensor]:
        assert return_tensors == "pt"
        self.last_prompt = prompt
        return {"input_ids": _FakeTensor([1, 2, 3])}

    def decode(self, _ids: object, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "decoded-output"


class _FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.generate_calls: list[dict[str, object]] = []

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_calls.append(dict(kwargs))
        return [[1, 2, 3, 4]]

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self


def _request() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="hello")], model="demo")


def test_transformers_local_backend_generate_and_caching(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    calls: dict[str, int] = {"tokenizer": 0, "model": 0}
    captured_model_kwargs: dict[str, object] = {}

    transformers = ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeTokenizer:
            calls["tokenizer"] += 1
            return tokenizer

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args: object, **kwargs: object) -> _FakeModel:
            calls["model"] += 1
            captured_model_kwargs.update(dict(kwargs))
            return model

    transformers.AutoTokenizer = _AutoTokenizer
    transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    transformers.TextIteratorStreamer = object

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(float16="f16", bfloat16="bf16", float32="f32"),
    )

    backend = TransformersLocalBackend(
        name="local-transformers",
        model_id="demo-model",
        default_model="demo-model",
        device="cuda",
        dtype="float16",
        quantization="8bit",
        trust_remote_code=True,
        revision="main",
        config_hash="cfg",
    )

    response = backend._generate(_request())
    assert response.text == "decoded-output"
    assert model.generate_calls
    first_call = model.generate_calls[0]
    assert "temperature" not in first_call
    assert "do_sample" not in first_call
    assert calls == {"tokenizer": 1, "model": 1}
    assert captured_model_kwargs["load_in_8bit"] is True
    assert captured_model_kwargs["device_map"] == "auto"
    assert captured_model_kwargs["trust_remote_code"] is True
    assert captured_model_kwargs["revision"] == "main"
    assert captured_model_kwargs["torch_dtype"] == "f16"
    assert model.device == "cuda"

    backend._ensure_model()
    assert calls == {"tokenizer": 1, "model": 1}


def test_transformers_local_backend_generate_temperature_controls() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    backend = TransformersLocalBackend(
        name="local-transformers",
        model_id="demo-model",
        default_model="demo-model",
        device=None,
        dtype=None,
        quantization="none",
        trust_remote_code=False,
        revision=None,
        config_hash="cfg",
    )
    backend._tokenizer = tokenizer
    backend._model = model

    backend._generate(
        LLMRequest(
            messages=[LLMMessage(role="user", content="hi")],
            model="demo",
            temperature=0.3,
        )
    )
    sampled_call = model.generate_calls[-1]
    assert sampled_call["do_sample"] is True
    assert sampled_call["temperature"] == 0.3
    assert "top_p" not in sampled_call
    assert "top_k" not in sampled_call

    backend._generate(
        LLMRequest(
            messages=[LLMMessage(role="user", content="hi")],
            model="demo",
            temperature=0.0,
        )
    )
    greedy_call = model.generate_calls[-1]
    assert greedy_call["do_sample"] is False
    assert greedy_call["temperature"] == 1.0
    assert greedy_call["top_p"] == 1.0
    assert greedy_call["top_k"] == 50


def test_transformers_local_backend_stream_import_error_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersLocalBackend(
        name="local-transformers",
        model_id="demo-model",
        default_model="demo-model",
        device=None,
        dtype=None,
        quantization="none",
        trust_remote_code=False,
        revision=None,
        config_hash="cfg",
    )

    backend._tokenizer = _FakeTokenizer()
    backend._model = _FakeModel()

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert list(backend._stream(_request())) == []


def test_transformers_local_backend_ensure_model_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = TransformersLocalBackend(
        name="local-transformers",
        model_id="demo-model",
        default_model="demo-model",
        device=None,
        dtype=None,
        quantization="none",
        trust_remote_code=False,
        revision=None,
        config_hash="cfg",
    )

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="transformers"):
        backend._ensure_model()


def test_move_to_device_no_device_returns_input_mapping() -> None:
    payload = {"input_ids": _FakeTensor([1, 2, 3])}
    model = SimpleNamespace()
    assert _move_to_device(payload, model) == payload


def test_transformers_local_backend_capabilities_and_healthcheck() -> None:
    backend = TransformersLocalBackend(
        name="local-transformers",
        model_id="demo-model",
        default_model="demo-model",
        device=None,
        dtype=None,
        quantization="none",
        trust_remote_code=False,
        revision=None,
        config_hash="cfg",
    )

    capabilities = backend.capabilities()
    status = backend.healthcheck()
    assert capabilities.tool_calling == "best_effort"
    assert status.ok is True
