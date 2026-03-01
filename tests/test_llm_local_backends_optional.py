from __future__ import annotations

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

from design_research_agents._contracts._llm import LLMMessage, LLMRequest
from design_research_agents.llm._backends._providers import _mlx_local, _transformers_local


def _request() -> LLMRequest:
    return LLMRequest(messages=[LLMMessage(role="user", content="hello")], model="m")


def test_transformers_quantization_and_prompt_fallback() -> None:
    assert _transformers_local._quantization_kwargs("8bit") == {"load_in_8bit": True}
    assert _transformers_local._quantization_kwargs("4bit") == {"load_in_4bit": True}
    assert _transformers_local._quantization_kwargs("none") == {}

    class _BadTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

    prompt = _transformers_local._format_prompt(_request(), _BadTokenizer())
    assert "user: hello" in prompt


def test_transformers_resolve_dtype_success_and_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(float16="f16", bfloat16="bf16", float32="f32")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert _transformers_local._resolve_dtype("float16") == "f16"
    assert _transformers_local._resolve_dtype("bfloat16") == "bf16"
    assert _transformers_local._resolve_dtype("float32") == "f32"
    with pytest.raises(ValueError, match="Unsupported dtype"):
        _transformers_local._resolve_dtype("int8")


def test_transformers_resolve_dtype_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "torch":
            raise ImportError("missing torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="torch is required"):
        _transformers_local._resolve_dtype("float16")


def test_transformers_streaming_available_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = ModuleType("transformers")
    fake.TextIteratorStreamer = object
    monkeypatch.setitem(sys.modules, "transformers", fake)
    assert _transformers_local._streaming_available() is True

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "transformers":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert _transformers_local._streaming_available() is False


def test_transformers_generation_control_kwargs() -> None:
    assert _transformers_local._generation_control_kwargs(None) == {}
    assert _transformers_local._generation_control_kwargs(0.2) == {
        "do_sample": True,
        "temperature": 0.2,
    }
    assert _transformers_local._generation_control_kwargs(0.0) == {
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
    }


def test_mlx_prompt_fallback_and_streaming_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BadTokenizer:
        def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("boom")

    prompt = _mlx_local._format_prompt(_request(), _BadTokenizer())
    assert "user: hello" in prompt

    fake_module = ModuleType("mlx_lm")

    def _generate_with_stream(
        model: object,
        tokenizer: object,
        prompt: str,
        *,
        stream: bool = False,
    ):
        del model, tokenizer, prompt
        return stream

    fake_module.generate = _generate_with_stream
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    assert _mlx_local._mlx_supports_streaming() is True


def test_mlx_generate_stream_kwarg_and_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("mlx_lm")
    calls: list[dict[str, object]] = []

    def _generate_no_stream(
        model: object,
        tokenizer: object,
        prompt: str,
        **kwargs: object,
    ):
        del model, tokenizer, prompt
        calls.append(dict(kwargs))
        return "ok"

    fake_module.generate = _generate_no_stream
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)

    assert _mlx_local._mlx_generate(object(), object(), "p", max_tokens=10, temperature=0.2) == "ok"
    assert "stream" not in calls[-1]

    def _generate_with_stream(
        model: object,
        tokenizer: object,
        prompt: str,
        *,
        stream: bool = False,
        **kwargs: object,
    ):
        del model, tokenizer, prompt
        call = dict(kwargs)
        call["stream"] = stream
        calls.append(call)
        return "ok"

    fake_module.generate = _generate_with_stream
    assert _mlx_local._mlx_generate(object(), object(), "p", max_tokens=10, temperature=0.2, stream=True) == "ok"
    assert calls[-1]["stream"] is True

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "mlx_lm":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert _mlx_local._mlx_supports_streaming() is False


def test_mlx_generate_uses_sampler_when_sample_utils_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = ModuleType("mlx_lm")
    fake_module.__path__ = []  # Mark as package so submodule import path is valid.
    fake_sample_utils = ModuleType("mlx_lm.sample_utils")
    calls: list[dict[str, object]] = []

    def _generate_no_stream(
        model: object,
        tokenizer: object,
        prompt: str,
        **kwargs: object,
    ) -> str:
        del model, tokenizer, prompt
        calls.append(dict(kwargs))
        return "ok"

    def _make_sampler(temp: float) -> object:
        return {"sampler_temp": temp}

    fake_module.generate = _generate_no_stream
    fake_sample_utils.make_sampler = _make_sampler
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", fake_sample_utils)

    assert _mlx_local._mlx_generate(object(), object(), "p", max_tokens=10, temperature=0.2) == "ok"
    assert calls[-1]["sampler"] == {"sampler_temp": 0.2}
    assert "temp" not in calls[-1]
