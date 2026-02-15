"""Transformers local backend for in-process model inference."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.utils import messages_to_prompt


class TransformersLocalBackend(BaseLLMBackend):
    """Transformers backend using AutoModelForCausalLM."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        default_model: str,
        device: str | None,
        dtype: str | None,
        quantization: str,
        trust_remote_code: bool,
        revision: str | None,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Configure local Transformers backend and deferred model loading."""
        super().__init__(
            name=name,
            kind="transformers_local",
            default_model=default_model,
            base_url=None,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._model_id = model_id
        self._device = device
        self._dtype = dtype
        self._quantization = quantization
        self._trust_remote_code = trust_remote_code
        self._revision = revision
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def capabilities(self) -> BackendCapabilities:
        """Return capabilities inferred from installed Transformers features."""
        return BackendCapabilities(
            streaming=_streaming_available(),
            tool_calling="best_effort",
            json_mode="prompt+validate",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        """Return static health status for configured backend."""
        return BackendStatus(ok=True, message="Transformers backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        tokenizer, model = self._ensure_model()
        prompt = _format_prompt(request, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = _move_to_device(inputs, model)
        input_length = inputs["input_ids"].shape[-1]
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens or 256,
            temperature=request.temperature if request.temperature is not None else 0.7,
            do_sample=request.temperature is not None and request.temperature > 0,
        )
        generated_ids = output_ids[0][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return LLMResponse(text=text, model=request.model, provider=self.name)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        tokenizer, model = self._ensure_model()
        try:
            from transformers import TextIteratorStreamer
        except ImportError:
            yield from ()
            return
        prompt = _format_prompt(request, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = _move_to_device(inputs, model)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": request.max_tokens or 256,
            "temperature": request.temperature if request.temperature is not None else 0.7,
            "do_sample": request.temperature is not None and request.temperature > 0,
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        for text in streamer:
            if text:
                yield LLMDelta(text_delta=text)
        thread.join(timeout=1.0)

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "The 'transformers' package is required for transformers_local backends. "
                "Install with: pip install -e '.[local]'"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            revision=self._revision,
            trust_remote_code=self._trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {}
        if self._quantization in {"8bit", "4bit"}:
            model_kwargs.update(_quantization_kwargs(self._quantization))
            model_kwargs.setdefault("device_map", "auto")
        if self._dtype and self._dtype != "auto":
            model_kwargs["torch_dtype"] = _resolve_dtype(self._dtype)
        if self._revision:
            model_kwargs["revision"] = self._revision
        if self._trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        model = AutoModelForCausalLM.from_pretrained(self._model_id, **model_kwargs)
        if self._device and self._device not in {"auto"} and hasattr(model, "to"):
            model = model.to(self._device)
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model


def _format_prompt(request: LLMRequest, tokenizer: Any) -> str:
    messages = [{"role": message.role, "content": message.content} for message in request.messages]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return str(formatted)
        except Exception:
            return messages_to_prompt(request.messages)
    return messages_to_prompt(request.messages)


def _resolve_dtype(dtype: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for transformers_local backend dtype settings."
        ) from exc
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'.")
    return mapping[dtype]


def _quantization_kwargs(quantization: str) -> dict[str, Any]:
    if quantization == "8bit":
        return {"load_in_8bit": True}
    if quantization == "4bit":
        return {"load_in_4bit": True}
    return {}


def _move_to_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def _streaming_available() -> bool:
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        return False
    return TextIteratorStreamer is not None
