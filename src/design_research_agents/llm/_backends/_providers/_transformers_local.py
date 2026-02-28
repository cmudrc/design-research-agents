"""Transformers local backend for in-process model inference."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any, cast, override

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm._backends._utils import messages_to_prompt


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
        """Configure local Transformers backend and deferred model loading.

        Args:
            name: Unique backend name for tracing and routing.
            model_id: Hugging Face model identifier to load locally.
            default_model: Default model returned by ``default_model()``.
            device: Optional explicit target device (for example ``cpu``).
            dtype: Optional explicit torch dtype string (for example ``float16``).
            quantization: Quantization mode (``none``, ``8bit``, ``4bit``).
            trust_remote_code: Whether to allow model-defined remote code.
            revision: Optional model revision or branch name.
            config_hash: Stable hash of backend config inputs.
            max_retries: Maximum retry attempts for request execution.
            model_patterns: Optional glob-like patterns for selector routing.
        """
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

    @override
    def capabilities(self) -> BackendCapabilities:
        """Return capabilities inferred from installed Transformers features.

        Returns:
            Capability metadata for this backend instance.
        """
        return BackendCapabilities(
            streaming=_streaming_available(),
            tool_calling="best_effort",
            json_mode="prompt+validate",
            vision=False,
            max_context_tokens=None,
        )

    @override
    def healthcheck(self) -> BackendStatus:
        """Return static health status for configured backend.

        Returns:
            Healthy backend status when configuration is valid.
        """
        return BackendStatus(ok=True, message="Transformers backend configured.")

    @override
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a non-streaming response from the local model.

        Args:
            request: Normalized request including messages and generation limits.

        Returns:
            Completed response text and metadata.
        """
        tokenizer, model = self._ensure_model()
        prompt = _format_prompt(request, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = _move_to_device(inputs, model)
        input_length = inputs["input_ids"].shape[-1]
        generation_kwargs = {
            **inputs,
            "max_new_tokens": request.max_tokens or 256,
            **_generation_control_kwargs(request.temperature),
        }
        output_ids = model.generate(**generation_kwargs)
        generated_ids = output_ids[0][input_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return LLMResponse(text=text, model=request.model, provider=self.name)

    @override
    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Generate and yield streaming deltas from the local model.

        Args:
            request: Normalized request including messages and generation limits.

        Yields:
            Incremental text chunks emitted by the streamer.
        """
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
            **_generation_control_kwargs(request.temperature),
            "streamer": streamer,
        }

        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()
        for text in streamer:
            if text:
                yield LLMDelta(text_delta=text)
        thread.join(timeout=1.0)

    def _ensure_model(self) -> tuple[Any, Any]:
        """Load tokenizer/model once and reuse cached instances.

        Returns:
            Tuple of ``(tokenizer, model)`` for generation calls.

        Raises:
            RuntimeError: If ``transformers`` is not installed.
        """
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
            move_to = getattr(model, "to", None)
            if callable(move_to):
                model = cast(Any, move_to)(self._device)
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model


def _format_prompt(request: LLMRequest, tokenizer: Any) -> str:
    """Format request messages into one prompt string.

    Args:
        request: Request carrying user/system messages.
        tokenizer: Tokenizer that may expose ``apply_chat_template``.

    Returns:
        Prompt string suitable for model generation.
    """
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
    """Resolve a supported dtype name to a torch dtype object.

    Args:
        dtype: Dtype name such as ``float16``, ``bfloat16``, or ``float32``.

    Returns:
        Torch dtype constant matching ``dtype``.

    Raises:
        RuntimeError: If torch is not installed.
        ValueError: If ``dtype`` is unsupported.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for transformers_local backend dtype settings.") from exc
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype}'.")
    return mapping[dtype]


def _quantization_kwargs(quantization: str) -> dict[str, Any]:
    """Translate quantization mode into ``from_pretrained`` kwargs.

    Args:
        quantization: Requested quantization level.

    Returns:
        Keyword arguments for model loading.
    """
    if quantization == "8bit":
        return {"load_in_8bit": True}
    if quantization == "4bit":
        return {"load_in_4bit": True}
    return {}


def _generation_control_kwargs(temperature: float | None) -> dict[str, Any]:
    """Return generation controls derived from one optional temperature override.

    When temperature is omitted we defer to the model generation config.
    A non-positive temperature requests greedy decoding, so we also neutralize
    sampling-only knobs to avoid Transformers warning about ignored flags.
    """
    if temperature is None:
        return {}
    if temperature > 0:
        return {"do_sample": True, "temperature": float(temperature)}
    return {"do_sample": False, "temperature": 1.0, "top_p": 1.0, "top_k": 50}


def _move_to_device(inputs: dict[str, Any], model: Any) -> dict[str, Any]:
    """Move tokenized inputs onto the model device when available.

    Args:
        inputs: Tokenizer output mapping.
        model: Model object that may expose ``device``.

    Returns:
        Possibly device-moved tensor mapping.
    """
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return {key: value.to(device) for key, value in inputs.items()}


def _streaming_available() -> bool:
    """Streaming available.

    Returns:
        ``True`` when ``TextIteratorStreamer`` can be imported.
    """
    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        return False
    return TextIteratorStreamer is not None
