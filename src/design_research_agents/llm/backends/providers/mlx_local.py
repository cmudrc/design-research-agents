"""MLX local backend for Apple Silicon inference via mlx-lm."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import Any, cast

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    BackendStatus,
    LLMDelta,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.llm.backends.base import BaseLLMBackend
from design_research_agents.llm.backends.utils import messages_to_prompt


class MlxLocalBackend(BaseLLMBackend):
    """MLX-LM backend for local Apple Silicon inference."""

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        default_model: str,
        quantization: str,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] = (),
    ) -> None:
        """Configure MLX backend parameters and deferred model loading.

        Args:
            name: Unique name for this backend configuration.
            model_id: Identifier for the MLX model to load (e.g. "gemma-2b-it").
            default_model: Default model name for prompts that don't specify one.
            quantization: Quantization level to use when loading the model (e.g. "4-bit", "8-bit",
                "fp16").
            config_hash: Unique hash of the configuration for caching and invalidation purposes.
            max_retries: Maximum number of retries for generation attempts.
            model_patterns: Optional tuple of glob patterns to match against
                model names for routing purposes.
        """
        super().__init__(
            name=name,
            kind="mlx_local",
            default_model=default_model,
            base_url=None,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=model_patterns,
        )
        self._model_id = model_id
        self._quantization = quantization
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def capabilities(self) -> BackendCapabilities:
        """Return capabilities inferred from installed MLX version."""
        return BackendCapabilities(
            streaming=_mlx_supports_streaming(),
            tool_calling="best_effort",
            json_mode="prompt+validate",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        """Return static health state for configured MLX backend."""
        return BackendStatus(ok=True, message="MLX backend configured.")

    def _generate(self, request: LLMRequest) -> LLMResponse:
        model, tokenizer = self._ensure_model()
        prompt = _format_prompt(request, tokenizer)
        output = _mlx_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature,
        )
        text = output if isinstance(output, str) else "".join(output)
        return LLMResponse(text=text, model=request.model, provider=self.name)

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        model, tokenizer = self._ensure_model()
        prompt = _format_prompt(request, tokenizer)
        output = _mlx_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=request.max_tokens or 256,
            temperature=request.temperature,
            stream=True,
        )
        if isinstance(output, str):
            if output:
                yield LLMDelta(text_delta=output)
            return
        for chunk in output:
            if chunk:
                yield LLMDelta(text_delta=str(chunk))

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "The 'mlx-lm' package is required for mlx_local backends. "
                "Install with: pip install mlx-lm"
            ) from exc
        model, tokenizer = load(self._model_id)
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer


def _format_prompt(request: LLMRequest, tokenizer: Any | None) -> str:
    messages = [{"role": message.role, "content": message.content} for message in request.messages]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
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


def _mlx_generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float | None,
    stream: bool | None = None,
) -> str | Iterator[str]:
    from mlx_lm import generate

    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temp": temperature if temperature is not None else 0.7,
    }
    if stream is not None and "stream" in inspect.signature(generate).parameters:
        kwargs["stream"] = stream
    return cast(str | Iterator[str], generate(model, tokenizer, prompt, **kwargs))


def _mlx_supports_streaming() -> bool:
    try:
        from mlx_lm import generate
    except ImportError:
        return False
    return "stream" in inspect.signature(generate).parameters
