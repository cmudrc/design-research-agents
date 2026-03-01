"""MLX local backend for Apple Silicon inference via mlx-lm."""

from __future__ import annotations

import inspect
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

    @override
    def capabilities(self) -> BackendCapabilities:
        """Return capabilities inferred from installed MLX version.

        Returns:
            Backend capability flags for streaming, tools, JSON behavior,
            and other runtime limits.
        """
        return BackendCapabilities(
            streaming=_mlx_supports_streaming(),
            tool_calling="best_effort",
            json_mode="prompt+validate",
            vision=False,
            max_context_tokens=None,
        )

    @override
    def healthcheck(self) -> BackendStatus:
        """Return static health state for configured MLX backend.

        Returns:
            A healthy status when configuration is valid.
        """
        return BackendStatus(ok=True, message="MLX backend configured.")

    @override
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a non-streaming completion for one request.

        Args:
            request: Normalized request payload containing messages and options.

        Returns:
            Completed response text with provider/model metadata.
        """
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

    @override
    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Generate and yield streaming text deltas for one request.

        Args:
            request: Normalized request payload containing messages and options.

        Yields:
            Incremental text chunks from model output.
        """
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
        """Lazily load and cache the MLX model/tokenizer pair.

        Returns:
            Tuple of loaded ``(model, tokenizer)`` objects.

        Raises:
            RuntimeError: If ``mlx-lm`` is unavailable or load returns
                an unexpected shape.
        """
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "The 'mlx-lm' package is required for mlx_local backends. Install with: pip install mlx-lm"
            ) from exc
        loaded = load(self._model_id)
        if not isinstance(loaded, tuple) or len(loaded) < 2:
            raise RuntimeError("mlx_lm.load() returned an unexpected result.")
        model, tokenizer = loaded[0], loaded[1]
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer


def _format_prompt(request: LLMRequest, tokenizer: Any | None) -> str:
    """Format chat messages into a model prompt string.

    Args:
        request: LLM request containing ordered chat messages.
        tokenizer: Optional tokenizer with ``apply_chat_template`` support.

    Returns:
        Prompt text ready for MLX generation.
    """
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
    """Call ``mlx_lm.generate`` with backward-compatible sampling kwargs.

    Args:
        model: Loaded MLX model instance.
        tokenizer: Tokenizer paired with ``model``.
        prompt: Prompt string to complete.
        max_tokens: Maximum completion token budget.
        temperature: Optional sampling temperature override.
        stream: Optional streaming flag if supported by installed ``mlx-lm``.

    Returns:
        Either a full string completion or an iterator of streamed chunks.
    """
    from mlx_lm import generate

    temp_value = float(temperature if temperature is not None else 0.7)
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
    }
    try:
        from mlx_lm.sample_utils import make_sampler
    except ImportError:
        # Backward-compatible path for mlx-lm versions that still accept ``temp`` directly.
        kwargs["temp"] = temp_value
    else:
        kwargs["sampler"] = make_sampler(temp_value)
    if stream is not None and "stream" in inspect.signature(generate).parameters:
        kwargs["stream"] = stream
    return cast(str | Iterator[str], generate(model, tokenizer, prompt, **kwargs))


def _mlx_supports_streaming() -> bool:
    """Mlx supports streaming.

    Returns:
        ``True`` when installed ``mlx_lm.generate`` exposes a ``stream`` kwarg.
    """
    try:
        from mlx_lm import generate
    except ImportError:
        return False
    return "stream" in inspect.signature(generate).parameters
