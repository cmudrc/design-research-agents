"""Provider-specific LLM client classes with constructor-first defaults."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Sequence
from hashlib import sha256

from design_research_agents.contracts.llm import (
    BackendCapabilities,
    LLMChatParams,
    LLMClient,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)

from .backends.base import BaseLLMBackend
from .backends.providers.llama_cpp import LlamaCppBackend
from .backends.providers.llama_cpp_server import (
    create_backend as create_llama_cpp_server,
)
from .backends.providers.mlx_local import MlxLocalBackend
from .backends.providers.openai_compatible_http import OpenAICompatibleHTTPBackend
from .backends.providers.openai_service import OpenAIServiceBackend
from .backends.providers.transformers_local import TransformersLocalBackend

_OPENAI_COMPAT_CAPABILITIES = BackendCapabilities(
    streaming=False,
    tool_calling="best_effort",
    json_mode="prompt+validate",
    vision=False,
    max_context_tokens=None,
)


class _SingleBackendLLMClient(LLMClient):
    """LLM client wrapper that delegates to one concrete backend."""

    def __init__(self, *, backend: BaseLLMBackend) -> None:
        """Initialize the client with a configured backend instance.

        Args:
            backend: The LLM backend instance to delegate calls to. This should be fully configured
                and ready to use, as the client will not perform any additional setup.
        """
        self._backend = backend

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response using the configured backend."""
        return self._backend.generate(request)

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Build and execute a request-object call from chat-style inputs."""
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            tools=(),
            response_schema=params.response_schema,
            response_format=None,
            metadata={},
            provider_options=dict(params.provider_options),
            task_profile=None,
        )
        return self.generate(request)

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream response deltas for one request."""
        return self._backend.stream(request)

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Build and execute a streaming request from chat-style inputs."""
        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            tools=(),
            response_schema=params.response_schema,
            response_format=None,
            metadata={},
            provider_options=dict(params.provider_options),
            task_profile=None,
        )
        stream = self.stream(request)
        full_text = ""
        for delta in stream:
            if delta.text_delta:
                full_text += delta.text_delta
                yield LLMStreamEvent(kind="delta", delta_text=delta.text_delta)
        completed = getattr(stream, "response", None)
        if not isinstance(completed, LLMResponse):
            completed = LLMResponse(
                text=full_text,
                model=model,
                provider=None,
                finish_reason=None,
                usage=None,
                latency_ms=None,
                raw_output=None,
                tool_calls=(),
                raw=None,
                provenance=None,
            )
        yield LLMStreamEvent(kind="completed", response=completed)

    def default_model(self) -> str:
        """Return the configured backend default model."""
        default_model = self._backend.default_model
        if not isinstance(default_model, str) or not default_model.strip():
            raise ValueError("LLM backend default_model is not configured.")
        return default_model


class LlamaCppServerLLMClient(_SingleBackendLLMClient):
    """Client for a managed local ``llama_cpp.server`` backend."""

    def __init__(
        self,
        *,
        name: str = "llama-local",
        model: str = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        hf_model_repo_id: str | None = "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        api_model: str = "qwen2-1.5b-q4",
        host: str = "127.0.0.1",
        port: int = 8001,
        context_window: int = 4096,
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        python_executable: str = sys.executable,
        extra_server_args: tuple[str, ...] = (),
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a local llama-cpp client with sensible defaults.

        Args:
            name: Logical name for this client instance, used in logging and provenance.
            model: Local model identifier or path for llama_cpp.server to load.
            hf_model_repo_id: Optional Hugging Face repo ID to auto-download the model from if not
                found locally.
            api_model: The model name to report in API responses, which can differ from the local
                model name.
            host: Host interface for the local server to bind to.
            port: Port for the local server to listen on.
            context_window: Context window size (n_ctx) to configure the llama_cpp.server with.
            startup_timeout_seconds: Max time to wait for the server process to start and become
                healthy.
            poll_interval_seconds: Time interval between health check polls during startup.
            python_executable: Python executable to use for running the server process.
            extra_server_args: Additional command-line arguments to pass when starting the server
                process.
            max_retries: Number of times to retry a request in case of failure before giving up.
            model_patterns: Optional tuple of model name patterns supported by this client, used for
                routing decisions. If None, defaults to (api_model,).
        """
        combined_server_args = ("--n_ctx", str(context_window), *extra_server_args)
        self._llama_server = create_llama_cpp_server(
            model=model,
            hf_model_repo_id=hf_model_repo_id,
            api_model=api_model,
            host=host,
            port=port,
            startup_timeout_seconds=startup_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            python_executable=python_executable,
            extra_server_args=combined_server_args,
        )
        config_hash = _config_hash(
            {
                "kind": "llama_cpp",
                "name": name,
                "model": model,
                "hf_model_repo_id": hf_model_repo_id,
                "api_model": api_model,
                "host": host,
                "port": port,
                "context_window": context_window,
                "startup_timeout_seconds": startup_timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "python_executable": python_executable,
                "extra_server_args": combined_server_args,
                "max_retries": max_retries,
            }
        )
        backend = LlamaCppBackend(
            name=name,
            llama_backend=self._llama_server,
            default_model=api_model,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, api_model),
        )
        super().__init__(backend=backend)

    def close(self) -> None:
        """Stop the managed local server process."""
        self._llama_server.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort cleanup for managed server process during GC."""
        self.close()


class OpenAIServiceLLMClient(_SingleBackendLLMClient):
    """Client for the official OpenAI API backend."""

    def __init__(
        self,
        *,
        name: str = "openai",
        default_model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an OpenAI service client with sensible defaults.

        Args:
            name: Logical name for this client instance, used in logging and provenance.
            default_model: Default model name to use for requests that don't specify one.
            api_key_env: Environment variable name to read the API key from if not provided
                directly.
            api_key: Optional API key string. If not provided, the client will attempt to read
                the key from the environment variable specified by `api_key_env`.
            base_url: Optional base URL for the OpenAI API endpoint, useful for testing or if
                using a proxy. If None, defaults to the official OpenAI API URL.
            max_retries: Number of times to retry a request in case of failure before giving up
            model_patterns: Optional tuple of model name patterns supported by this client,
                used for routing decisions. If None, defaults to (default_model,).
        """
        config_hash = _config_hash(
            {
                "kind": "openai_service",
                "name": name,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "base_url": base_url,
                "max_retries": max_retries,
            }
        )
        backend = OpenAIServiceBackend(
            name=name,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            base_url=base_url,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(backend=backend)


class OpenAICompatibleHTTPLLMClient(_SingleBackendLLMClient):
    """Client for OpenAI-compatible HTTP endpoints."""

    def __init__(
        self,
        *,
        name: str = "openai-compatible",
        base_url: str = "http://127.0.0.1:8001/v1",
        default_model: str = "qwen2-1.5b-q4",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an OpenAI-compatible HTTP client with sensible defaults.

        Args:
            name: Logical name for this client instance, used in logging and provenance.
            base_url: Base URL for the OpenAI-compatible API endpoint.
            default_model: Default model name to use for requests that don't specify one.
            api_key_env: Environment variable name to read the API key from if not provided
                directly.
            api_key: Optional API key string. If not provided, the client will attempt to
                read the key from the environment variable specified by `api_key_env`.
            max_retries: Number of times to retry a request in case of failure before giving up.
            model_patterns: Optional tuple of model name patterns supported by this client, used for
                routing decisions. If None, defaults to (default_model,).
        """
        config_hash = _config_hash(
            {
                "kind": "openai_compatible_http",
                "name": name,
                "base_url": base_url,
                "default_model": default_model,
                "api_key_env": api_key_env,
                "api_key": api_key,
                "max_retries": max_retries,
            }
        )
        backend = OpenAICompatibleHTTPBackend(
            name=name,
            base_url=base_url,
            default_model=default_model,
            api_key_env=api_key_env,
            api_key=api_key,
            capabilities=_OPENAI_COMPAT_CAPABILITIES,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(backend=backend)


class TransformersLocalLLMClient(_SingleBackendLLMClient):
    """Client for in-process Transformers local inference."""

    def __init__(
        self,
        *,
        name: str = "transformers-local",
        model_id: str = "distilgpt2",
        default_model: str = "distilgpt2",
        device: str | None = "auto",
        dtype: str | None = "auto",
        quantization: str = "none",
        trust_remote_code: bool = False,
        revision: str | None = None,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a local Transformers client with sensible defaults.

        Args:
            name: Logical name for this client instance, used in logging and provenance.
            model_id: Identifier for the model to load (e.g. "distilgpt2
                or a Hugging Face repo ID like "gpt2").
            default_model: Default model name for prompts that don't specify one.
            device: Device to load the model on (e.g. "cpu", "cuda",
                "mps", or "auto" to automatically select based on availability).
            dtype: Data type to use for model weights (e.g. "float16", "bfloat16",
                "int8", or "auto" to automatically select based on device).
            quantization: Quantization level to use when loading the model (e.g. "4
                bit", "8-bit", "fp16", or "none" for no quantization).
            trust_remote_code: Whether to allow execution of custom code from remote repositories
                when loading models, which may be required for some models but can be a security
                risk.
            revision: Optional model revision to load (e.g. a git branch, tag, or
                commit hash), if the model is being loaded from a Hugging Face repository that
                has multiple revisions.
            max_retries: Number of times to retry a request in case of failure before giving up
            model_patterns: Optional tuple of model name patterns supported by this client, used
                for routing decisions. If None, defaults to (default_model,).
        """
        config_hash = _config_hash(
            {
                "kind": "transformers_local",
                "name": name,
                "model_id": model_id,
                "default_model": default_model,
                "device": device,
                "dtype": dtype,
                "quantization": quantization,
                "trust_remote_code": trust_remote_code,
                "revision": revision,
                "max_retries": max_retries,
            }
        )
        backend = TransformersLocalBackend(
            name=name,
            model_id=model_id,
            default_model=default_model,
            device=device,
            dtype=dtype,
            quantization=quantization,
            trust_remote_code=trust_remote_code,
            revision=revision,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(backend=backend)


class MlxLocalLLMClient(_SingleBackendLLMClient):
    """Client for Apple MLX local inference."""

    def __init__(
        self,
        *,
        name: str = "mlx-local",
        model_id: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        default_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        quantization: str = "none",
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an MLX local client with sensible defaults.

        Args:
            name: Logical name for this client instance, used in logging and provenance.
            model_id: Identifier for the MLX model to load (e.g. "mlx-community
                /Qwen2.5-1.5B-Instruct-4bit").
            default_model: Default model name for prompts that don't specify one.
            quantization: Quantization level to use when loading the model (e.g. "4
                -bit", "8-bit", "fp16").
            max_retries: Number of times to retry a request in case of failure before giving up
            model_patterns: Optional tuple of model name patterns supported by this client, used for
                routing decisions. If None, defaults to (default_model,).
        """
        config_hash = _config_hash(
            {
                "kind": "mlx_local",
                "name": name,
                "model_id": model_id,
                "default_model": default_model,
                "quantization": quantization,
                "max_retries": max_retries,
            }
        )
        backend = MlxLocalBackend(
            name=name,
            model_id=model_id,
            default_model=default_model,
            quantization=quantization,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(backend=backend)


def _resolve_model_patterns(
    model_patterns: tuple[str, ...] | None,
    default_model: str,
) -> tuple[str, ...]:
    if model_patterns is not None:
        return model_patterns
    return (default_model,)


def _config_hash(config_payload: dict[str, object]) -> str:
    encoded = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


__all__ = [
    "LlamaCppServerLLMClient",
    "MlxLocalLLMClient",
    "OpenAICompatibleHTTPLLMClient",
    "OpenAIServiceLLMClient",
    "TransformersLocalLLMClient",
]
