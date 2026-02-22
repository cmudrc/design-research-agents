"""Provider-specific LLM client classes with constructor-first defaults."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from hashlib import sha256

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    LLMChatParams,
    LLMClient,
    LLMDelta,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMStreamEvent,
)
from design_research_agents._tracing import (
    emit_model_request_observed,
    emit_model_response_observed,
)

from .._backends._base import BaseLLMBackend
from .._backends._providers._llama_cpp import LlamaCppBackend
from .._backends._providers._llama_cpp_server import (
    create_backend as create_llama_cpp_server,
)
from .._backends._providers._mlx_local import MlxLocalBackend
from .._backends._providers._ollama_local import OllamaLocalBackend
from .._backends._providers._ollama_server import (
    create_backend as create_ollama_server,
)
from .._backends._providers._sglang_local import SglangLocalBackend
from .._backends._providers._sglang_server import (
    create_backend as create_sglang_server,
)
from .._backends._providers._transformers_local import TransformersLocalBackend
from .._backends._providers._vllm_local import VllmLocalBackend
from .._backends._providers._vllm_server import (
    create_backend as create_vllm_server,
)
from ._snapshot_helpers import (
    capabilities_to_dict,
    llama_config_snapshot,
    llama_server_snapshot,
    mlx_config_snapshot,
    normalize_snapshot_mapping,
    ollama_config_snapshot,
    ollama_server_snapshot,
    sglang_config_snapshot,
    sglang_server_snapshot,
    transformers_config_snapshot,
    vllm_config_snapshot,
    vllm_server_snapshot,
)


class _SingleBackendLLMClient(LLMClient):
    """LLM client wrapper that delegates to one concrete backend."""

    def __init__(
        self,
        *,
        backend: BaseLLMBackend,
        config_snapshot: Mapping[str, object] | None = None,
        server_snapshot: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize the client with a configured backend instance.

        Args:
            backend: The LLM backend instance to delegate calls to. This should be fully configured
                and ready to use, as the client will not perform any additional setup.
            config_snapshot: Additional client/backend config metadata fields.
            server_snapshot: Managed server metadata fields when server lifecycle is owned.
        """
        self._backend = backend
        snapshot: dict[str, object] = {
            "name": backend.name,
            "kind": backend.kind,
            "default_model": backend.default_model,
            "base_url": backend.base_url,
            "config_hash": backend.config_hash,
            "max_retries": backend.max_retries,
            "model_patterns": list(backend.model_patterns),
        }
        if config_snapshot is not None:
            snapshot.update(normalize_snapshot_mapping(config_snapshot))
        # Snapshot fields are intentionally plain JSON-compatible scalars for deterministic example output.
        self._config_snapshot = snapshot
        self._server_snapshot = normalize_snapshot_mapping(server_snapshot) if server_snapshot is not None else None

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response using the configured backend.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Normalized completion response.
        """
        emit_model_request_observed(
            source="_SingleBackendLLMClient.generate",
            model=request.model,
            request_payload=request,
            metadata={"backend_name": self._backend.name, "backend_kind": self._backend.kind},
        )
        try:
            response = self._backend.generate(request)
        except Exception as exc:
            emit_model_response_observed(
                source="_SingleBackendLLMClient.generate",
                error=str(exc),
                metadata={"backend_name": self._backend.name, "backend_kind": self._backend.kind},
            )
            raise
        emit_model_response_observed(
            source="_SingleBackendLLMClient.generate",
            response_payload=response,
            metadata={"backend_name": self._backend.name, "backend_kind": self._backend.kind},
        )
        return response

    def chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> LLMResponse:
        """Build and execute a request-object call from chat-style inputs.

        Args:
            messages: Ordered chat messages for this completion request.
            model: Target model identifier.
            params: Generation controls (temperature, max tokens, schema, options).

        Returns:
            Normalized completion response.
        """
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
        # Reuse generate() so chat() inherits shared tracing and error-observation behavior.
        return self.generate(request)

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream response deltas for one request.

        Args:
            request: Provider-neutral request payload.

        Returns:
            Iterator yielding normalized response deltas.
        """
        emit_model_request_observed(
            source="_SingleBackendLLMClient.stream",
            model=request.model,
            request_payload=request,
            metadata={"backend_name": self._backend.name, "backend_kind": self._backend.kind},
        )
        stream = self._backend.stream(request)

        def _observed_stream() -> Iterator[LLMDelta]:
            delta_count = 0
            text_delta_count = 0
            try:
                for delta in stream:
                    delta_count += 1
                    if delta.text_delta:
                        text_delta_count += 1
                    yield delta
            except Exception as exc:
                emit_model_response_observed(
                    source="_SingleBackendLLMClient.stream",
                    error=str(exc),
                    metadata={
                        "backend_name": self._backend.name,
                        "backend_kind": self._backend.kind,
                        "delta_count": delta_count,
                        "text_delta_count": text_delta_count,
                    },
                )
                raise
            emit_model_response_observed(
                source="_SingleBackendLLMClient.stream",
                response_payload=getattr(stream, "response", None),
                metadata={
                    "backend_name": self._backend.name,
                    "backend_kind": self._backend.kind,
                    "delta_count": delta_count,
                    "text_delta_count": text_delta_count,
                },
            )

        return _observed_stream()

    def stream_chat(
        self,
        messages: Sequence[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> Iterator[LLMStreamEvent]:
        """Build and execute a streaming request from chat-style inputs.

        Args:
            messages: Ordered chat messages for this completion request.
            model: Target model identifier.
            params: Generation controls (temperature, max tokens, schema, options).

        Yields:
            Streaming events containing deltas and one final completed response.
        """
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
            # If backend streaming does not expose a final response object, synthesize one from deltas.
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
        """Return the configured backend default model.

        Returns:
            Non-empty model identifier configured as backend default.

        Raises:
            ValueError: If the backend default model is missing or blank.
        """
        default_model = self._backend.default_model
        if not isinstance(default_model, str) or not default_model.strip():
            raise ValueError("LLM backend default_model is not configured.")
        return default_model

    def capabilities(self) -> BackendCapabilities:
        """Return declared backend capabilities for this client.

        Returns:
            Backend capability payload.
        """
        return self._backend.capabilities()

    def config_snapshot(self) -> dict[str, object]:
        """Return stable client/backend configuration metadata.

        Returns:
            Snapshot dictionary safe for diagnostics and example output.
        """
        return normalize_snapshot_mapping(self._config_snapshot)

    def server_snapshot(self) -> dict[str, object] | None:
        """Return managed-server metadata when this client owns a server.

        Returns:
            Server snapshot dictionary, or ``None`` when not managed.
        """
        if self._server_snapshot is None:
            return None
        return normalize_snapshot_mapping(self._server_snapshot)

    def describe(self) -> dict[str, object]:
        """Return composed client configuration and capability summary.

        Returns:
            JSON-serializable description of client runtime characteristics.
        """
        return {
            "client_class": self.__class__.__name__,
            "default_model": self.default_model(),
            "backend": self.config_snapshot(),
            "capabilities": capabilities_to_dict(self.capabilities()),
            "server": self.server_snapshot(),
        }


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
        super().__init__(
            backend=backend,
            config_snapshot=llama_config_snapshot(
                model=model,
                hf_model_repo_id=hf_model_repo_id,
                api_model=api_model,
                context_window=context_window,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                python_executable=python_executable,
                extra_server_args=combined_server_args,
            ),
            server_snapshot=llama_server_snapshot(server=self._llama_server),
        )

    def close(self) -> None:
        """Stop the managed local server process."""
        self._llama_server.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort cleanup for managed server process during GC."""
        self.close()


class VllmServerLLMClient(_SingleBackendLLMClient):
    """Client for local or self-hosted vLLM OpenAI-compatible inference."""

    def __init__(
        self,
        *,
        name: str = "vllm-local",
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        api_model: str = "qwen2.5-1.5b-instruct",
        host: str = "127.0.0.1",
        port: int = 8002,
        manage_server: bool = True,
        startup_timeout_seconds: float = 90.0,
        poll_interval_seconds: float = 0.5,
        python_executable: str = sys.executable,
        extra_server_args: tuple[str, ...] = (),
        base_url: str | None = None,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize a vLLM client in managed-server or connect mode.

        Args:
            name: Logical name for this client instance.
            model: Model identifier passed to managed vLLM server startup.
            api_model: Model alias exposed by vLLM OpenAI-compatible API.
            host: Host interface used in managed mode.
            port: TCP port used in managed mode.
            manage_server: Whether this client manages the vLLM server lifecycle.
            startup_timeout_seconds: Maximum startup wait time in managed mode.
            poll_interval_seconds: Delay between readiness probes in managed mode.
            python_executable: Python executable used to launch managed vLLM process.
            extra_server_args: Additional CLI flags forwarded to vLLM server.
            base_url: Optional connect-mode endpoint URL. Required only for remote/self-managed
                deployments; defaults to ``http://{host}:{port}/v1``.
            request_timeout_seconds: HTTP timeout for generate and stream requests.
            max_retries: Number of retries for retryable provider/transport errors.
            model_patterns: Optional tuple of model patterns for routing decisions.

        Raises:
            ValueError: If ``manage_server`` and ``base_url`` are both configured.
        """
        if manage_server and base_url is not None:
            raise ValueError("base_url cannot be provided when manage_server is True.")

        self._vllm_server = (
            create_vllm_server(
                model=model,
                api_model=api_model,
                host=host,
                port=port,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                python_executable=python_executable,
                extra_server_args=extra_server_args,
            )
            if manage_server
            else None
        )
        resolved_base_url = (
            self._vllm_server.base_url if self._vllm_server is not None else (base_url or f"http://{host}:{port}/v1")
        )
        config_hash = _config_hash(
            {
                "kind": "vllm_local",
                "name": name,
                "model": model,
                "api_model": api_model,
                "host": host,
                "port": port,
                "manage_server": manage_server,
                "startup_timeout_seconds": startup_timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "python_executable": python_executable,
                "extra_server_args": extra_server_args,
                "base_url": base_url,
                "request_timeout_seconds": request_timeout_seconds,
                "max_retries": max_retries,
            }
        )
        backend = VllmLocalBackend(
            name=name,
            base_url=resolved_base_url,
            default_model=api_model,
            request_timeout_seconds=request_timeout_seconds,
            managed_server=self._vllm_server,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, api_model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=vllm_config_snapshot(
                model=model,
                api_model=api_model,
                host=host,
                port=port,
                manage_server=manage_server,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                python_executable=python_executable,
                extra_server_args=extra_server_args,
                request_timeout_seconds=request_timeout_seconds,
            ),
            server_snapshot=vllm_server_snapshot(server=self._vllm_server),
        )

    def close(self) -> None:
        """Stop the managed vLLM server process when present."""
        server = getattr(self, "_vllm_server", None)
        if server is not None:
            server.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort managed server cleanup during garbage collection."""
        self.close()


class OllamaLLMClient(_SingleBackendLLMClient):
    """Client for local or self-hosted Ollama chat inference."""

    def __init__(
        self,
        *,
        name: str = "ollama-local",
        default_model: str = "qwen2.5:1.5b-instruct",
        host: str = "127.0.0.1",
        port: int = 11434,
        manage_server: bool = True,
        ollama_executable: str = "ollama",
        auto_pull_model: bool = False,
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.25,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an Ollama client in managed-server or connect mode.

        Args:
            name: Logical name for this client instance.
            default_model: Default model id used when requests omit model.
            host: Host interface used in managed mode or connect mode.
            port: TCP port used in managed mode or connect mode.
            manage_server: Whether this client manages ``ollama serve`` lifecycle.
            ollama_executable: Executable used to invoke ``ollama`` commands.
            auto_pull_model: Whether to pull ``default_model`` after startup.
            startup_timeout_seconds: Maximum startup wait time in managed mode.
            poll_interval_seconds: Delay between readiness probes in managed mode.
            request_timeout_seconds: HTTP timeout for generate and stream requests.
            max_retries: Number of retries for retryable provider/transport errors.
            model_patterns: Optional tuple of model patterns for routing decisions.
        """
        self._ollama_server = (
            create_ollama_server(
                host=host,
                port=port,
                ollama_executable=ollama_executable,
                auto_pull_model=auto_pull_model,
                default_model=default_model,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
            )
            if manage_server
            else None
        )
        resolved_base_url = self._ollama_server.base_url if self._ollama_server is not None else f"http://{host}:{port}"
        config_hash = _config_hash(
            {
                "kind": "ollama_local",
                "name": name,
                "default_model": default_model,
                "host": host,
                "port": port,
                "manage_server": manage_server,
                "ollama_executable": ollama_executable,
                "auto_pull_model": auto_pull_model,
                "startup_timeout_seconds": startup_timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "request_timeout_seconds": request_timeout_seconds,
                "max_retries": max_retries,
            }
        )
        backend = OllamaLocalBackend(
            name=name,
            base_url=resolved_base_url,
            default_model=default_model,
            request_timeout_seconds=request_timeout_seconds,
            managed_server=self._ollama_server,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, default_model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=ollama_config_snapshot(
                host=host,
                port=port,
                manage_server=manage_server,
                ollama_executable=ollama_executable,
                auto_pull_model=auto_pull_model,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                request_timeout_seconds=request_timeout_seconds,
            ),
            server_snapshot=ollama_server_snapshot(server=self._ollama_server),
        )

    def close(self) -> None:
        """Stop the managed Ollama daemon when present."""
        server = getattr(self, "_ollama_server", None)
        if server is not None:
            server.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort managed daemon cleanup during garbage collection."""
        self.close()


class SglangServerLLMClient(_SingleBackendLLMClient):
    """Client for local or self-hosted SGLang OpenAI-compatible inference."""

    def __init__(
        self,
        *,
        name: str = "sglang-local",
        model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        host: str = "127.0.0.1",
        port: int = 30000,
        manage_server: bool = True,
        startup_timeout_seconds: float = 90.0,
        poll_interval_seconds: float = 0.5,
        python_executable: str = sys.executable,
        extra_server_args: tuple[str, ...] = (),
        base_url: str | None = None,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        model_patterns: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize an SGLang client in managed-server or connect mode.

        Args:
            name: Logical name for this client instance.
            model: Model identifier passed to managed SGLang server startup.
            host: Host interface used in managed mode.
            port: TCP port used in managed mode.
            manage_server: Whether this client manages the SGLang server lifecycle.
            startup_timeout_seconds: Maximum startup wait time in managed mode.
            poll_interval_seconds: Delay between readiness probes in managed mode.
            python_executable: Python executable used to launch managed SGLang process.
            extra_server_args: Additional CLI flags forwarded to SGLang server.
            base_url: Optional connect-mode endpoint URL. Required only for remote/self-managed
                deployments; defaults to ``http://{host}:{port}/v1``.
            request_timeout_seconds: HTTP timeout for generate and stream requests.
            max_retries: Number of retries for retryable provider/transport errors.
            model_patterns: Optional tuple of model patterns for routing decisions.

        Raises:
            ValueError: If ``manage_server`` and ``base_url`` are both configured.
        """
        if manage_server and base_url is not None:
            raise ValueError("base_url cannot be provided when manage_server is True.")

        self._sglang_server = (
            create_sglang_server(
                model=model,
                host=host,
                port=port,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                python_executable=python_executable,
                extra_server_args=extra_server_args,
            )
            if manage_server
            else None
        )
        resolved_base_url = (
            self._sglang_server.base_url
            if self._sglang_server is not None
            else (base_url or f"http://{host}:{port}/v1")
        )
        config_hash = _config_hash(
            {
                "kind": "sglang_local",
                "name": name,
                "model": model,
                "host": host,
                "port": port,
                "manage_server": manage_server,
                "startup_timeout_seconds": startup_timeout_seconds,
                "poll_interval_seconds": poll_interval_seconds,
                "python_executable": python_executable,
                "extra_server_args": extra_server_args,
                "base_url": base_url,
                "request_timeout_seconds": request_timeout_seconds,
                "max_retries": max_retries,
            }
        )
        backend = SglangLocalBackend(
            name=name,
            base_url=resolved_base_url,
            default_model=model,
            request_timeout_seconds=request_timeout_seconds,
            managed_server=self._sglang_server,
            config_hash=config_hash,
            max_retries=max_retries,
            model_patterns=_resolve_model_patterns(model_patterns, model),
        )
        super().__init__(
            backend=backend,
            config_snapshot=sglang_config_snapshot(
                model=model,
                host=host,
                port=port,
                manage_server=manage_server,
                startup_timeout_seconds=startup_timeout_seconds,
                poll_interval_seconds=poll_interval_seconds,
                python_executable=python_executable,
                extra_server_args=extra_server_args,
                request_timeout_seconds=request_timeout_seconds,
            ),
            server_snapshot=sglang_server_snapshot(server=self._sglang_server),
        )

    def close(self) -> None:
        """Stop the managed SGLang server process when present."""
        server = getattr(self, "_sglang_server", None)
        if server is not None:
            server.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup.
        """Best-effort managed server cleanup during garbage collection."""
        self.close()


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
        super().__init__(
            backend=backend,
            config_snapshot=transformers_config_snapshot(
                model_id=model_id,
                device=device,
                dtype=dtype,
                quantization=quantization,
                trust_remote_code=trust_remote_code,
                revision=revision,
            ),
        )


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
        super().__init__(
            backend=backend,
            config_snapshot=mlx_config_snapshot(model_id=model_id, quantization=quantization),
        )


def _resolve_model_patterns(
    model_patterns: tuple[str, ...] | None,
    default_model: str,
) -> tuple[str, ...]:
    """Resolve model patterns, defaulting to the configured default model.

    Args:
        model_patterns: Optional explicit model-match patterns.
        default_model: Fallback model used when no patterns are provided.

    Returns:
        Non-empty tuple of model patterns used by selectors/routers.
    """
    if model_patterns is not None:
        return model_patterns
    return (default_model,)


def _config_hash(config_payload: dict[str, object]) -> str:
    """Create a short stable hash for backend configuration payloads.

    Args:
        config_payload: JSON-serializable configuration mapping.

    Returns:
        Deterministic 12-character SHA-256 prefix.
    """
    encoded = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()[:12]


__all__ = [
    "LlamaCppServerLLMClient",
    "MlxLocalLLMClient",
    "OllamaLLMClient",
    "SglangServerLLMClient",
    "TransformersLocalLLMClient",
    "VllmServerLLMClient",
    "_SingleBackendLLMClient",
]
