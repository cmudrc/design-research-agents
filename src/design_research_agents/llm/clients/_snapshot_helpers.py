"""Snapshot helpers for public LLM client introspection APIs."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents._contracts._llm import BackendCapabilities


def normalize_snapshot_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Return shallow-normalized snapshot mapping with string keys."""
    return {str(key): normalize_snapshot_value(value) for key, value in values.items()}


def normalize_snapshot_value(value: object) -> object:
    """Normalize snapshot values to JSON-serializable containers."""
    if isinstance(value, Mapping):
        return {str(key): normalize_snapshot_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [normalize_snapshot_value(item) for item in value]
    if isinstance(value, list):
        return [normalize_snapshot_value(item) for item in value]
    return value


def capabilities_to_dict(capabilities: BackendCapabilities) -> dict[str, object]:
    """Convert backend capabilities to JSON-serializable mapping."""
    return {
        "streaming": capabilities.streaming,
        "tool_calling": capabilities.tool_calling,
        "json_mode": capabilities.json_mode,
        "vision": capabilities.vision,
        "max_context_tokens": capabilities.max_context_tokens,
    }


def llama_config_snapshot(
    *,
    model: str,
    hf_model_repo_id: str | None,
    api_model: str,
    context_window: int,
    startup_timeout_seconds: float,
    poll_interval_seconds: float,
    python_executable: str,
    extra_server_args: tuple[str, ...],
) -> dict[str, object]:
    """Build llama-cpp client config snapshot payload."""
    return {
        "model": model,
        "hf_model_repo_id": hf_model_repo_id,
        "api_model": api_model,
        "context_window": context_window,
        "startup_timeout_seconds": startup_timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "python_executable": python_executable,
        "extra_server_args": list(extra_server_args),
    }


def llama_server_snapshot(*, server: object) -> dict[str, object]:
    """Build llama-cpp managed server snapshot payload."""
    return {
        "host": getattr(server, "host", None),
        "port": getattr(server, "port", None),
        "base_url": getattr(server, "base_url", None),
        "model": getattr(server, "model", None),
        "hf_model_repo_id": getattr(server, "hf_model_repo_id", None),
        "api_model": getattr(server, "api_model", None),
        "startup_timeout_seconds": getattr(server, "startup_timeout_seconds", None),
        "poll_interval_seconds": getattr(server, "poll_interval_seconds", None),
        "python_executable": getattr(server, "python_executable", None),
        "extra_server_args": list(getattr(server, "extra_server_args", ())),
    }


def vllm_config_snapshot(
    *,
    model: str,
    api_model: str,
    host: str,
    port: int,
    manage_server: bool,
    startup_timeout_seconds: float,
    poll_interval_seconds: float,
    python_executable: str,
    extra_server_args: tuple[str, ...],
    request_timeout_seconds: float,
) -> dict[str, object]:
    """Build vLLM client config snapshot payload."""
    return {
        "model": model,
        "api_model": api_model,
        "host": host,
        "port": port,
        "manage_server": manage_server,
        "startup_timeout_seconds": startup_timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "python_executable": python_executable,
        "extra_server_args": list(extra_server_args),
        "request_timeout_seconds": request_timeout_seconds,
    }


def vllm_server_snapshot(*, server: object | None) -> dict[str, object] | None:
    """Build vLLM managed server snapshot payload."""
    if server is None:
        return None
    return {
        "host": getattr(server, "host", None),
        "port": getattr(server, "port", None),
        "base_url": getattr(server, "base_url", None),
        "model": getattr(server, "model", None),
        "api_model": getattr(server, "api_model", None),
        "startup_timeout_seconds": getattr(server, "startup_timeout_seconds", None),
        "poll_interval_seconds": getattr(server, "poll_interval_seconds", None),
        "python_executable": getattr(server, "python_executable", None),
        "extra_server_args": list(getattr(server, "extra_server_args", ())),
    }


def ollama_config_snapshot(
    *,
    host: str,
    port: int,
    manage_server: bool,
    ollama_executable: str,
    auto_pull_model: bool,
    startup_timeout_seconds: float,
    poll_interval_seconds: float,
    request_timeout_seconds: float,
) -> dict[str, object]:
    """Build Ollama client config snapshot payload."""
    return {
        "host": host,
        "port": port,
        "manage_server": manage_server,
        "ollama_executable": ollama_executable,
        "auto_pull_model": auto_pull_model,
        "startup_timeout_seconds": startup_timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "request_timeout_seconds": request_timeout_seconds,
    }


def ollama_server_snapshot(*, server: object | None) -> dict[str, object] | None:
    """Build Ollama managed server snapshot payload."""
    if server is None:
        return None
    return {
        "host": getattr(server, "host", None),
        "port": getattr(server, "port", None),
        "base_url": getattr(server, "base_url", None),
        "ollama_executable": getattr(server, "ollama_executable", None),
        "startup_timeout_seconds": getattr(server, "startup_timeout_seconds", None),
        "poll_interval_seconds": getattr(server, "poll_interval_seconds", None),
        "default_model": getattr(server, "default_model", None),
        "auto_pull_model": getattr(server, "auto_pull_model", None),
    }


def sglang_config_snapshot(
    *,
    model: str,
    host: str,
    port: int,
    manage_server: bool,
    startup_timeout_seconds: float,
    poll_interval_seconds: float,
    python_executable: str,
    extra_server_args: tuple[str, ...],
    request_timeout_seconds: float,
) -> dict[str, object]:
    """Build SGLang client config snapshot payload."""
    return {
        "model": model,
        "host": host,
        "port": port,
        "manage_server": manage_server,
        "startup_timeout_seconds": startup_timeout_seconds,
        "poll_interval_seconds": poll_interval_seconds,
        "python_executable": python_executable,
        "extra_server_args": list(extra_server_args),
        "request_timeout_seconds": request_timeout_seconds,
    }


def sglang_server_snapshot(*, server: object | None) -> dict[str, object] | None:
    """Build SGLang managed server snapshot payload."""
    if server is None:
        return None
    return {
        "host": getattr(server, "host", None),
        "port": getattr(server, "port", None),
        "base_url": getattr(server, "base_url", None),
        "model": getattr(server, "model", None),
        "startup_timeout_seconds": getattr(server, "startup_timeout_seconds", None),
        "poll_interval_seconds": getattr(server, "poll_interval_seconds", None),
        "python_executable": getattr(server, "python_executable", None),
        "extra_server_args": list(getattr(server, "extra_server_args", ())),
    }


def openai_service_config_snapshot(*, api_key_env: str, api_key: str | None) -> dict[str, object]:
    """Build OpenAI service client config snapshot payload."""
    return {
        "api_key_env": api_key_env,
        "has_api_key": bool(api_key),
    }


def azure_openai_service_config_snapshot(
    *,
    api_key_env: str,
    api_key: str | None,
    azure_endpoint_env: str,
    azure_endpoint: str | None,
    api_version_env: str,
    api_version: str | None,
) -> dict[str, object]:
    """Build Azure OpenAI service client config snapshot payload."""
    return {
        "api_key_env": api_key_env,
        "has_api_key": bool(api_key),
        "azure_endpoint_env": azure_endpoint_env,
        "has_azure_endpoint": bool(azure_endpoint),
        "api_version_env": api_version_env,
        "api_version": api_version,
    }


def anthropic_service_config_snapshot(
    *,
    api_key_env: str,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, object]:
    """Build Anthropic service client config snapshot payload."""
    return {
        "api_key_env": api_key_env,
        "has_api_key": bool(api_key),
        "base_url": base_url,
    }


def gemini_service_config_snapshot(*, api_key_env: str, api_key: str | None) -> dict[str, object]:
    """Build Gemini service client config snapshot payload."""
    return {
        "api_key_env": api_key_env,
        "has_api_key": bool(api_key),
    }


def groq_service_config_snapshot(
    *,
    api_key_env: str,
    api_key: str | None,
    base_url: str | None,
) -> dict[str, object]:
    """Build Groq service client config snapshot payload."""
    return {
        "api_key_env": api_key_env,
        "has_api_key": bool(api_key),
        "base_url": base_url,
    }


def openai_compatible_http_config_snapshot(
    *,
    api_key_env: str,
    api_key: str | None,
    base_url: str,
) -> dict[str, object]:
    """Build OpenAI-compatible HTTP client config snapshot payload."""
    return {
        "api_key_env": api_key_env,
        "has_api_key": bool(api_key),
        "chat_url": openai_compatible_http_chat_url(base_url),
    }


def openai_compatible_http_chat_url(base_url: str) -> str:
    """Compute OpenAI-compatible ``chat/completions`` endpoint URL."""
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    if base_url.endswith("/v1/"):
        return f"{base_url}chat/completions"
    if base_url.endswith("/"):
        return f"{base_url}v1/chat/completions"
    return f"{base_url}/v1/chat/completions"


def transformers_config_snapshot(
    *,
    model_id: str,
    device: str | None,
    dtype: str | None,
    quantization: str,
    trust_remote_code: bool,
    revision: str | None,
) -> dict[str, object]:
    """Build Transformers local client config snapshot payload."""
    return {
        "model_id": model_id,
        "device": device,
        "dtype": dtype,
        "quantization": quantization,
        "trust_remote_code": trust_remote_code,
        "revision": revision,
    }


def mlx_config_snapshot(*, model_id: str, quantization: str) -> dict[str, object]:
    """Build MLX local client config snapshot payload."""
    return {
        "model_id": model_id,
        "quantization": quantization,
    }
