"""YAML-based LLM backend configuration loader and validator."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Literal

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency during lightweight installs.
    yaml = None

from design_research_agents.contracts.llm import BackendCapabilities, JSONMode, ToolCallingMode

BackendKind = Literal[
    "openai_service",
    "openai_compatible_http",
    "transformers_local",
    "mlx_local",
    "llama_cpp",
    "echo_test",
]


@dataclass(slots=True, frozen=True)
class CapabilityOverrides:
    """Optional per-backend capability overrides from YAML config."""

    streaming: bool | None = None
    tool_calling: ToolCallingMode | bool | None = None
    json_mode: JSONMode | bool | None = None
    vision: bool | None = None
    max_context_tokens: int | None = None

    def apply(self, base: BackendCapabilities) -> BackendCapabilities:
        """Apply override values onto a base capability declaration."""
        return BackendCapabilities(
            streaming=base.streaming if self.streaming is None else bool(self.streaming),
            tool_calling=(
                base.tool_calling
                if self.tool_calling is None
                else _normalize_tool_calling(self.tool_calling)
            ),
            json_mode=(
                base.json_mode if self.json_mode is None else _normalize_json_mode(self.json_mode)
            ),
            vision=base.vision if self.vision is None else bool(self.vision),
            max_context_tokens=(
                base.max_context_tokens
                if self.max_context_tokens is None
                else self.max_context_tokens
            ),
        )


@dataclass(slots=True, frozen=True)
class BackendConfig:
    """Shared backend configuration fields used by all backend kinds."""

    name: str
    kind: BackendKind
    default_model: str | None = None
    model_patterns: tuple[str, ...] = ()
    capabilities: CapabilityOverrides = field(default_factory=CapabilityOverrides)
    max_retries: int = 2


@dataclass(slots=True, frozen=True)
class OpenAIServiceConfig(BackendConfig):
    """Configuration for the official OpenAI SDK backend."""

    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None


@dataclass(slots=True, frozen=True)
class OpenAICompatibleHTTPConfig(BackendConfig):
    """Configuration for generic OpenAI-compatible HTTP endpoints."""

    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None


@dataclass(slots=True, frozen=True)
class TransformersLocalConfig(BackendConfig):
    """Configuration for in-process Transformers local inference backend."""

    model_id: str = ""
    device: str | None = "auto"
    dtype: str | None = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False
    revision: str | None = None


@dataclass(slots=True, frozen=True)
class MlxLocalConfig(BackendConfig):
    """Configuration for Apple MLX local inference backend."""

    model_id: str = ""
    quantization: str = "none"


@dataclass(slots=True, frozen=True)
class LlamaCppConfig(BackendConfig):
    """Configuration for managed llama.cpp server backend."""

    model_path: str = ""
    hf_model_repo_id: str | None = None
    api_model: str = "local-model"
    host: str = "127.0.0.1"
    port: int = 8001
    startup_timeout_seconds: float = 60.0
    poll_interval_seconds: float = 0.25
    extra_server_args: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class EchoTestConfig(BackendConfig):
    """Configuration for deterministic echo backend used in tests."""

    model: str = "echo-test"


@dataclass(slots=True, frozen=True)
class LLMConfig:
    """Top-level container for configured backend declarations."""

    backends: tuple[BackendConfig, ...]


def load_config(path: str) -> LLMConfig:
    """Load and validate YAML configuration from disk."""
    if yaml is None:
        raise RuntimeError("YAML support requires PyYAML. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Config root must be a mapping.")
    raw_backends = payload.get("backends")
    if not isinstance(raw_backends, list) or not raw_backends:
        raise ValueError("Config must include a non-empty 'backends' list.")
    backends = tuple(_parse_backend(item, index) for index, item in enumerate(raw_backends))
    _ensure_unique_backend_names(backends)
    return LLMConfig(backends=backends)


def backend_config_hash(backend_config: BackendConfig) -> str:
    """Return a stable hash for backend configuration."""
    config_payload = json.dumps(
        asdict(backend_config),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return sha256(config_payload).hexdigest()[:12]


def _parse_backend(raw_backend: object, index: int) -> BackendConfig:
    if not isinstance(raw_backend, dict):
        raise ValueError(f"backends[{index}] must be a mapping.")
    name = _require_str(raw_backend, "name", index)
    kind = _require_str(raw_backend, "kind", index)
    default_model = _optional_str(raw_backend.get("default_model"))
    model_patterns = _parse_model_patterns(raw_backend.get("models"))
    max_retries = _optional_int(raw_backend.get("max_retries"), default=2)
    capabilities = _parse_capabilities(raw_backend.get("capabilities"))

    if kind == "openai_service":
        return OpenAIServiceConfig(
            name=name,
            kind="openai_service",
            default_model=default_model,
            model_patterns=model_patterns,
            capabilities=capabilities,
            max_retries=max_retries,
            api_key_env=_optional_str(raw_backend.get("api_key_env")) or "OPENAI_API_KEY",
            api_key=_optional_str(raw_backend.get("api_key")),
            base_url=_optional_str(raw_backend.get("base_url")),
        )
    if kind == "openai_compatible_http":
        base_url = _optional_str(raw_backend.get("base_url")) or ""
        if not base_url:
            raise ValueError(f"backends[{index}].base_url is required for {kind}.")
        return OpenAICompatibleHTTPConfig(
            name=name,
            kind="openai_compatible_http",
            default_model=default_model,
            model_patterns=model_patterns,
            capabilities=capabilities,
            max_retries=max_retries,
            base_url=base_url,
            api_key_env=_optional_str(raw_backend.get("api_key_env")) or "OPENAI_API_KEY",
            api_key=_optional_str(raw_backend.get("api_key")),
        )
    if kind == "transformers_local":
        model_id = _optional_str(raw_backend.get("model_id")) or ""
        if not model_id:
            raise ValueError(f"backends[{index}].model_id is required for {kind}.")
        return TransformersLocalConfig(
            name=name,
            kind="transformers_local",
            default_model=default_model or model_id,
            model_patterns=model_patterns or (model_id,),
            capabilities=capabilities,
            max_retries=max_retries,
            model_id=model_id,
            device=_optional_str(raw_backend.get("device")) or "auto",
            dtype=_optional_str(raw_backend.get("dtype")) or "auto",
            quantization=_optional_str(raw_backend.get("quantization")) or "none",
            trust_remote_code=bool(raw_backend.get("trust_remote_code", False)),
            revision=_optional_str(raw_backend.get("revision")),
        )
    if kind == "mlx_local":
        model_id = _optional_str(raw_backend.get("model_id")) or ""
        if not model_id:
            raise ValueError(f"backends[{index}].model_id is required for {kind}.")
        return MlxLocalConfig(
            name=name,
            kind="mlx_local",
            default_model=default_model or model_id,
            model_patterns=model_patterns or (model_id,),
            capabilities=capabilities,
            max_retries=max_retries,
            model_id=model_id,
            quantization=_optional_str(raw_backend.get("quantization")) or "none",
        )
    if kind == "llama_cpp":
        model_path = _optional_str(raw_backend.get("model_path")) or ""
        if not model_path:
            raise ValueError(f"backends[{index}].model_path is required for {kind}.")
        return LlamaCppConfig(
            name=name,
            kind="llama_cpp",
            default_model=default_model or "local-model",
            model_patterns=model_patterns or ("local-model",),
            capabilities=capabilities,
            max_retries=max_retries,
            model_path=model_path,
            hf_model_repo_id=_optional_str(raw_backend.get("hf_model_repo_id")),
            api_model=_optional_str(raw_backend.get("api_model")) or "local-model",
            host=_optional_str(raw_backend.get("host")) or "127.0.0.1",
            port=_optional_int(raw_backend.get("port"), default=8001),
            startup_timeout_seconds=_optional_float(
                raw_backend.get("startup_timeout_seconds"),
                60.0,
            ),
            poll_interval_seconds=_optional_float(
                raw_backend.get("poll_interval_seconds"),
                0.25,
            ),
            extra_server_args=_optional_str_list(raw_backend.get("extra_server_args")),
        )
    if kind == "echo_test":
        return EchoTestConfig(
            name=name,
            kind="echo_test",
            default_model=default_model or "echo-test",
            model_patterns=model_patterns or ("echo-test",),
            capabilities=capabilities,
            max_retries=max_retries,
            model=_optional_str(raw_backend.get("model")) or "echo-test",
        )
    raise ValueError(f"backends[{index}].kind '{kind}' is not supported.")


def _ensure_unique_backend_names(backends: tuple[BackendConfig, ...]) -> None:
    seen: set[str] = set()
    for backend in backends:
        if backend.name in seen:
            raise ValueError(f"Duplicate backend name '{backend.name}'.")
        seen.add(backend.name)


def _require_str(raw_mapping: dict[str, object], key: str, index: int) -> str:
    raw_field_value = raw_mapping.get(key)
    if not isinstance(raw_field_value, str) or not raw_field_value.strip():
        raise ValueError(f"backends[{index}].{key} must be a non-empty string.")
    return raw_field_value.strip()


def _optional_str(raw_field_value: object) -> str | None:
    if not isinstance(raw_field_value, str):
        return None
    normalized = raw_field_value.strip()
    return normalized or None


def _optional_int(raw_field_value: object, *, default: int) -> int:
    if raw_field_value is None:
        return default
    if isinstance(raw_field_value, bool) or not isinstance(raw_field_value, int):
        raise ValueError("Expected integer value.")
    return raw_field_value


def _optional_float(raw_field_value: object, default: float) -> float:
    if raw_field_value is None:
        return default
    if isinstance(raw_field_value, bool):
        raise ValueError("Expected float value.")
    if isinstance(raw_field_value, (int, float)):
        return float(raw_field_value)
    raise ValueError("Expected float value.")


def _optional_str_list(raw_field_value: object) -> tuple[str, ...]:
    if raw_field_value is None:
        return ()
    if not isinstance(raw_field_value, list):
        raise ValueError("Expected list of strings.")
    normalized: list[str] = []
    for item in raw_field_value:
        if not isinstance(item, str):
            raise ValueError("Expected list of strings.")
        item_norm = item.strip()
        if item_norm:
            normalized.append(item_norm)
    return tuple(normalized)


def _parse_model_patterns(raw_model_patterns: object) -> tuple[str, ...]:
    if raw_model_patterns is None:
        return ()
    if isinstance(raw_model_patterns, str):
        return (raw_model_patterns.strip(),) if raw_model_patterns.strip() else ()
    if not isinstance(raw_model_patterns, list):
        raise ValueError("models must be a string or list of strings.")
    patterns: list[str] = []
    for item in raw_model_patterns:
        if isinstance(item, str) and item.strip():
            patterns.append(item.strip())
    return tuple(patterns)


def _parse_capabilities(raw_capabilities: object) -> CapabilityOverrides:
    if raw_capabilities is None:
        return CapabilityOverrides()
    if not isinstance(raw_capabilities, dict):
        raise ValueError("capabilities must be a mapping.")
    return CapabilityOverrides(
        streaming=raw_capabilities.get("streaming"),
        tool_calling=raw_capabilities.get("tool_calling"),
        json_mode=raw_capabilities.get("json_mode"),
        vision=raw_capabilities.get("vision"),
        max_context_tokens=raw_capabilities.get("max_context_tokens"),
    )


def _normalize_tool_calling(
    tool_calling_setting: ToolCallingMode | bool | None,
) -> ToolCallingMode:
    if tool_calling_setting is None:
        return "none"
    if isinstance(tool_calling_setting, bool):
        return "native" if tool_calling_setting else "none"
    return tool_calling_setting


def _normalize_json_mode(json_mode_setting: JSONMode | bool | None) -> JSONMode:
    if json_mode_setting is None:
        return "none"
    if isinstance(json_mode_setting, bool):
        return "native" if json_mode_setting else "none"
    return json_mode_setting
