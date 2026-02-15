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
    streaming: bool | None = None
    tool_calling: ToolCallingMode | bool | None = None
    json_mode: JSONMode | bool | None = None
    vision: bool | None = None
    max_context_tokens: int | None = None

    def apply(self, base: BackendCapabilities) -> BackendCapabilities:
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
    name: str
    kind: BackendKind
    default_model: str | None = None
    model_patterns: tuple[str, ...] = ()
    capabilities: CapabilityOverrides = field(default_factory=CapabilityOverrides)
    max_retries: int = 2


@dataclass(slots=True, frozen=True)
class OpenAIServiceConfig(BackendConfig):
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    base_url: str | None = None


@dataclass(slots=True, frozen=True)
class OpenAICompatibleHTTPConfig(BackendConfig):
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None


@dataclass(slots=True, frozen=True)
class TransformersLocalConfig(BackendConfig):
    model_id: str = ""
    device: str | None = "auto"
    dtype: str | None = "auto"
    quantization: str = "none"
    trust_remote_code: bool = False
    revision: str | None = None


@dataclass(slots=True, frozen=True)
class MlxLocalConfig(BackendConfig):
    model_id: str = ""
    quantization: str = "none"


@dataclass(slots=True, frozen=True)
class LlamaCppConfig(BackendConfig):
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
    model: str = "echo-test"


@dataclass(slots=True, frozen=True)
class LLMConfig:
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


def backend_config_hash(config: BackendConfig) -> str:
    """Return a stable hash for backend configuration."""
    payload = json.dumps(asdict(config), sort_keys=True, default=str).encode("utf-8")
    return sha256(payload).hexdigest()[:12]


def _parse_backend(raw: object, index: int) -> BackendConfig:
    if not isinstance(raw, dict):
        raise ValueError(f"backends[{index}] must be a mapping.")
    name = _require_str(raw, "name", index)
    kind = _require_str(raw, "kind", index)
    default_model = _optional_str(raw.get("default_model"))
    model_patterns = _parse_model_patterns(raw.get("models"))
    max_retries = _optional_int(raw.get("max_retries"), default=2)
    capabilities = _parse_capabilities(raw.get("capabilities"))

    if kind == "openai_service":
        return OpenAIServiceConfig(
            name=name,
            kind="openai_service",
            default_model=default_model,
            model_patterns=model_patterns,
            capabilities=capabilities,
            max_retries=max_retries,
            api_key_env=_optional_str(raw.get("api_key_env")) or "OPENAI_API_KEY",
            api_key=_optional_str(raw.get("api_key")),
            base_url=_optional_str(raw.get("base_url")),
        )
    if kind == "openai_compatible_http":
        base_url = _optional_str(raw.get("base_url")) or ""
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
            api_key_env=_optional_str(raw.get("api_key_env")) or "OPENAI_API_KEY",
            api_key=_optional_str(raw.get("api_key")),
        )
    if kind == "transformers_local":
        model_id = _optional_str(raw.get("model_id")) or ""
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
            device=_optional_str(raw.get("device")) or "auto",
            dtype=_optional_str(raw.get("dtype")) or "auto",
            quantization=_optional_str(raw.get("quantization")) or "none",
            trust_remote_code=bool(raw.get("trust_remote_code", False)),
            revision=_optional_str(raw.get("revision")),
        )
    if kind == "mlx_local":
        model_id = _optional_str(raw.get("model_id")) or ""
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
            quantization=_optional_str(raw.get("quantization")) or "none",
        )
    if kind == "llama_cpp":
        model_path = _optional_str(raw.get("model_path")) or ""
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
            hf_model_repo_id=_optional_str(raw.get("hf_model_repo_id")),
            api_model=_optional_str(raw.get("api_model")) or "local-model",
            host=_optional_str(raw.get("host")) or "127.0.0.1",
            port=_optional_int(raw.get("port"), default=8001),
            startup_timeout_seconds=_optional_float(raw.get("startup_timeout_seconds"), 60.0),
            poll_interval_seconds=_optional_float(raw.get("poll_interval_seconds"), 0.25),
            extra_server_args=_optional_str_list(raw.get("extra_server_args")),
        )
    if kind == "echo_test":
        return EchoTestConfig(
            name=name,
            kind="echo_test",
            default_model=default_model or "echo-test",
            model_patterns=model_patterns or ("echo-test",),
            capabilities=capabilities,
            max_retries=max_retries,
            model=_optional_str(raw.get("model")) or "echo-test",
        )
    raise ValueError(f"backends[{index}].kind '{kind}' is not supported.")


def _ensure_unique_backend_names(backends: tuple[BackendConfig, ...]) -> None:
    seen: set[str] = set()
    for backend in backends:
        if backend.name in seen:
            raise ValueError(f"Duplicate backend name '{backend.name}'.")
        seen.add(backend.name)


def _require_str(raw: dict[str, object], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"backends[{index}].{key} must be a non-empty string.")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _optional_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected integer value.")
    return value


def _optional_float(value: object, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("Expected float value.")
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError("Expected float value.")


def _optional_str_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Expected list of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Expected list of strings.")
        item_norm = item.strip()
        if item_norm:
            normalized.append(item_norm)
    return tuple(normalized)


def _parse_model_patterns(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if not isinstance(value, list):
        raise ValueError("models must be a string or list of strings.")
    patterns: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            patterns.append(item.strip())
    return tuple(patterns)


def _parse_capabilities(value: object) -> CapabilityOverrides:
    if value is None:
        return CapabilityOverrides()
    if not isinstance(value, dict):
        raise ValueError("capabilities must be a mapping.")
    return CapabilityOverrides(
        streaming=value.get("streaming"),
        tool_calling=value.get("tool_calling"),
        json_mode=value.get("json_mode"),
        vision=value.get("vision"),
        max_context_tokens=value.get("max_context_tokens"),
    )


def _normalize_tool_calling(value: ToolCallingMode | bool | None) -> ToolCallingMode:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "native" if value else "none"
    return value


def _normalize_json_mode(value: JSONMode | bool | None) -> JSONMode:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "native" if value else "none"
    return value
