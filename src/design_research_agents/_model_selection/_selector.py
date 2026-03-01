"""Public model selection facade with flattened constructor-first ergonomics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Literal, cast

from design_research_agents._contracts._llm import LLMClient

from ._catalog import ModelCatalog
from ._hardware import HardwareProfile
from ._policy import ModelSelectionPolicy
from ._types import (
    ModelSelectionConstraints,
    ModelSelectionDecision,
    ModelSelectionIntent,
    ModelSelectionPolicyConfig,
)

type Priority = Literal["quality", "balanced", "speed"]
type SelectionOutput = Literal["client", "decision", "client_config"]
type LocalClientResolver = Callable[[ModelSelectionDecision], dict[str, object]]

_CLIENT_CLASS_REFS: dict[str, str] = {
    "AzureOpenAIServiceLLMClient": "design_research_agents.llm:AzureOpenAIServiceLLMClient",
    "LlamaCppServerLLMClient": "design_research_agents.llm:LlamaCppServerLLMClient",
    "OpenAIServiceLLMClient": "design_research_agents.llm:OpenAIServiceLLMClient",
    "OpenAICompatibleHTTPLLMClient": "design_research_agents.llm:OpenAICompatibleHTTPLLMClient",
    "TransformersLocalLLMClient": "design_research_agents.llm:TransformersLocalLLMClient",
    "MLXLocalLLMClient": "design_research_agents.llm:MLXLocalLLMClient",
    "VLLMServerLLMClient": "design_research_agents.llm:VLLMServerLLMClient",
    "OllamaLLMClient": "design_research_agents.llm:OllamaLLMClient",
    "SGLangServerLLMClient": "design_research_agents.llm:SGLangServerLLMClient",
}


class ModelSelector:
    """Flat model selection interface with client/config resolution helpers."""

    def __init__(
        self,
        *,
        catalog: ModelCatalog | None = None,
        prefer_local: bool = True,
        ram_reserve_gb: float = 2.0,
        vram_reserve_gb: float = 0.5,
        max_load_ratio: float = 0.85,
        remote_cost_floor_usd: float = 0.02,
        default_max_latency_ms: int | None = None,
        local_client_resolver: LocalClientResolver | None = None,
    ) -> None:
        """Initialize model selector policy controls and optional resolver hook.

        Args:
            catalog: Optional model catalog to use for selection.
            prefer_local: Whether to prefer local models over remote ones when all else is equal.
            ram_reserve_gb: Amount of RAM (in GB) to reserve when evaluating local candidates.
            vram_reserve_gb: Amount of GPU VRAM (in GB) to reserve when evaluating local
                candidates.
            max_load_ratio: Maximum system load ratio to consider a local candidate viable
                (0.0 to 1.0).
            remote_cost_floor_usd: Minimum cost threshold (in USD) for remote models to be
                considered viable.
            default_max_latency_ms: Default maximum latency (in milliseconds) to consider when
                evaluating candidates, if not specified in selection constraints.
            local_client_resolver: Optional callable that takes a ModelSelectionDecision and
                returns a dict with 'client_class' and 'kwargs' for constructing a local client
                when the provider is not recognized by the built-in resolver. This allows for
                custom local providers to be integrated without modifying the ModelSelector code.
        """
        self._policy = ModelSelectionPolicy(
            catalog=catalog or ModelCatalog.default(),
            config=ModelSelectionPolicyConfig(
                policy_id="default",
                prefer_local=prefer_local,
                ram_reserve_gb=ram_reserve_gb,
                vram_reserve_gb=vram_reserve_gb,
                max_load_ratio=max_load_ratio,
                remote_cost_floor_usd=remote_cost_floor_usd,
                default_max_latency_ms=default_max_latency_ms,
            ),
        )
        self._local_client_resolver = local_client_resolver

    def select(
        self,
        *,
        task: str,
        priority: Priority = "balanced",
        require_local: bool = False,
        preferred_provider: str | None = None,
        max_cost_usd: float | None = None,
        max_latency_ms: int | None = None,
        hardware_profile: Mapping[str, object] | HardwareProfile | None = None,
        output: SelectionOutput = "client",
    ) -> LLMClient | ModelSelectionDecision | dict[str, object]:
        """Select a model and return a decision, config mapping, or live client.

        Args:
            task: Description of the task or use case for which a model is being selected.
            priority: Selection priority, which may influence the trade-off between quality,
                latency, and cost in the decision process.
            require_local: If True, only consider local models as viable candidates.
            preferred_provider: Optional provider name to prioritize in the selection process.
            max_cost_usd: Optional maximum cost threshold (in USD) for candidate models.
            max_latency_ms: Optional maximum latency threshold (in milliseconds) for
                candidate models.
            hardware_profile: Optional mapping or HardwareProfile instance describing the
                current hardware state, which may be used to evaluate local candidates.
            output: Determines the format of the selection result. "client" returns an
                instantiated LLMClient ready for use, "decision" returns the
                raw ModelSelectionDecision object with details of the selection
                rationale, and "client_config" returns a dict containing the
                information needed to construct an LLMClient (including 'client_class'
                and 'kwargs') without actually instantiating it.

        Returns:
            Depending on the 'output' parameter:
            - If output is "client": An instantiated LLMClient configured according to the
              selection decision, ready for use in making requests.
            - If output is "decision": A ModelSelectionDecision object containing details
              about the selected model, provider, rationale, and policy information.
            - If output is "client_config": A dict containing the resolved client configuration,
              including 'client_class', 'kwargs', and metadata from the selection decision,
              which can be used to instantiate an LLMClient at a later time or in a different
              context.

        Raises:
            ValueError: If ``output`` is unsupported or selection/config coercion fails.
        """
        if output not in {"client", "decision", "client_config"}:
            raise ValueError("output must be one of: 'client', 'decision', 'client_config'.")

        decision = self._policy.select_model(
            intent=ModelSelectionIntent(task=task, priority=priority),
            constraints=ModelSelectionConstraints(
                require_local=require_local,
                preferred_provider=preferred_provider,
                max_cost_usd=max_cost_usd,
                max_latency_ms=max_latency_ms,
            ),
            hardware_profile=_coerce_hardware_profile(hardware_profile),
        )
        if output == "decision":
            return decision

        client_config = self._resolve_client_config(decision)
        if output == "client_config":
            return client_config
        # Build concrete client only for "client" output; other modes stay side-effect free.
        return _build_client_from_config(client_config)

    def _resolve_client_config(self, decision: ModelSelectionDecision) -> dict[str, object]:
        """Resolve one selection decision into a concrete client configuration payload.

        Args:
            decision: Selection decision produced by the policy layer.

        Returns:
            Configuration payload containing ``client_class`` and ``kwargs`` plus
            metadata used for downstream tracing and reporting.

        Raises:
            ValueError: If resolved ``client_class`` or ``kwargs`` are invalid.
        """
        provider = decision.provider.strip()
        default_config: dict[str, object] | None = None
        # Map provider identifiers to concrete client constructors with minimal required kwargs.
        if provider == "openai":
            default_config = {
                "client_class": "OpenAIServiceLLMClient",
                "kwargs": {"default_model": decision.model_id},
            }
        elif provider in {"azure", "azure_openai", "azure-openai"}:
            default_config = {
                "client_class": "AzureOpenAIServiceLLMClient",
                "kwargs": {"default_model": decision.model_id},
            }
        elif provider in {
            "openai_compatible_http",
            "openai-compatible-http",
            "openai-compatible",
        }:
            default_config = {
                "client_class": "OpenAICompatibleHTTPLLMClient",
                "kwargs": {"default_model": decision.model_id},
            }
        elif provider == "transformers_local":
            default_config = {
                "client_class": "TransformersLocalLLMClient",
                "kwargs": {
                    "model_id": decision.model_id,
                    "default_model": decision.model_id,
                },
            }
        elif provider == "mlx_local":
            default_config = {
                "client_class": "MLXLocalLLMClient",
                "kwargs": {
                    "model_id": decision.model_id,
                    "default_model": decision.model_id,
                },
            }
        elif provider == "vllm_local":
            default_config = {
                "client_class": "VLLMServerLLMClient",
                "kwargs": {
                    "api_model": decision.model_id,
                    "manage_server": False,
                    "base_url": "http://127.0.0.1:8002/v1",
                },
            }
        elif provider == "ollama_local":
            default_config = {
                "client_class": "OllamaLLMClient",
                "kwargs": {
                    "default_model": decision.model_id,
                    "manage_server": False,
                },
            }
        elif provider == "sglang_local":
            default_config = {
                "client_class": "SGLangServerLLMClient",
                "kwargs": {
                    "model": decision.model_id,
                    "manage_server": False,
                    "base_url": "http://127.0.0.1:30000/v1",
                },
            }

        resolved_config: dict[str, object]
        # Unknown providers are delegated to caller-supplied resolver for extensibility.
        resolved_config = default_config if default_config is not None else self._resolve_local_client_config(decision)

        client_class = resolved_config.get("client_class")
        kwargs = resolved_config.get("kwargs")
        if not isinstance(client_class, str) or client_class not in _CLIENT_CLASS_REFS:
            supported = ", ".join(sorted(_CLIENT_CLASS_REFS))
            raise ValueError(f"ModelSelector resolver returned unsupported client_class. Expected one of: {supported}.")
        if not isinstance(kwargs, dict):
            raise ValueError("ModelSelector resolver returned invalid kwargs (must be a dict).")

        full_config = dict(resolved_config)
        full_config.update(
            {
                "provider": decision.provider,
                "model_id": decision.model_id,
                "client_class": client_class,
                "kwargs": dict(kwargs),
                "rationale": decision.rationale,
                "policy_id": decision.policy_id,
                "catalog_signature": decision.catalog_signature,
            }
        )
        return full_config

    def _resolve_local_client_config(self, decision: ModelSelectionDecision) -> dict[str, object]:
        """Resolve local-provider client configuration via user-supplied resolver.

        Args:
            decision: Selection decision that could not be mapped by built-in providers.

        Returns:
            Resolver payload containing ``client_class`` and ``kwargs``.

        Raises:
            ValueError: If resolver is missing, returns a non-dict, or misses required keys.
        """
        if self._local_client_resolver is None:
            raise ValueError(
                "ModelSelector cannot map selected provider "
                f"'{decision.provider}' (model '{decision.model_id}') to a client config. "
                "Provide local_client_resolver returning {'client_class': ..., 'kwargs': {...}}."
            )
        resolved = self._local_client_resolver(decision)
        if not isinstance(resolved, dict):
            raise ValueError("local_client_resolver must return a dict payload.")
        if "client_class" not in resolved or "kwargs" not in resolved:
            raise ValueError("local_client_resolver result must include 'client_class' and 'kwargs'.")
        return resolved


def _build_client_from_config(config: dict[str, object]) -> LLMClient:
    """Instantiate an ``LLMClient`` from a resolved configuration payload.

    Args:
        config: Mapping with ``client_class`` and constructor ``kwargs``.

    Returns:
        Instantiated ``LLMClient`` for immediate use.

    Raises:
        ValueError: If client class name or kwargs payload is invalid.
    """
    client_class = config.get("client_class")
    kwargs = config.get("kwargs")
    if not isinstance(client_class, str) or client_class not in _CLIENT_CLASS_REFS:
        raise ValueError("client_config has unsupported client_class.")
    if not isinstance(kwargs, dict):
        raise ValueError("client_config has invalid kwargs (must be dict).")
    client_ctor = _resolve_client_class(client_class)
    client = client_ctor(**kwargs)
    return cast(LLMClient, client)


def _resolve_client_class(client_class_name: str) -> type[object]:
    """Resolve one client class name into the concrete constructor."""
    export_ref = _CLIENT_CLASS_REFS[client_class_name]
    module_name, attr_name = export_ref.split(":", maxsplit=1)
    return cast(type[object], getattr(import_module(module_name), attr_name))


def _coerce_hardware_profile(
    value: Mapping[str, object] | HardwareProfile | None,
) -> HardwareProfile | None:
    """Normalize optional hardware-profile input into ``HardwareProfile``.

    Args:
        value: Existing ``HardwareProfile`` instance, mapping, or ``None``.

    Returns:
        Normalized hardware profile instance or ``None``.

    Raises:
        ValueError: If ``value`` has unsupported type or invalid field shapes.
    """
    if value is None:
        return None
    if isinstance(value, HardwareProfile):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("hardware_profile must be a mapping, HardwareProfile, or None.")

    load_average = _coerce_load_average(value.get("load_average"))
    return HardwareProfile(
        total_ram_gb=_coerce_optional_float(value.get("total_ram_gb")),
        available_ram_gb=_coerce_optional_float(value.get("available_ram_gb")),
        cpu_count=_coerce_optional_int(value.get("cpu_count")),
        load_average=load_average,
        gpu_present=_coerce_optional_bool(value.get("gpu_present")),
        gpu_vram_gb=_coerce_optional_float(value.get("gpu_vram_gb")),
        gpu_name=_coerce_optional_str(value.get("gpu_name")),
        platform_name=_coerce_optional_str(value.get("platform_name")),
    )


def _coerce_load_average(raw: object) -> tuple[float, float, float] | None:
    """Normalize optional load-average values to a 3-float tuple.

    Args:
        raw: Optional 3-item sequence representing system load averages.

    Returns:
        ``None`` when unset, otherwise a ``(1m, 5m, 15m)`` float tuple.

    Raises:
        ValueError: If value is not a 3-item sequence.
    """
    if raw is None:
        return None
    if not isinstance(raw, (tuple, list)) or len(raw) != 3:
        raise ValueError("hardware_profile.load_average must be a 3-item sequence when provided.")
    coerced = tuple(float(item) for item in raw)
    return cast(tuple[float, float, float], coerced)


def _coerce_optional_float(raw: object) -> float | None:
    """Coerce an optional value into ``float``.

    Args:
        raw: Input value to normalize.

    Returns:
        ``None`` when unset, otherwise parsed float.

    Raises:
        ValueError: If value cannot be interpreted as float.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("Expected float-compatible value, got bool.")
    if isinstance(raw, (int, float, str, bytes, bytearray)):
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected float-compatible value.") from exc
    raise ValueError("Expected float-compatible value.")


def _coerce_optional_int(raw: object) -> int | None:
    """Coerce an optional value into ``int``.

    Args:
        raw: Input value to normalize.

    Returns:
        ``None`` when unset, otherwise parsed int.

    Raises:
        ValueError: If value cannot be interpreted as int.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("Expected int-compatible value, got bool.")
    if isinstance(raw, (int, float, str, bytes, bytearray)):
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected int-compatible value.") from exc
    raise ValueError("Expected int-compatible value.")


def _coerce_optional_bool(raw: object) -> bool | None:
    """Coerce an optional value into ``bool``.

    Args:
        raw: Input value to normalize.

    Returns:
        ``None`` when unset, otherwise bool value.

    Raises:
        ValueError: If provided value is not a bool.
    """
    if raw is None:
        return None
    if not isinstance(raw, bool):
        raise ValueError("Expected bool value when provided.")
    return raw


def _coerce_optional_str(raw: object) -> str | None:
    """Coerce an optional value into normalized ``str``.

    Args:
        raw: Input value to normalize.

    Returns:
        ``None`` when unset or blank, otherwise stripped string value.

    Raises:
        ValueError: If provided value is not a string.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise ValueError("Expected str value when provided.")
    normalized = raw.strip()
    return normalized or None


__all__ = ["ModelSelector"]
