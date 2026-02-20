"""Unified multi-step agent facade with explicit mode selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.memory import MemoryStore
from design_research_agents.contracts.tools import ToolRuntime, ToolSpec
from design_research_agents.implementations.shared.agent_internal.multi_step_modes.code import (
    MultiStepCodeToolCallingAgent as _CodeModeStrategy,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_modes.direct import (
    MultiStepDirectLLMAgent as _DirectModeStrategy,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_modes.direct import (
    _coerce_state_records,
    _parse_controller_decision,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_modes.json import (
    MultiStepJsonToolCallingAgent as _JsonModeStrategy,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_modes.router import (
    MultiStepToolRouterAgent as _RouterModeStrategy,
)
from design_research_agents.implementations.shared.agent_internal.prompt_alternatives import (
    AlternativesPromptTarget,
)
from design_research_agents.tracing import Tracer

MultiStepMode = Literal["direct", "json", "code"]


class MultiStepAgent(Agent):
    """Single multi-step runtime entrypoint for direct/json/code strategies."""

    def __init__(
        self,
        *,
        mode: MultiStepMode,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime | None = None,
        max_steps: int = 5,
        stop_on_step_failure: bool = True,
        controller_system_prompt: str | None = None,
        controller_user_prompt_template: str | None = None,
        continuation_system_prompt: str | None = None,
        continuation_user_prompt_template: str | None = None,
        step_user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        continuation_memory_tail_items: int = 6,
        step_memory_tail_items: int = 8,
        memory_store: MemoryStore | None = None,
        memory_namespace: str = "default",
        memory_read_top_k: int = 4,
        memory_write_observations: bool = True,
        max_tool_calls_per_step: int = 5,
        execution_timeout_seconds: int = 5,
        validate_tool_input_schema: bool = False,
        normalize_generated_code_per_step: bool = False,
        default_tools_per_step: Sequence[Mapping[str, object]] | None = None,
        router_system_prompt: str | None = None,
        router_user_prompt_template: str | None = None,
        allowed_routes: Sequence[str] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize one mode-specific multi-step strategy.

        Args:
            mode: Required strategy mode (``direct``, ``json``, or ``code``).
            llm_client: LLM client shared by all strategy modes.
            tool_runtime: Tool runtime required for ``json`` and ``code`` modes.
            max_steps: Maximum number of multi-step iterations.
            stop_on_step_failure: Whether to stop loop execution on failed steps.
            controller_system_prompt: Direct-mode controller system prompt override.
            controller_user_prompt_template: Direct-mode controller user prompt override.
            continuation_system_prompt: Continuation system prompt override.
            continuation_user_prompt_template: Continuation user prompt override.
            step_user_prompt_template: Step action user prompt override.
            alternatives_prompt_target: Prompt insertion target for alternatives blocks.
            continuation_memory_tail_items: Continuation memory tail item count.
            step_memory_tail_items: Step memory tail item count.
            memory_store: Optional persistent memory dependency.
            memory_namespace: Memory namespace for read/write operations.
            memory_read_top_k: Memory retrieval top-k.
            memory_write_observations: Whether to persist per-step observations.
            max_tool_calls_per_step: Code-mode per-step tool call cap.
            execution_timeout_seconds: Code-mode sandbox timeout.
            validate_tool_input_schema: Code-mode tool input schema validation toggle.
            normalize_generated_code_per_step: Code-mode code normalization toggle.
            default_tools_per_step: Code-mode default tool allowlist.
            router_system_prompt: Router-special-case system prompt override.
            router_user_prompt_template: Router-special-case user prompt override.
            allowed_routes: Optional router-special-case route allowlist.
            tracer: Optional tracer dependency.

        Raises:
            ValueError: Raised when mode/tool configuration is invalid.
        """
        normalized_mode = mode.strip().lower() if isinstance(mode, str) else ""
        if normalized_mode not in {"direct", "json", "code"}:
            raise ValueError("mode must be one of: 'direct', 'json', 'code'.")

        if normalized_mode in {"json", "code"}:
            if tool_runtime is None:
                raise ValueError("tool_runtime is required when mode is 'json' or 'code'.")
            runtime_tools = tuple(tool_runtime.list_tools())
            if not runtime_tools:
                raise ValueError(
                    "tool_runtime must expose at least one tool when mode is 'json' or 'code'."
                )
        else:
            runtime_tools = ()

        self._mode: MultiStepMode = normalized_mode  # type: ignore[assignment]
        self._strategy: Agent
        if self._mode == "direct":
            self._strategy = _DirectModeStrategy(
                llm_client=llm_client,
                max_steps=max_steps,
                controller_system_prompt=controller_system_prompt,
                controller_user_prompt_template=controller_user_prompt_template,
                step_memory_tail_items=step_memory_tail_items,
                tracer=tracer,
            )
            return

        assert tool_runtime is not None
        if self._mode == "code":
            self._strategy = _CodeModeStrategy(
                llm_client=llm_client,
                tool_runtime=tool_runtime,
                max_steps=max_steps,
                max_tool_calls_per_step=max_tool_calls_per_step,
                execution_timeout_seconds=execution_timeout_seconds,
                validate_tool_input_schema=validate_tool_input_schema,
                normalize_generated_code_per_step=normalize_generated_code_per_step,
                stop_on_step_failure=stop_on_step_failure,
                default_tools_per_step=default_tools_per_step,
                continuation_system_prompt=continuation_system_prompt,
                continuation_user_prompt_template=continuation_user_prompt_template,
                step_user_prompt_template=step_user_prompt_template,
                alternatives_prompt_target=alternatives_prompt_target,
                continuation_memory_tail_items=continuation_memory_tail_items,
                step_memory_tail_items=step_memory_tail_items,
                memory_store=memory_store,
                memory_namespace=memory_namespace,
                memory_read_top_k=memory_read_top_k,
                memory_write_observations=memory_write_observations,
                tracer=tracer,
            )
            return

        if _all_tools_are_argless(runtime_tools):
            self._strategy = _RouterModeStrategy(
                llm_client=llm_client,
                tool_runtime=tool_runtime,
                max_steps=max_steps,
                stop_on_step_failure=stop_on_step_failure,
                system_prompt=router_system_prompt,
                user_prompt_template=router_user_prompt_template,
                alternatives_prompt_target=alternatives_prompt_target,
                allowed_routes=allowed_routes,
                step_memory_tail_items=step_memory_tail_items,
                tracer=tracer,
            )
            return

        self._strategy = _JsonModeStrategy(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=max_steps,
            stop_on_step_failure=stop_on_step_failure,
            continuation_system_prompt=continuation_system_prompt,
            continuation_user_prompt_template=continuation_user_prompt_template,
            step_user_prompt_template=step_user_prompt_template,
            alternatives_prompt_target=alternatives_prompt_target,
            continuation_memory_tail_items=continuation_memory_tail_items,
            step_memory_tail_items=step_memory_tail_items,
            memory_store=memory_store,
            memory_namespace=memory_namespace,
            memory_read_top_k=memory_read_top_k,
            memory_write_observations=memory_write_observations,
            tracer=tracer,
        )

    @property
    def workflow(self) -> object | None:
        """Expose underlying strategy workflow for runtime inspection.

        Returns:
            Underlying mode-strategy workflow object when available.
        """
        return getattr(self._strategy, "workflow", None)

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one run through the selected strategy mode.

        Args:
            prompt: Prompt text for this run.
            request_id: Optional request identifier for tracing/correlation.
            dependencies: Optional dependency mapping passed through to runtime calls.

        Returns:
            Execution result produced by the selected mode strategy.
        """
        return self._strategy.run(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )


def _all_tools_are_argless(tool_specs: Sequence[ToolSpec]) -> bool:
    """Return whether all runtime tools accept no structured arguments.

    Args:
        tool_specs: Runtime tool specifications.

    Returns:
        ``True`` when every tool schema is argument-less.
    """
    if not tool_specs:
        return False
    return all(not _tool_takes_structured_args(spec.input_schema) for spec in tool_specs)


def _tool_takes_structured_args(schema: object) -> bool:
    """Return whether a JSON schema clearly accepts structured argument fields.

    Args:
        schema: JSON-schema-like object.

    Returns:
        ``True`` when the schema allows or requires structured fields.
    """
    if not isinstance(schema, Mapping):
        return True
    schema_type = schema.get("type")
    if isinstance(schema_type, str) and schema_type != "object":
        return True

    required = schema.get("required")
    if isinstance(required, list) and len(required) > 0:
        return True

    properties = schema.get("properties")
    if isinstance(properties, Mapping) and len(properties) > 0:
        return True

    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        return True
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        return True
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        return True

    additional_properties = schema.get("additionalProperties")
    return additional_properties not in (None, False)


__all__ = [
    "MultiStepAgent",
    "MultiStepMode",
    "_coerce_state_records",
    "_parse_controller_decision",
]
