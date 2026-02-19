"""Workflow-native JSON tool-calling single-step agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import LLMClient, LLMMessage, LLMRequest, LLMResponse
from design_research_agents.contracts.tools import ToolError, ToolResult, ToolRuntime
from design_research_agents.contracts.workflow import (
    LogicStep,
    ToolStep,
    ToolStepInputBuilder,
    WorkflowStepResult,
)
from design_research_agents.implementations.shared.agent_internal.execution_context import (
    finish_agent_execution,
    prepare_agent_execution,
)
from design_research_agents.implementations.shared.agent_internal.json_tool_agent_helpers import (
    build_tool_call_prompt,
    build_tool_choices_text,
    clone_tool_choice,
    extract_tool_choices,
    parse_tool_call,
    parse_tool_call_from_response,
    request_tool_call_response,
    resolve_allowed_tool_names,
    resolve_tool_input,
    select_tool_choice,
    tool_call_response_schema,
)
from design_research_agents.implementations.shared.agent_internal.model_resolution import (
    resolve_agent_model,
)
from design_research_agents.implementations.shared.agent_internal.prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    normalize_alternatives_prompt_target,
)
from design_research_agents.implementations.shared.agent_internal.prompt_overrides import (
    resolve_prompt_text,
)
from design_research_agents.implementations.shared.agent_internal.result_builders import (
    build_failure_result,
)
from design_research_agents.implementations.shared.agent_internal.tool_input import extract_prompt
from design_research_agents.implementations.shared.agent_internal.workflow_first_envelope import (
    build_workflow_first_output,
)
from design_research_agents.tracing import (
    Tracer,
    emit_guardrail_decision,
    emit_tool_selection_decision,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import Workflow


class SingleStepJsonToolCallingAgent(Agent):
    """Agent that asks the model to select one tool and structured arguments."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        allowed_tools: Sequence[str] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a tool-calling agent with injected runtime dependencies.

        Args:
            llm_client: LLM client used for prompt execution.
            tool_runtime: Tool runtime used for tool invocation.
            system_prompt: Optional system prompt override.
            user_prompt_template: Optional user prompt template override.
            alternatives_prompt_target: Prompt target for tools block.
            allowed_tools: Optional tool allowlist.
            tracer: Optional explicit tracer dependency.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self._system_prompt = resolve_prompt_text(
            override=system_prompt,
            default_prompt_name="tool_calling_system",
            field_name="system_prompt",
        )
        self._user_prompt_template = resolve_prompt_text(
            override=user_prompt_template,
            default_prompt_name="tool_calling_user_select_tool",
            field_name="user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        self._allowed_tool_names = resolve_allowed_tool_names(
            runtime_specs=self._runtime_specs,
            allowed_tools=allowed_tools,
        )
        self._compiled_tool_choices = extract_tool_choices(
            tool_specs=self._runtime_specs,
            allowed_tool_names=self._allowed_tool_names,
        )
        self._default_tool_call_response_schema = tool_call_response_schema(
            [choice.tool_name for choice in self._compiled_tool_choices]
        )
        self._tool_step_ids = {
            choice.tool_name: _tool_step_id(choice.tool_name)
            for choice in self._compiled_tool_choices
        }
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run one workflow-native tool selection and invocation cycle.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.

        Raises:
            Exception: Raised when execution fails.
        """
        execution_context = prepare_agent_execution(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            agent_name="SingleStepJsonToolCallingAgent",
            tracer=self._tracer,
        )
        self.workflow = self._build_workflow()

        try:
            workflow_result = self.workflow.run(
                {
                    "normalized_input": execution_context.normalized_input,
                    "request_id": execution_context.request_id,
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
                request_id=f"{execution_context.request_id}:single_step_json",
                dependencies=execution_context.dependencies,
            )
            result = self._build_result(
                workflow_result=workflow_result,
                request_id=execution_context.request_id,
                dependencies=execution_context.dependencies,
            )
        except Exception as exc:
            finish_agent_execution(trace_scope=execution_context.trace_scope, error=str(exc))
            raise

        finish_agent_execution(trace_scope=execution_context.trace_scope, result=result)
        return result

    def _build_workflow(self) -> Workflow:
        """Compose a route-selection + per-tool invocation workflow graph.

        Returns:
            Workflow configured for selection, routed tool execution,
            and invalid selection handling.
        """
        route_map = {
            choice.tool_name: (self._tool_step_ids[choice.tool_name],)
            for choice in self._compiled_tool_choices
        }
        route_map["__invalid__"] = ("invalid_selection",)

        steps: list[LogicStep | ToolStep] = [
            LogicStep(
                step_id="select_tool",
                handler=self._select_tool_handler,
                route_map=route_map,
            ),
        ]

        for choice in self._compiled_tool_choices:
            step_id = self._tool_step_ids[choice.tool_name]
            steps.append(
                ToolStep(
                    step_id=step_id,
                    tool_name=choice.tool_name,
                    dependencies=("select_tool",),
                    input_builder=_build_tool_input_builder(expected_tool_name=choice.tool_name),
                )
            )

        steps.append(
            LogicStep(
                step_id="invalid_selection",
                dependencies=("select_tool",),
                handler=_invalid_selection_handler,
            )
        )

        return Workflow(
            tool_runtime=self._tool_runtime,
            tracer=self._tracer,
            input_mode="schema",
            steps=steps,
            default_execution_mode="sequential",
            default_failure_policy="skip_dependents",
        )

    def _select_tool_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Run model selection logic and emit a route for tool branch activation.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping containing route, selected tool metadata, and model response payload.

        Raises:
            TypeError: If schema-mode input payloads are missing or malformed.
            Exception: Propagated when the model request fails.
        """
        inputs = context.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("Single-step JSON workflow requires schema input mapping.")
        normalized_input = inputs.get("normalized_input")
        request_id = inputs.get("request_id")
        if not isinstance(normalized_input, Mapping):
            raise TypeError("normalized_input must be a mapping.")

        resolved_request_id = str(request_id) if request_id is not None else ""
        prompt = extract_prompt(normalized_input)
        resolved_model = resolve_agent_model(llm_client=self._llm_client)
        choices = [clone_tool_choice(choice) for choice in self._compiled_tool_choices]
        alternatives_prompt_target = self._alternatives_prompt_target
        choices_text = build_tool_choices_text(choices=choices)
        choices_block = build_user_prompt_alternatives_block(
            section_label="Available tools",
            alternatives_text=choices_text,
            target=alternatives_prompt_target,
        )
        user_prompt = build_tool_call_prompt(
            prompt=prompt,
            choices_block=choices_block,
            prompt_template=self._user_prompt_template,
        )
        system_prompt = self._system_prompt
        if alternatives_prompt_target == "system":
            system_prompt = append_alternatives_block(
                prompt_text=system_prompt,
                section_label="Available tools",
                alternatives_text=choices_text,
            )

        model_messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        llm_request = LLMRequest(
            messages=model_messages,
            model=resolved_model,
            tools=list(self._runtime_specs.values()),
            response_schema=dict(self._default_tool_call_response_schema),
            metadata={
                "request_id": resolved_request_id,
                "agent": "SingleStepJsonToolCallingAgent",
            },
            provider_options={"agent": "SingleStepJsonToolCallingAgent"},
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=model_messages,
            params=llm_request,
            metadata={"agent": "SingleStepJsonToolCallingAgent"},
        )
        try:
            llm_response = request_tool_call_response(
                llm_client=self._llm_client,
                llm_request=llm_request,
            )
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            raise
        finish_model_call(model_span_id, response=llm_response)

        parsed_tool_call = parse_tool_call_from_response(llm_response)
        if parsed_tool_call is None:
            parsed_tool_call = parse_tool_call(llm_response.text)
        tool_selection = select_tool_choice(
            parsed_tool_call=parsed_tool_call,
            choices=choices,
        )

        available_tools = [choice.tool_name for choice in choices]
        if tool_selection is None:
            emit_guardrail_decision(
                guardrail="tool_selection_output",
                decision="reject",
                reason="invalid model tool selection",
                details={"stage": "tool_selection"},
            )
            emit_tool_selection_decision(
                source="model_invalid",
                tool_name="",
                reason="invalid model tool selection",
                parsed_tool_call=parsed_tool_call,
            )
            return {
                "route": "__invalid__",
                "selection_valid": False,
                "error": (
                    "Model tool selection was invalid. Expected one allowed "
                    "`tool_name` in structured output."
                ),
                "model_text": llm_response.text,
                "tool_name": None,
                "tool_input": {},
                "available_tools": available_tools,
                "parsed_tool_call": parsed_tool_call,
                "metadata_tool_call": {
                    "source": "model_invalid",
                    "reason": "invalid model tool selection",
                    "available_tools": available_tools,
                    "parsed_tool_call": parsed_tool_call,
                },
                "model_response": llm_response,
            }

        selected_choice, tool_call_source, tool_call_reason = tool_selection
        emit_tool_selection_decision(
            source=tool_call_source,
            tool_name=selected_choice.tool_name,
            reason=tool_call_reason,
            parsed_tool_call=parsed_tool_call,
        )
        tool_input = resolve_tool_input(
            selected_choice=selected_choice,
            parsed_tool_call=parsed_tool_call,
            input_payload=normalized_input,
        )
        return {
            "route": selected_choice.tool_name,
            "selection_valid": True,
            "error": None,
            "model_text": llm_response.text,
            "tool_name": selected_choice.tool_name,
            "tool_input": tool_input,
            "available_tools": available_tools,
            "parsed_tool_call": parsed_tool_call,
            "metadata_tool_call": {
                "source": tool_call_source,
                "reason": tool_call_reason,
                "available_tools": available_tools,
                "parsed_tool_call": parsed_tool_call,
            },
            "model_response": llm_response,
        }

    def _build_result(
        self,
        *,
        workflow_result: ExecutionResult,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Build public ``ExecutionResult`` from workflow outcomes.

        Args:
            workflow_result: Aggregated workflow runtime result.
            request_id: Stable request identifier for the run.
            dependencies: Dependency payload mapping passed at run-time.

        Returns:
            Final agent execution result with normalized workflow-first envelope fields.

        Raises:
            RuntimeError: If required workflow step outputs are missing.
        """
        select_step = workflow_result.step_results.get("select_tool")
        if select_step is None:
            raise RuntimeError("JSON single-step workflow missing select_tool result.")
        select_output = select_step.output

        raw_model_response = select_output.get("model_response")
        model_response = raw_model_response if isinstance(raw_model_response, LLMResponse) else None
        tool_call_metadata = select_output.get("metadata_tool_call")
        tool_call_info = dict(tool_call_metadata) if isinstance(tool_call_metadata, Mapping) else {}

        selected_tool_name = select_output.get("tool_name")
        selected_tool_name_text = selected_tool_name if isinstance(selected_tool_name, str) else ""
        selected_tool_step_id = self._tool_step_ids.get(selected_tool_name_text, "")
        selected_tool_step = workflow_result.step_results.get(selected_tool_step_id)

        if not workflow_result.success:
            typed_selected_step = (
                selected_tool_step if isinstance(selected_tool_step, WorkflowStepResult) else None
            )
            error_text = _resolve_failure_error(
                select_output=select_output,
                selected_step=typed_selected_step,
            )
            failed_output = {
                "model_text": str(select_output.get("model_text", "")),
                "tool_name": selected_tool_name if selected_tool_name_text else None,
                "tool_input": _mapping_or_empty(select_output.get("tool_input")),
                "tool_output": _resolve_failed_tool_output(typed_selected_step),
            }
            base_result = build_failure_result(
                error=error_text,
                model_response=model_response,
                tool_results=_resolve_tool_results_for_failure(typed_selected_step),
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "tool_selection", "tool_call": tool_call_info},
                output=failed_output,
            )
            output = build_workflow_first_output(
                base_output=base_result.output,
                workflow_result=workflow_result,
                final_output=failed_output["tool_output"],
            )
            return ExecutionResult(
                output=output,
                success=False,
                tool_results=list(base_result.tool_results),
                model_response=model_response,
                metadata=dict(base_result.metadata),
            )

        if selected_tool_step is None:
            raise RuntimeError("JSON single-step workflow missing selected tool invocation step.")
        tool_result = _tool_result_from_step_output(
            step_output=selected_tool_step.output,
            fallback_tool_name=selected_tool_name_text,
        )
        tool_output = tool_result.result
        base_output: dict[str, object] = {
            "model_text": str(select_output.get("model_text", "")),
            "tool_name": selected_tool_name_text,
            "tool_input": _mapping_or_empty(select_output.get("tool_input")),
            "tool_output": tool_output,
        }
        output = build_workflow_first_output(
            base_output=base_output,
            workflow_result=workflow_result,
            final_output=tool_output,
        )
        return ExecutionResult(
            output=output,
            success=workflow_result.success and tool_result.ok,
            tool_results=[tool_result],
            model_response=model_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "tool_call": tool_call_info,
            },
        )


def _invalid_selection_handler(context: Mapping[str, object]) -> Mapping[str, object]:
    """Raise a deterministic error when model tool selection is invalid.

    Args:
        context: Workflow step execution context payload.

    Returns:
        Mapping payload for the logic-step contract; this function always raises.

    Raises:
        ValueError: If selection output is missing or invalid.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        raise ValueError("Missing dependency_results for invalid_selection step.")
    select_result = dependency_results.get("select_tool")
    if not isinstance(select_result, Mapping):
        raise ValueError("Missing select_tool result for invalid_selection step.")
    select_output = select_result.get("output")
    if not isinstance(select_output, Mapping):
        raise ValueError("Invalid select_tool output for invalid_selection step.")
    error_text = select_output.get("error")
    resolved_error = (
        str(error_text)
        if isinstance(error_text, str) and error_text.strip()
        else "invalid model tool selection"
    )
    raise ValueError(resolved_error)


def _build_tool_input_builder(*, expected_tool_name: str) -> ToolStepInputBuilder:
    """Build ``ToolStep.input_builder`` callback for one routed tool.

    Args:
        expected_tool_name: Tool name bound to the generated builder callback.

    Returns:
        Tool-step input builder that forwards selected tool input payloads.
    """

    def _builder(context: Mapping[str, object]) -> Mapping[str, object]:
        """Extract selected tool input payload from routed dependency outputs.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Tool input payload for the expected tool, or an empty mapping.
        """
        dependency_results = context.get("dependency_results")
        if not isinstance(dependency_results, Mapping):
            return {}
        select_result = dependency_results.get("select_tool")
        if not isinstance(select_result, Mapping):
            return {}
        select_output = select_result.get("output")
        if not isinstance(select_output, Mapping):
            return {}
        selected_tool_name = select_output.get("tool_name")
        if selected_tool_name != expected_tool_name:
            return {}
        return _mapping_or_empty(select_output.get("tool_input"))

    return _builder


def _mapping_or_empty(value: object) -> dict[str, object]:
    """Normalize optional mapping payloads to plain dictionaries.

    Args:
        value: Candidate mapping value.

    Returns:
        Plain dictionary when ``value`` is a mapping, otherwise an empty dictionary.
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _tool_result_from_step_output(
    *,
    step_output: Mapping[str, object],
    fallback_tool_name: str,
) -> ToolResult:
    """Rebuild typed ``ToolResult`` from ``ToolStep`` serialized output payload.

    Args:
        step_output: Serialized tool step output mapping.
        fallback_tool_name: Tool name used when payload omits explicit tool name.

    Returns:
        Typed ``ToolResult`` reconstructed from serialized step output.
    """
    metadata = step_output.get("metadata")
    artifacts = step_output.get("artifacts")
    warnings = step_output.get("warnings")
    return ToolResult(
        tool_name=str(step_output.get("tool_name", fallback_tool_name)),
        ok=bool(step_output.get("ok", False)),
        result=step_output.get("result", {}),
        artifacts=artifacts if isinstance(artifacts, Sequence) else (),
        warnings=warnings if isinstance(warnings, Sequence) else (),
        error=_tool_error_payload(step_output.get("error")),
        metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def _resolve_failure_error(
    *,
    select_output: Mapping[str, object],
    selected_step: WorkflowStepResult | None,
) -> str:
    """Resolve one stable failure message for invalid selection/tool failure paths.

    Args:
        select_output: Output payload emitted by the selection step.
        selected_step: Optional selected tool-step result.

    Returns:
        Stable failure message string.
    """
    select_error = select_output.get("error")
    if isinstance(select_error, str) and select_error.strip():
        return select_error
    if selected_step is not None:
        selected_error = selected_step.error
        if isinstance(selected_error, str) and selected_error.strip():
            return selected_error
    return "Tool selection or execution failed."


def _resolve_failed_tool_output(selected_step: WorkflowStepResult | None) -> object:
    """Extract failed tool payload when a selected ``ToolStep`` returned output.

    Args:
        selected_step: Optional selected tool-step result.

    Returns:
        Failed tool result payload when available, otherwise an empty mapping.
    """
    if selected_step is None:
        return {}
    step_output = selected_step.output
    if isinstance(step_output, Mapping):
        return step_output.get("result", {})
    return {}


def _resolve_tool_results_for_failure(selected_step: WorkflowStepResult | None) -> list[ToolResult]:
    """Return tool results accumulated before/at failure.

    Args:
        selected_step: Optional selected tool-step result.

    Returns:
        List of reconstructed tool results collected before or at failure.
    """
    if selected_step is None:
        return []
    step_output = selected_step.output
    if not isinstance(step_output, Mapping):
        return []
    tool_name = step_output.get("tool_name")
    if not isinstance(tool_name, str):
        return []
    return [
        _tool_result_from_step_output(
            step_output=step_output,
            fallback_tool_name=tool_name,
        )
    ]


def _tool_error_payload(value: object) -> ToolError | Mapping[str, object] | str | None:
    """Narrow raw tool error payload into ``ToolResult`` accepted input types.

    Args:
        value: Candidate tool error payload.

    Returns:
        Value narrowed to accepted ``ToolResult.error`` input types.
    """
    if isinstance(value, (ToolError, str)):
        return value
    if isinstance(value, Mapping):
        return value
    return None


def _tool_step_id(tool_name: str) -> str:
    """Normalize tool names into stable workflow step ids.

    Args:
        tool_name: Tool name to normalize.

    Returns:
        Stable workflow step identifier for routed invocation.
    """
    sanitized = [ch if ch.isalnum() else "_" for ch in tool_name]
    suffix = "".join(sanitized).strip("_") or "tool"
    return f"invoke_{suffix}"


__all__ = [
    "SingleStepJsonToolCallingAgent",
]
