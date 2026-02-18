"""Tool-calling agent that chooses a tool and arguments from model output.

The agent prompts the model with runtime-backed tool options, validates the
structured response, and executes one tool call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents.agent.internal.json_tool_agent_helpers import (
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
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    normalize_alternatives_prompt_target,
)
from design_research_agents.agent.internal.prompt_overrides import (
    resolve_prompt_text,
)
from design_research_agents.agent.internal.result_builders import build_failure_result
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.internal.tool_input import extract_prompt
from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMClient, LLMMessage, LLMRequest
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import (
    Tracer,
    emit_guardrail_decision,
    emit_tool_selection_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


class SingleStepJsonToolCallingAgent(Agent):
    """Agent that asks the model to select a tool and structured arguments.

    The execution path is: gather choices, request strict JSON tool call, parse
    and validate, then invoke exactly one selected tool.
    """

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

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run one tool-calling step from planning through tool execution.

        The run prompts for a structured tool call, validates selection, resolves
        tool input, executes the tool, and returns unified output/metadata.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.

        Raises:
            Exception: Raised when execution fails.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="SingleStepJsonToolCallingAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        prompt = extract_prompt(normalized_input)
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
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
            LLMMessage(
                role="system",
                content=system_prompt,
            ),
            LLMMessage(
                role="user",
                content=user_prompt,
            ),
        ]
        llm_request = LLMRequest(
            messages=model_messages,
            model=resolved_model,
            tools=list(self._runtime_specs.values()),
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
            finish_trace_run(trace_scope, error=str(exc))
            raise
        finish_model_call(model_span_id, response=llm_response)

        parsed_tool_call = parse_tool_call_from_response(llm_response)
        if parsed_tool_call is None:
            parsed_tool_call = parse_tool_call(llm_response.text)
        tool_selection = select_tool_choice(
            parsed_tool_call=parsed_tool_call,
            choices=choices,
        )
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
            result = build_failure_result(
                error=(
                    "Model tool selection was invalid. Expected one allowed "
                    "`tool_name` in structured output."
                ),
                model_response=llm_response,
                tool_results=[],
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                metadata={
                    "stage": "tool_selection",
                    "tool_call": {
                        "source": "model_invalid",
                        "reason": "invalid model tool selection",
                        "available_tools": [choice.tool_name for choice in choices],
                        "parsed_tool_call": parsed_tool_call,
                    },
                },
                output={
                    "model_text": llm_response.text,
                    "tool_name": None,
                    "tool_input": {},
                    "tool_output": {},
                },
            )
            finish_trace_run(trace_scope, result=result)
            return result

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

        tool_result = self._tool_runtime.invoke(
            selected_choice.tool_name,
            tool_input,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
        )
        output: dict[str, object] = {
            "model_text": llm_response.text,
            "tool_name": selected_choice.tool_name,
            "tool_input": tool_input,
            "tool_output": tool_result.result,
        }
        result = ExecutionResult(
            output=output,
            success=tool_result.ok,
            tool_results=[tool_result],
            model_response=llm_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "tool_call": {
                    "source": tool_call_source,
                    "reason": tool_call_reason,
                    "available_tools": [choice.tool_name for choice in choices],
                    "parsed_tool_call": parsed_tool_call,
                },
            },
        )
        finish_trace_run(trace_scope, result=result)
        return result


__all__ = [
    "SingleStepJsonToolCallingAgent",
]
