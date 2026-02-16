"""Single-step code-writing agent with strict sandboxed tool execution.

This agent generates one Python action program, validates it against a restricted
AST policy, executes it in a constrained runtime, and returns structured tool
and final-output artifacts.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence

from design_research_agents.agent.internal.code_tool_agent_execution import (
    compile_sandboxed_code,
    execute_compiled_code,
    failure_result,
)
from design_research_agents.agent.internal.code_tool_agent_parsing import (
    AllowedTool,
    CodeNormalizationResult,
    canonicalize_generated_code,
    compile_default_allowed_tools,
    extract_allowed_tools,
    extract_boolean,
    extract_positive_int,
    extract_prompt,
    extract_python_code,
    match_fenced_code_block,
    normalize_allowed_tools,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    normalize_alternatives_prompt_target,
)
from design_research_agents.agent.internal.prompt_overrides import (
    render_template_text,
    resolve_prompt_text,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.tracing import (
    Tracer,
    emit_guardrail_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


class SingleStepCodeToolCallingAgent(Agent):
    """Agent that writes and executes one sandboxed Python action program.

    The agent is designed for deterministic single-turn execution with strict
    controls around tool access, syntax, runtime builtins, and wall-clock time.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        max_tool_calls: int = 5,
        execution_timeout_seconds: int = 5,
        validate_tool_input_schema: bool = False,
        normalize_generated_code: bool = False,
        default_tools: Sequence[Mapping[str, object]] | None = None,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        alternatives_prompt_target: AlternativesPromptTarget = "user",
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a single-step code agent.

        Args:
            llm_client: LLM client used to generate one action program.
            tool_runtime: Tool runtime used for allowed tool invocation.
            max_tool_calls: Maximum number of tool calls allowed in one run.
            execution_timeout_seconds: Max wall-clock seconds for executing generated code.
            validate_tool_input_schema: Whether to validate tool args against tool input schemas.
            normalize_generated_code: Whether to apply conservative pre-validation
                rewrites for common non-canonical tool-call patterns.
            default_tools: Optional default allowed-tool list compiled at init time.
                When omitted, all runtime-registered tools are allowed by default.
            system_prompt: Optional system prompt override.
            user_prompt_template: Optional user prompt template override.
            alternatives_prompt_target: Prompt target for allowed tools block.
            tracer: Optional explicit tracer dependency.
        """
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1.")
        if execution_timeout_seconds < 1:
            raise ValueError("execution_timeout_seconds must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._max_tool_calls = max_tool_calls
        self._execution_timeout_seconds = execution_timeout_seconds
        self._tracer = tracer
        self._validate_tool_input_schema = validate_tool_input_schema
        self._normalize_generated_code = normalize_generated_code
        self._system_prompt = resolve_prompt_text(
            override=system_prompt,
            default_prompt_name="single_step_code_system",
            field_name="system_prompt",
        )
        self._user_prompt_template = resolve_prompt_text(
            override=user_prompt_template,
            default_prompt_name="single_step_code_user_plan",
            field_name="user_prompt_template",
        )
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(
            alternatives_prompt_target
        )
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        self._compiled_default_allowed_tools = compile_default_allowed_tools(
            runtime_specs=self._runtime_specs,
            default_tools=default_tools,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run one LLM generation and one sandboxed code execution pass.

        The method resolves runtime options, generates code, validates AST safety,
        executes within strict constraints, and returns structured artifacts.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="SingleStepCodeToolCallingAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        allowed_tools, allowed_tools_source = extract_allowed_tools(
            default_allowed_tools=self._compiled_default_allowed_tools,
        )
        if not allowed_tools:
            emit_guardrail_decision(
                guardrail="allowed_tools",
                decision="deny",
                reason="no allowed tools resolved",
            )
            result = failure_result(
                error=(
                    "No allowed tools were resolved from init-time defaults or runtime "
                    "tool registration."
                ),
                model_response=None,
                tool_results=[],
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                metadata={"stage": "input_validation"},
                generated_code="",
            )
            finish_trace_run(trace_scope, result=result)
            return result

        max_tool_calls = extract_positive_int(
            input_payload=normalized_input,
            key="max_tool_calls",
            default_value=self._max_tool_calls,
        )
        execution_timeout_seconds = extract_positive_int(
            input_payload=normalized_input,
            key="execution_timeout_seconds",
            default_value=self._execution_timeout_seconds,
        )
        validate_tool_input_schema = extract_boolean(
            input_payload=normalized_input,
            key="validate_tool_input_schema",
            default_value=self._validate_tool_input_schema,
        )
        normalize_generated_code = self._normalize_generated_code
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        prompt = extract_prompt(normalized_input)
        alternatives_prompt_target = self._alternatives_prompt_target

        try:
            llm_response = self._generate_code(
                prompt=prompt,
                allowed_tools=allowed_tools,
                model=resolved_model,
                alternatives_prompt_target=alternatives_prompt_target,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise
        raw_code_text = extract_python_code(llm_response.text)
        if normalize_generated_code:
            code_normalization = canonicalize_generated_code(
                code_text=raw_code_text,
                allowed_tools=allowed_tools,
            )
        else:
            code_normalization = CodeNormalizationResult(
                code_text=raw_code_text,
                raw_code_text=raw_code_text,
                stripped_safe_tool_imports=0,
                rewritten_tool_calls=0,
                rewritten_direct_name_calls=0,
                rewritten_module_attr_calls=0,
                parse_error=None,
            )
        code_text = code_normalization.code_text
        raw_generated_code = raw_code_text if code_normalization.changed else None

        try:
            compiled_code = compile_sandboxed_code(code_text)
        except Exception as exc:
            emit_guardrail_decision(
                guardrail="code_validation",
                decision="reject",
                reason=str(exc),
                details={"stage": "code_validation"},
            )
            result = failure_result(
                error=f"Generated code failed sandbox validation: {exc}",
                model_response=llm_response,
                tool_results=[],
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                metadata={
                    "stage": "code_validation",
                    "code_normalization": code_normalization.metadata(),
                },
                generated_code=code_text,
                raw_generated_code=raw_generated_code,
            )
            finish_trace_run(trace_scope, result=result)
            return result

        tool_results: list[ToolResult] = []
        try:
            final_output = execute_compiled_code(
                compiled_code=compiled_code,
                prompt=prompt,
                input_payload=normalized_input,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                allowed_tools=allowed_tools,
                tool_runtime=self._tool_runtime,
                max_tool_calls=max_tool_calls,
                execution_timeout_seconds=execution_timeout_seconds,
                validate_tool_input_schema=validate_tool_input_schema,
                tool_results=tool_results,
            )
        except Exception as exc:
            result = failure_result(
                error=f"Sandboxed code execution failed: {exc}",
                model_response=llm_response,
                tool_results=tool_results,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                metadata={
                    "stage": "code_execution",
                    "code_normalization": code_normalization.metadata(),
                    "max_tool_calls": max_tool_calls,
                    "execution_timeout_seconds": execution_timeout_seconds,
                },
                generated_code=code_text,
                raw_generated_code=raw_generated_code,
            )
            finish_trace_run(trace_scope, result=result)
            return result

        output: dict[str, object] = {
            "model_text": llm_response.text,
            "generated_code": code_text,
            "final_output": final_output,
            "tool_name": tool_results[-1].tool_name if tool_results else None,
            "tool_output": tool_results[-1].result if tool_results else {},
        }
        if raw_generated_code is not None:
            output["raw_generated_code"] = raw_generated_code
        result = AgentResult(
            output=output,
            success=all(tool_result.ok for tool_result in tool_results),
            tool_results=tool_results,
            model_response=llm_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "code_execution": {
                    "allowed_tools": [tool.tool_name for tool in allowed_tools],
                    "allowed_tools_source": allowed_tools_source,
                    "tool_call_count": len(tool_results),
                    "max_tool_calls": max_tool_calls,
                    "execution_timeout_seconds": execution_timeout_seconds,
                    "validate_tool_input_schema": validate_tool_input_schema,
                    "normalize_generated_code": normalize_generated_code,
                },
                "code_normalization": code_normalization.metadata(),
            },
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Emit a deterministic stream wrapper around ``run``.

        The wrapper emits one full delta and then a completion event containing
        the final ``AgentResult``.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Yields:
            Streaming events through completion.
        """
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)

    def _generate_code(
        self,
        *,
        prompt: str,
        allowed_tools: Sequence[AllowedTool],
        model: str,
        alternatives_prompt_target: AlternativesPromptTarget,
    ) -> LLMResponse:
        """Generate one Python action program from the model.

        The prompt enumerates allowed tools and their schemas so the model can
        produce executable code aligned with runtime constraints.

        Args:
            prompt: User prompt text.
            allowed_tools: Allowed tool list for this run.
            model: Model identifier for the call.
            alternatives_prompt_target: Prompt target for allowed-tool context.

        Returns:
            LLM response containing the generated code.
        """
        tool_lines: list[str] = []
        for allowed_tool in allowed_tools:
            tool_lines.append(
                "\n".join(
                    [
                        f"- tool_name: {allowed_tool.tool_name}",
                        f"  description: {allowed_tool.description or '(none)'}",
                        f"  input_schema: {json.dumps(allowed_tool.input_schema, sort_keys=True)}",
                    ]
                )
            )
        tools_text = "\n".join(tool_lines)
        tools_block = build_user_prompt_alternatives_block(
            section_label="Allowed tools",
            alternatives_text=tools_text,
            target=alternatives_prompt_target,
        )
        user_prompt = render_template_text(
            template_text=self._user_prompt_template,
            variables={"tools_block": tools_block, "user_prompt": prompt},
            field_name="user_prompt_template",
        )
        system_prompt = self._system_prompt
        if alternatives_prompt_target == "system":
            system_prompt = append_alternatives_block(
                prompt_text=system_prompt,
                section_label="Allowed tools",
                alternatives_text=tools_text,
            )
        llm_params = LLMChatParams(
            provider_options={"agent": "SingleStepCodeToolCallingAgent"},
        )
        messages = [
            LLMMessage(
                role="system",
                content=system_prompt,
            ),
            LLMMessage(role="user", content=user_prompt),
        ]
        model_span_id = start_model_call(
            model=model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "SingleStepCodeToolCallingAgent"},
        )
        try:
            response = self._llm_client.chat(messages, model=model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=model)
            raise
        finish_model_call(model_span_id, response=response)
        return response


# Backward-compatible helper aliases used by internal tests.
_AllowedTool = AllowedTool
_compile_default_allowed_tools = compile_default_allowed_tools
_extract_allowed_tools = extract_allowed_tools
_extract_boolean = extract_boolean
_extract_positive_int = extract_positive_int
_extract_python_code = extract_python_code
_match_fenced_code_block = match_fenced_code_block
_normalize_allowed_tools = normalize_allowed_tools

__all__ = [
    "SingleStepCodeToolCallingAgent",
]
