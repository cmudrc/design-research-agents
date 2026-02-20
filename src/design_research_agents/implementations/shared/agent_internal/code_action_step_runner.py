"""Workflow-native code-writing action-step runner."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.contracts.workflow import LogicStep
from design_research_agents.implementations.shared.agent_internal.code_tool_agent_execution import (
    compile_sandboxed_code,
    execute_compiled_code,
    failure_result,
)
from design_research_agents.implementations.shared.agent_internal.code_tool_agent_parsing import (
    AllowedTool,
    CodeNormalizationResult,
    canonicalize_generated_code,
    compile_default_allowed_tools,
    extract_allowed_tools,
    extract_boolean,
    extract_positive_int,
    extract_prompt,
    extract_python_code,
)
from design_research_agents.implementations.shared.agent_internal.execution_context import (
    finish_agent_execution,
    prepare_agent_execution,
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
    render_template_text,
    resolve_prompt_text,
)
from design_research_agents.implementations.shared.agent_internal.workflow_first_envelope import (
    build_workflow_first_output,
)
from design_research_agents.tracing import (
    Tracer,
    emit_guardrail_decision,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import Workflow

from .code_action_step_workflow_helpers import (
    assert_success_handler,
    dependency_output,
    int_or_default,
    llm_response_or_none,
    mapping_or_empty,
)


class CodeActionStepRunner(Agent):
    """Agent that writes and executes one sandboxed Python action program."""

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
        """Initialize a code action-step runner.

        Args:
            llm_client: LLM client used to generate one action program.
            tool_runtime: Tool runtime used for allowed tool invocation.
            max_tool_calls: Maximum number of tool calls allowed in one run.
            execution_timeout_seconds: Max wall-clock seconds for executing generated code.
            validate_tool_input_schema: Whether to validate tool args against tool input schemas.
            normalize_generated_code: Whether to apply conservative pre-validation rewrites.
            default_tools: Optional default allowed-tool list compiled at init time.
            system_prompt: Optional system prompt override.
            user_prompt_template: Optional user prompt template override.
            alternatives_prompt_target: Prompt target for allowed tools block.
            tracer: Optional explicit tracer dependency.

        Raises:
            ValueError: If constructor bounds are invalid.
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
            default_prompt_name="code_action_step_system",
            field_name="system_prompt",
        )
        self._user_prompt_template = resolve_prompt_text(
            override=user_prompt_template,
            default_prompt_name="code_action_step_user_plan",
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
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run one workflow-native code-generation and sandbox execution pass.

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
            agent_name="CodeActionStepRunner",
            tracer=self._tracer,
        )
        self.workflow = self._build_workflow()

        try:
            workflow_result = self.workflow.run(
                {
                    "normalized_input": execution_context.normalized_input,
                    "request_id": execution_context.request_id,
                    "dependencies": dict(execution_context.dependencies),
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
                request_id=f"{execution_context.request_id}:action_step_code",
                dependencies=execution_context.dependencies,
            )
            finalize_step = workflow_result.step_results.get("finalize")
            if finalize_step is None:
                raise RuntimeError("Code action-step workflow missing finalize step result.")
            finalize_output = finalize_step.output
            raw_agent_result = finalize_output.get("agent_result")
            if not isinstance(raw_agent_result, ExecutionResult):
                raise TypeError("Code action-step workflow finalize result is invalid.")
            output = build_workflow_first_output(
                base_output=raw_agent_result.output,
                workflow_result=workflow_result,
                final_output=raw_agent_result.output.get("final_output", {}),
            )
            result = ExecutionResult(
                output=output,
                success=raw_agent_result.success and workflow_result.success,
                tool_results=list(raw_agent_result.tool_results),
                model_response=raw_agent_result.model_response,
                metadata=dict(raw_agent_result.metadata),
            )
        except Exception as exc:
            finish_agent_execution(trace_scope=execution_context.trace_scope, error=str(exc))
            raise

        finish_agent_execution(trace_scope=execution_context.trace_scope, result=result)
        return result

    def _build_workflow(self) -> Workflow:
        """Build the code generation/validation/execution workflow graph.

        Returns:
            Workflow configured for input resolution, code generation, validation, and execution.
        """
        return Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_mode="schema",
            steps=[
                LogicStep(step_id="resolve_inputs", handler=self._resolve_inputs_handler),
                LogicStep(
                    step_id="generate_code",
                    handler=self._generate_code_handler,
                    dependencies=("resolve_inputs",),
                ),
                LogicStep(
                    step_id="validate_code",
                    handler=self._validate_code_handler,
                    dependencies=("resolve_inputs", "generate_code"),
                ),
                LogicStep(
                    step_id="execute_code",
                    handler=self._execute_code_handler,
                    dependencies=("resolve_inputs", "generate_code", "validate_code"),
                ),
                LogicStep(
                    step_id="finalize",
                    handler=self._finalize_handler,
                    dependencies=(
                        "resolve_inputs",
                        "generate_code",
                        "validate_code",
                        "execute_code",
                    ),
                ),
                LogicStep(
                    step_id="assert_success",
                    handler=assert_success_handler,
                    dependencies=("finalize",),
                ),
            ],
            default_execution_mode="sequential",
            default_failure_policy="skip_dependents",
        )

    def _resolve_inputs_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Resolve per-run code execution configuration and guardrails.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping containing resolved execution settings and normalized input payloads.

        Raises:
            TypeError: If schema-mode input payloads are missing or malformed.
        """
        inputs = context.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("Code action-step workflow requires schema input mapping.")
        normalized_input = inputs.get("normalized_input")
        request_id = inputs.get("request_id")
        dependencies = inputs.get("dependencies")
        if not isinstance(normalized_input, Mapping):
            raise TypeError("normalized_input must be a mapping.")
        dependencies_dict = dict(dependencies) if isinstance(dependencies, Mapping) else {}
        resolved_request_id = str(request_id) if request_id is not None else ""

        allowed_tools, allowed_tools_source = extract_allowed_tools(
            default_allowed_tools=self._compiled_default_allowed_tools,
        )
        if not allowed_tools:
            emit_guardrail_decision(
                guardrail="allowed_tools",
                decision="deny",
                reason="no allowed tools resolved",
            )
            return {
                "success": False,
                "error": (
                    "No allowed tools were resolved from init-time defaults or runtime "
                    "tool registration."
                ),
                "metadata": {"stage": "input_validation"},
                "generated_code": "",
                "raw_generated_code": None,
                "model_response": None,
                "tool_results": [],
                "dependencies": dependencies_dict,
                "request_id": resolved_request_id,
                "code_normalization": {
                    "changed": False,
                    "stripped_safe_tool_imports": 0,
                    "rewritten_tool_calls": 0,
                    "rewritten_direct_name_calls": 0,
                    "rewritten_module_attr_calls": 0,
                    "parse_error": None,
                },
            }

        prompt = extract_prompt(normalized_input)
        return {
            "success": True,
            "request_id": resolved_request_id,
            "dependencies": dependencies_dict,
            "normalized_input": dict(normalized_input),
            "prompt": prompt,
            "resolved_model": resolve_agent_model(llm_client=self._llm_client),
            "allowed_tools": allowed_tools,
            "allowed_tools_source": allowed_tools_source,
            "max_tool_calls": extract_positive_int(
                input_payload=normalized_input,
                key="max_tool_calls",
                default_value=self._max_tool_calls,
            ),
            "execution_timeout_seconds": extract_positive_int(
                input_payload=normalized_input,
                key="execution_timeout_seconds",
                default_value=self._execution_timeout_seconds,
            ),
            "validate_tool_input_schema": extract_boolean(
                input_payload=normalized_input,
                key="validate_tool_input_schema",
                default_value=self._validate_tool_input_schema,
            ),
            "normalize_generated_code": self._normalize_generated_code,
        }

    def _generate_code_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Generate candidate code and apply optional canonicalization.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping containing generated code and optional normalization metadata.

        Raises:
            TypeError: If dependency payloads are missing or malformed.
        """
        resolved = dependency_output(context=context, step_id="resolve_inputs")
        if not bool(resolved.get("success", False)):
            return {"success": False}

        prompt = str(resolved.get("prompt", ""))
        allowed_tools_raw = resolved.get("allowed_tools")
        resolved_model = str(resolved.get("resolved_model", ""))
        normalize_generated_code = bool(resolved.get("normalize_generated_code", False))
        if not isinstance(allowed_tools_raw, Sequence):
            raise TypeError("allowed_tools payload is invalid.")
        allowed_tools = [tool for tool in allowed_tools_raw if isinstance(tool, AllowedTool)]

        llm_response = self._generate_code(
            prompt=prompt,
            allowed_tools=allowed_tools,
            model=resolved_model,
            alternatives_prompt_target=self._alternatives_prompt_target,
        )
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
        return {
            "success": True,
            "llm_response": llm_response,
            "generated_code": code_text,
            "raw_generated_code": raw_generated_code,
            "code_normalization": code_normalization.metadata(),
        }

    def _validate_code_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Validate generated code against sandbox compilation policy.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping indicating whether generated code passed validation.
        """
        resolved = dependency_output(context=context, step_id="resolve_inputs")
        generated = dependency_output(context=context, step_id="generate_code")
        if not bool(resolved.get("success", False)):
            return {"success": False}
        if not bool(generated.get("success", False)):
            return {"success": False}

        code_text = str(generated.get("generated_code", ""))
        try:
            compile_sandboxed_code(code_text)
        except Exception as exc:
            emit_guardrail_decision(
                guardrail="code_validation",
                decision="reject",
                reason=str(exc),
                details={"stage": "code_validation"},
            )
            return {
                "success": False,
                "error": f"Generated code failed sandbox validation: {exc}",
                "metadata": {
                    "stage": "code_validation",
                    "code_normalization": mapping_or_empty(generated.get("code_normalization")),
                },
                "generated_code": code_text,
                "raw_generated_code": generated.get("raw_generated_code"),
                "model_response": generated.get("llm_response"),
                "tool_results": [],
            }
        return {
            "success": True,
        }

    def _execute_code_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute validated code in sandbox and collect tool call artifacts.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping containing execution output, tool results, and runtime metadata.

        Raises:
            TypeError: If dependency payloads are missing required typed values.
        """
        resolved = dependency_output(context=context, step_id="resolve_inputs")
        generated = dependency_output(context=context, step_id="generate_code")
        validated = dependency_output(context=context, step_id="validate_code")
        if not bool(resolved.get("success", False)):
            return {"success": False}
        if not bool(generated.get("success", False)):
            return {"success": False}
        if not bool(validated.get("success", False)):
            return {
                "success": False,
                "error": validated.get("error", "Generated code failed sandbox validation."),
                "metadata": mapping_or_empty(validated.get("metadata")),
                "generated_code": generated.get("generated_code", ""),
                "raw_generated_code": generated.get("raw_generated_code"),
                "model_response": generated.get("llm_response"),
                "tool_results": [],
            }

        tool_results: list[ToolResult] = []
        code_text = str(generated.get("generated_code", ""))
        llm_response = generated.get("llm_response")
        if not isinstance(llm_response, LLMResponse):
            raise TypeError("Generated step missing llm_response payload.")

        allowed_tools_raw = resolved.get("allowed_tools")
        if not isinstance(allowed_tools_raw, Sequence):
            raise TypeError("allowed_tools payload is invalid.")
        allowed_tools = [tool for tool in allowed_tools_raw if isinstance(tool, AllowedTool)]

        try:
            compiled_code = compile_sandboxed_code(code_text)
            final_output = execute_compiled_code(
                compiled_code=compiled_code,
                prompt=str(resolved.get("prompt", "")),
                input_payload=mapping_or_empty(resolved.get("normalized_input")),
                request_id=str(resolved.get("request_id", "")),
                dependencies=mapping_or_empty(resolved.get("dependencies")),
                allowed_tools=allowed_tools,
                tool_runtime=self._tool_runtime,
                max_tool_calls=int_or_default(
                    resolved.get("max_tool_calls"),
                    default=self._max_tool_calls,
                ),
                execution_timeout_seconds=int_or_default(
                    resolved.get("execution_timeout_seconds"),
                    default=self._execution_timeout_seconds,
                ),
                validate_tool_input_schema=bool(resolved.get("validate_tool_input_schema", False)),
                tool_results=tool_results,
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"Sandboxed code execution failed: {exc}",
                "metadata": {
                    "stage": "code_execution",
                    "code_normalization": mapping_or_empty(generated.get("code_normalization")),
                    "max_tool_calls": int_or_default(
                        resolved.get("max_tool_calls"),
                        default=self._max_tool_calls,
                    ),
                    "execution_timeout_seconds": int_or_default(
                        resolved.get("execution_timeout_seconds"),
                        default=self._execution_timeout_seconds,
                    ),
                },
                "generated_code": code_text,
                "raw_generated_code": generated.get("raw_generated_code"),
                "model_response": llm_response,
                "tool_results": tool_results,
            }

        return {
            "success": True,
            "model_response": llm_response,
            "generated_code": code_text,
            "raw_generated_code": generated.get("raw_generated_code"),
            "final_output": final_output,
            "tool_results": tool_results,
            "code_normalization": mapping_or_empty(generated.get("code_normalization")),
        }

    def _finalize_handler(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Assemble final agent result from staged workflow outputs.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping that carries the finalized ``ExecutionResult`` and success signal.

        Raises:
            TypeError: If expected generated model response payload is missing.
        """
        resolved = dependency_output(context=context, step_id="resolve_inputs")
        generated = dependency_output(context=context, step_id="generate_code")
        validated = dependency_output(context=context, step_id="validate_code")
        executed = dependency_output(context=context, step_id="execute_code")

        request_id = str(resolved.get("request_id", ""))
        dependencies = mapping_or_empty(resolved.get("dependencies"))

        if not bool(resolved.get("success", False)):
            result = failure_result(
                error=str(resolved.get("error", "No allowed tools were resolved.")),
                model_response=None,
                tool_results=[],
                request_id=request_id,
                dependencies=dependencies,
                metadata=mapping_or_empty(resolved.get("metadata")),
                generated_code=str(resolved.get("generated_code", "")),
                raw_generated_code=(
                    str(resolved.get("raw_generated_code"))
                    if isinstance(resolved.get("raw_generated_code"), str)
                    else None
                ),
            )
            return {"agent_result": result, "success": result.success}

        if not bool(validated.get("success", False)):
            generated_model_response = llm_response_or_none(generated.get("llm_response"))
            result = failure_result(
                error=str(validated.get("error", "Generated code failed sandbox validation.")),
                model_response=generated_model_response,
                tool_results=[],
                request_id=request_id,
                dependencies=dependencies,
                metadata=mapping_or_empty(validated.get("metadata")),
                generated_code=str(
                    validated.get("generated_code", generated.get("generated_code", ""))
                ),
                raw_generated_code=(
                    str(validated.get("raw_generated_code"))
                    if isinstance(validated.get("raw_generated_code"), str)
                    else (
                        str(generated.get("raw_generated_code"))
                        if isinstance(generated.get("raw_generated_code"), str)
                        else None
                    )
                ),
            )
            return {"agent_result": result, "success": result.success}

        if not bool(executed.get("success", False)):
            failed_tool_results = executed.get("tool_results")
            executed_model_response = llm_response_or_none(executed.get("model_response"))
            generated_model_response = llm_response_or_none(generated.get("llm_response"))
            result = failure_result(
                error=str(executed.get("error", "Sandboxed code execution failed.")),
                model_response=(
                    executed_model_response
                    if executed_model_response is not None
                    else generated_model_response
                ),
                tool_results=(
                    list(failed_tool_results) if isinstance(failed_tool_results, list) else []
                ),
                request_id=request_id,
                dependencies=dependencies,
                metadata=mapping_or_empty(executed.get("metadata")),
                generated_code=str(
                    executed.get("generated_code", generated.get("generated_code", ""))
                ),
                raw_generated_code=(
                    str(executed.get("raw_generated_code"))
                    if isinstance(executed.get("raw_generated_code"), str)
                    else (
                        str(generated.get("raw_generated_code"))
                        if isinstance(generated.get("raw_generated_code"), str)
                        else None
                    )
                ),
            )
            return {"agent_result": result, "success": result.success}

        llm_response = generated.get("llm_response")
        if not isinstance(llm_response, LLMResponse):
            raise TypeError("Finalize step missing llm_response payload.")

        tool_results_raw = executed.get("tool_results")
        tool_results: list[ToolResult] = (
            [result for result in tool_results_raw if isinstance(result, ToolResult)]
            if isinstance(tool_results_raw, list)
            else []
        )
        final_output = executed.get("final_output", {})
        output: dict[str, object] = {
            "model_text": llm_response.text,
            "generated_code": str(
                executed.get("generated_code", generated.get("generated_code", ""))
            ),
            "final_output": final_output,
            "tool_name": tool_results[-1].tool_name if tool_results else None,
            "tool_output": tool_results[-1].result if tool_results else {},
        }
        raw_generated_code = executed.get("raw_generated_code")
        if isinstance(raw_generated_code, str):
            output["raw_generated_code"] = raw_generated_code

        allowed_tools_raw = resolved.get("allowed_tools")
        allowed_tool_names = (
            [tool.tool_name for tool in allowed_tools_raw if isinstance(tool, AllowedTool)]
            if isinstance(allowed_tools_raw, Sequence)
            else []
        )
        result = ExecutionResult(
            output=output,
            success=all(tool_result.ok for tool_result in tool_results),
            tool_results=tool_results,
            model_response=llm_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "code_execution": {
                    "allowed_tools": allowed_tool_names,
                    "allowed_tools_source": str(
                        resolved.get("allowed_tools_source", "init_default")
                    ),
                    "tool_call_count": len(tool_results),
                    "max_tool_calls": int_or_default(
                        resolved.get("max_tool_calls"),
                        default=self._max_tool_calls,
                    ),
                    "execution_timeout_seconds": int_or_default(
                        resolved.get("execution_timeout_seconds"),
                        default=self._execution_timeout_seconds,
                    ),
                    "validate_tool_input_schema": bool(
                        resolved.get("validate_tool_input_schema", self._validate_tool_input_schema)
                    ),
                    "normalize_generated_code": bool(
                        resolved.get("normalize_generated_code", self._normalize_generated_code)
                    ),
                },
                "code_normalization": mapping_or_empty(generated.get("code_normalization")),
            },
        )
        return {
            "agent_result": result,
            "success": result.success,
        }

    def _generate_code(
        self,
        *,
        prompt: str,
        allowed_tools: Sequence[AllowedTool],
        model: str,
        alternatives_prompt_target: AlternativesPromptTarget,
    ) -> LLMResponse:
        """Generate one Python action program from the model.

        Args:
            prompt: User prompt text.
            allowed_tools: Allowed tool list for this run.
            model: Model identifier for the call.
            alternatives_prompt_target: Prompt target for allowed-tool context.

        Returns:
            LLM response containing the generated code.

        Raises:
            Exception: Raised when execution fails.
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
            provider_options={"agent": "CodeActionStepRunner"},
        )
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        model_span_id = start_model_call(
            model=model,
            messages=messages,
            params=llm_params,
            metadata={"agent": "CodeActionStepRunner"},
        )
        try:
            response = self._llm_client.chat(messages, model=model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=model)
            raise
        finish_model_call(model_span_id, response=response)
        return response


__all__ = [
    "CodeActionStepRunner",
]
