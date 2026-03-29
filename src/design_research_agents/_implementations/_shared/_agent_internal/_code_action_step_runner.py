"""Workflow-native code-writing action-step runner."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

from design_research_agents._contracts._delegate import Delegate
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents._contracts._tools import ToolResult, ToolRuntime
from design_research_agents._contracts._workflow import LogicStep
from design_research_agents._implementations._shared._agent_internal._code_tool_agent_execution import (
    _clamp_tool_result,
    compile_sandboxed_code,
    execute_compiled_code,
    failure_result,
)
from design_research_agents._implementations._shared._agent_internal._code_tool_agent_parsing import (
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
from design_research_agents._implementations._shared._agent_internal._execution_context import (
    resolve_agent_execution_context,
)
from design_research_agents._implementations._shared._agent_internal._model_resolution import (
    resolve_agent_model,
)
from design_research_agents._implementations._shared._agent_internal._prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    normalize_alternatives_prompt_target,
)
from design_research_agents._implementations._shared._agent_internal._prompt_overrides import (
    render_template_text,
    resolve_prompt_text,
)
from design_research_agents._implementations._shared._agent_internal._tool_input import (
    resolve_known_tool_input,
)
from design_research_agents._implementations._shared._agent_internal._workflow_first_envelope import (
    build_workflow_first_output,
)
from design_research_agents._tracing import (
    Tracer,
    emit_guardrail_decision,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import CompiledExecution, Workflow

from ._code_action_step_workflow_helpers import (
    assert_success_handler,
    dependency_output,
    int_or_default,
    llm_response_or_none,
    mapping_or_empty,
)

_DISALLOWED_TOOL_ERROR_FRAGMENT = "is not in the allowed tool list"
_ARITHMETIC_CANDIDATE_PATTERN = re.compile(r"^[0-9\.\+\-\*\/%\(\)\s]+$")
_FINAL_ANSWER_TOOL_NAME = "final_answer"


class CodeActionStepRunner(Delegate):
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
        self._alternatives_prompt_target = normalize_alternatives_prompt_target(alternatives_prompt_target)
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        if _FINAL_ANSWER_TOOL_NAME in self._runtime_specs:
            raise ValueError(f"ToolRuntime cannot expose reserved tool name '{_FINAL_ANSWER_TOOL_NAME}'.")
        self._compiled_default_allowed_tools = compile_default_allowed_tools(
            runtime_specs=self._runtime_specs,
            default_tools=default_tools,
        )
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """One workflow-native code-generation and sandbox execution pass."""
        return self.compile(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one bound code-generation execution."""
        execution_context = resolve_agent_execution_context(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        workflow = self._build_workflow()
        self.workflow = workflow
        return CompiledExecution(
            workflow=workflow,
            input={
                "normalized_input": execution_context.normalized_input,
                "request_id": execution_context.request_id,
                "dependencies": dict(execution_context.dependencies),
            },
            request_id=execution_context.request_id,
            workflow_request_id=f"{execution_context.request_id}:action_step_code",
            dependencies=execution_context.dependencies,
            delegate_name="CodeActionStepRunner",
            tracer=self._tracer,
            trace_input=execution_context.normalized_input,
            finalize=self._build_result,
        )

    def _build_result(self, workflow_result: ExecutionResult) -> ExecutionResult:
        """Build one public execution result from compiled workflow output."""
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
        return ExecutionResult(
            output=output,
            success=raw_agent_result.success and workflow_result.success,
            tool_results=list(raw_agent_result.tool_results),
            model_response=raw_agent_result.model_response,
            metadata=dict(raw_agent_result.metadata),
        )

    def _build_workflow(self) -> Workflow:
        """Build the code generation/validation/execution workflow graph.

        Returns:
            Workflow configured for input resolution, code generation, validation, and execution.
        """
        return Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
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
                "error": ("No allowed tools were resolved from init-time defaults or runtime tool registration."),
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
            execution_outcome = execute_compiled_code(
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
            recovered_output = self._recover_from_execution_error(
                execution_error=exc,
                prompt=str(resolved.get("prompt", "")),
                normalized_input=mapping_or_empty(resolved.get("normalized_input")),
                request_id=str(resolved.get("request_id", "")),
                dependencies=mapping_or_empty(resolved.get("dependencies")),
                allowed_tools=allowed_tools,
                tool_results=tool_results,
                llm_response=llm_response,
                generated_code=code_text,
                raw_generated_code=generated.get("raw_generated_code"),
                code_normalization=generated.get("code_normalization"),
            )
            if recovered_output is not None:
                return recovered_output
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
            "final_output": execution_outcome.final_output,
            "final_answer_called": execution_outcome.final_answer_called,
            "used_tool_output_fallback": execution_outcome.used_tool_output_fallback,
            "action_type": "final_answer" if execution_outcome.final_answer_called else "tool_call",
            "tool_results": tool_results,
            "code_normalization": mapping_or_empty(generated.get("code_normalization")),
        }

    def _recover_from_execution_error(
        self,
        *,
        execution_error: Exception,
        prompt: str,
        normalized_input: Mapping[str, object],
        request_id: str,
        dependencies: Mapping[str, object],
        allowed_tools: Sequence[AllowedTool],
        tool_results: list[ToolResult],
        llm_response: LLMResponse,
        generated_code: str,
        raw_generated_code: object,
        code_normalization: object,
    ) -> Mapping[str, object] | None:
        """Attempt guardrail-safe fallback execution for disallowed tool calls.

        Args:
            execution_error: Underlying execution exception from sandbox runtime.
            prompt: Prompt text associated with this code step.
            normalized_input: Normalized step input payload.
            request_id: Request id for runtime tool calls.
            dependencies: Dependency mapping for runtime tool calls.
            allowed_tools: Allowed tool descriptors for this step.
            tool_results: Mutable step-level tool result collector.
            llm_response: Model response that produced the generated code.
            generated_code: Generated code text.
            raw_generated_code: Raw generated code text before normalization.
            code_normalization: Optional code-normalization metadata payload.

        Returns:
            Successful execute-step payload when recovery succeeds, else ``None``.
        """
        error_text = str(execution_error)
        if _DISALLOWED_TOOL_ERROR_FRAGMENT not in error_text:
            return None
        if tool_results:
            return None

        recovered_final_output = self._resolve_known_tool_recovery_output(
            prompt=prompt,
            normalized_input=normalized_input,
            request_id=request_id,
            dependencies=dependencies,
            allowed_tools=allowed_tools,
            tool_results=tool_results,
        )
        if recovered_final_output is None:
            return None

        emit_guardrail_decision(
            guardrail="tool_call_allowed_recovery",
            decision="recover",
            reason="used known-tool fallback after disallowed tool call",
            details={
                "allowed_tools": [allowed_tool.tool_name for allowed_tool in allowed_tools],
                "error": error_text,
            },
        )
        payload: dict[str, object] = {
            "success": True,
            "model_response": llm_response,
            "generated_code": generated_code,
            "final_output": recovered_final_output,
            "final_answer_called": False,
            "used_tool_output_fallback": False,
            "action_type": "tool_call",
            "tool_results": tool_results,
            "code_normalization": mapping_or_empty(code_normalization),
        }
        if isinstance(raw_generated_code, str):
            payload["raw_generated_code"] = raw_generated_code
        return payload

    def _resolve_known_tool_recovery_output(
        self,
        *,
        prompt: str,
        normalized_input: Mapping[str, object],
        request_id: str,
        dependencies: Mapping[str, object],
        allowed_tools: Sequence[AllowedTool],
        tool_results: list[ToolResult],
    ) -> dict[str, object] | None:
        """A deterministic fallback plan when exactly one known tool is allowed.

        Args:
            prompt: Prompt text associated with this code step.
            normalized_input: Normalized step input payload.
            request_id: Request id for runtime tool calls.
            dependencies: Dependency mapping for runtime tool calls.
            allowed_tools: Allowed tool descriptors for this step.
            tool_results: Mutable step-level tool result collector.

        Returns:
            Final output mapping when fallback execution succeeds, else ``None``.
        """
        if len(allowed_tools) != 1:
            return None
        selected_tool_name = allowed_tools[0].tool_name
        fallback_tool_results: list[ToolResult] = []

        if selected_tool_name == "calculator":
            expressions = _resolve_calculator_expressions(
                prompt=prompt,
                input_payload=normalized_input,
            )
            if not expressions:
                return None

            calculator_results: list[dict[str, object]] = []
            for expression in expressions:
                tool_result = self._tool_runtime.invoke(
                    selected_tool_name,
                    {"expression": expression},
                    request_id=request_id,
                    dependencies=dependencies,
                )
                fallback_tool_results.append(tool_result)
                if not tool_result.ok or not isinstance(tool_result.result, Mapping):
                    return None
                calculator_results.append(dict(tool_result.result))
            tool_results.extend(_clamp_tool_result(tool_result) for tool_result in fallback_tool_results)

            if len(calculator_results) == 1:
                return calculator_results[0]
            return {
                "results": calculator_results,
                "result": calculator_results[-1].get("result"),
            }

        known_tool_input = resolve_known_tool_input(
            tool_name=selected_tool_name,
            input_payload={**dict(normalized_input), "prompt": prompt},
        )
        if known_tool_input is None:
            return None

        tool_result = self._tool_runtime.invoke(
            selected_tool_name,
            known_tool_input,
            request_id=request_id,
            dependencies=dependencies,
        )
        fallback_tool_results.append(tool_result)
        if not tool_result.ok or not isinstance(tool_result.result, Mapping):
            return None
        tool_results.extend(_clamp_tool_result(tool_result) for tool_result in fallback_tool_results)
        return dict(tool_result.result)

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
                generated_code=str(validated.get("generated_code", generated.get("generated_code", ""))),
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
                    executed_model_response if executed_model_response is not None else generated_model_response
                ),
                tool_results=(list(failed_tool_results) if isinstance(failed_tool_results, list) else []),
                request_id=request_id,
                dependencies=dependencies,
                metadata=mapping_or_empty(executed.get("metadata")),
                generated_code=str(executed.get("generated_code", generated.get("generated_code", ""))),
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
            "generated_code": str(executed.get("generated_code", generated.get("generated_code", ""))),
            "final_output": final_output,
            "final_answer_called": bool(executed.get("final_answer_called", False)),
            "action_type": str(executed.get("action_type", "tool_call")),
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
                    "allowed_tools_source": str(resolved.get("allowed_tools_source", "init_default")),
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
            Exception: Propagated when the underlying LLM client chat request fails.
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
            metadata={"agent": "CodeActionStepRunner", "step_id": "code_generation"},
        )
        try:
            response = self._llm_client.chat(messages, model=model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=model)
            raise
        finish_model_call(model_span_id, response=response)
        return response


def _resolve_calculator_expressions(
    *,
    prompt: str,
    input_payload: Mapping[str, object],
) -> list[str]:
    """Resolve arithmetic expressions for deterministic calculator fallback.

    Args:
        prompt: Prompt text that may contain arithmetic expressions.
        input_payload: Step input payload that may include expression overrides.

    Returns:
        Ordered list of normalized arithmetic expressions.
    """
    candidates: list[str] = []
    raw_expressions = input_payload.get("expressions")
    if isinstance(raw_expressions, Sequence) and not isinstance(raw_expressions, (str, bytes)):
        for raw_expression in raw_expressions:
            if isinstance(raw_expression, str):
                candidates.append(raw_expression)

    raw_expression = input_payload.get("expression")
    if isinstance(raw_expression, str):
        candidates.append(raw_expression)

    candidates.extend(_extract_parenthesized_expressions(prompt))
    if not candidates:
        known_tool_input = resolve_known_tool_input(
            tool_name="calculator",
            input_payload={**dict(input_payload), "prompt": prompt},
        )
        expression = known_tool_input.get("expression") if isinstance(known_tool_input, Mapping) else None
        if isinstance(expression, str):
            candidates.append(expression)

    normalized_candidates: list[str] = []
    for candidate in candidates:
        normalized_expression = _normalize_arithmetic_expression(candidate)
        if normalized_expression is not None:
            normalized_candidates.append(normalized_expression)
    return list(dict.fromkeys(normalized_candidates))


def _extract_parenthesized_expressions(text: str) -> list[str]:
    """Extract balanced parenthesized substrings from text in encounter order.

    Args:
        text: Source text that may contain arithmetic expressions.

    Returns:
        Parenthesized substring candidates.
    """
    expressions: list[str] = []
    depth = 0
    start_index: int | None = None
    for index, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start_index = index
            depth += 1
            continue
        if char != ")" or depth == 0:
            continue
        depth -= 1
        if depth == 0 and start_index is not None:
            expressions.append(text[start_index : index + 1])
            start_index = None
    return expressions


def _normalize_arithmetic_expression(candidate: str) -> str | None:
    """Normalize one arithmetic-expression candidate when it matches safe heuristics.

    Args:
        candidate: Candidate expression text.

    Returns:
        Normalized expression string, or ``None`` when candidate is not arithmetic.
    """
    normalized = candidate.strip().strip(".,;:")
    if not normalized:
        return None
    if not any(operator in normalized for operator in "+-*/%"):
        return None
    if not any(character.isdigit() for character in normalized):
        return None
    if _ARITHMETIC_CANDIDATE_PATTERN.fullmatch(normalized) is None:
        return None
    return normalized


__all__ = [
    "CodeActionStepRunner",
]
