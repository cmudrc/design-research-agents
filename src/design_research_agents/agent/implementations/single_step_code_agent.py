"""Single-step code-writing agent with strict sandboxed tool execution.

This agent generates one Python action program, validates it against a restricted
AST policy, executes it in a constrained runtime, and returns structured tool
and final-output artifacts.
"""

from __future__ import annotations

import ast
import json
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import CodeType

from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.prompt_alternatives import (
    AlternativesPromptTarget,
    append_alternatives_block,
    build_user_prompt_alternatives_block,
    resolve_alternatives_prompt_target,
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
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.prompts import load_prompt, render_prompt
from design_research_agents.tracing import (
    emit_guardrail_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


@dataclass(slots=True, frozen=True)
class _AllowedTool:
    """Normalized allowed-tool definition used during one run.

    Attributes:
        tool_name: Runtime tool identifier.
        description: Tool description shown to the model.
        input_schema: JSON-schema-like input constraints for optional validation.
        default_tool_input: Optional default arguments when code passes empty input.
    """

    tool_name: str
    description: str
    input_schema: dict[str, object]
    default_tool_input: dict[str, object] | None = None


@dataclass(slots=True, frozen=True)
class _CodeNormalizationResult:
    """Captures optional pre-validation code normalization details."""

    code_text: str
    raw_code_text: str
    stripped_safe_tool_imports: int
    rewritten_tool_calls: int
    rewritten_direct_name_calls: int
    rewritten_module_attr_calls: int
    parse_error: str | None = None

    @property
    def changed(self) -> bool:
        return self.stripped_safe_tool_imports > 0 or self.rewritten_tool_calls > 0

    def metadata(self) -> dict[str, object]:
        return {
            "changed": self.changed,
            "stripped_safe_tool_imports": self.stripped_safe_tool_imports,
            "rewritten_tool_calls": self.rewritten_tool_calls,
            "rewritten_direct_name_calls": self.rewritten_direct_name_calls,
            "rewritten_module_attr_calls": self.rewritten_module_attr_calls,
            "parse_error": self.parse_error,
        }


class SingleStepCodeAgent(Agent):
    """Agent that writes and executes one sandboxed Python action program.

    The agent is designed for deterministic single-turn execution with strict
    controls around tool access, syntax, runtime builtins, and wall-clock time.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        model: str | None = None,
        max_tool_calls: int = 5,
        execution_timeout_seconds: int = 5,
        validate_tool_input_schema: bool = False,
        normalize_generated_code: bool = False,
        default_tools: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Initialize a single-step code agent.

        Args:
            llm_client: LLM client used to generate one action program.
            tool_runtime: Tool runtime used for allowed tool invocation.
            model: Optional model override applied to all runs when provided.
            max_tool_calls: Maximum number of tool calls allowed in one run.
            execution_timeout_seconds: Max wall-clock seconds for executing generated code.
            validate_tool_input_schema: Whether to validate tool args against tool input schemas.
            normalize_generated_code: Whether to apply conservative pre-validation
                rewrites for common non-canonical tool-call patterns.
            default_tools: Optional default allowed-tool list compiled at init time.
                When omitted, all runtime-registered tools are allowed by default.
        """
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1.")
        if execution_timeout_seconds < 1:
            raise ValueError("execution_timeout_seconds must be >= 1.")

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._model = model
        self._max_tool_calls = max_tool_calls
        self._execution_timeout_seconds = execution_timeout_seconds
        self._validate_tool_input_schema = validate_tool_input_schema
        self._normalize_generated_code = normalize_generated_code
        self._runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        self._compiled_default_allowed_tools = _compile_default_allowed_tools(
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
            agent_name="SingleStepCodeAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
        )
        allowed_tools, allowed_tools_source = _extract_allowed_tools(
            default_allowed_tools=self._compiled_default_allowed_tools,
        )
        if not allowed_tools:
            emit_guardrail_decision(
                guardrail="allowed_tools",
                decision="deny",
                reason="no allowed tools resolved",
            )
            result = _failure_result(
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

        max_tool_calls = _extract_positive_int(
            input_payload=normalized_input,
            key="max_tool_calls",
            default_value=self._max_tool_calls,
        )
        execution_timeout_seconds = _extract_positive_int(
            input_payload=normalized_input,
            key="execution_timeout_seconds",
            default_value=self._execution_timeout_seconds,
        )
        validate_tool_input_schema = _extract_boolean(
            input_payload=normalized_input,
            key="validate_tool_input_schema",
            default_value=self._validate_tool_input_schema,
        )
        normalize_generated_code = self._normalize_generated_code
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
            input_payload=normalized_input,
            init_model=self._model,
        )
        prompt = _extract_prompt(normalized_input)
        alternatives_prompt_target = resolve_alternatives_prompt_target(
            input_payload=normalized_input
        )

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
        raw_code_text = _extract_python_code(llm_response.text)
        if normalize_generated_code:
            code_normalization = _canonicalize_generated_code(
                code_text=raw_code_text,
                allowed_tools=allowed_tools,
            )
        else:
            code_normalization = _CodeNormalizationResult(
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
            compiled_code = _compile_sandboxed_code(code_text)
        except Exception as exc:
            emit_guardrail_decision(
                guardrail="code_validation",
                decision="reject",
                reason=str(exc),
                details={"stage": "code_validation"},
            )
            result = _failure_result(
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
            final_output = _execute_compiled_code(
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
            result = _failure_result(
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
            "tool_output": tool_results[-1].output if tool_results else {},
        }
        if raw_generated_code is not None:
            output["raw_generated_code"] = raw_generated_code
        result = AgentResult(
            output=output,
            success=all(tool_result.success for tool_result in tool_results),
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
        allowed_tools: Sequence[_AllowedTool],
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
        user_prompt = render_prompt(
            "single_step_code_user_plan",
            variables={"tools_block": tools_block, "user_prompt": prompt},
        )
        system_prompt = load_prompt("single_step_code_system")
        if alternatives_prompt_target == "system":
            system_prompt = append_alternatives_block(
                prompt_text=system_prompt,
                section_label="Allowed tools",
                alternatives_text=tools_text,
            )
        llm_params = LLMChatParams(
            provider_options={"agent": "SingleStepCodeAgent"},
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
            metadata={"agent": "SingleStepCodeAgent"},
        )
        try:
            response = self._llm_client.chat(messages, model=model, params=llm_params)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=model)
            raise
        finish_model_call(model_span_id, response=response)
        return response


def _extract_allowed_tools(
    *,
    default_allowed_tools: Sequence[_AllowedTool],
) -> tuple[list[_AllowedTool], str]:
    """Return allowed tools compiled at initialization time.

    Runtime input payload does not override allowed tools; tool access is a
    construction-time concern.

    Args:
        default_allowed_tools: Allowed tools compiled at initialization.

    Returns:
        Tuple of allowed tools list and source label.
    """
    return (
        [_clone_allowed_tool(tool) for tool in default_allowed_tools],
        "init_default",
    )


def _compile_default_allowed_tools(
    *,
    runtime_specs: Mapping[str, ToolSpec],
    default_tools: Sequence[Mapping[str, object]] | None,
) -> tuple[_AllowedTool, ...]:
    """Compile default allowed tools from init config and runtime tool specs.

    When no init defaults are provided, all runtime-registered tools are allowed.

    Args:
        runtime_specs: Tool specs available in the runtime.
        default_tools: Optional init-time allowed tool configuration.

    Returns:
        Tuple of compiled allowed tools.
    """
    if default_tools is not None:
        compiled_from_input = _normalize_allowed_tools(
            raw_tools=default_tools,
            runtime_specs=runtime_specs,
        )
        return tuple(compiled_from_input)

    return tuple(
        _AllowedTool(
            tool_name=spec.name,
            description=spec.description,
            input_schema=dict(spec.input_schema),
        )
        for spec in runtime_specs.values()
    )


def _normalize_allowed_tools(
    *,
    raw_tools: object,
    runtime_specs: Mapping[str, ToolSpec],
) -> list[_AllowedTool]:
    """Normalize explicit allowed-tool payload into runtime-backed tool entries.

    Unknown or malformed tools are dropped to prevent unsafe dynamic invocation.

    Args:
        raw_tools: Raw tool configuration payload.
        runtime_specs: Tool specs available in the runtime.

    Returns:
        Normalized list of allowed tools.
    """
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        return []

    normalized: list[_AllowedTool] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        raw_tool_name = raw_tool.get("tool_name", raw_tool.get("name"))
        if not isinstance(raw_tool_name, str):
            continue
        tool_name = raw_tool_name.strip()
        if not tool_name:
            continue

        runtime_spec = runtime_specs.get(tool_name)
        if runtime_spec is None:
            # Only runtime-registered tools can be called.
            continue

        raw_description = raw_tool.get("description")
        description = (
            str(raw_description) if raw_description is not None else runtime_spec.description
        )

        raw_input_schema = raw_tool.get("input_schema")
        if isinstance(raw_input_schema, Mapping):
            input_schema = dict(raw_input_schema)
        else:
            input_schema = dict(runtime_spec.input_schema)

        raw_default_tool_input = raw_tool.get("tool_input")
        default_tool_input = (
            dict(raw_default_tool_input) if isinstance(raw_default_tool_input, Mapping) else None
        )
        normalized.append(
            _AllowedTool(
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                default_tool_input=default_tool_input,
            )
        )

    deduped: dict[str, _AllowedTool] = {}
    for allowed_tool in normalized:
        deduped[allowed_tool.tool_name] = allowed_tool
    return list(deduped.values())


def _clone_allowed_tool(allowed_tool: _AllowedTool) -> _AllowedTool:
    """Clone one allowed tool to isolate run-level payload mutations.

    Mutable dictionaries are copied so per-run writes do not leak globally.

    Args:
        allowed_tool: Allowed tool to clone.

    Returns:
        Cloned allowed tool instance.
    """
    return _AllowedTool(
        tool_name=allowed_tool.tool_name,
        description=allowed_tool.description,
        input_schema=dict(allowed_tool.input_schema),
        default_tool_input=(
            dict(allowed_tool.default_tool_input)
            if isinstance(allowed_tool.default_tool_input, Mapping)
            else None
        ),
    )


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input.

    Falls back to ``text`` and then a default string when missing.

    Args:
        input_payload: Normalized run input payload mapping.

    Returns:
        Prompt text for the run.
    """
    raw_prompt = input_payload.get(
        "prompt", input_payload.get("text", "Provide a concise response.")
    )
    return str(raw_prompt)


def _extract_positive_int(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: int,
) -> int:
    """Extract a positive integer option from run input.

    Invalid, missing, or boolean values resolve to the provided default.

    Args:
        input_payload: Normalized run input payload mapping.
        key: Input payload key to extract.
        default_value: Default value when extraction fails.

    Returns:
        Positive integer value for the option.
    """
    raw_value = input_payload.get(key)
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return default_value
    if isinstance(raw_value, int) and raw_value >= 1:
        return raw_value
    return default_value


def _extract_boolean(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: bool,
) -> bool:
    """Extract a boolean option from run input.

    Non-boolean values resolve to the provided default.

    Args:
        input_payload: Normalized run input payload mapping.
        key: Input payload key to extract.
        default_value: Default value when extraction fails.

    Returns:
        Boolean value for the option.
    """
    raw_value = input_payload.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def _extract_python_code(raw_model_text: str) -> str:
    """Extract Python code from model output text, preferring fenced blocks.

    If no valid fenced block is found, the raw trimmed model text is used.

    Args:
        raw_model_text: Raw model response text.

    Returns:
        Extracted Python code text.
    """
    fenced_match = _match_fenced_code_block(raw_model_text)
    if fenced_match is not None:
        return fenced_match.strip()
    return raw_model_text.strip()


def _match_fenced_code_block(raw_text: str) -> str | None:
    """Return first Python-like fenced code block when present and well-formed.

    Only empty, ``python``, or ``py`` fence headers are accepted.

    Args:
        raw_text: Raw model response text.

    Returns:
        Code block content when found, otherwise ``None``.
    """
    fence = "```"
    start_index = raw_text.find(fence)
    if start_index == -1:
        return None
    end_of_fence_header = raw_text.find("\n", start_index + len(fence))
    if end_of_fence_header == -1:
        return None

    fence_header = raw_text[start_index + len(fence) : end_of_fence_header].strip().lower()
    end_index = raw_text.find(fence, end_of_fence_header + 1)
    if end_index == -1:
        return None

    if fence_header not in {"", "python", "py"}:
        # Only consume the block when header is absent or python-like.
        return None
    return raw_text[end_of_fence_header + 1 : end_index]


class _CodeCanonicalizer(ast.NodeTransformer):
    """Conservative AST normalizer for common tool-call variants.

    The transformer only rewrites explicitly-supported patterns and leaves all
    other code intact so unsupported behavior still fails at validation/runtime.
    """

    def __init__(self, *, allowed_tool_names: set[str]) -> None:
        self._allowed_tool_names = allowed_tool_names
        self.stripped_safe_tool_imports = 0
        self.rewritten_tool_calls = 0
        self.rewritten_direct_name_calls = 0
        self.rewritten_module_attr_calls = 0

    def visit_Module(self, node: ast.Module) -> ast.Module:
        new_body: list[ast.stmt] = []
        for statement in node.body:
            if self._is_strippable_tool_import(statement):
                self.stripped_safe_tool_imports += 1
                continue
            visited = self.visit(statement)
            if visited is None:
                continue
            if isinstance(visited, list):
                new_body.extend(visited)
                continue
            new_body.append(visited)
        node.body = new_body
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        visited_node = self.generic_visit(node)
        if not isinstance(visited_node, ast.Call):
            return visited_node

        target = self._resolve_rewrite_target(visited_node.func)
        if target is None:
            return visited_node

        tool_name, call_style = target
        tool_input_arg = self._build_tool_input_argument(visited_node)
        if tool_input_arg is None:
            return visited_node

        rewritten_call = ast.Call(
            func=ast.Name(id="call_tool", ctx=ast.Load()),
            args=[ast.Constant(value=tool_name), tool_input_arg],
            keywords=[],
        )
        rewritten_call = ast.copy_location(rewritten_call, visited_node)
        self.rewritten_tool_calls += 1
        if call_style == "direct_name":
            self.rewritten_direct_name_calls += 1
        else:
            self.rewritten_module_attr_calls += 1
        return rewritten_call

    def _is_strippable_tool_import(self, statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.Import):
            return False
        if not statement.names:
            return False
        for alias in statement.names:
            if alias.asname is not None:
                return False
            if alias.name not in self._allowed_tool_names:
                return False
        return True

    def _resolve_rewrite_target(
        self,
        call_target: ast.expr,
    ) -> tuple[str, str] | None:
        if isinstance(call_target, ast.Name):
            if call_target.id in self._allowed_tool_names:
                return call_target.id, "direct_name"
            return None
        if not isinstance(call_target, ast.Attribute):
            return None
        if not isinstance(call_target.value, ast.Name):
            return None
        if call_target.value.id != call_target.attr:
            return None
        if call_target.attr not in self._allowed_tool_names:
            return None
        return call_target.attr, "module_attr"

    def _build_tool_input_argument(self, node: ast.Call) -> ast.expr | None:
        if len(node.args) == 1 and not node.keywords:
            if isinstance(node.args[0], ast.Starred):
                return None
            return node.args[0]
        if node.args:
            return None
        if not node.keywords:
            return ast.Dict(keys=[], values=[])

        keys: list[ast.expr | None] = []
        values: list[ast.expr] = []
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            keys.append(ast.Constant(value=keyword.arg))
            values.append(keyword.value)
        return ast.Dict(keys=keys, values=values)


def _canonicalize_generated_code(
    *,
    code_text: str,
    allowed_tools: Sequence[_AllowedTool],
) -> _CodeNormalizationResult:
    """Rewrite narrow, known-safe tool call variants into canonical form.

    Args:
        code_text: Raw generated code text.
        allowed_tools: Allowed tool list for this run.

    Returns:
        Normalization result containing rewritten code and metadata.
    """
    if not code_text:
        return _CodeNormalizationResult(
            code_text=code_text,
            raw_code_text=code_text,
            stripped_safe_tool_imports=0,
            rewritten_tool_calls=0,
            rewritten_direct_name_calls=0,
            rewritten_module_attr_calls=0,
            parse_error=None,
        )

    try:
        syntax_tree = ast.parse(code_text, mode="exec")
    except SyntaxError as exc:
        return _CodeNormalizationResult(
            code_text=code_text,
            raw_code_text=code_text,
            stripped_safe_tool_imports=0,
            rewritten_tool_calls=0,
            rewritten_direct_name_calls=0,
            rewritten_module_attr_calls=0,
            parse_error=str(exc),
        )

    canonicalizer = _CodeCanonicalizer(
        allowed_tool_names={tool.tool_name for tool in allowed_tools}
    )
    normalized_tree = canonicalizer.visit(syntax_tree)
    ast.fix_missing_locations(normalized_tree)

    if canonicalizer.stripped_safe_tool_imports == 0 and canonicalizer.rewritten_tool_calls == 0:
        normalized_code = code_text
    else:
        normalized_code = ast.unparse(normalized_tree).strip()

    return _CodeNormalizationResult(
        code_text=normalized_code,
        raw_code_text=code_text,
        stripped_safe_tool_imports=canonicalizer.stripped_safe_tool_imports,
        rewritten_tool_calls=canonicalizer.rewritten_tool_calls,
        rewritten_direct_name_calls=canonicalizer.rewritten_direct_name_calls,
        rewritten_module_attr_calls=canonicalizer.rewritten_module_attr_calls,
        parse_error=None,
    )


def _compile_sandboxed_code(code_text: str) -> CodeType:
    """Validate and compile generated code under strict sandbox constraints.

    The function enforces syntax safety before compilation to bytecode.

    Args:
        code_text: Python code text to compile.

    Returns:
        Compiled code object.

    Raises:
        ValueError: If the code is empty or fails sandbox validation.
    """
    if not code_text:
        raise ValueError("Generated code is empty.")

    syntax_tree = ast.parse(code_text, mode="exec")
    _validate_sandbox_syntax_tree(syntax_tree)
    return compile(syntax_tree, filename="<single_step_code_agent>", mode="exec")


def _validate_sandbox_syntax_tree(syntax_tree: ast.AST) -> None:
    """Validate AST uses only explicitly allowed constructs and names.

    Disallows imports, dynamic execution helpers, and suspicious dunder access.

    Args:
        syntax_tree: Parsed AST to validate.

    Raises:
        ValueError: If the syntax tree contains unsupported constructs.
    """
    banned_node_types: tuple[type[ast.AST], ...] = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.With,
        ast.AsyncWith,
        ast.Try,
        ast.Raise,
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Delete,
    )
    banned_names = {
        "__import__",
        "open",
        "exec",
        "eval",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "help",
        "type",
        "object",
        "super",
        "breakpoint",
    }

    for node in ast.walk(syntax_tree):
        if isinstance(node, banned_node_types):
            raise ValueError(f"Unsupported syntax node: {type(node).__name__}")

        if isinstance(node, ast.Name) and node.id in banned_names:
            raise ValueError(f"Use of banned name: {node.id}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed.")

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("__")
        ):
            raise ValueError("Calling dunder names is not allowed.")


class _FinalOutputProxy(dict[str, object]):
    """Mutable placeholder used to detect whether ``final_output`` was touched."""

    def __init__(self) -> None:
        super().__init__()
        self._was_mutated = False

    @property
    def was_mutated(self) -> bool:
        return self._was_mutated

    def __setitem__(self, key: str, value: object) -> None:
        self._was_mutated = True
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        self._was_mutated = True
        super().__delitem__(key)

    def clear(self) -> None:
        self._was_mutated = True
        super().clear()

    def pop(self, key: str, default: object = None) -> object:
        self._was_mutated = True
        return super().pop(key, default)

    def popitem(self) -> tuple[str, object]:
        self._was_mutated = True
        return super().popitem()

    def setdefault(self, key: str, default: object = None) -> object:
        self._was_mutated = True
        return super().setdefault(key, default)

    def update(self, *args: object, **kwargs: object) -> None:
        self._was_mutated = True
        super().update(*args, **kwargs)


def _execute_compiled_code(
    *,
    compiled_code: CodeType,
    prompt: str,
    input_payload: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    allowed_tools: Sequence[_AllowedTool],
    tool_runtime: ToolRuntime,
    max_tool_calls: int,
    execution_timeout_seconds: int,
    validate_tool_input_schema: bool,
    tool_results: list[ToolResult],
) -> dict[str, object]:
    """Execute compiled code with strict runtime sandbox and tool guardrails.

    Enforces allowed tools, tool-call limits, optional schema validation, timeout
    controls, and JSON-serializable final output.

    Args:
        compiled_code: Compiled code object to execute.
        prompt: User prompt text.
        input_payload: Normalized run input payload mapping.
        request_id: Request identifier for tracing.
        dependencies: Dependency payload mapping.
        allowed_tools: Allowed tool list for this run.
        tool_runtime: Tool runtime used for tool invocation.
        max_tool_calls: Maximum allowed tool invocations.
        execution_timeout_seconds: Execution timeout in seconds.
        validate_tool_input_schema: Whether to validate tool input schemas.
        tool_results: List that will be populated with tool results.

    Returns:
        Final output mapping produced by the executed code.
    """
    allowed_tools_map = {tool.tool_name: tool for tool in allowed_tools}
    tool_call_count = 0

    def call_tool(tool_name: str, tool_input: Mapping[str, object]) -> dict[str, object]:
        """Invoke one allowed tool with validation, limits, and error mapping.

        Tool failures are surfaced as runtime exceptions so generated code cannot
        silently ignore failed calls.

        Args:
            tool_name: Tool name to invoke.
            tool_input: Tool input payload mapping.

        Returns:
            Tool output mapping.

        Raises:
            ValueError: If tool name or input is invalid.
            RuntimeError: If tool invocation fails or limits are exceeded.
        """
        nonlocal tool_call_count
        if not isinstance(tool_name, str):
            emit_guardrail_decision(
                guardrail="tool_call_name",
                decision="reject",
                reason="call_tool tool_name must be a string.",
                details={"tool_name": tool_name},
            )
            raise ValueError("call_tool tool_name must be a string.")
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name not in allowed_tools_map:
            emit_guardrail_decision(
                guardrail="tool_call_allowed",
                decision="reject",
                reason="tool not in allowed tool list",
                details={"tool_name": normalized_tool_name},
            )
            raise ValueError(f"Tool '{normalized_tool_name}' is not in the allowed tool list.")
        if tool_call_count >= max_tool_calls:
            emit_guardrail_decision(
                guardrail="tool_call_limit",
                decision="reject",
                reason="tool call limit exceeded",
                details={"max_tool_calls": max_tool_calls},
            )
            raise RuntimeError(f"Tool call limit exceeded ({max_tool_calls}).")

        if not isinstance(tool_input, Mapping):
            emit_guardrail_decision(
                guardrail="tool_call_input_type",
                decision="reject",
                reason="call_tool tool_input must be a mapping/object.",
            )
            raise ValueError("call_tool tool_input must be a mapping/object.")
        allowed_tool = allowed_tools_map[normalized_tool_name]
        normalized_tool_input = dict(tool_input)
        if not normalized_tool_input and allowed_tool.default_tool_input is not None:
            normalized_tool_input = dict(allowed_tool.default_tool_input)
        if validate_tool_input_schema:
            try:
                _validate_input_against_schema(
                    input_payload=normalized_tool_input,
                    input_schema=allowed_tool.input_schema,
                )
            except Exception as exc:
                emit_guardrail_decision(
                    guardrail="tool_input_schema",
                    decision="reject",
                    reason=str(exc),
                    details={"tool_name": normalized_tool_name},
                )
                raise

        tool_call_count += 1
        tool_result = tool_runtime.invoke(
            normalized_tool_name,
            normalized_tool_input,
            request_id=request_id,
            dependencies=dependencies,
        )
        tool_results.append(tool_result)
        if not tool_result.success:
            error = tool_result.error or "Unknown tool runtime error."
            raise RuntimeError(f"Tool '{normalized_tool_name}' failed: {error}")
        return dict(tool_result.output)

    sandbox_globals = {
        "__builtins__": {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "range": range,
            "enumerate": enumerate,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "sorted": sorted,
            "abs": abs,
            "all": all,
            "any": any,
        },
        "call_tool": call_tool,
    }
    sandbox_locals: dict[str, object] = {
        "prompt": prompt,
        "input_payload": dict(input_payload),
        "request_id": request_id,
        "dependencies": dict(dependencies),
        "allowed_tools": [tool.tool_name for tool in allowed_tools],
        "final_output": _FinalOutputProxy(),
    }

    with _execution_timeout(seconds=execution_timeout_seconds):
        exec(compiled_code, sandbox_globals, sandbox_locals)

    if not tool_results:
        emit_guardrail_decision(
            guardrail="tool_call_required",
            decision="reject",
            reason="generated code must call at least one tool",
        )
        raise ValueError("Generated code must call at least one tool.")

    raw_final_output = sandbox_locals.get("final_output")
    final_output: object | None
    if isinstance(raw_final_output, _FinalOutputProxy):
        final_output = dict(raw_final_output) if raw_final_output.was_mutated else None
    else:
        final_output = raw_final_output
    if final_output is None:
        # Local models occasionally omit the required assignment.
        # Fall back to the last successful tool output to keep execution usable.
        final_output = dict(tool_results[-1].output)
    if not isinstance(final_output, Mapping):
        emit_guardrail_decision(
            guardrail="final_output_type",
            decision="reject",
            reason="final_output must be a dict/object",
        )
        raise ValueError("Generated code must assign `final_output` to a dict/object.")

    # Force JSON-serializable dict-like result.
    serialized = json.loads(json.dumps(dict(final_output)))
    if not isinstance(serialized, dict):
        emit_guardrail_decision(
            guardrail="final_output_json",
            decision="reject",
            reason="final_output must serialize to a JSON object",
        )
        raise ValueError("final_output must serialize to a JSON object.")
    return serialized


@contextmanager
def _execution_timeout(*, seconds: int) -> Iterator[None]:
    """Enforce execution timeout via POSIX alarms when available.

    On platforms/threads where alarms are unavailable, the context manager
    degrades gracefully to no hard timeout.

    Args:
        seconds: Timeout duration in seconds.

    Yields:
        None.
    """
    if not hasattr(signal, "SIGALRM"):
        # Non-POSIX fallback: no hard timeout support.
        yield
        return

    def _on_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError(f"Execution exceeded timeout ({seconds}s).")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
    except ValueError:
        # Signals only work in the main thread; fallback to no hard timeout.
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_input_against_schema(
    *,
    input_payload: Mapping[str, object],
    input_schema: Mapping[str, object],
) -> None:
    """Validate tool input against constrained JSON-schema-like subset.

    Supports object type enforcement, required fields, property filtering, and
    primitive field type checks.

    Args:
        input_payload: Tool input payload mapping.
        input_schema: JSON-schema-like input schema mapping.

    Raises:
        ValueError: If the payload violates the schema constraints.
    """
    schema_type = input_schema.get("type")
    if isinstance(schema_type, str) and schema_type != "object":
        raise ValueError("Tool input schema type must be object.")

    required_fields = input_schema.get("required")
    if isinstance(required_fields, list):
        for field_name in required_fields:
            if isinstance(field_name, str) and field_name not in input_payload:
                raise ValueError(f"Missing required tool input field: {field_name}")

    raw_properties = input_schema.get("properties")
    properties = raw_properties if isinstance(raw_properties, Mapping) else {}
    additional_properties = input_schema.get("additionalProperties")
    if additional_properties is False:
        for field_name in input_payload:
            if field_name not in properties:
                raise ValueError(f"Unexpected tool input field: {field_name}")

    for field_name, field_schema in properties.items():
        if not isinstance(field_name, str) or not isinstance(field_schema, Mapping):
            continue
        if field_name not in input_payload:
            continue
        _validate_field_type(
            field_name=field_name,
            field_value=input_payload[field_name],
            field_schema=field_schema,
        )


def _validate_field_type(
    *,
    field_name: str,
    field_value: object,
    field_schema: Mapping[str, object],
) -> None:
    """Validate one input field value against supported schema type hints.

    Supported hints include string, number, integer, boolean, object, and array.

    Args:
        field_name: Field name being validated.
        field_value: Field value to validate.
        field_schema: Schema mapping for the field.

    Raises:
        ValueError: If the field value does not match the schema type.
    """
    field_type = field_schema.get("type")
    if not isinstance(field_type, str):
        return

    if field_type == "string" and not isinstance(field_value, str):
        raise ValueError(f"Field '{field_name}' must be a string.")
    if field_type == "number" and (
        isinstance(field_value, bool) or not isinstance(field_value, (int, float))
    ):
        raise ValueError(f"Field '{field_name}' must be a number.")
    if field_type == "integer" and (
        isinstance(field_value, bool) or not isinstance(field_value, int)
    ):
        raise ValueError(f"Field '{field_name}' must be an integer.")
    if field_type == "boolean" and not isinstance(field_value, bool):
        raise ValueError(f"Field '{field_name}' must be a boolean.")
    if field_type == "object" and not isinstance(field_value, Mapping):
        raise ValueError(f"Field '{field_name}' must be an object.")
    if field_type == "array" and not isinstance(field_value, list):
        raise ValueError(f"Field '{field_name}' must be an array.")


def _failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    generated_code: str,
    raw_generated_code: str | None = None,
) -> AgentResult:
    """Build a structured failure result for predictable error handling.

    Keeps failure output shape stable across validation and execution failures.

    Args:
        error: Error message describing the failure.
        model_response: Model response payload, if available.
        tool_results: Tool results collected before failure.
        request_id: Request identifier for tracing.
        dependencies: Dependency payload mapping.
        metadata: Additional metadata to include in the result.
        generated_code: Generated code text.
        raw_generated_code: Optional raw unnormalized generated code text.

    Returns:
        Agent result payload describing the failure.
    """
    output: dict[str, object] = {
        "error": error,
        "model_text": model_response.text if model_response is not None else "",
        "generated_code": generated_code,
        "final_output": {},
    }
    if raw_generated_code is not None:
        output["raw_generated_code"] = raw_generated_code
    return AgentResult(
        output=output,
        success=False,
        tool_results=list(tool_results),
        model_response=model_response,
        metadata={
            "request_id": request_id,
            "dependency_keys": sorted(dependencies.keys()),
            **dict(metadata),
        },
    )
