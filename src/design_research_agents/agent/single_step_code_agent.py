"""Single-step code-writing agent with strict sandboxed tool execution."""

from __future__ import annotations

import ast
import json
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import CodeType

from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec


@dataclass(slots=True, frozen=True)
class _AllowedTool:
    """Normalized input-allowed tool definition."""

    tool_name: str
    description: str
    input_schema: dict[str, object]
    default_tool_input: dict[str, object] | None = None


class SingleStepCodeAgent(Agent):
    """Agent that writes and executes one sandboxed Python action program."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        model: str = "gpt-4o-mini",
        max_tool_calls: int = 5,
        execution_timeout_seconds: int = 5,
        validate_tool_input_schema: bool = False,
    ) -> None:
        """Initialize a single-step code agent.

        Args:
            llm_client: LLM client used to generate one action program.
            tool_runtime: Tool runtime used for allowed tool invocation.
            model: Model name used for LLM calls.
            max_tool_calls: Maximum number of tool calls allowed in one run.
            execution_timeout_seconds: Max wall-clock seconds for executing generated code.
            validate_tool_input_schema: Whether to validate tool args against tool input schemas.
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

    def run(self, input: Mapping[str, object], context: Mapping[str, object]) -> AgentResult:
        """Run one LLM generation and one sandboxed code execution pass."""
        runtime_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        allowed_tools = _extract_allowed_tools(input_payload=input, runtime_specs=runtime_specs)
        if not allowed_tools:
            return _failure_result(
                error=(
                    "Input must include at least one allowed tool in `tools` (or `alternatives`)."
                ),
                model_response=None,
                tool_results=[],
                context=context,
                metadata={"stage": "input_validation"},
                generated_code="",
            )

        max_tool_calls = _extract_positive_int(
            input_payload=input,
            key="max_tool_calls",
            default_value=self._max_tool_calls,
        )
        execution_timeout_seconds = _extract_positive_int(
            input_payload=input,
            key="execution_timeout_seconds",
            default_value=self._execution_timeout_seconds,
        )
        validate_tool_input_schema = _extract_boolean(
            input_payload=input,
            key="validate_tool_input_schema",
            default_value=self._validate_tool_input_schema,
        )
        prompt = _extract_prompt(input)

        llm_response = self._generate_code(prompt=prompt, allowed_tools=allowed_tools)
        code_text = _extract_python_code(llm_response.text)

        try:
            compiled_code = _compile_sandboxed_code(code_text)
        except Exception as exc:
            return _failure_result(
                error=f"Generated code failed sandbox validation: {exc}",
                model_response=llm_response,
                tool_results=[],
                context=context,
                metadata={"stage": "code_validation", "generated_code": code_text},
                generated_code=code_text,
            )

        tool_results: list[ToolResult] = []
        try:
            final_output = _execute_compiled_code(
                compiled_code=compiled_code,
                prompt=prompt,
                input_payload=input,
                context=context,
                allowed_tools=allowed_tools,
                tool_runtime=self._tool_runtime,
                max_tool_calls=max_tool_calls,
                execution_timeout_seconds=execution_timeout_seconds,
                validate_tool_input_schema=validate_tool_input_schema,
                tool_results=tool_results,
            )
        except Exception as exc:
            return _failure_result(
                error=f"Sandboxed code execution failed: {exc}",
                model_response=llm_response,
                tool_results=tool_results,
                context=context,
                metadata={
                    "stage": "code_execution",
                    "generated_code": code_text,
                    "max_tool_calls": max_tool_calls,
                    "execution_timeout_seconds": execution_timeout_seconds,
                },
                generated_code=code_text,
            )

        output: dict[str, object] = {
            "model_text": llm_response.text,
            "generated_code": code_text,
            "final_output": final_output,
            "tool_name": tool_results[-1].tool_name if tool_results else None,
            "tool_output": tool_results[-1].output if tool_results else {},
        }
        return AgentResult(
            output=output,
            success=all(tool_result.success for tool_result in tool_results),
            tool_results=tool_results,
            model_response=llm_response,
            metadata={
                "context_keys": sorted(context.keys()),
                "code_execution": {
                    "allowed_tools": [tool.tool_name for tool in allowed_tools],
                    "tool_call_count": len(tool_results),
                    "max_tool_calls": max_tool_calls,
                    "execution_timeout_seconds": execution_timeout_seconds,
                    "validate_tool_input_schema": validate_tool_input_schema,
                },
            },
        )

    def run_stream(
        self,
        input: Mapping[str, object],
        context: Mapping[str, object],
    ) -> Iterator[AgentStreamEvent]:
        """Stream a deterministic event pair around ``run``."""
        result = self.run(input, context)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)

    def _generate_code(self, *, prompt: str, allowed_tools: Sequence[_AllowedTool]) -> LLMResponse:
        """Generate one Python code action from the model."""
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
        plan_prompt = "\n".join(
            [
                "Write Python code that solves the request with one or more `call_tool` calls.",
                "Rules:",
                "- Use only call_tool(tool_name: str, tool_input: dict).",
                "- Use only allowed tools listed below.",
                "- Assign the final result to `final_output` as a dict.",
                "- Always end with a line that assigns `final_output`.",
                "- If your last tool call returns the answer as a dict, set `final_output` to it.",
                "- Return code only. No markdown. No prose.",
                "",
                "Required output pattern:",
                'final_output = {"key": "value"}',
                "",
                "Allowed tools:",
                tools_text,
                "",
                "User request:",
                prompt,
            ]
        )
        llm_params = LLMChatParams(
            provider_options={"agent": "SingleStepCodeAgent"},
        )
        messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a practical coding assistant for a strict Python sandbox. "
                    "Output only valid Python code and always assign final_output to a dict."
                ),
            ),
            LLMMessage(role="user", content=plan_prompt),
        ]
        return self._llm_client.chat(messages, model=self._model, params=llm_params)


def _extract_allowed_tools(
    *,
    input_payload: Mapping[str, object],
    runtime_specs: Mapping[str, ToolSpec],
) -> list[_AllowedTool]:
    """Extract tools explicitly allowed by input payload."""
    raw_tools = input_payload.get("tools", input_payload.get("alternatives"))
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        return []

    allowed_tools: list[_AllowedTool] = []
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
        allowed_tools.append(
            _AllowedTool(
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                default_tool_input=default_tool_input,
            )
        )

    deduped: dict[str, _AllowedTool] = {}
    for allowed_tool in allowed_tools:
        deduped[allowed_tool.tool_name] = allowed_tool
    return list(deduped.values())


def _extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from the input payload."""
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
    """Extract positive integer option from input payload."""
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
    """Extract boolean option from input payload."""
    raw_value = input_payload.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def _extract_python_code(raw_model_text: str) -> str:
    """Extract Python code from model output text."""
    fenced_match = _match_fenced_code_block(raw_model_text)
    if fenced_match is not None:
        return fenced_match.strip()
    return raw_model_text.strip()


def _match_fenced_code_block(raw_text: str) -> str | None:
    """Return the first fenced code block if present."""
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


def _compile_sandboxed_code(code_text: str) -> CodeType:
    """Validate and compile generated code under strict sandbox constraints."""
    if not code_text:
        raise ValueError("Generated code is empty.")

    syntax_tree = ast.parse(code_text, mode="exec")
    _validate_sandbox_syntax_tree(syntax_tree)
    return compile(syntax_tree, filename="<single_step_code_agent>", mode="exec")


def _validate_sandbox_syntax_tree(syntax_tree: ast.AST) -> None:
    """Validate that AST only uses explicitly allowed constructs."""
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

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise ValueError("Dunder attribute access is not allowed.")

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id.startswith("__"):
                raise ValueError("Calling dunder names is not allowed.")


def _execute_compiled_code(
    *,
    compiled_code: CodeType,
    prompt: str,
    input_payload: Mapping[str, object],
    context: Mapping[str, object],
    allowed_tools: Sequence[_AllowedTool],
    tool_runtime: ToolRuntime,
    max_tool_calls: int,
    execution_timeout_seconds: int,
    validate_tool_input_schema: bool,
    tool_results: list[ToolResult],
) -> dict[str, object]:
    """Execute compiled code with a strict runtime sandbox."""
    allowed_tools_map = {tool.tool_name: tool for tool in allowed_tools}
    tool_call_count = 0

    def call_tool(tool_name: str, tool_input: Mapping[str, object]) -> dict[str, object]:
        """Invoke one allowed tool with validated input."""
        nonlocal tool_call_count
        if not isinstance(tool_name, str):
            raise ValueError("call_tool tool_name must be a string.")
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name not in allowed_tools_map:
            raise ValueError(f"Tool '{normalized_tool_name}' is not in the allowed tool list.")
        if tool_call_count >= max_tool_calls:
            raise RuntimeError(f"Tool call limit exceeded ({max_tool_calls}).")

        if not isinstance(tool_input, Mapping):
            raise ValueError("call_tool tool_input must be a mapping/object.")
        allowed_tool = allowed_tools_map[normalized_tool_name]
        normalized_tool_input = dict(tool_input)
        if not normalized_tool_input and allowed_tool.default_tool_input is not None:
            normalized_tool_input = dict(allowed_tool.default_tool_input)
        if validate_tool_input_schema:
            _validate_input_against_schema(
                input_payload=normalized_tool_input,
                input_schema=allowed_tool.input_schema,
            )

        tool_call_count += 1
        tool_result = tool_runtime.invoke(normalized_tool_name, normalized_tool_input, context)
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
        "context": dict(context),
        "allowed_tools": [tool.tool_name for tool in allowed_tools],
        "final_output": None,
    }

    with _execution_timeout(seconds=execution_timeout_seconds):
        exec(compiled_code, sandbox_globals, sandbox_locals)

    if not tool_results:
        raise ValueError("Generated code must call at least one tool.")

    final_output = sandbox_locals.get("final_output")
    if final_output is None:
        # Local models occasionally omit the required assignment.
        # Fall back to the last successful tool output to keep execution usable.
        final_output = dict(tool_results[-1].output)
    if not isinstance(final_output, Mapping):
        raise ValueError("Generated code must assign `final_output` to a dict/object.")

    # Force JSON-serializable dict-like result.
    serialized = json.loads(json.dumps(dict(final_output)))
    if not isinstance(serialized, dict):
        raise ValueError("final_output must serialize to a JSON object.")
    return serialized


@contextmanager
def _execution_timeout(*, seconds: int) -> Iterator[None]:
    """Context manager enforcing execution timeout using POSIX alarms when available."""
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
    """Validate tool input against a small JSON-schema-like subset."""
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
    """Validate one field value type against supported schema type hints."""
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
    context: Mapping[str, object],
    metadata: Mapping[str, object],
    generated_code: str,
) -> AgentResult:
    """Build a structured failure result for predictable error handling."""
    output: dict[str, object] = {
        "error": error,
        "model_text": model_response.text if model_response is not None else "",
        "generated_code": generated_code,
        "final_output": {},
    }
    return AgentResult(
        output=output,
        success=False,
        tool_results=list(tool_results),
        model_response=model_response,
        metadata={
            "context_keys": sorted(context.keys()),
            **dict(metadata),
        },
    )
