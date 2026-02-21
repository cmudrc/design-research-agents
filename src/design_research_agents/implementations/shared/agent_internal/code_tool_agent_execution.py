"""Sandbox execution helpers for code-writing tool-calling agents."""

from __future__ import annotations

import ast
import json
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from types import CodeType

from design_research_agents.contracts.agent import ExecutionResult
from design_research_agents.contracts.llm import LLMResponse
from design_research_agents.contracts.tools import ToolResult, ToolRuntime
from design_research_agents.implementations.shared.agent_internal.result_builders import (
    build_failure_result,
)
from design_research_agents.tracing import emit_guardrail_decision

from .code_tool_agent_parsing import AllowedTool


def compile_sandboxed_code(code_text: str) -> CodeType:
    """Validate and compile generated code under strict sandbox constraints.

    Args:
        code_text: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when validation or execution fails.
    """
    if not code_text:
        raise ValueError("Generated code is empty.")

    syntax_tree = ast.parse(code_text, mode="exec")
    validate_sandbox_syntax_tree(syntax_tree)
    return compile(syntax_tree, filename="<code_action_step_runner>", mode="exec")


def validate_sandbox_syntax_tree(syntax_tree: ast.AST) -> None:
    """Validate AST uses only explicitly allowed constructs and names.

    Args:
        syntax_tree: Input value for this parameter.

    Raises:
        Exception: Raised when validation or execution fails.
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
        """Execute init."""
        super().__init__()
        self._was_mutated = False

    @property
    def was_mutated(self) -> bool:
        """Execute was mutated.

        Returns:
            Computed return value.
        """
        return self._was_mutated

    def __setitem__(self, key: str, value: object) -> None:
        """Execute setitem.

        Args:
            key: Input value for this parameter.
            value: Input value for this parameter.
        """
        self._was_mutated = True
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Execute delitem.

        Args:
            key: Input value for this parameter.
        """
        self._was_mutated = True
        super().__delitem__(key)

    def clear(self) -> None:
        """Execute clear."""
        self._was_mutated = True
        super().clear()

    def pop(self, key: str, default: object = None) -> object:
        """Execute pop.

        Args:
            key: Input value for this parameter.
            default: Input value for this parameter.

        Returns:
            Computed return value.
        """
        self._was_mutated = True
        return super().pop(key, default)

    def popitem(self) -> tuple[str, object]:
        """Execute popitem.

        Returns:
            Computed return value.
        """
        self._was_mutated = True
        return super().popitem()

    def setdefault(self, key: str, default: object = None) -> object:
        """Execute setdefault.

        Args:
            key: Input value for this parameter.
            default: Input value for this parameter.

        Returns:
            Computed return value.
        """
        self._was_mutated = True
        return super().setdefault(key, default)

    def update(self, *args: object, **kwargs: object) -> None:
        """Execute update.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._was_mutated = True
        super().update(*args, **kwargs)


def _normalize_tool_name(
    *,
    tool_name: str,
    allowed_tools_map: Mapping[str, AllowedTool],
) -> str:
    """Validate and normalize one requested tool name.

    Args:
        tool_name: Raw tool name argument passed to ``call_tool``.
        allowed_tools_map: Allowed tool mapping keyed by tool name.

    Returns:
        Normalized tool name.

    Raises:
        ValueError: If the tool name is missing or not allowed.
    """
    if not isinstance(tool_name, str):
        emit_guardrail_decision(
            guardrail="tool_call_name",
            decision="reject",
            reason="call_tool tool_name must be a string.",
            details={"tool_name": tool_name},
        )
        raise ValueError("call_tool tool_name must be a string.")

    normalized_tool_name = tool_name.strip()
    if normalized_tool_name in allowed_tools_map:
        return normalized_tool_name

    emit_guardrail_decision(
        guardrail="tool_call_allowed",
        decision="reject",
        reason="tool not in allowed tool list",
        details={"tool_name": normalized_tool_name},
    )
    raise ValueError(f"Tool '{normalized_tool_name}' is not in the allowed tool list.")


def _enforce_tool_call_budget(*, tool_call_count: int, max_tool_calls: int) -> None:
    """Enforce per-step maximum tool call count.

    Args:
        tool_call_count: Number of tool calls executed so far.
        max_tool_calls: Maximum allowed tool calls for this execution.

    Raises:
        RuntimeError: If the call budget is exhausted.
    """
    if tool_call_count < max_tool_calls:
        return
    emit_guardrail_decision(
        guardrail="tool_call_limit",
        decision="reject",
        reason="tool call limit exceeded",
        details={"max_tool_calls": max_tool_calls},
    )
    raise RuntimeError(f"Tool call limit exceeded ({max_tool_calls}).")


def _normalize_tool_input(
    *,
    tool_input: object,
    allowed_tool: AllowedTool,
    validate_tool_input_schema: bool,
) -> dict[str, object]:
    """Validate and normalize ``call_tool`` input payload.

    Args:
        tool_input: Raw tool input payload provided by sandbox code.
        allowed_tool: Allowed-tool descriptor for the selected tool.
        validate_tool_input_schema: Whether to enforce input schema validation.

    Returns:
        Normalized tool input mapping.

    Raises:
        ValueError: If input payload type or schema validation is invalid.
    """
    if not isinstance(tool_input, Mapping):
        emit_guardrail_decision(
            guardrail="tool_call_input_type",
            decision="reject",
            reason="call_tool tool_input must be a mapping/object.",
        )
        raise ValueError("call_tool tool_input must be a mapping/object.")

    normalized_tool_input = dict(tool_input)
    if not normalized_tool_input and allowed_tool.default_tool_input is not None:
        normalized_tool_input = dict(allowed_tool.default_tool_input)
    if not validate_tool_input_schema:
        return normalized_tool_input

    try:
        validate_input_against_schema(
            input_payload=normalized_tool_input,
            input_schema=allowed_tool.input_schema,
        )
    except Exception as exc:
        emit_guardrail_decision(
            guardrail="tool_input_schema",
            decision="reject",
            reason=str(exc),
            details={"tool_name": allowed_tool.tool_name},
        )
        raise
    return normalized_tool_input


def _invoke_tool_runtime(
    *,
    tool_runtime: ToolRuntime,
    tool_name: str,
    tool_input: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    tool_results: list[ToolResult],
) -> dict[str, object]:
    """Invoke one runtime tool and normalize result payload.

    Args:
        tool_runtime: Tool runtime dependency.
        tool_name: Normalized tool name to invoke.
        tool_input: Normalized tool input mapping.
        request_id: Request id for runtime invocation.
        dependencies: Dependency mapping for runtime invocation.
        tool_results: Mutable tool result collector for this execution.

    Returns:
        Normalized tool result mapping.

    Raises:
        RuntimeError: If the tool fails or returns a non-mapping payload.
    """
    tool_result = tool_runtime.invoke(
        tool_name,
        dict(tool_input),
        request_id=request_id,
        dependencies=dependencies,
    )
    tool_results.append(tool_result)
    if not tool_result.ok:
        if tool_result.error is not None:
            error_message = tool_result.error.message
        else:
            error_message = "Unknown tool runtime error."
        raise RuntimeError(f"Tool '{tool_name}' failed: {error_message}")

    if isinstance(tool_result.result, Mapping):
        return dict(tool_result.result)
    raise RuntimeError(
        f"Tool '{tool_name}' returned a non-object payload: {type(tool_result.result).__name__}."
    )


def _serialize_final_output(
    *,
    sandbox_locals: Mapping[str, object],
    tool_results: Sequence[ToolResult],
) -> dict[str, object]:
    """Resolve and serialize the final output mapping from sandbox locals.

    Args:
        sandbox_locals: Sandbox local variable mapping after code execution.
        tool_results: Tool results collected during execution.

    Returns:
        JSON-serializable final output mapping.

    Raises:
        ValueError: If final output is missing or not object-serializable.
    """
    raw_final_output = sandbox_locals.get("final_output")
    if isinstance(raw_final_output, _FinalOutputProxy):
        final_output: object | None
        final_output = dict(raw_final_output) if raw_final_output.was_mutated else None
    else:
        final_output = raw_final_output

    if final_output is None:
        if not tool_results:
            raise ValueError("Generated code must call at least one tool.")
        if not isinstance(tool_results[-1].result, Mapping):
            raise ValueError("final_output fallback requires the last tool result to be an object.")
        final_output = dict(tool_results[-1].result)

    if not isinstance(final_output, Mapping):
        emit_guardrail_decision(
            guardrail="final_output_type",
            decision="reject",
            reason="final_output must be a dict/object",
        )
        raise ValueError("Generated code must assign `final_output` to a dict/object.")

    serialized = json.loads(json.dumps(dict(final_output)))
    if isinstance(serialized, dict):
        return serialized

    emit_guardrail_decision(
        guardrail="final_output_json",
        decision="reject",
        reason="final_output must serialize to a JSON object",
    )
    raise ValueError("final_output must serialize to a JSON object.")


def execute_compiled_code(
    *,
    compiled_code: CodeType,
    prompt: str,
    input_payload: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
    allowed_tools: Sequence[AllowedTool],
    tool_runtime: ToolRuntime,
    max_tool_calls: int,
    execution_timeout_seconds: int,
    validate_tool_input_schema: bool,
    tool_results: list[ToolResult],
) -> dict[str, object]:
    """Execute compiled code with strict runtime sandbox and tool guardrails.

    Args:
        compiled_code: Input value for this parameter.
        prompt: Input value for this parameter.
        input_payload: Input value for this parameter.
        request_id: Input value for this parameter.
        dependencies: Input value for this parameter.
        allowed_tools: Input value for this parameter.
        tool_runtime: Input value for this parameter.
        max_tool_calls: Input value for this parameter.
        execution_timeout_seconds: Input value for this parameter.
        validate_tool_input_schema: Input value for this parameter.
        tool_results: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when validation or execution fails.
    """
    allowed_tools_map = {tool.tool_name: tool for tool in allowed_tools}
    tool_call_count = 0

    def call_tool(tool_name: str, tool_input: object) -> dict[str, object]:
        """Execute call tool.

        Args:
            tool_name: Input value for this parameter.
            tool_input: Input value for this parameter.

        Returns:
            Computed return value.

        Raises:
            Exception: Raised when validation or execution fails.
        """
        nonlocal tool_call_count
        normalized_tool_name = _normalize_tool_name(
            tool_name=tool_name,
            allowed_tools_map=allowed_tools_map,
        )
        _enforce_tool_call_budget(
            tool_call_count=tool_call_count,
            max_tool_calls=max_tool_calls,
        )
        allowed_tool = allowed_tools_map[normalized_tool_name]
        normalized_tool_input = _normalize_tool_input(
            tool_input=tool_input,
            allowed_tool=allowed_tool,
            validate_tool_input_schema=validate_tool_input_schema,
        )

        tool_call_count += 1
        return _invoke_tool_runtime(
            tool_runtime=tool_runtime,
            tool_name=normalized_tool_name,
            tool_input=normalized_tool_input,
            request_id=request_id,
            dependencies=dependencies,
            tool_results=tool_results,
        )

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

    with execution_timeout(seconds=execution_timeout_seconds):
        exec(compiled_code, sandbox_globals, sandbox_locals)

    if not tool_results:
        emit_guardrail_decision(
            guardrail="tool_call_required",
            decision="reject",
            reason="generated code must call at least one tool",
        )
        raise ValueError("Generated code must call at least one tool.")

    return _serialize_final_output(sandbox_locals=sandbox_locals, tool_results=tool_results)


@contextmanager
def execution_timeout(*, seconds: int) -> Iterator[None]:
    """Enforce execution timeout via POSIX alarms when available.

    Args:
        seconds: Input value for this parameter.

    Yields:
        The yielded values.
    """
    if not hasattr(signal, "SIGALRM"):
        # Non-POSIX fallback: no hard timeout support.
        yield
        return

    def _on_timeout(signum: int, frame: object) -> None:
        """Execute on timeout.

        Args:
            signum: Input value for this parameter.
            frame: Input value for this parameter.

        Raises:
            Exception: Raised when validation or execution fails.
        """
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


def validate_input_against_schema(
    *,
    input_payload: Mapping[str, object],
    input_schema: Mapping[str, object],
) -> None:
    """Validate tool input against constrained JSON-schema-like subset.

    Args:
        input_payload: Input value for this parameter.
        input_schema: Input value for this parameter.

    Raises:
        Exception: Raised when validation or execution fails.
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
        validate_field_type(
            field_name=field_name,
            field_value=input_payload[field_name],
            field_schema=field_schema,
        )


def validate_field_type(
    *,
    field_name: str,
    field_value: object,
    field_schema: Mapping[str, object],
) -> None:
    """Validate one input field value against supported schema type hints.

    Args:
        field_name: Input value for this parameter.
        field_value: Input value for this parameter.
        field_schema: Input value for this parameter.

    Raises:
        Exception: Raised when validation or execution fails.
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


def failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    tool_results: Sequence[ToolResult],
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    generated_code: str,
    raw_generated_code: str | None = None,
) -> ExecutionResult:
    """Build a structured failure result for predictable error handling.

    Args:
        error: Input value for this parameter.
        model_response: Input value for this parameter.
        tool_results: Input value for this parameter.
        request_id: Input value for this parameter.
        dependencies: Input value for this parameter.
        metadata: Input value for this parameter.
        generated_code: Input value for this parameter.
        raw_generated_code: Input value for this parameter.

    Returns:
        Computed return value.
    """
    output: dict[str, object] = {
        "error": error,
        "model_text": model_response.text if model_response is not None else "",
        "generated_code": generated_code,
        "final_output": {},
    }
    if raw_generated_code is not None:
        output["raw_generated_code"] = raw_generated_code
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=tool_results,
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )


__all__ = [
    "compile_sandboxed_code",
    "execute_compiled_code",
    "failure_result",
]
