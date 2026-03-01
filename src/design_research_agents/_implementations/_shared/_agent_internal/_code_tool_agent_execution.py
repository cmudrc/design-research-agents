"""Sandbox execution helpers for code-writing tool-calling agents."""

from __future__ import annotations

import ast
import json
import signal
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from types import CodeType

from design_research_agents._contracts._delegate import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import ToolResult, ToolRuntime
from design_research_agents._implementations._shared._agent_internal._result_builders import (
    build_failure_result,
)
from design_research_agents._tracing import emit_guardrail_decision

from ._code_tool_agent_parsing import AllowedTool

_FINAL_ANSWER_TOOL_NAME = "final_answer"
_MAX_SERIALIZED_PAYLOAD_BYTES = 65_536
_TRUNCATED_PAYLOAD_MESSAGE = f"[truncated: serialized payload exceeded {_MAX_SERIALIZED_PAYLOAD_BYTES} bytes]"


@dataclass(slots=True, frozen=True)
class _PayloadBounds:
    """Recursive normalization limits used for sandbox-visible payloads."""

    max_depth: int
    max_items: int
    max_string_chars: int


_SANDBOX_VIEW_BOUNDS = _PayloadBounds(max_depth=12, max_items=512, max_string_chars=262_144)
_SERIALIZATION_BOUNDS_PROFILES = (
    _PayloadBounds(max_depth=8, max_items=128, max_string_chars=8_192),
    _PayloadBounds(max_depth=6, max_items=32, max_string_chars=2_048),
    _PayloadBounds(max_depth=4, max_items=8, max_string_chars=512),
)


def compile_sandboxed_code(code_text: str) -> CodeType:
    """Validate and compile generated code under strict sandbox constraints.

    Args:
        code_text: Generated Python source emitted by the model.

    Returns:
        Compiled code object ready for sandbox execution.

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
        syntax_tree: Parsed syntax tree to validate before execution.

    Raises:
        ValueError: If the tree contains banned syntax or names.
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

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("__"):
            raise ValueError("Calling dunder names is not allowed.")


class _FinalOutputProxy(dict[str, object]):
    """Mutable placeholder used to detect whether ``final_output`` was touched."""

    def __init__(self) -> None:
        """Initialize the proxy with mutation tracking disabled."""
        super().__init__()
        self._was_mutated = False

    @property
    def was_mutated(self) -> bool:
        """Return whether any mutating dictionary API has been used.

        Returns:
            ``True`` after the proxy has been modified at least once.
        """
        return self._was_mutated

    def __setitem__(self, key: str, value: object) -> None:
        """Mark the proxy as mutated before setting one item.

        Args:
            key: Dictionary key to assign.
            value: Value to store under ``key``.
        """
        self._was_mutated = True
        super().__setitem__(key, value)

    def __delitem__(self, key: str) -> None:
        """Mark the proxy as mutated before deleting one item.

        Args:
            key: Dictionary key to remove.
        """
        self._was_mutated = True
        super().__delitem__(key)

    def clear(self) -> None:
        """Mark the proxy as mutated before clearing all items."""
        self._was_mutated = True
        super().clear()

    def pop(self, key: str, default: object = None) -> object:
        """Mark the proxy as mutated before popping one item.

        Args:
            key: Dictionary key to remove.
            default: Fallback value when ``key`` is absent.

        Returns:
            Removed value, or ``default`` when the key is absent.
        """
        self._was_mutated = True
        return super().pop(key, default)

    def popitem(self) -> tuple[str, object]:
        """Mark the proxy as mutated before removing the last item.

        Returns:
            Removed ``(key, value)`` pair.
        """
        self._was_mutated = True
        return super().popitem()

    def setdefault(self, key: str, default: object = None) -> object:
        """Mark the proxy as mutated before applying ``setdefault``.

        Args:
            key: Dictionary key to look up or initialize.
            default: Value to insert when ``key`` is absent.

        Returns:
            Existing or inserted value associated with ``key``.
        """
        self._was_mutated = True
        return super().setdefault(key, default)

    def update(self, *args: object, **kwargs: object) -> None:
        """Mark the proxy as mutated before applying a bulk update.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._was_mutated = True
        super().update(*args, **kwargs)


@dataclass(slots=True, frozen=True)
class CodeExecutionOutcome:
    """Structured outcome returned from sandboxed code execution."""

    final_output: dict[str, object]
    """Per-step output payload resolved after sandbox execution."""

    final_answer_called: bool
    """Whether the built-in ``final_answer()`` helper terminated the step."""

    used_tool_output_fallback: bool
    """Whether the last tool result was used as the step output fallback."""


class _FinalAnswerSignal(Exception):
    """Internal control-flow signal used to stop execution after ``final_answer()``."""


def _truncate_string(value: str, *, max_chars: int) -> tuple[str, bool]:
    """Clamp one string to a bounded preview while preserving original length."""
    if len(value) <= max_chars:
        return value, False
    preview_len = max(0, max_chars - 16)
    return f"{value[:preview_len]}...[truncated:{len(value)}]", True


def _normalize_json_value(
    value: object,
    *,
    bounds: _PayloadBounds,
    depth: int = 0,
) -> tuple[object, bool]:
    """Normalize arbitrary values into bounded JSON-safe structures."""
    if depth >= bounds.max_depth:
        return "[truncated: max depth reached]", True

    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        changed = False
        items = list(value.items())
        for index, (key, raw_val) in enumerate(items):
            if index >= bounds.max_items:
                normalized["__truncated_items__"] = len(items) - bounds.max_items
                changed = True
                break
            key_text = str(key)
            normalized_key, key_changed = _truncate_string(key_text, max_chars=128)
            normalized_value, value_changed = _normalize_json_value(raw_val, bounds=bounds, depth=depth + 1)
            normalized[normalized_key] = normalized_value
            changed = changed or key_changed or value_changed or normalized_key != key_text
        return normalized, changed

    if isinstance(value, (list, tuple, set)):
        sequence = list(value)
        normalized_items: list[object] = []
        changed = not isinstance(value, list)
        for index, item in enumerate(sequence):
            if index >= bounds.max_items:
                normalized_items.append(f"[truncated_items:{len(sequence) - bounds.max_items}]")
                changed = True
                break
            normalized_item, item_changed = _normalize_json_value(item, bounds=bounds, depth=depth + 1)
            normalized_items.append(normalized_item)
            changed = changed or item_changed
        return normalized_items, changed

    if isinstance(value, str):
        return _truncate_string(value, max_chars=bounds.max_string_chars)

    if isinstance(value, (int, float, bool)) or value is None:
        return value, False

    text_value = str(value)
    truncated_text, _ = _truncate_string(text_value, max_chars=bounds.max_string_chars)
    return truncated_text, True


def _serialized_size_bytes(value: object) -> int:
    """Return UTF-8 byte length for one JSON-serializable value."""
    return len(json.dumps(value, ensure_ascii=True, sort_keys=True).encode("utf-8"))


def _bounded_json_clone(
    value: object,
    *,
    bounds_profiles: Sequence[_PayloadBounds],
    fallback_root: str,
) -> tuple[object, bool]:
    """Return a bounded JSON clone using progressively tighter normalization profiles."""
    clamped = False
    last_serialized: object | None = None

    for index, bounds in enumerate(bounds_profiles):
        normalized, changed = _normalize_json_value(value, bounds=bounds)
        serialized = json.loads(json.dumps(normalized))
        last_serialized = serialized
        if _serialized_size_bytes(serialized) <= _MAX_SERIALIZED_PAYLOAD_BYTES:
            return serialized, clamped or changed or index > 0
        clamped = True

    if fallback_root == "mapping":
        return {
            "_truncated": True,
            "_reason": _TRUNCATED_PAYLOAD_MESSAGE,
        }, True
    if fallback_root == "sequence":
        return [_TRUNCATED_PAYLOAD_MESSAGE], True
    if isinstance(last_serialized, str):
        return _truncate_string(last_serialized, max_chars=256)[0], True
    return _TRUNCATED_PAYLOAD_MESSAGE, True


def _sandbox_visible_copy(value: object) -> object:
    """Return a JSON-safe copy for values exposed directly to sandboxed code."""
    normalized, _ = _normalize_json_value(value, bounds=_SANDBOX_VIEW_BOUNDS)
    return json.loads(json.dumps(normalized))


def _clamp_tool_result(tool_result: ToolResult) -> ToolResult:
    """Return a bounded clone safe for result storage and downstream serialization."""
    result_root = (
        "mapping"
        if isinstance(tool_result.result, Mapping)
        else "sequence"
        if isinstance(tool_result.result, (list, tuple, set))
        else "scalar"
    )
    result_payload, result_clamped = _bounded_json_clone(
        tool_result.result,
        bounds_profiles=_SERIALIZATION_BOUNDS_PROFILES,
        fallback_root=result_root,
    )
    metadata_payload, metadata_clamped = _bounded_json_clone(
        tool_result.metadata,
        bounds_profiles=_SERIALIZATION_BOUNDS_PROFILES,
        fallback_root="mapping",
    )
    warnings_payload, warnings_clamped = _bounded_json_clone(
        list(tool_result.warnings),
        bounds_profiles=_SERIALIZATION_BOUNDS_PROFILES,
        fallback_root="sequence",
    )

    metadata = dict(metadata_payload) if isinstance(metadata_payload, Mapping) else {}
    if result_clamped:
        metadata["result_clamped"] = True
    if metadata_clamped:
        metadata["metadata_clamped"] = True
    if warnings_clamped:
        metadata["warnings_clamped"] = True

    warnings = (
        tuple(str(item) for item in warnings_payload) if isinstance(warnings_payload, list) else tool_result.warnings
    )
    return ToolResult(
        tool_name=tool_result.tool_name,
        ok=tool_result.ok,
        result=result_payload,
        artifacts=tool_result.artifacts,
        warnings=warnings,
        error=tool_result.error,
        metadata=metadata,
    )


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
    tool_results.append(_clamp_tool_result(tool_result))
    if not tool_result.ok:
        error_message = tool_result.error.message if tool_result.error is not None else "Unknown tool runtime error."
        raise RuntimeError(f"Tool '{tool_name}' failed: {error_message}")

    if isinstance(tool_result.result, Mapping):
        return dict(tool_result.result)
    raise RuntimeError(f"Tool '{tool_name}' returned a non-object payload: {type(tool_result.result).__name__}.")


def _serialize_mapping_payload(
    *,
    raw_payload: object,
    type_guardrail: str,
    type_reason: str,
    type_error: str,
    json_guardrail: str,
    json_reason: str,
    json_error: str,
) -> dict[str, object]:
    """Validate and serialize one mapping payload into a JSON object.

    Args:
        raw_payload: Candidate payload to validate.
        type_guardrail: Guardrail id emitted for invalid payload types.
        type_reason: Guardrail reason for invalid payload types.
        type_error: Exception message for invalid payload types.
        json_guardrail: Guardrail id emitted for invalid JSON serialization shape.
        json_reason: Guardrail reason for invalid JSON serialization shape.
        json_error: Exception message for invalid JSON serialization shape.

    Returns:
        JSON-serializable mapping payload.

    Raises:
        ValueError: If the payload is not a JSON-serializable mapping.
    """
    if not isinstance(raw_payload, Mapping):
        emit_guardrail_decision(
            guardrail=type_guardrail,
            decision="reject",
            reason=type_reason,
        )
        raise ValueError(type_error)

    serialized, was_clamped = _bounded_json_clone(
        dict(raw_payload),
        bounds_profiles=_SERIALIZATION_BOUNDS_PROFILES,
        fallback_root="mapping",
    )
    if not isinstance(serialized, dict):
        raise AssertionError(f"Guardrail '{json_guardrail}' returned a non-mapping payload.")
    if was_clamped:
        emit_guardrail_decision(
            guardrail=f"{json_guardrail}_size",
            decision="clamp",
            reason=f"payload was clamped to stay within {_MAX_SERIALIZED_PAYLOAD_BYTES} bytes",
        )
    return serialized


def _resolve_execution_outcome(
    *,
    sandbox_locals: Mapping[str, object],
    tool_results: Sequence[ToolResult],
    final_answer_payload: dict[str, object] | None,
) -> CodeExecutionOutcome:
    """Resolve the canonical per-step outcome after sandbox execution."""
    if final_answer_payload is not None:
        return CodeExecutionOutcome(
            final_output=dict(final_answer_payload),
            final_answer_called=True,
            used_tool_output_fallback=False,
        )

    raw_final_output = sandbox_locals.get("final_output")
    if isinstance(raw_final_output, _FinalOutputProxy):
        explicit_final_output: object | None
        explicit_final_output = dict(raw_final_output) if raw_final_output.was_mutated else None
    else:
        explicit_final_output = raw_final_output

    if explicit_final_output is not None:
        if not tool_results:
            raise ValueError("Generated code must call at least one tool or use `final_answer({...})`.")
        return CodeExecutionOutcome(
            final_output=_serialize_mapping_payload(
                raw_payload=explicit_final_output,
                type_guardrail="final_output_type",
                type_reason="final_output must be a dict/object",
                type_error="Generated code must assign `final_output` to a dict/object.",
                json_guardrail="final_output_json",
                json_reason="final_output must serialize to a JSON object",
                json_error="final_output must serialize to a JSON object.",
            ),
            final_answer_called=False,
            used_tool_output_fallback=False,
        )

    if not tool_results:
        raise ValueError("Generated code must call at least one tool or use `final_answer({...})`.")
    if not isinstance(tool_results[-1].result, Mapping):
        raise ValueError("final_output fallback requires the last tool result to be an object.")
    return CodeExecutionOutcome(
        final_output=_serialize_mapping_payload(
            raw_payload=tool_results[-1].result,
            type_guardrail="final_output_type",
            type_reason="final_output must be a dict/object",
            type_error="Generated code must assign `final_output` to a dict/object.",
            json_guardrail="final_output_json",
            json_reason="final_output must serialize to a JSON object",
            json_error="final_output must serialize to a JSON object.",
        ),
        final_answer_called=False,
        used_tool_output_fallback=True,
    )


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
) -> CodeExecutionOutcome:
    """Execute compiled code with strict runtime sandbox and tool guardrails.

    Args:
        compiled_code: Prevalidated code object to execute in the sandbox.
        prompt: Original user prompt exposed to generated code.
        input_payload: Structured input payload exposed to generated code.
        request_id: Request id forwarded into tool-runtime calls.
        dependencies: Runtime dependency bag forwarded into tool calls.
        allowed_tools: Tool definitions the generated code is allowed to call.
        tool_runtime: Tool runtime used to execute normalized tool invocations.
        max_tool_calls: Hard limit on tool calls for this execution.
        execution_timeout_seconds: Hard timeout applied to sandbox execution when supported.
        validate_tool_input_schema: Whether tool inputs must satisfy declared schemas.
        tool_results: Mutable collector that accumulates tool results in order.

    Returns:
        Structured per-step outcome produced by the generated code.

    Raises:
        Exception: Propagated when sandbox validation or execution fails.
    """
    allowed_tools_map = {tool.tool_name: tool for tool in allowed_tools}
    if _FINAL_ANSWER_TOOL_NAME in allowed_tools_map:
        raise ValueError(f"Allowed tools cannot include reserved tool name '{_FINAL_ANSWER_TOOL_NAME}'.")
    tool_call_count = 0
    final_answer_payload: dict[str, object] | None = None

    def call_tool(tool_name: str, tool_input: object) -> dict[str, object]:
        """Sandbox-visible helper that validates and executes one tool call.

        Args:
            tool_name: Tool name requested by generated code.
            tool_input: Tool input payload requested by generated code.

        Returns:
            Normalized tool result payload.

        Raises:
            Exception: Propagated when guardrails or tool execution fail.
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

    def final_answer(payload: object) -> None:
        """Sandbox-visible helper that ends the step with a terminal payload."""
        nonlocal final_answer_payload
        final_answer_payload = _serialize_mapping_payload(
            raw_payload=payload,
            type_guardrail="final_answer_type",
            type_reason="final_answer payload must be a dict/object",
            type_error="final_answer() payload must be a dict/object.",
            json_guardrail="final_answer_json",
            json_reason="final_answer payload must serialize to a JSON object",
            json_error="final_answer() payload must serialize to a JSON object.",
        )
        raise _FinalAnswerSignal()

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
        "final_answer": final_answer,
    }
    sandbox_locals: dict[str, object] = {
        "prompt": _sandbox_visible_copy(prompt),
        "input_payload": _sandbox_visible_copy(dict(input_payload)),
        "request_id": _sandbox_visible_copy(request_id),
        "dependencies": _sandbox_visible_copy(dict(dependencies)),
        "allowed_tools": _sandbox_visible_copy([tool.tool_name for tool in allowed_tools]),
        "final_output": _FinalOutputProxy(),
    }

    with execution_timeout(seconds=execution_timeout_seconds), suppress(_FinalAnswerSignal):
        exec(compiled_code, sandbox_globals, sandbox_locals)

    if not tool_results and final_answer_payload is None:
        emit_guardrail_decision(
            guardrail="tool_call_required",
            decision="reject",
            reason="generated code must call at least one tool or use final_answer",
        )

    return _resolve_execution_outcome(
        sandbox_locals=sandbox_locals,
        tool_results=tool_results,
        final_answer_payload=final_answer_payload,
    )


@contextmanager
def execution_timeout(*, seconds: int) -> Iterator[None]:
    """Enforce execution timeout via POSIX alarms when available.

    Args:
        seconds: Maximum allowed execution time in seconds.

    Yields:
        Control to the wrapped execution block.
    """
    if not hasattr(signal, "SIGALRM"):
        # Non-POSIX fallback: no hard timeout support.
        emit_guardrail_decision(
            guardrail="execution_timeout_enforced",
            decision="warn",
            reason="hard timeout unavailable because SIGALRM is not supported on this platform",
            details={"seconds": seconds, "enforced": False, "fallback": "unsupported_platform"},
        )
        yield
        return

    def _on_timeout(signum: int, frame: object) -> None:
        """Raise a timeout error when the alarm signal fires.

        Args:
            signum: POSIX signal number received from ``SIGALRM``.
            frame: Current execution frame supplied by the signal handler.

        Raises:
            TimeoutError: Always raised when the alarm fires.
        """
        del signum, frame
        raise TimeoutError(f"Execution exceeded timeout ({seconds}s).")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
    except ValueError:
        # Signals only work in the main thread; fallback to no hard timeout.
        emit_guardrail_decision(
            guardrail="execution_timeout_enforced",
            decision="warn",
            reason="hard timeout unavailable because SIGALRM only works in the main thread",
            details={"seconds": seconds, "enforced": False, "fallback": "not_main_thread"},
        )
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
        input_payload: Tool input payload to validate.
        input_schema: Declared tool input schema subset.

    Raises:
        ValueError: If the payload violates the supported schema subset.
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
        field_name: Name of the field being validated.
        field_value: Field value supplied by generated code.
        field_schema: Schema fragment describing the expected field type.

    Raises:
        ValueError: If the field does not match its declared primitive type.
    """
    field_type = field_schema.get("type")
    if not isinstance(field_type, str):
        return

    if field_type == "string" and not isinstance(field_value, str):
        raise ValueError(f"Field '{field_name}' must be a string.")
    if field_type == "number" and (isinstance(field_value, bool) or not isinstance(field_value, (int, float))):
        raise ValueError(f"Field '{field_name}' must be a number.")
    if field_type == "integer" and (isinstance(field_value, bool) or not isinstance(field_value, int)):
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
        error: Human-readable error message for the failure.
        model_response: Optional model response associated with the failed step.
        tool_results: Tool results collected before the failure occurred.
        request_id: Request id associated with the failed execution.
        dependencies: Runtime dependency bag forwarded through the step.
        metadata: Additional execution metadata to preserve in the result.
        generated_code: Normalized generated code text, if available.
        raw_generated_code: Unnormalized generated code text before rewriting, if available.

    Returns:
        Structured failure result that matches the standard execution contract.
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
    "CodeExecutionOutcome",
    "compile_sandboxed_code",
    "execute_compiled_code",
    "failure_result",
]
