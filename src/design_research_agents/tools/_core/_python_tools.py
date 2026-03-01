"""Restricted Python execution tooling for JSON action agents."""

from __future__ import annotations

import ast
import json
import math
import signal
import statistics
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from types import CodeType

from design_research_agents._contracts._tools import (
    ToolCostHints,
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str

_MAX_TIMEOUT_SECONDS = 10
_MAX_STDOUT_BYTES = 65_536
_BANNED_NODE_TYPES: tuple[type[ast.AST], ...] = (
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
_BANNED_NAMES = {
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
    "breakpoint",
}


def register_python_tools(source: InProcessToolSource) -> None:
    """Register strict in-process python sandbox tooling.

    Args:
        source: In-process tool source receiving the registrations.
    """
    source.register_tool(
        spec=ToolSpec(
            name="python.sandbox",
            description=(
                "Run guarded Python snippets for local numeric/data transformations with no "
                "imports, no filesystem access, and bounded execution time."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {"type": "string"},
                    "context": {"type": "object"},
                    "timeout_s": {"type": "integer"},
                    "result_key": {"type": "string"},
                },
                "required": ["code"],
            },
            output_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "result": {},
                    "stdout": {"type": "string"},
                    "truncated": {"type": "boolean"},
                    "execution_ms": {"type": "integer"},
                },
                "required": ["result", "stdout", "truncated", "execution_ms"],
            },
            metadata=ToolMetadata(
                source="core",
                side_effects=ToolSideEffects(),
                timeout_s=_MAX_TIMEOUT_SECONDS,
                max_output_bytes=_MAX_STDOUT_BYTES,
                risky=True,
            ),
            permissions=("compute:python_sandbox",),
            cost_hints=ToolCostHints(
                token_cost_estimate=0,
                latency_ms_estimate=20,
                usd_cost_estimate=0.0,
            ),
        ),
        handler=_python_sandbox_handler,
    )


def _python_sandbox_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    """Execute restricted Python code and return structured result payload.

    Args:
        input_dict: Tool input payload.
        request_id: Request identifier.
        dependencies: Additional runtime dependencies.

    Returns:
        Structured sandbox output payload.

    Raises:
        ValueError: When guardrails or payload contracts are violated.
        TimeoutError: When execution exceeds timeout.
    """
    del request_id, dependencies
    code = get_str(input_dict, "code").strip()
    if not code:
        raise ValueError("code is required.")

    context_raw = input_dict.get("context", {})
    if not isinstance(context_raw, Mapping):
        raise ValueError("context must be an object.")
    context_payload = dict(context_raw)

    timeout_s = get_int(input_dict, "timeout_s", default=2)
    if timeout_s < 1 or timeout_s > _MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_s must be within 1..{_MAX_TIMEOUT_SECONDS}.")

    result_key = get_str(input_dict, "result_key", default="result").strip()
    if not result_key:
        raise ValueError("result_key must be a non-empty string.")

    compiled = _compile_guarded_code(code)
    stdout_capture = StringIO()
    sandbox_globals = _build_sandbox_globals()
    sandbox_locals: dict[str, object] = {"context": context_payload, "result": None}

    started = time.perf_counter()
    with execution_timeout(seconds=timeout_s), redirect_stdout(stdout_capture):
        exec(compiled, sandbox_globals, sandbox_locals)
    execution_ms = round((time.perf_counter() - started) * 1000)

    raw_result = sandbox_locals.get(result_key)
    serialized_result = _coerce_json_object(raw_result)
    stdout_text, truncated = _truncate_stdout(
        stdout_capture.getvalue(),
        max_bytes=_MAX_STDOUT_BYTES,
    )
    return {
        "result": serialized_result,
        "stdout": stdout_text,
        "truncated": truncated,
        "execution_ms": execution_ms,
    }


def _compile_guarded_code(code: str) -> CodeType:
    """Compile code after validating AST guardrails.

    Args:
        code: Raw Python source text.

    Returns:
        Compiled code object.
    """
    syntax_tree = ast.parse(code, mode="exec")
    _validate_syntax_tree(syntax_tree)
    return compile(syntax_tree, filename="<python.sandbox>", mode="exec")


def _validate_syntax_tree(syntax_tree: ast.AST) -> None:
    """Validate that source AST follows sandbox restrictions.

    Args:
        syntax_tree: Parsed syntax tree.

    Raises:
        ValueError: When unsupported syntax is detected.
    """
    for node in ast.walk(syntax_tree):
        if isinstance(node, _BANNED_NODE_TYPES):
            raise ValueError(f"Unsupported syntax node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ValueError(f"Use of banned name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Dunder attribute access is not allowed.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("__"):
            raise ValueError("Calling dunder names is not allowed.")


def _build_sandbox_globals() -> dict[str, object]:
    """Build globals payload available to sandboxed snippets.

    Returns:
        Restricted globals dictionary.
    """
    safe_builtins = {
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
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
        "round": round,
        "all": all,
        "any": any,
        "print": print,
    }
    helpers = {
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "json_loads": json.loads,
        "json_dumps": json.dumps,
    }
    return {"__builtins__": safe_builtins, **helpers}


def _coerce_json_object(value: object) -> object:
    """Normalize one value into JSON-serializable form.

    Args:
        value: Raw value.

    Returns:
        JSON-serializable value.

    Raises:
        ValueError: When value is not JSON serializable.
    """
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("result value must be JSON serializable.") from exc


def _truncate_stdout(text: str, *, max_bytes: int) -> tuple[str, bool]:
    """Clamp stdout size to configured byte limit.

    Args:
        text: Captured stdout text.
        max_bytes: Maximum UTF-8 bytes allowed in output.

    Returns:
        Tuple ``(clipped_stdout, truncated)``.
    """
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return clipped, True


@contextmanager
def execution_timeout(*, seconds: int) -> Iterator[None]:
    """Enforce execution timeout using POSIX alarms when available.

    Args:
        seconds: Timeout in seconds.

    Yields:
        Context manager yield for guarded execution.
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _on_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise TimeoutError(f"Execution exceeded timeout ({seconds}s).")

    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _on_timeout)
    except ValueError:
        yield
        return
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


__all__ = ["execution_timeout", "register_python_tools"]
