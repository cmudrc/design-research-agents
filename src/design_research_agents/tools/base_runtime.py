"""Base in-memory tool runtime with functional built-in tools."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping, Sequence

from design_research_agents.contracts.tools import ToolCostHints, ToolResult, ToolRuntime, ToolSpec

ToolHandler = Callable[[Mapping[str, object], Mapping[str, object]], Mapping[str, object]]


class BaseToolRuntime(ToolRuntime):
    """In-memory registry of tools for examples and local workflows."""

    def __init__(self) -> None:
        """Initialize runtime with built-in functional tools."""
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}
        # Register concrete default tools so examples run without external setup.
        self.register_tool(spec=create_calculator_tool_spec(), handler=_calculator_tool_handler)
        self.register_tool(spec=create_text_stats_tool_spec(), handler=_text_stats_tool_handler)

    def register_tool(self, *, spec: ToolSpec, handler: ToolHandler) -> None:
        """Register a tool specification and callable handler."""
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def list_tools(self) -> Sequence[ToolSpec]:
        """Return all registered tool specifications."""
        return tuple(self._specs.values())

    def invoke(
        self,
        tool_name: str,
        input_dict: Mapping[str, object],
        context: Mapping[str, object],
    ) -> ToolResult:
        """Invoke one registered tool by name."""
        handler = self._handlers.get(tool_name)
        if handler is None:
            # Return structured failure instead of raising to keep agent flows deterministic.
            return ToolResult(
                tool_name=tool_name,
                output={},
                success=False,
                error=f"Tool '{tool_name}' is not registered.",
            )

        try:
            # Normalize handler output to a plain dict for schema serialization.
            tool_output = dict(handler(input_dict, context))
        except Exception as exc:
            # Tool exceptions are surfaced as tool errors to avoid aborting the full run.
            return ToolResult(
                tool_name=tool_name,
                output={},
                success=False,
                error=str(exc),
            )

        return ToolResult(
            tool_name=tool_name,
            output=tool_output,
            success=True,
            metadata={"context_keys": sorted(context.keys())},
        )


def create_calculator_tool_spec() -> ToolSpec:
    """Create the default arithmetic calculator tool specification."""
    return ToolSpec(
        name="calculator_tool",
        description="Safely evaluates arithmetic expressions.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "expression": {"type": "string"},
                "result": {"type": "number"},
            },
            "required": ["expression", "result"],
        },
        permissions=("compute:arithmetic",),
        cost_hints=ToolCostHints(
            token_cost_estimate=0, latency_ms_estimate=1, usd_cost_estimate=0.0
        ),
    )


def create_text_stats_tool_spec() -> ToolSpec:
    """Create a text statistics tool specification."""
    return ToolSpec(
        name="text_stats_tool",
        description="Computes basic statistics for a text input.",
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "char_count": {"type": "integer"},
                "word_count": {"type": "integer"},
                "line_count": {"type": "integer"},
                "unique_word_count": {"type": "integer"},
            },
            "required": ["char_count", "word_count", "line_count", "unique_word_count"],
        },
        permissions=("analyze:text",),
        cost_hints=ToolCostHints(
            token_cost_estimate=0, latency_ms_estimate=1, usd_cost_estimate=0.0
        ),
    )


def _calculator_tool_handler(
    input_dict: Mapping[str, object],
    context: Mapping[str, object],
) -> Mapping[str, object]:
    """Evaluate one arithmetic expression safely."""
    del context
    expression = str(input_dict.get("expression", "")).strip()
    if not expression:
        raise ValueError("calculator_tool requires a non-empty 'expression'.")
    result = _safe_eval_arithmetic(expression)
    return {"expression": expression, "result": float(result)}


def _text_stats_tool_handler(
    input_dict: Mapping[str, object],
    context: Mapping[str, object],
) -> Mapping[str, object]:
    """Compute text statistics for one input string."""
    del context
    text = str(input_dict.get("text", ""))
    words = [word for word in text.split() if word]
    # Normalize punctuation/casing so unique counts are semantically meaningful.
    normalized_words = {word.strip(".,!?;:").lower() for word in words if word.strip(".,!?;:")}
    line_count = 0 if not text else text.count("\n") + 1
    return {
        "char_count": len(text),
        "word_count": len(words),
        "line_count": line_count,
        "unique_word_count": len(normalized_words),
    }


def _safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a strict arithmetic expression with a safe AST walker."""
    # Parse expression only in eval mode to block statements and assignments.
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> int | float:
    """Recursively evaluate one arithmetic AST node."""
    # Whitelists intentionally limit evaluation to deterministic arithmetic only.
    binary_operations: dict[
        type[ast.operator], Callable[[int | float, int | float], int | float]
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    unary_operations: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    if isinstance(node, ast.BinOp):
        binary_operation = binary_operations.get(type(node.op))
        if binary_operation is None:
            raise ValueError("Unsupported arithmetic operator.")
        # Evaluate children recursively before applying the resolved operator.
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return binary_operation(left, right)

    if isinstance(node, ast.UnaryOp):
        unary_operation = unary_operations.get(type(node.op))
        if unary_operation is None:
            raise ValueError("Unsupported unary operator.")
        return unary_operation(_eval_node(node.operand))

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    raise ValueError("Expression contains unsupported syntax.")
