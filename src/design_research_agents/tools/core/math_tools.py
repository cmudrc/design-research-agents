"""Arithmetic tools used by demos and deterministic baselines."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable, Mapping

from design_research_agents.contracts.tools import (
    ToolCostHints,
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from ._helpers import get_str


def register_math_tools(source: InProcessToolSource) -> None:
    """Register calculator/math tools on an in-process source."""
    metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=False, filesystem_write=False),
        timeout_s=5,
        max_output_bytes=8_192,
        risky=False,
    )

    for name in ("math.eval", "calculator"):
        source.register_tool(
            spec=ToolSpec(
                name=name,
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
                metadata=metadata,
                permissions=("compute:arithmetic",),
                cost_hints=ToolCostHints(
                    token_cost_estimate=0,
                    latency_ms_estimate=1,
                    usd_cost_estimate=0.0,
                ),
            ),
            handler=_calculator_handler,
        )


def _calculator_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    del request_id, dependencies
    expression = get_str(input_dict, "expression").strip()
    if not expression:
        raise ValueError("calculator requires a non-empty 'expression'.")
    result = _safe_eval_arithmetic(expression)
    return {"expression": expression, "result": float(result)}


def _safe_eval_arithmetic(expression: str) -> float:
    """Evaluate strict arithmetic expression with a safe AST walker."""
    tree = ast.parse(expression, mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> int | float:
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
