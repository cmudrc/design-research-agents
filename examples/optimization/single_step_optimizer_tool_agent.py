"""Minimal single-step tool-calling example for one-dimensional optimization."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents import CallableTool, LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepJsonToolCallingAgent

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - optional dependency in examples
    minimize = None


def _search_1d(payload: Mapping[str, object]) -> dict[str, object]:
    """Minimize ``f(x)=x^2`` from one initial guess.

    Args:
        payload: Tool input mapping with optional ``initial_guess``.

    Returns:
        Optimizer summary payload.
    """
    raw_initial = payload.get("initial_guess", 7.0)
    initial_guess = float(raw_initial) if isinstance(raw_initial, (int, float)) else 7.0

    if minimize is None:
        best_x = 0.0
        return {
            "method": "closed_form_fallback",
            "scipy_available": False,
            "initial_guess": initial_guess,
            "best_x": best_x,
            "best_objective": best_x * best_x,
        }

    result = minimize(lambda x: x[0] ** 2, x0=[initial_guess], method="BFGS")
    best_x = float(result.x[0])
    return {
        "method": "scipy.optimize.minimize",
        "scipy_available": True,
        "initial_guess": initial_guess,
        "best_x": best_x,
        "best_objective": best_x * best_x,
        "success": bool(result.success),
        "iterations": int(getattr(result, "nit", 0)),
    }


def main() -> None:
    """Run a single-step agent that calls one optimizer tool."""
    tools = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="optimizer.search_1d",
                description="Run a scalar optimizer for f(x)=x^2 from initial_guess.",
                handler=_search_1d,
            ),
        ),
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = SingleStepJsonToolCallingAgent(llm_client=llm_client, tool_runtime=tools)
        result = agent.run(
            prompt=(
                "Call optimizer.search_1d with a good initial guess to minimize "
                "the function f of x equals x squared."
            ),
            request_id="example-single-step-optimizer-tool-agent-001",
        )
    finally:
        llm_client.close()

    payload = {
        "agent": "SingleStepJsonToolCallingAgent",
        "selected_tool": result.output.get("tool_name"),
        "tool_input": result.output.get("tool_input"),
        "tool_result": result.tool_results[0].result if result.tool_results else {},
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
