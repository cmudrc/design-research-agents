"""Minimal multi-step ToolRouting example for one-dimensional optimization."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents import CallableTool, LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import MultiStepToolRouterAgent


def _objective(x: float) -> float:
    """Return the objective value for one scalar input.

    Args:
        x: Scalar parameter.

    Returns:
        Objective value.
    """
    return x * x


def _build_controller_diagnostics(step_outputs: object) -> dict[str, object]:
    """Summarize whether controller behavior matched monotonic improvement.

    Args:
        step_outputs: Agent step outputs from ``result.output["step_outputs"]``.

    Returns:
        Diagnostic flags describing STOP behavior and non-improving steps.
    """
    if not isinstance(step_outputs, list):
        return {
            "stop_step": None,
            "first_non_improving_step": None,
            "continued_after_non_improving_step": False,
        }

    stop_step: int | None = None
    previous_f_x: float | None = None
    first_non_improving_step: int | None = None
    continued_after_non_improving_step = False

    for raw_step in step_outputs:
        if not isinstance(raw_step, Mapping):
            continue
        action = raw_step.get("action")
        raw_step_number = raw_step.get("step")
        step_number = raw_step_number if isinstance(raw_step_number, int) else None

        if action == "STOP" and stop_step is None and step_number is not None:
            stop_step = step_number

        if action != "TOOL_CALL":
            continue
        tool_output = raw_step.get("tool_output")
        if not isinstance(tool_output, Mapping):
            continue
        raw_f_x = tool_output.get("f_x")
        if not isinstance(raw_f_x, (int, float)):
            continue
        f_x = float(raw_f_x)

        if (
            previous_f_x is not None
            and f_x >= previous_f_x
            and first_non_improving_step is None
            and step_number is not None
        ):
            first_non_improving_step = step_number
        if (
            first_non_improving_step is not None
            and step_number is not None
            and step_number > first_non_improving_step
        ):
            continued_after_non_improving_step = True
        previous_f_x = f_x

    return {
        "stop_step": stop_step,
        "first_non_improving_step": first_non_improving_step,
        "continued_after_non_improving_step": continued_after_non_improving_step,
    }


def main() -> None:
    """Run a multi-step optimizer that moves ``x`` with increase/decrease tools."""
    initial_x = 3.0
    state = {
        "x": initial_x,
        "history": [initial_x],
        "objective_history": [_objective(initial_x)],
    }

    def _step(delta: float, payload: Mapping[str, object]) -> dict[str, object]:
        """Apply one signed step update and report current objective state.

        Args:
            delta: Direction sign (+1 or -1).
            payload: Tool input that may include ``step``.

        Returns:
            Updated scalar state snapshot.
        """
        previous_x = float(state["x"])
        previous_f_x = _objective(previous_x)
        raw_step = payload.get("step", 1.0)
        step = float(raw_step) if isinstance(raw_step, (int, float)) else 1.0
        state["x"] = previous_x + delta * abs(step)
        x = float(state["x"])
        f_x = _objective(x)
        state["history"].append(x)
        state["objective_history"].append(f_x)
        best_index = min(
            range(len(state["objective_history"])),
            key=lambda index: float(state["objective_history"][index]),
        )

        return {
            "x": x,
            "f_x": f_x,
            "previous_x": previous_x,
            "previous_f_x": previous_f_x,
            "improved": f_x < previous_f_x,
            "best_x": float(state["history"][best_index]),
            "best_objective": float(state["objective_history"][best_index]),
            "history": list(state["history"]),
        }

    def _increase(payload: Mapping[str, object]) -> dict[str, object]:
        """Increase ``x`` by one step.

        Args:
            payload: Tool input mapping.

        Returns:
            Updated scalar state snapshot.
        """
        return _step(1.0, payload)

    def _decrease(payload: Mapping[str, object]) -> dict[str, object]:
        """Decrease ``x`` by one step.

        Args:
            payload: Tool input mapping.

        Returns:
            Updated scalar state snapshot.
        """
        return _step(-1.0, payload)

    tools = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="optimizer.decrease_x",
                description="Decrease x by step (default 1).",
                handler=_decrease,
            ),
            CallableTool(
                name="optimizer.increase_x",
                description="Increase x by step (default 1).",
                handler=_increase,
            ),
        ),
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = MultiStepToolRouterAgent(llm_client=llm_client, tool_runtime=tools, max_steps=6)
        result = agent.run(
            prompt=(
                "Minimize f(x)=x^2 from x=3 using optimizer.increase_x or "
                "optimizer.decrease_x with step=1. Stop when no one-step move "
                "improves the value. Use memory observations at each step, and "
                "when stopping return final_output with best_x and best_objective."
            ),
            request_id="example-multi-step-tool-router-1d-optimization-001",
        )
    finally:
        llm_client.close()

    objective_history = state["objective_history"]
    best_index = min(
        range(len(objective_history)),
        key=lambda index: float(objective_history[index]),
    )
    step_outputs = result.output.get("step_outputs")
    memory = result.output.get("memory")
    memory_tail = memory[-6:] if isinstance(memory, list) else []
    payload = {
        "agent": "MultiStepToolRouterAgent",
        "objective": "x^2",
        "final_output": result.output.get("final_output"),
        "best_seen": {
            "best_x": float(state["history"][best_index]),
            "best_objective": float(objective_history[best_index]),
            "best_history_index": best_index,
        },
        "terminated_reason": result.output.get("terminated_reason"),
        "steps_executed": result.output.get("steps_executed"),
        "controller_diagnostics": _build_controller_diagnostics(step_outputs),
        "step_outputs": step_outputs,
        "memory_tail": memory_tail,
        "history": state["history"],
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
