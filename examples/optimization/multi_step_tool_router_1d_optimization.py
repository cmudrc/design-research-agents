"""Run traced LLM-driven optimization with callable increase/decrease tools.

Expected observations:
- ``best_seen`` captures the lowest objective encountered.
- ``history`` and ``objective_history`` show the optimization path.
- ``trace.trace_path`` points to emitted trace JSONL.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import (
    CallableTool,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    Toolbox,
    Tracer,
)


def _objective(x: float) -> float:
    return x * x


def main() -> None:
    """Optimize ``x^2`` from ``x=3`` by letting the LLM choose each tool step."""
    request_id = "example-optimization-router-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    initial_x = 3.0
    x_value = initial_x
    history: list[float] = [initial_x]
    objective_history: list[float] = [_objective(initial_x)]

    def _step(delta: float, payload: Mapping[str, object]) -> dict[str, object]:
        nonlocal x_value
        previous_x = x_value
        previous_f_x = _objective(previous_x)
        raw_step = payload.get("step", 1.0)
        step = float(raw_step) if isinstance(raw_step, (int, float)) else 1.0
        x_value = previous_x + delta * abs(step)
        f_x = _objective(x_value)
        history.append(x_value)
        objective_history.append(f_x)
        best_index = min(
            range(len(objective_history)),
            key=lambda index: objective_history[index],
        )
        return {
            "x": x_value,
            "f_x": f_x,
            "previous_x": previous_x,
            "previous_f_x": previous_f_x,
            "improved": f_x < previous_f_x,
            "best_x": history[best_index],
            "best_objective": objective_history[best_index],
            "history": list(history),
        }

    def _increase(payload: Mapping[str, object]) -> dict[str, object]:
        return _step(1.0, payload)

    def _decrease(payload: Mapping[str, object]) -> dict[str, object]:
        return _step(-1.0, payload)

    tools = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="optimizer.decrease_x",
                description="Decrease x by step (default 1).",
                handler=_decrease,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"step": {"type": "number"}},
                    "required": [],
                },
            ),
            CallableTool(
                name="optimizer.increase_x",
                description="Increase x by step (default 1).",
                handler=_increase,
                input_schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"step": {"type": "number"}},
                    "required": [],
                },
            ),
        ),
    )

    llm_client = LlamaCppServerLLMClient()
    try:
        agent = MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tools,
            max_steps=6,
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "Your job is to find a value of x to minimize the blackbox function f(x). "
                "Start at x=3 and use optimizer.increase_x or optimizer.decrease_x to search. "
                "Keep iterating until no one-step move improves the value."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()
        tools.close()

    output = result.output if isinstance(result.output, Mapping) else {}
    best_index = min(
        range(len(objective_history)),
        key=lambda index: objective_history[index],
    )
    memory = output.get("memory")
    payload = {
        "example": "optimization/multi_step_tool_router_1d_optimization.py",
        "agent": "MultiStepAgent(mode=json)",
        "success": result.success,
        "objective": "x^2",
        "final_output": result.final_output,
        "best_seen": {
            "best_x": history[best_index],
            "best_objective": objective_history[best_index],
            "best_history_index": best_index,
        },
        "terminated_reason": result.terminated_reason,
        "steps_executed": output.get("steps_executed"),
        "tool_results_count": len(result.tool_results),
        "memory_tail": memory[-6:] if isinstance(memory, list) else [],
        "history": history,
        "objective_history": objective_history,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
