r"""# Optimization / Multi Step JSON Tool Calling 1d Optimization.

## Introduction
Practical Bayesian optimization motivates iterative search over expensive objective evaluations, while
Toolformer and Plan-and-Solve motivate explicit action/reason loops for model-guided exploration. This
example operationalizes that idea as a JSON tool-calling optimization workflow with traceable proposals and
evaluations.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MultiStepAgent.run(...)"]
    C --> D["optimization loop combines callable tools with continuation control"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results
Example output captured with ``DRA_EXAMPLE_LLM_MODE=deterministic``
(timestamps, durations, and trace filenames vary by run):

.. code-block:: text

   {
     "agent": "MultiStepAgent(mode=json)",
     "best_seen": {
       "best_history_index": 3,
       "best_objective": 0.0,
       "best_x": 0.0
     },
     "example": "optimization/multi_step_json_tool_calling_1d_optimization.py",
     "final_output": {
       "best_objective": 0.0,
       "best_x": 0.0,
       "f_x": 0.0,
       "history": [
         3.0,
         2.0,
         1.0,
         0.0
       ],
       "improved": true,
       "previous_f_x": 1.0,
       "previous_x": 1.0,
       "x": 0.0
     },
     "history": [
       3.0,
       2.0,
       1.0,
       0.0
     ],
     "memory_tail": [
       {
         "kind": "action",
         "step": 2,
         "tool_input": {
           "step": 1
         },
         "tool_name": "optimizer.decrease_x"
       },
       {
         "error": "Step execution failed.",
         "final_output": {
           "best_objective": 1.0,
           "best_x": 1.0,
           "f_x": 1.0,
           "history": [
             3.0,
             2.0,
             1.0
           ],
           "improved": true,
           "previous_f_x": 4.0,
           "previous_x": 2.0,
           "x": 1.0
         },
         "kind": "observation",
         "step": 2,
         "success": true
       },
       {
         "continue": true,
         "kind": "thought",
         "source": "model",
         "step": 3,
         "text": "One more decrease should reach zero."
       },
       {
         "kind": "action",
         "step": 3,
         "tool_input": {
           "step": 1
         },
         "tool_name": "optimizer.decrease_x"
       },
       {
         "error": "Step execution failed.",
         "final_output": {
           "best_objective": 0.0,
           "best_x": 0.0,
           "f_x": 0.0,
           "history": [
             3.0,
             2.0,
             1.0,
             0.0
           ],
           "improved": true,
           "previous_f_x": 1.0,
           "previous_x": 1.0,
           "x": 0.0
         },
         "kind": "observation",
         "step": 3,
         "success": true
       },
       {
         "continue": false,
         "kind": "thought",
         "source": "model",
         "step": 4,
         "text": "No better one-step move remains."
       }
     ],
     "objective": "x^2",
     "objective_history": [
       9.0,
       4.0,
       1.0,
       0.0
     ],
     "steps_executed": 3,
     "success": true,
     "terminated_reason": "continuation_stopped:model",
     "tool_results_count": 3,
     "trace": {
       "request_id": "example-optimization-json-tool-calling-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
     }
   }


## References
- `Practical Bayesian Optimization of Machine Learning Algorithms <https://arxiv.org/abs/1012.2599>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
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
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-optimization-json-tool-calling-design-001"
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
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        llm_client.close()
        tools.close()

    best_index = min(
        range(len(objective_history)),
        key=lambda index: objective_history[index],
    )
    memory = result.output_list("memory")
    summary = result.summary(
        details={
            "agent": "MultiStepAgent(mode=json)",
            "objective": "x^2",
            "best_seen": {
                "best_x": history[best_index],
                "best_objective": objective_history[best_index],
                "best_history_index": best_index,
            },
            "steps_executed": result.output_value("steps_executed"),
            "tool_results_count": len(result.tool_results),
            "memory_tail": memory[-6:] if isinstance(memory, list) else [],
            "history": history,
            "objective_history": objective_history,
        },
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
