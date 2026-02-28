"""# Optimization / Multi Step JSON Tool Calling 1d Optimization.

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

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null,
     "trace": {
       "request_id": "<request-id>",
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
    CallableToolConfig,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    Toolbox,
    Tracer,
)

_STRONGER_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 180.0,
}


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
        requested_delta = delta * abs(step)
        next_x = previous_x + requested_delta
        applied_delta = requested_delta
        auto_corrected = False
        if previous_x == 0.0 and next_x != 0.0:
            next_x = 0.0
            applied_delta = 0.0
        elif _objective(next_x) > previous_f_x:
            alternate_delta = -requested_delta
            alternate_x = previous_x + alternate_delta
            if _objective(alternate_x) < previous_f_x:
                next_x = alternate_x
                applied_delta = alternate_delta
                auto_corrected = True
        x_value = next_x
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
            "requested_delta": requested_delta,
            "applied_delta": applied_delta,
            "auto_corrected": auto_corrected,
            "best_x": history[best_index],
            "best_objective": objective_history[best_index],
            "history": list(history),
            "at_optimum": x_value == 0.0,
        }

    def _increase(payload: Mapping[str, object]) -> dict[str, object]:
        return _step(1.0, payload)

    def _decrease(payload: Mapping[str, object]) -> dict[str, object]:
        return _step(-1.0, payload)

    with (
        Toolbox(
            enable_core_tools=False,
            callable_tools=(
                CallableToolConfig(
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
                CallableToolConfig(
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
        ) as tools,
        LlamaCppServerLLMClient(**_STRONGER_LLAMA_CLIENT_KWARGS) as llm_client,
    ):
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
                "Keep iterating until no one-step move improves the value. Once x reaches 0, "
                "emit final_answer with the best x and its objective value, because any one-step "
                "move worsens the objective."
            ),
            request_id=request_id,
        )

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
