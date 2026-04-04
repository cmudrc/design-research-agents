"""# Optimization / Multi Step JSON Tool Calling 1d Optimization.

## Introduction
Practical Bayesian optimization motivates iterative search over expensive objective evaluations, while
Toolformer and Plan-and-Solve motivate explicit action/reason loops for model-guided exploration. This
example operationalizes that idea as a JSON tool-calling optimization workflow with traceable proposals and
evaluations.

.. note::

   This example's checked-in local ``LlamaCppServerLLMClient`` config uses a
   ``Qwen3-4B`` GGUF model. On lower-RAM machines, swap in a smaller local
   model or start with :doc:`../clients/ollama_local_client`.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``MultiStepAgent.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["MultiStepAgent.run(...)"]
    C --> D["optimization loop combines callable tools with explicit final answers"]
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

import design_research_agents as drag

# This checked-in local config uses a Qwen3-4B GGUF model to exercise a richer
# multi-step path. On lower-RAM machines, swap in a smaller local model or
# start with the lighter Ollama local client example first.
_EXAMPLE_LLAMA_CLIENT_KWARGS = {
    "model": "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "hf_model_repo_id": "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF",
    "api_model": "qwen3-4b-instruct-2507-q4km",
    "context_window": 8192,
    "startup_timeout_seconds": 240.0,
    "request_timeout_seconds": 240.0,
}


def _objective(x: float) -> float:
    return x * x


def main() -> None:
    """Optimize ``x^2`` from ``x=3`` by letting the LLM choose each tool step."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-optimization-json-tool-calling-design-001"
    tracer = drag.Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    initial_x = 3.0
    evaluation_history: list[dict[str, float]] = []

    def _evaluate(payload: Mapping[str, object]) -> dict[str, object]:
        raw_x = payload.get("x", initial_x)
        x_value = float(raw_x) if isinstance(raw_x, (int, float)) else initial_x
        f_x = _objective(x_value)
        evaluation_record = {"x": x_value, "f_x": f_x}
        evaluation_history.append(evaluation_record)
        best_record = min(evaluation_history, key=lambda record: record["f_x"])
        previous_record = evaluation_history[-2] if len(evaluation_history) > 1 else None
        return {
            "x": x_value,
            "f_x": f_x,
            "evaluations": len(evaluation_history),
            "previous_x": None if previous_record is None else previous_record["x"],
            "previous_f_x": None if previous_record is None else previous_record["f_x"],
            "best_x": best_record["x"],
            "best_objective": best_record["f_x"],
            "improved_best": best_record is evaluation_record,
            "history": list(evaluation_history),
        }

    # Run the optimization example using public runtime surfaces. Using this with statement will automatically
    # shut down the managed client and tool runtime when the example is done.
    with (
        drag.Toolbox(
            enable_core_tools=False,
            callable_tools=(
                drag.CallableToolConfig(
                    name="optimizer.evaluate",
                    description="Evaluate f(x) = x^2 at a proposed x and return the best observation so far.",
                    handler=_evaluate,
                    input_schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"x": {"type": "number"}},
                        "required": ["x"],
                    },
                ),
            ),
        ) as tools,
        drag.LlamaCppServerLLMClient(**_EXAMPLE_LLAMA_CLIENT_KWARGS) as llm_client,
    ):
        optimization_agent = drag.MultiStepAgent(
            mode="json",
            llm_client=llm_client,
            tool_runtime=tools,
            max_steps=6,
            # This example uses prompt guidance rather than tool-enforced step directions.
            tool_calling_system_prompt=(
                "You are solving a simple one-dimensional black-box minimization problem. "
                "Use optimizer.evaluate to test concrete x values, and rely on observed tool results instead "
                "of guessing numeric outcomes. Prefer a short, informative search that moves toward lower "
                "observed objective values, then emit final_answer once the best observed x is well-supported."
            ),
            tracer=tracer,
        )
        result = optimization_agent.run(
            prompt=(
                "Minimize the black-box function f(x). Begin by evaluating x=3. "
                "Use the observed results to choose a few better candidate x values, keeping the search efficient. "
                "When you have enough evidence, emit final_answer with exactly the keys best_x, "
                "best_objective, and evaluations, and use only values that came from tool observations."
            ),
            request_id=request_id,
        )

    # Print the results
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
