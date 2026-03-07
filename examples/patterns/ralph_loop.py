"""# Patterns / Ralph Loop.

## Introduction
Nominal-team style deliberation motivates structured role handoffs where proposal, critique, and evaluation
signals stay explicit. This example demonstrates a dynamic-role Ralph loop with evaluator-threshold stopping
and deterministic output traces.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build role delegates that emit deterministic proposal/critique/evaluation payloads.
3. Execute ``RalphLoopPattern.run(...)`` with typed ``LoopConfig`` and fixed ``request_id``.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["RalphLoopPattern.run(...)"]
    C --> D["role batch executes proposer/critic/evaluator each round"]
    C --> E["evaluator score compared to consensus threshold"]
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
- `Nominal Group Technique <https://doi.org/10.1177/104649648300400202>`_
- `Self-Refine <https://arxiv.org/abs/2303.17651>`_
- `Reflexion <https://arxiv.org/abs/2303.11366>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import ExecutionResult, Tracer
from design_research_agents.patterns import RalphLoopPattern


class _StaticRoleDelegate:
    """Role delegate returning deterministic output sequence."""

    def __init__(self, outputs: list[dict[str, object]]) -> None:
        self._outputs = list(outputs)
        self._index = 0

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: dict[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        output = self._outputs[min(self._index, len(self._outputs) - 1)]
        self._index += 1
        return ExecutionResult(output=dict(output), success=True, tool_results=[], model_response=None)


def main() -> None:
    """Run one Ralph loop workflow and print JSON summary."""
    request_id = "example-pattern-ralph-loop-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )

    proposer = _StaticRoleDelegate(
        outputs=[
            {"proposal": "Draft v1: modular enclosure with service hatch."},
            {"proposal": "Draft v2: modular enclosure with keyed service hatch and alignment tabs."},
        ]
    )
    critic = _StaticRoleDelegate(
        outputs=[
            {"feedback": "Add clearer serviceability details."},
            {"feedback": "Feedback resolved; proposal is actionable."},
        ]
    )
    evaluator = _StaticRoleDelegate(outputs=[{"score": 0.55}, {"score": 0.86}])

    pattern = RalphLoopPattern(
        roles=(
            RalphLoopPattern.RoleSpec(role_id="proposer", delegate=proposer),
            RalphLoopPattern.RoleSpec(role_id="critic", delegate=critic),
            RalphLoopPattern.RoleSpec(role_id="evaluator", delegate=evaluator),
        ),
        evaluator_role_id="evaluator",
        loop_config=RalphLoopPattern.LoopConfig(
            max_iterations=3,
            consensus_threshold=0.8,
            selection_strategy="best_score",
        ),
        tracer=tracer,
    )

    result = pattern.run(
        "Refine a field-serviceable edge-device enclosure concept.",
        request_id=request_id,
    )
    print(json.dumps(result.summary(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
