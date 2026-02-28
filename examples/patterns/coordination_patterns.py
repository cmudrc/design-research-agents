"""# Patterns / Coordination Patterns.

## Introduction
Blackboard-system architecture motivates shared-state collaboration among specialized problem solvers,
AutoGen informs practical multi-agent implementation choices, and Human-AI collaboration by design clarifies
governance value in shared workspace reasoning. This example compares round-based coordination and
blackboard-specialized runs with explicit execution records.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``RoundBasedCoordinationPattern.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["RoundBasedCoordinationPattern.run(...)"]
    C --> D["blackboard workers contribute and aggregate shared state"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "round_based_coordination": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "blackboard": {
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
   }

## References
- `Blackboard System (Wikipedia) <https://en.wikipedia.org/wiki/Blackboard_system>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    ExecutionResult,
    LlamaCppServerLLMClient,
    Tracer,
)
from design_research_agents.patterns import BlackboardPattern, RoundBasedCoordinationPattern


def _summarize(result: ExecutionResult) -> dict[str, object]:
    return result.summary()


def main() -> None:
    """Run one round-based coordination and one blackboard pass."""
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )

    llm_client = LlamaCppServerLLMClient()
    try:
        peer_a = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        peer_b = DirectLLMCall(llm_client=llm_client, tracer=tracer)

        # Split ids by pattern variant to keep networked and blackboard traces distinct.

        coordination_request_id = "example-workflow-round-based-coordination-design-001"
        coordination = RoundBasedCoordinationPattern(
            peers={
                "peer_b": peer_b,
                "peer_a": peer_a,
            },
            max_rounds=2,
            tracer=tracer,
        )
        coordination_result = coordination.run(
            "Coordinate candidate mechanisms for a field-serviceable sensor enclosure.",
            request_id=coordination_request_id,
        )

        # Split ids by pattern variant to keep networked and blackboard traces distinct.

        blackboard_request_id = "example-workflow-blackboard-design-001"
        blackboard = BlackboardPattern(
            peers={
                "peer_b": peer_b,
                "peer_a": peer_a,
            },
            max_rounds=3,
            stability_rounds=2,
            tracer=tracer,
        )
        blackboard_result = blackboard.run(
            "Compare two concept options and converge on a serviceable design direction.",
            request_id=blackboard_request_id,
        )
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        llm_client.close()

    print(
        json.dumps(
            {
                "blackboard": _summarize(blackboard_result),
                "round_based_coordination": _summarize(coordination_result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
