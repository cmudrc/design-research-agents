r"""# Patterns / Beam Search.

## Introduction
Tree of Thoughts motivates branching deliberation over single-chain prompting, while Plan-and-Solve and
ReAct provide complementary stepwise control principles. This example instantiates tree-search reasoning as
an inspectable pattern for comparing branch quality under fixed runtime controls.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``BeamSearchPattern.run(...)`` with a fixed ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["BeamSearchPattern.run(...)"]
    C --> D["generator/evaluator loop expands and prunes candidate tree"]
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
- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from design_research_agents import Tracer
from design_research_agents.patterns import BeamSearchPattern


def _generator(context: Mapping[str, object]) -> list[dict[str, object]]:
    """Generate deterministic design candidates for one search depth."""
    depth = int(context.get("depth", 0))
    if depth == 1:
        return [
            {"concept": "lightweight frame", "score_hint": 0.45},
            {"concept": "modular frame", "score_hint": 0.7},
        ]
    return [
        {"concept": "modular frame + fail-safe", "score_hint": 0.92},
        {"concept": "modular frame + low cost", "score_hint": 0.61},
    ]


def _evaluator(context: Mapping[str, object]) -> float:
    """Return deterministic score for one candidate payload."""
    candidate = context.get("candidate")
    if isinstance(candidate, Mapping):
        score = candidate.get("score_hint")
        if isinstance(score, (int, float)):
            return float(score)
    return 0.0


def main() -> None:
    """Run one beam-search workflow and print JSON summary."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-workflow-beam-search-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    pattern = BeamSearchPattern(
        generator_delegate=_generator,
        evaluator_delegate=_evaluator,
        max_depth=2,
        branch_factor=2,
        beam_width=1,
        tracer=tracer,
    )
    result = pattern.run(
        "Find the most robust concept architecture for a serviceable edge-device enclosure.",
        request_id=request_id,
    )
    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
