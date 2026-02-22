r"""# Patterns / Debate Pattern.

## Introduction
Multiagent Debate shows how adversarial dialogue can improve answer quality, AutoGen provides practical
orchestration motifs, and Human-AI collaboration by design situates debate outputs within reviewable
decision pipelines. This example runs a proposer-vs-critic debate pattern over shared tool/runtime
interfaces.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``DebatePattern.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["DebatePattern.run(...)"]
    C --> D["position agents debate before synthesis"]
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
     "error": null,
     "example": "patterns/debate_pattern.py",
     "final_output": {
       "synthesis": "Use local models for sensitive data and hosted APIs for burst capacity."
     },
     "rounds": [
       {
         "affirmative_argument": "Local models improve data control and predictable costs for many research workloads.",
         "negative_argument": "Hosted APIs can ship faster and often provide higher quality with less ops burden.",
         "round": 1
       }
     ],
     "success": true,
     "terminated_reason": "completed",
     "trace": {
       "request_id": "example-workflow-debate-design-001",
       "trace_dir": "artifacts/examples/traces",
       "trace_path": "artifacts/examples/traces/run_20260222T162207Z_example-workflow-debate-design-001.jsonl"
     },
     "verdict": {
       "rationale": "Both positions are compelling with different tradeoffs.",
       "synthesis": "Use local models for sensitive data and hosted APIs for burst capacity.",
       "winner": "tie"
     },
     "winner": "tie"
   }


## References
- `Multiagent Debate <https://arxiv.org/abs/2305.14325>`_
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, Toolbox, Tracer
from design_research_agents.patterns import DebatePattern


def main() -> None:
    """Run one debate round with final judge verdict."""
    request_id = "example-workflow-debate-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = DebatePattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_rounds=1,
            tracer=tracer,
        )
        result = workflow.run(
            prompt=(
                "Should an engineering design team prioritize local models over hosted APIs when "
                "reviewing sensitive prototype telemetry?"
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "patterns/debate_pattern.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "final_output": result.final_output,
        "rounds": result.output_value("rounds"),
        "winner": result.output_value("winner"),
        "verdict": result.output_value("verdict"),
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
