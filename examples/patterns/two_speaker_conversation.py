"""# Patterns / Two Speaker Conversation.

## Introduction
AutoGen-style multi-agent conversations can externalize reasoning roles, Human-AI collaboration by design
explains why role separation matters for oversight, and AI-assisted design synthesis work motivates
structured dialogue in design ideation. This example implements a two-agent conversation loop with trace
visibility at each turn.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``TwoSpeakerConversationPattern.run(...)`` with a fixed
   ``request_id``.
3. Capture structured outputs from runtime execution and preserve termination metadata for analysis.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["TwoSpeakerConversationPattern.run(...)"]
    C --> D["turn-based conversation state drives each step"]
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
- `AutoGen <https://arxiv.org/abs/2308.08155>`_
- `Human-AI collaboration by design <https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/humanai-collaboration-by-design/45BC30ADFF2FE3B204D4A29DD67F6353>`_
- `AI-assisted design synthesis and human creativity in engineering education <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1714523/full>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, Tracer
from design_research_agents.patterns import TwoSpeakerConversationPattern


def main() -> None:
    """Run two-speaker brainstorming loop for a serviceable device enclosure."""
    # Fixed request id keeps traces and docs output deterministic across runs.
    request_id = "example-workflow-two-speaker-conversation-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        pattern = TwoSpeakerConversationPattern(
            llm_client_a=llm_client,
            max_turns=5,
            speaker_a_name="Concept Designer",
            speaker_b_name="Validation Engineer",
            speaker_a_system_prompt=(
                "You are Concept Designer. Propose practical ideas for a field-serviceable sensor enclosure."
            ),
            speaker_b_system_prompt=(
                "You are Validation Engineer. Stress-test ideas for manufacturability, safety, and maintenance time."
            ),
            tracer=tracer,
        )
        result = pattern.run(
            prompt=(
                "Brainstorm a modular enclosure for a wearable biosensor. Cover sealing strategy, "
                "fastener choices, and quick battery replacement."
            ),
            request_id=request_id,
        )
    # Always close runtime resources explicitly to avoid handle leakage in repeated runs.
    finally:
        llm_client.close()

    summary = result.summary()
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
