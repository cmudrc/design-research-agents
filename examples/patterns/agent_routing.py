"""Example script.

Motivation
Run traced ``RouterPattern`` across design-focused delegate agents.

Diagram
```mermaid
flowchart LR
    A["Pattern prompt"] --> B["Pattern orchestration"]
    B --> C["agent routing result"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `patterns` use-cases and run `agent_routing`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/patterns/agent_routing.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    DirectLLMCall,
    LlamaCppServerLLMClient,
    MultiStepAgent,
    RouterPattern,
    Toolbox,
    Tracer,
)


def main() -> None:
    """Route one design prompt to the best delegate and print summary."""
    request_id = "example-workflow-agent-routing-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()

    direct_llm_agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)
    json_tool_agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        max_steps=1,
        tracer=tracer,
    )

    workflow = RouterPattern(
        llm_client=llm_client,
        tool_runtime=tool_runtime,
        alternatives={
            "direct_llm_agent": direct_llm_agent,
            "json_tool_agent": json_tool_agent,
        },
        alternative_descriptions={
            "direct_llm_agent": "Use for concise textual design summaries with no runtime tools.",
            "json_tool_agent": ("Use for design requests needing runtime text analysis or tool calls."),
        },
        tracer=tracer,
    )

    try:
        result = workflow.run(
            prompt=("Count the words in this design phrase and return the tool result: modular field service workflow"),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "patterns/agent_routing.py",
        "success": result.success,
        "selected_alternative": result.output_value("selected_alternative"),
        "final_output": result.final_output,
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
