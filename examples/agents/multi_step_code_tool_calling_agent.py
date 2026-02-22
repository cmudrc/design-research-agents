"""Example script.

Motivation
Run traced ``MultiStepAgent(mode="code")`` for design-metric analysis.

Diagram
```mermaid
flowchart LR
    A["Prompt"] --> B["Agent run"]
    B --> C["multi step code tool calling agent output"]
    C --> D["JSON payload and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `agents` use-cases and run `multi_step_code_tool_calling_agent`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/agents/multi_step_code_tool_calling_agent.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer


def main() -> None:
    """Execute one multi-step code-mode run and print compact result."""
    request_id = "example-multi-step-code-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        agent = MultiStepAgent(
            mode="code",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            normalize_generated_code_per_step=True,
            default_tools_per_step=({"tool_name": "text.word_count"},),
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "No imports. Use call_tool only. Compute two design-review metrics using "
                "text.word_count on these phrases: 'design review metrics' and "
                "'runtime tool boundaries'. Return final_output with both counts."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    step_outputs = result.output_list("step_outputs")
    payload = {
        "example": "agents/multi_step_code_tool_calling_agent.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": result.output_value("steps_executed"),
        "step_outputs_count": len(step_outputs),
        "tool_results_count": len(result.tool_results),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
