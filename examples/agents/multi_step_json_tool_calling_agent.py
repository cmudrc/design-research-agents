"""Example script.

Motivation
Run traced ``MultiStepAgent(mode="json")`` with explicit core-tool allowlist.

Diagram
```mermaid
flowchart LR
    A["Prompt"] --> B["Agent run"]
    B --> C["multi step json tool calling agent output"]
    C --> D["JSON payload and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `agents` use-cases and run `multi_step_json_tool_calling_agent`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/agents/multi_step_json_tool_calling_agent.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Toolbox, Tracer

_JSON_ALLOWED_TOOLS: tuple[str, ...] = (
    "fs.read_text",
    "text.word_count",
    "python.sandbox",
    "memory.search",
    "memory.write",
    "memory.stats",
    "eval.decision_matrix",
    "eval.pairwise_rank",
)


def main() -> None:
    """Execute one traced multi-step JSON tool-calling run."""
    request_id = "example-multi-step-json-design-001"
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
            mode="json",
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_steps=3,
            allowed_tools=_JSON_ALLOWED_TOOLS,
            tracer=tracer,
        )
        result = agent.run(
            prompt="Read README.md and summarize one implementation insight from the text.",
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "agents/multi_step_json_tool_calling_agent.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": result.output_value("steps_executed"),
        "tool_results_count": len(result.tool_results),
        "allowed_tools": list(_JSON_ALLOWED_TOOLS),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
