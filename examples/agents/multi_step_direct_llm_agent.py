"""Example script.

Motivation
Run traced ``MultiStepAgent(mode="direct")`` for design-brief drafting.

Diagram
```mermaid
flowchart LR
    A["Prompt"] --> B["Agent run"]
    B --> C["multi step direct llm agent output"]
    C --> D["JSON payload and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `agents` use-cases and run `multi_step_direct_llm_agent`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/agents/multi_step_direct_llm_agent.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, MultiStepAgent, Tracer


def main() -> None:
    """Execute one multi-step direct run and print summary."""
    request_id = "example-multi-step-direct-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = MultiStepAgent(
            mode="direct",
            llm_client=llm_client,
            max_steps=3,
            tracer=tracer,
        )
        result = agent.run(
            prompt=(
                "Draft then finalize a short design memo title for reducing maintenance time in a modular lab rig."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "agents/multi_step_direct_llm_agent.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "steps_executed": result.output_value("steps_executed"),
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
