"""Example script.

Motivation
Run traced ``ReflexionPattern`` for iterative design-summary refinement.

Diagram
```mermaid
flowchart LR
    A["Pattern prompt"] --> B["Pattern orchestration"]
    B --> C["propose critic result"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `patterns` use-cases and run `propose_critic`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/patterns/propose_critic.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import LlamaCppServerLLMClient, ReflexionPattern, Toolbox, Tracer


def main() -> None:
    """Run propose/critique refinement orchestration with tracing."""
    request_id = "example-workflow-propose-critic-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = ReflexionPattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            tracer=tracer,
        )
        result = workflow.run(
            prompt=(
                "Write and iteratively improve a short engineering design rationale for using "
                "modular connectors in field-serviceable devices."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "patterns/propose_critic.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "final_output": result.final_output,
        "approved": result.output_value("approved"),
        "critique_iterations": result.output_value("critique_iterations"),
        "proposal": result.output_value("proposal"),
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
