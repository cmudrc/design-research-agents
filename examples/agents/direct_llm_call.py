"""Example script.

Motivation
Run one traced ``DirectLLMCall`` for an engineering-design prompt.

Diagram
```mermaid
flowchart LR
    A["Prompt"] --> B["Agent run"]
    B --> C["direct llm call output"]
    C --> D["JSON payload and trace"]
```

Technical Walkthrough
1. Configure the runtime surface for `agents` use-cases and run `direct_llm_call`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/agents/direct_llm_call.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import DirectLLMCall, LlamaCppServerLLMClient, Tracer, __version__


def main() -> None:
    """Execute one direct model call with explicit tracing."""
    request_id = "example-direct-llm-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        agent = DirectLLMCall(llm_client=llm_client, tracer=tracer)
        result = agent.run(
            prompt=(
                "Write one sentence describing the primary engineering objective for a "
                "field-repairable wearable sensor enclosure."
            ),
            request_id=request_id,
        )
    finally:
        llm_client.close()

    payload = {
        "example": "agents/direct_llm_call.py",
        "package_version": __version__,
        "success": result.success,
        "final_output": result.final_output,
        "model": result.output_value("model"),
        "terminated_reason": result.terminated_reason,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
