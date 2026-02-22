"""Example script.

Motivation
Run traced ``NetworkedPattern`` and ``BlackboardPattern`` design coordination.

Diagram
```mermaid
flowchart LR
    A["Pattern prompt"] --> B["Pattern orchestration"]
    B --> C["networked blackboard result"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `patterns` use-cases and run `networked_blackboard`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/patterns/networked_blackboard.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import (
    BlackboardPattern,
    DirectLLMCall,
    ExecutionResult,
    LlamaCppServerLLMClient,
    NetworkedPattern,
    Tracer,
)


def _summarize(result: ExecutionResult, request_id: str, tracer: Tracer) -> dict[str, object]:
    blackboard = result.output_dict("blackboard")
    messages = blackboard.get("messages") if isinstance(blackboard, dict) else []
    return {
        "example": "patterns/networked_blackboard.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "rounds_executed": result.output_value("rounds_executed"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "final_output": result.final_output,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }


def main() -> None:
    """Run one networked and one blackboard coordination pass."""
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

        network_request_id = "example-workflow-networked-pattern-design-001"
        networked = NetworkedPattern(
            peers={
                "peer_b": peer_b,
                "peer_a": peer_a,
            },
            max_rounds=2,
            tracer=tracer,
        )
        network_result = networked.run(
            "Coordinate candidate mechanisms for a field-serviceable sensor enclosure.",
            request_id=network_request_id,
        )

        blackboard_request_id = "example-workflow-blackboard-pattern-design-001"
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
    finally:
        llm_client.close()

    print(
        json.dumps(
            {
                "networked_pattern": _summarize(network_result, network_request_id, tracer),
                "blackboard_pattern": _summarize(blackboard_result, blackboard_request_id, tracer),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
