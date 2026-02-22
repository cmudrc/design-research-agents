"""Example script.

Motivation
Run traced ``ConversationPattern`` for engineering concept iteration.

Diagram
```mermaid
flowchart LR
    A["Pattern prompt"] --> B["Pattern orchestration"]
    B --> C["conversation pattern result"]
    C --> D["Trace metadata"]
```

Technical Walkthrough
1. Configure the runtime surface for `patterns` use-cases and run `conversation_pattern`.
2. Execute the example with direct public APIs and capture trace metadata.
3. Print a JSON payload that is easy to inspect in docs and tests.

Expected Results
- The script exits successfully and prints a non-empty JSON payload.
- The payload includes the example identity and trace metadata.
- Deterministic test runs can monkeypatch model backends without changing this script.

Discussion
Run with `PYTHONPATH=src python3 examples/patterns/conversation_pattern.py`.
In tests, deterministic monkeypatching can replace live client behavior while preserving
this script's capability-first structure.
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import ConversationPattern, LlamaCppServerLLMClient, Tracer


def main() -> None:
    """Run two-speaker brainstorming loop for a serviceable device enclosure."""
    request_id = "example-workflow-conversation-design-001"
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=False,
    )
    llm_client = LlamaCppServerLLMClient()
    try:
        pattern = ConversationPattern(
            llm_client_a=llm_client,
            max_turns=2,
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
    finally:
        llm_client.close()

    transcript = result.output_list("transcript")
    transcript_preview = transcript[-2:]
    payload = {
        "example": "patterns/conversation_pattern.py",
        "success": result.success,
        "terminated_reason": result.terminated_reason,
        "turns_executed": result.output_value("turns_executed"),
        "participants": result.output_value("participants"),
        "final_output": result.final_output,
        "transcript_preview": transcript_preview,
        "error": result.error,
        "trace": tracer.trace_info(request_id),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
