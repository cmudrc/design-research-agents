"""Run traced ``ConversationPattern`` for engineering concept iteration.

Expected observations:
- ``turns_executed`` indicates iterative conversation depth.
- ``transcript_preview`` shows latest role contributions.
- ``trace.trace_path`` points to emitted trace JSONL.
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
        "example": "workflow/conversation_pattern.py",
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
