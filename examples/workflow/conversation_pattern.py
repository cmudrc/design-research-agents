"""Runnable example for ``ConversationPattern`` brainstorming orchestration."""

import json

from design_research_agents import ConversationPattern, LlamaCppServerLLMClient, Tracer


def main() -> None:
    """Run a two-speaker brainstorming loop for a peanut sheller design."""
    llm_client = LlamaCppServerLLMClient()
    try:
        pattern = ConversationPattern(
            llm_client_a=llm_client,
            max_turns=2,
            speaker_a_name="Concept Designer",
            speaker_b_name="Prototype Reviewer",
            conversation_speaker_a_system_prompt=(
                "You are Concept Designer. Propose practical ideas for a low-cost peanut sheller "
                "for small farms."
            ),
            conversation_speaker_b_system_prompt=(
                "You are Prototype Reviewer. Stress-test each idea for manufacturability, safety, "
                "and maintainability."
            ),
            tracer=Tracer(),
        )
        result = pattern.run(
            prompt=(
                "Brainstorm a hand-crank peanut sheller for small farms. "
                "Cover shelling mechanism, adjustability for different peanut sizes, "
                "and easy field maintenance."
            ),
            request_id="example-conversation-pattern-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    transcript = output.get("transcript")
    transcript_preview = transcript[-2:] if isinstance(transcript, list) else []
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "turns_executed": output.get("turns_executed"),
        "participants": output.get("participants"),
        "final_output": output.get("final_output"),
        "transcript_preview": transcript_preview,
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
