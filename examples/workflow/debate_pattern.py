"""Runnable example for the reusable ``DebatePattern`` orchestration chunk."""

import json

from design_research_agents import DebatePattern, LlamaCppServerLLMClient, Toolbox


def main() -> None:
    """Run ``DebatePattern`` with one debate round and a final judge verdict."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox()
    try:
        workflow = DebatePattern(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            max_rounds=1,
        )
        result = workflow.run(
            prompt="Should a research team prioritize local models over hosted APIs?",
            request_id="example-debate-pattern-001",
        )
    finally:
        llm_client.close()

    output = result.output if isinstance(result.output, dict) else {}
    payload = {
        "success": result.success,
        "terminated_reason": output.get("terminated_reason"),
        "rounds": output.get("rounds"),
        "winner": output.get("winner"),
        "verdict": output.get("verdict"),
        "error": output.get("error"),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
