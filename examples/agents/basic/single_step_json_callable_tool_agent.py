"""Runnable example using a callable tool with a basic JSON tool-calling agent."""

from __future__ import annotations

import json
from collections.abc import Mapping

from design_research_agents import CallableTool, LlamaCppServerLLMClient, Toolbox
from design_research_agents.agent import SingleStepJsonToolCallingAgent


def _normalize_title(payload: Mapping[str, object]) -> dict[str, object]:
    """Run normalize title.

    Args:
        payload: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    raw_title = str(payload.get("title", "")).strip()
    if not raw_title:
        raise ValueError("title is required.")
    normalized = " ".join(part.capitalize() for part in raw_title.split())
    return {
        "normalized_title": normalized,
        "word_count": len(normalized.split()),
        "original_title": raw_title,
    }


def main() -> None:
    """Run one callable-tool flow through ``SingleStepJsonToolCallingAgent``."""
    llm_client = LlamaCppServerLLMClient()
    tool_runtime = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableTool(
                name="normalize.title",
                description="Normalize title casing and return compact title stats.",
                handler=_normalize_title,
                input_schema={
                    "type": "object",
                    "required": ["title"],
                    "properties": {"title": {"type": "string"}},
                    "additionalProperties": False,
                },
                output_schema={
                    "type": "object",
                    "required": ["normalized_title", "word_count", "original_title"],
                    "properties": {
                        "normalized_title": {"type": "string"},
                        "word_count": {"type": "integer"},
                        "original_title": {"type": "string"},
                    },
                    "additionalProperties": False,
                },
            ),
        ),
    )
    try:
        agent = SingleStepJsonToolCallingAgent(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
        )
        result = agent.run(
            prompt="Normalize the title 'the old man and the sea' and return title stats.",
            request_id="example-callable-tool-agent-001",
        )
    finally:
        llm_client.close()

    tool_result = result.tool_results[0].result if result.tool_results else {}
    payload = {
        "agent": "SingleStepJsonToolCallingAgent",
        "selected_tool": result.output.get("tool_name"),
        "tool_input": result.output.get("tool_input"),
        "tool_result": tool_result,
        "model_text": (result.model_response.text if result.model_response is not None else ""),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
