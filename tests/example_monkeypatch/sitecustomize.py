"""Test-only monkeypatches for running examples with deterministic LLM responses.

Loaded automatically by Python when this directory is prepended to PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

_DETERMINISTIC_MODE = os.environ.get("DRA_EXAMPLE_LLM_MODE", "").strip().lower() == "deterministic"

if _DETERMINISTIC_MODE:
    import design_research_agents.llm as llm_module
    from design_research_agents.contracts.llm import (
        LLMChatParams,
        LLMDelta,
        LLMMessage,
        LLMRequest,
        LLMResponse,
        LLMStreamEvent,
    )

_SCRIPT_RESPONSES: dict[str, tuple[str, ...]] = {
    "single_step_direct_llm_agent.py": ("4",),
    "single_step_router_agent.py": (
        '{"selection":"text.word_count","reason":"Analyze text content."}',
    ),
    "single_step_json_tool_calling_agent.py": (
        '{"tool_name":"calculator","tool_input":{"expression":"12 * (4 + 1)"}}',
    ),
    "single_step_json_callable_tool_agent.py": (
        (
            '{"tool_name":"normalize.title","tool_input":{"title":"the old man and the sea"},'
            '"reason":"Normalize the provided title casing."}'
        ),
    ),
    "single_step_json_script_rubric_score_agent.py": (
        (
            '{"tool_name":"script::rubric_score","tool_input":{"text":"Agents can quickly score '
            'this sample summary.","max_score":12}}'
        ),
    ),
    "single_step_json_script_repo_quickscan_agent.py": (
        '{"tool_name":"script::repo_quickscan","tool_input":{"include_hidden":false}}',
    ),
    "single_step_code_tool_calling_agent.py": (
        "\n".join(
            [
                'csv_text = "tool,source\\ncalculator,core\\nrepo_quickscan,script\\n"',
                'write_result = call_tool("fs.write_text", {"path": '
                '"artifacts/examples/single_step_tool_inventory.csv", "content": csv_text, '
                '"overwrite": True})',
                'describe_result = call_tool("data.describe", {"path": '
                '"artifacts/examples/single_step_tool_inventory.csv", "kind": "csv"})',
                'search_result = call_tool("search.ripgrep", {"query": '
                '"Toolbox", "root": "src/design_research_agents/tools", '
                '"max_matches": 3})',
                "final_output = {",
                '    "csv_path": write_result["path"],',
                '    "row_count": describe_result["rows"],',
                '    "column_count": describe_result["column_count"],',
                '    "match_count": search_result["count"],',
                "}",
            ]
        ),
    ),
    "multi_step_code_tool_calling_agent.py": (
        '{"continue": true, "thought": "Read README and analyze text stats."}',
        "\n".join(
            [
                'readme = call_tool("fs.read_text", {"path": "README.md", "max_bytes": 2400})',
                'stats = call_tool("text.word_count", {"text": readme["text"]})',
                'diff_result = call_tool("text.diff", {"a": "core tools only", '
                '"b": "core + script + mcp tools"})',
                "final_output = {",
                '    "word_count": stats["word_count"],',
                '    "line_count": stats["line_count"],',
                '    "diff_preview": diff_result["diff"].splitlines()[:4],',
                "}",
            ]
        ),
        '{"continue": false, "thought": "Task complete."}',
    ),
    "multi_step_json_tool_calling_agent.py": (
        '{"continue": true, "thought": "Read README first."}',
        (
            '{"tool_name":"fs.read_text","tool_input":{"path":"README.md","max_bytes":800},'
            '"reason":"Need repository text context."}'
        ),
        '{"continue": false, "thought": "Task complete."}',
    ),
    "plan_execute.py": (
        (
            '{"steps":[{"step_id":"analyze_repo_tools","instruction":"Write a small CSV artifact '
            "describing tool sources, then describe that CSV and search the tools package for "
            'Toolbox.","success_criteria":"Return csv row stats and source-code '
            'match count."}]}'
        ),
        "\n".join(
            [
                'csv_text = "tool,source\\ncalculator,core\\nrubric_score,script\\n'
                'text.word_count,mcp\\n"',
                'write_result = call_tool("fs.write_text", {"path": '
                '"artifacts/examples/plan_execute_runtime_inventory.csv", "content": csv_text, '
                '"overwrite": True})',
                'describe_result = call_tool("data.describe", {"path": '
                '"artifacts/examples/plan_execute_runtime_inventory.csv", "kind": "csv"})',
                'search_result = call_tool("search.ripgrep", {"query": '
                '"Toolbox", "root": "src/design_research_agents/tools", '
                '"max_matches": 4})',
                "final_output = {",
                '  "csv_path": write_result["path"],',
                '  "row_count": describe_result["rows"],',
                '  "column_count": describe_result["column_count"],',
                '  "search_hits": search_result["count"],',
                "}",
            ]
        ),
    ),
    "propose_critic.py": (
        "Draft v1: simple proposal.",
        (
            '{"approved": false, "feedback": "Add more detail.", '
            '"revision_goals": ["expand rationale"]}'
        ),
        "Draft v2: proposal with more detail.",
        '{"approved": true, "feedback": "Looks good.", "revision_goals": []}',
    ),
    "agent_routing.py": (
        '{"selection":"json_tool_agent","reason":"Arithmetic request uses tools."}',
        '{"tool_name":"calculator","tool_input":{"expression":"12 * (4 + 1)"}}',
    ),
    "mixed_agent_workflow.py": (
        (
            '{"title":"Deterministic workflow memo","summary":"Use one runtime that fuses core, '
            'script, and MCP tools.","priority":"high"}'
        ),
    ),
    "single_step_direct_llm_agent_stream.py": ("The answer is 4.",),
    "single_step_router_agent_stream.py": (
        '{"selection":"calculator","reason":"Arithmetic request."}',
    ),
    "single_step_json_tool_calling_agent_stream.py": (
        (
            '{"tool_name":"calculator","tool_input":{"expression":"12 * (4 + 1)"},'
            '"reason":"Arithmetic request."}'
        ),
    ),
    "single_step_code_tool_calling_agent_stream.py": (
        "\n".join(
            [
                'repo_files = call_tool("fs.list_dir", {"path": ".", "max_entries": 30})',
                'search_result = call_tool("search.ripgrep", {"query": "Toolbox", '
                '"root": "src", "max_matches": 2})',
                "final_output = {",
                '  "top_level_entry_count": repo_files["count"],',
                '  "search_hits": search_result["count"]',
                "}",
            ]
        ),
    ),
    "multi_step_code_tool_calling_agent_stream.py": (
        '{"continue": true, "thought": "Run one action step."}',
        "\n".join(
            [
                'readme = call_tool("fs.read_text", {"path": "README.md", "max_bytes": 1200})',
                'stats = call_tool("text.word_count", {"text": readme["text"]})',
                'final_output = {"word_count": stats["word_count"], "summary": "README measured."}',
            ]
        ),
    ),
    "multi_step_json_tool_calling_agent_stream.py": (
        '{"continue": true, "thought": "Run one action step."}',
        (
            '{"tool_name":"text.word_count","tool_input":{"text":"README measured"},'
            '"reason":"Compute compact metric."}'
        ),
    ),
}


if _DETERMINISTIC_MODE:

    class _DeterministicExampleClient:
        def __init__(self, *, response_texts: Sequence[str]) -> None:
            self._responses = list(response_texts)

        def chat(
            self,
            messages: Sequence[LLMMessage],
            *,
            model: str,
            params: LLMChatParams,
        ) -> LLMResponse:
            del messages, params
            return self._next(model)

        def stream_chat(
            self,
            messages: Sequence[LLMMessage],
            *,
            model: str,
            params: LLMChatParams,
        ) -> Iterator[LLMStreamEvent]:
            response = self.chat(messages, model=model, params=params)
            yield LLMStreamEvent(kind="delta", delta_text=response.text)
            yield LLMStreamEvent(kind="completed", response=response)

        def generate(self, request: LLMRequest) -> LLMResponse:
            return self._next(request.model or self.default_model())

        def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
            response = self.generate(request)
            yield LLMDelta(text_delta=response.text)

        def default_model(self) -> str:
            return "example-model"

        def close(self) -> None:
            return None

        def _next(self, model: str) -> LLMResponse:
            if not self._responses:
                raise RuntimeError("No deterministic responses remaining for this example script.")
            return LLMResponse(
                model=model,
                text=self._responses.pop(0),
                provider="example-test-monkeypatch",
            )

    def _patched_LlamaCppServerLLMClient() -> _DeterministicExampleClient:
        script_name = Path(sys.argv[0]).name
        responses = _SCRIPT_RESPONSES.get(script_name)
        if responses is None:
            raise RuntimeError(
                f"No deterministic LLM response profile configured for script '{script_name}'."
            )
        return _DeterministicExampleClient(response_texts=responses)

    llm_module.LlamaCppServerLLMClient = _patched_LlamaCppServerLLMClient

    # Patch top-level exported accessor for tests that import it directly.
    try:
        import design_research_agents as package_api
    except Exception:
        package_api = None
    if package_api is not None:
        package_api.LlamaCppServerLLMClient = _patched_LlamaCppServerLLMClient
