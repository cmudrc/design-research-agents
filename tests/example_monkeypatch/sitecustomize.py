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
    from design_research_agents.contracts.llm import (
        LLMChatParams,
        LLMDelta,
        LLMMessage,
        LLMRequest,
        LLMResponse,
        LLMStreamEvent,
    )
    from design_research_agents.llm.backends import default as default_backend

_SCRIPT_RESPONSES: dict[str, tuple[str, ...]] = {
    "single_step_direct_llm_agent.py": ("4",),
    "single_step_router_agent.py": (
        '{"selection":"text.word_count","reason":"Analyze text content."}',
    ),
    "single_step_json_tool_calling_agent.py": (
        '{"tool_name":"calculator","tool_input":{"expression":"12 * (4 + 1)"}}',
    ),
    "single_step_json_lazy_rubric_score_agent.py": (
        (
            '{"tool_name":"lazy::rubric_score","tool_input":{"text":"Agents can quickly score '
            'this sample summary.","max_score":12}}'
        ),
    ),
    "single_step_json_lazy_repo_quickscan_agent.py": (
        '{"tool_name":"lazy::repo_quickscan","tool_input":{"include_hidden":false}}',
    ),
    "single_step_code_tool_calling_agent.py": (
        "\n".join(
            [
                'csv_text = "tool,source\\ncalculator,core\\nrepo_quickscan,lazy\\n"',
                'write_result = call_tool("fs.write_text", {"path": '
                '"artifacts/examples/single_step_tools.csv", "content": csv_text, '
                '"overwrite": True})',
                'describe_result = call_tool("data.describe", {"path": '
                '"artifacts/examples/single_step_tools.csv", "kind": "csv"})',
                'search_result = call_tool("search.ripgrep", {"query": '
                '"UnifiedToolRuntime", "root": "src/design_research_agents/tools", '
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
                '"b": "core + lazy + mcp tools"})',
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
            'UnifiedToolRuntime.","success_criteria":"Return csv row stats and source-code '
            'match count."}]}'
        ),
        "\n".join(
            [
                'csv_text = "tool,source\\ncalculator,core\\nrubric_score,lazy\\n'
                'text.word_count,mcp\\n"',
                'write_result = call_tool("fs.write_text", {"path": '
                '"artifacts/examples/runtime_plan_execute.csv", "content": csv_text, '
                '"overwrite": True})',
                'describe_result = call_tool("data.describe", {"path": '
                '"artifacts/examples/runtime_plan_execute.csv", "kind": "csv"})',
                'search_result = call_tool("search.ripgrep", {"query": '
                '"UnifiedToolRuntime", "root": "src/design_research_agents/tools", '
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
            'lazy, and MCP tools.","priority":"high"}'
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
                'search_result = call_tool("search.ripgrep", {"query": "UnifiedToolRuntime", '
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

        def _next(self, model: str) -> LLMResponse:
            if not self._responses:
                raise RuntimeError("No deterministic responses remaining for this example script.")
            return LLMResponse(
                model=model,
                text=self._responses.pop(0),
                provider="example-test-monkeypatch",
            )

    def _patched_create_default_llm_client() -> _DeterministicExampleClient:
        script_name = Path(sys.argv[0]).name
        responses = _SCRIPT_RESPONSES.get(script_name)
        if responses is None:
            raise RuntimeError(
                f"No deterministic LLM response profile configured for script '{script_name}'."
            )
        return _DeterministicExampleClient(response_texts=responses)

    default_backend.create_default_llm_client = _patched_create_default_llm_client

    # `design_research_agents._public_api` captures the factory during package import.
    # Patch that captured reference too so examples calling `dra.llm.create_default_llm_client()`
    # resolve to deterministic test responses.
    try:
        from design_research_agents import _public_api as public_api
    except Exception:
        public_api = None
    if public_api is not None:
        public_api.create_default_llm_client = _patched_create_default_llm_client
        object.__setattr__(
            public_api.llm,
            "create_default_llm_client",
            _patched_create_default_llm_client,
        )
