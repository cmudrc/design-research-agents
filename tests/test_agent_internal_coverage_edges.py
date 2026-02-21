from __future__ import annotations

import pytest

from design_research_agents.contracts import MemoryRecord
from design_research_agents.contracts.llm import LLMChatParams, LLMMessage
from design_research_agents.implementations.shared.agent_internal.multi_step_continuation import (
    llm_should_continue,
)
from design_research_agents.implementations.shared.agent_internal.multi_step_memory import (
    retrieve_memory_context,
)
from design_research_agents.implementations.shared.agent_internal.response_schemas import (
    build_continuation_response_schema,
    build_multi_step_tool_router_response_schema,
    build_router_selection_response_schema,
)


class _RaisingLLMClient:
    def chat(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        params: LLMChatParams,
    ) -> object:
        del messages, model, params
        raise RuntimeError("chat failed")


class _SearchErrorStore:
    def write(self, records: list[object], *, namespace: str = "default") -> list[MemoryRecord]:
        del records, namespace
        return []

    def search(self, query: object) -> list[MemoryRecord]:
        del query
        raise RuntimeError("search failed")


class _EmptySearchStore:
    def write(self, records: list[object], *, namespace: str = "default") -> list[MemoryRecord]:
        del records, namespace
        return []

    def search(self, query: object) -> list[MemoryRecord]:
        del query
        return []


def test_schema_builders_cover_router_variants() -> None:
    router_schema = build_router_selection_response_schema(
        alternative_identifiers=("core", "remote")
    )
    tool_router_schema = build_multi_step_tool_router_response_schema(
        tool_names=("tool_a", "tool_b")
    )

    assert router_schema["required"] == ["tool_names"]
    assert router_schema["properties"]["tool_names"]["items"]["enum"] == ["core", "remote"]
    assert tool_router_schema["required"] == ["action"]
    assert tool_router_schema["properties"]["action"]["enum"] == ["TOOL_CALL", "STOP"]


def test_retrieve_memory_context_returns_error_on_store_exception() -> None:
    rendered, matches, error = retrieve_memory_context(
        memory_store=_SearchErrorStore(),
        namespace="design",
        top_k=2,
        task_prompt="summarize",
        memory=[{"kind": "task"}],
    )

    assert rendered == "(none)"
    assert matches == []
    assert error == "search failed"


def test_retrieve_memory_context_returns_none_for_empty_results() -> None:
    rendered, matches, error = retrieve_memory_context(
        memory_store=_EmptySearchStore(),
        namespace="design",
        top_k=2,
        task_prompt="summarize",
        memory=[{"kind": "task"}],
    )

    assert rendered == "(none)"
    assert matches == []
    assert error is None


def test_llm_should_continue_reraises_chat_exception() -> None:
    with pytest.raises(RuntimeError, match="chat failed"):
        llm_should_continue(
            llm_client=_RaisingLLMClient(),
            prompt="task prompt",
            memory=[{"kind": "task", "prompt": "task prompt"}],
            step_index=0,
            max_steps=3,
            model="test-model",
            alternatives_prompt_target="user",
            alternatives_text="alt-1\nalt-2",
            retrieved_context="",
            continuation_system_prompt="decide continue",
            continuation_user_prompt_template=(
                "Step {step_number}\nTask: {task_prompt}\nMemory: {memory_tail}\n"
                "Context: {retrieved_context}"
            ),
            continuation_response_schema=build_continuation_response_schema(),
            continuation_memory_tail_items=3,
            alternatives_section_label="Alternatives",
            agent_name="MultiStepAgent",
        )
