from __future__ import annotations

import pytest

from design_research_agents._contracts import MemoryRecord
from design_research_agents._contracts._llm import LLMChatParams, LLMMessage
from design_research_agents._implementations._shared._agent_internal._multi_step_continuation import (
    llm_should_continue,
)
from design_research_agents._implementations._shared._agent_internal._multi_step_memory import (
    retrieve_memory_context,
)
from design_research_agents._implementations._shared._agent_internal._response_schemas import (
    build_continuation_response_schema,
    build_multi_step_direct_controller_response_schema,
    build_tool_call_response_schema,
)


class _RaisingLLMClient:
    def default_model(self) -> str:
        return "test-model"

    def close(self) -> None:
        return None

    def __enter__(self) -> _RaisingLLMClient:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None

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

    def close(self) -> None:
        return None

    def __enter__(self) -> _SearchErrorStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


class _EmptySearchStore:
    def write(self, records: list[object], *, namespace: str = "default") -> list[MemoryRecord]:
        del records, namespace
        return []

    def search(self, query: object) -> list[MemoryRecord]:
        del query
        return []

    def close(self) -> None:
        return None

    def __enter__(self) -> _EmptySearchStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()
        return None


def test_schema_builders_cover_active_variants() -> None:
    tool_call_schema = build_tool_call_response_schema(tool_names=("tool_a", "final_answer"))
    direct_controller_schema = build_multi_step_direct_controller_response_schema()

    assert tool_call_schema["required"] == ["tool_name"]
    assert tool_call_schema["properties"]["tool_name"]["enum"] == ["tool_a", "final_answer"]
    assert direct_controller_schema["required"] == ["decision", "content"]
    assert direct_controller_schema["properties"]["decision"]["enum"] == ["CONTINUE", "STOP"]


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
                "Step {step_number}\nTask: {task_prompt}\nMemory: {memory_tail}\nContext: {retrieved_context}"
            ),
            continuation_response_schema=build_continuation_response_schema(),
            continuation_memory_tail_items=3,
            alternatives_section_label="Alternatives",
            agent_name="MultiStepAgent",
        )
