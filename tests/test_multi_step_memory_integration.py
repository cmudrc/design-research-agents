"""Integration tests for multi-step agent memory read/write behavior."""

from __future__ import annotations

from design_research_agents._contracts._memory import MemorySearchQuery, MemoryWriteRecord
from design_research_agents._memory._stores._sqlite_store import SQLiteMemoryStore
from design_research_agents.agent import MultiStepAgent
from design_research_agents.tools import Toolbox
from tests.helpers.workflow_stubs import SequenceLLMClient


def test_multi_step_json_behavior_unchanged_without_memory_store() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            '{"tool_name": "text.word_count", "tool_input": {"text": "design research agents"}}',
            '{"tool_name": "final_answer", "tool_input": {"word_count": 3}, "reason": "done"}',
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        memory_store=None,
    )

    result = agent.run("Count words in design research agents.")

    assert result.success
    assert result.output["final_output"]["word_count"] == 3
    assert result.output["steps_executed"] == 2


def test_multi_step_json_reads_memory_context_and_writes_observations(tmp_path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [MemoryWriteRecord(content="Prior note: use text.word_count for quick text metrics.")],
        namespace="agent-memory",
    )

    llm_client = SequenceLLMClient(
        response_texts=[
            '{"tool_name": "text.word_count", "tool_input": {"text": "design systems research"}}',
            '{"tool_name": "final_answer", "tool_input": {"word_count": 3}, "reason": "done"}',
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        memory_store=store,
        memory_namespace="agent-memory",
        memory_read_top_k=2,
        memory_write_observations=True,
    )

    result = agent.run("Count words in design systems research.")

    assert result.success
    assert result.metadata["memory"]["enabled"] is True
    assert result.metadata["memory"]["retrieval_trace"][0]["count"] >= 1

    observation_matches = store.search(
        MemorySearchQuery(
            text="Count words in design systems research",
            namespace="agent-memory",
            metadata_filters={"kind": "multi_step_observation"},
            top_k=10,
        )
    )
    store.close()

    assert len(observation_matches) >= 1


def test_multi_step_code_writes_observations_when_memory_enabled(tmp_path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [MemoryWriteRecord(content="Prior note: tool output should be summarized.")],
        namespace="code-memory",
    )

    llm_client = SequenceLLMClient(
        response_texts=[
            "\n".join(
                [
                    'stats = call_tool("text.word_count", {"text": "design memory trace"})',
                    'final_output = {"result": stats["word_count"]}',
                ]
            ),
            'final_answer({"result": 3})',
        ]
    )
    agent = MultiStepAgent(
        mode="code",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        memory_store=store,
        memory_namespace="code-memory",
        memory_read_top_k=2,
        memory_write_observations=True,
    )

    result = agent.run("Count words in design memory trace.")

    assert result.success
    assert result.metadata["memory"]["enabled"] is True
    assert result.metadata["memory"]["retrieval_trace"][0]["count"] >= 1

    observation_matches = store.search(
        MemorySearchQuery(
            text="Count words in design memory trace",
            namespace="code-memory",
            metadata_filters={"kind": "multi_step_observation"},
            top_k=10,
        )
    )
    store.close()

    assert len(observation_matches) >= 1
