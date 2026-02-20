"""Integration tests for multi-step agent memory read/write behavior."""

from __future__ import annotations

from design_research_agents.agent import MultiStepAgent
from design_research_agents.contracts.memory import MemorySearchQuery, MemoryWriteRecord
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from design_research_agents.tools import Toolbox
from tests.helpers.workflow_stubs import SequenceLLMClient


def test_multi_step_json_behavior_unchanged_without_memory_store() -> None:
    llm_client = SequenceLLMClient(
        response_texts=[
            '{"continue": true, "thought": "start"}',
            '{"tool_name": "calculator", "tool_input": {"expression": "6 * 7"}}',
            '{"continue": false, "thought": "done"}',
        ]
    )
    agent = MultiStepAgent(
        mode="json",
        llm_client=llm_client,
        tool_runtime=Toolbox(),
        max_steps=3,
        memory_store=None,
    )

    result = agent.run("Compute 6 * 7.")

    assert result.success
    assert result.output["final_output"]["result"] == 42.0
    assert result.output["steps_executed"] == 1


def test_multi_step_json_reads_memory_context_and_writes_observations(tmp_path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [MemoryWriteRecord(content="Prior note: use calculator for arithmetic.")],
        namespace="agent-memory",
    )

    llm_client = SequenceLLMClient(
        response_texts=[
            '{"continue": true, "thought": "start"}',
            '{"tool_name": "calculator", "tool_input": {"expression": "8 * 8"}}',
            '{"continue": false, "thought": "done"}',
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

    result = agent.run("Compute 8 * 8.")

    assert result.success
    assert result.metadata["memory"]["enabled"] is True
    assert result.metadata["memory"]["retrieval_trace"][0]["count"] >= 1

    observation_matches = store.search(
        MemorySearchQuery(
            text="Compute 8 * 8",
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
            '{"continue": true, "thought": "start"}',
            "\n".join(
                [
                    'calc = call_tool("calculator", {"expression": "9 * 9"})',
                    'final_output = {"result": calc["result"]}',
                ]
            ),
            '{"continue": false, "thought": "done"}',
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

    result = agent.run("Compute 9 * 9.")

    assert result.success
    assert result.metadata["memory"]["enabled"] is True
    assert result.metadata["memory"]["retrieval_trace"][0]["count"] >= 1

    observation_matches = store.search(
        MemorySearchQuery(
            text="Compute 9 * 9",
            namespace="code-memory",
            metadata_filters={"kind": "multi_step_observation"},
            top_k=10,
        )
    )
    store.close()

    assert len(observation_matches) >= 1
