"""Tests for tree-search and RAG reasoning patterns."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.memory import MemorySearchQuery, MemoryWriteRecord
from design_research_agents.implementations.patterns import (
    rag_reasoning as rag_reasoning_impl,
)
from design_research_agents.implementations.patterns import (
    tree_search as tree_search_impl,
)
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from design_research_agents.workflow import RagReasoningPattern, TreeSearchPattern


class _CaptureReasoningAgent(Agent):
    """Reasoning delegate that captures prompt text for assertions."""

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del request_id, dependencies
        self.last_prompt = prompt
        return ExecutionResult(
            output={
                "answer": "reasoned",
                "prompt_seen": prompt,
            },
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate": "capture"},
        )


class _StaticGeneratorAgent(Agent):
    """Generator agent with deterministic candidate payloads."""

    def __init__(self, *, output: Mapping[str, object], success: bool = True) -> None:
        self._output = dict(output)
        self._success = success

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(
            output=dict(self._output),
            success=self._success,
            tool_results=[],
            model_response=None,
            metadata={"role": "generator"},
        )


class _StaticEvaluatorAgent(Agent):
    """Evaluator agent with deterministic score payloads."""

    def __init__(self, *, output: Mapping[str, object], success: bool = True) -> None:
        self._output = dict(output)
        self._success = success

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        del prompt, request_id, dependencies
        return ExecutionResult(
            output=dict(self._output),
            success=self._success,
            tool_results=[],
            model_response=None,
            metadata={"role": "evaluator"},
        )


def test_tree_search_pattern_expands_scores_and_returns_best_candidate() -> None:
    def _generator(context: Mapping[str, object]) -> list[dict[str, object]]:
        depth = int(context.get("depth", 0))
        if depth == 1:
            return [
                {"name": "alpha", "score_hint": 0.2},
                {"name": "beta", "score_hint": 0.8},
            ]
        return [
            {"name": "beta_refined", "score_hint": 0.9},
            {"name": "beta_alt", "score_hint": 0.3},
        ]

    def _evaluator(context: Mapping[str, object]) -> float:
        candidate = context.get("candidate")
        if isinstance(candidate, Mapping):
            score = candidate.get("score_hint")
            if isinstance(score, (int, float)):
                return float(score)
        return 0.0

    pattern = TreeSearchPattern(
        generator_delegate=_generator,
        evaluator_delegate=_evaluator,
        max_depth=2,
        branch_factor=2,
        beam_width=1,
    )

    result = pattern.run("Find the best concept.")

    assert result.success
    assert pattern.workflow is not None
    assert result.output["best_candidate"]["name"] == "beta_refined"
    assert result.output["best_score"] == 0.9
    assert result.output["explored_nodes"] == 4
    assert len(result.output["frontier_trace"]) == 2


def test_tree_search_pattern_supports_agent_delegates() -> None:
    generator_agent = _StaticGeneratorAgent(
        output={
            "model_text": json.dumps(
                {
                    "candidates": [
                        {"name": "candidate_a", "score_hint": 0.4},
                        {"name": "candidate_b", "score_hint": 0.7},
                    ]
                }
            )
        }
    )
    evaluator_agent = _StaticEvaluatorAgent(output={"model_text": json.dumps({"score": 0.75})})
    pattern = TreeSearchPattern(
        generator_delegate=generator_agent,
        evaluator_delegate=evaluator_agent,
        max_depth=1,
        branch_factor=2,
        beam_width=2,
    )

    result = pattern.run("Rank concept options.")

    assert result.success
    assert pattern.workflow is not None
    assert result.output["explored_nodes"] == 2
    assert result.output["best_score"] == 0.75


def test_tree_search_helpers_cover_candidate_score_and_conversion_branches() -> None:
    assert tree_search_impl._extract_candidate_list({"candidates": [{"x": 1}, object()]}) == [
        {"x": 1}
    ]
    assert tree_search_impl._extract_candidate_list({"candidate": "single"}) == ["single"]
    assert tree_search_impl._extract_candidate_list(
        {"model_text": json.dumps([{"a": 1}, 2, None])}
    ) == [{"a": 1}, 2]
    assert tree_search_impl._extract_candidate_list(
        {"model_text": json.dumps({"candidate": {"b": 2}})}
    ) == [{"b": 2}]
    assert tree_search_impl._extract_candidate_list({"model_text": "not-json"}) == []

    assert tree_search_impl._extract_score({"score": 1}) == 1.0
    assert tree_search_impl._extract_score({"model_text": json.dumps({"score": 2.5})}) == 2.5
    assert tree_search_impl._extract_score({"model_text": "not-json"}) == 0.0

    assert tree_search_impl._safe_float(True) == 1.0
    assert tree_search_impl._safe_float(" 2.25 ") == 2.25
    assert tree_search_impl._safe_float("bad") == 0.0
    assert tree_search_impl._safe_int(True) == 1
    assert tree_search_impl._safe_int(2.9) == 2
    assert tree_search_impl._safe_int(" 7 ") == 7
    assert tree_search_impl._safe_int("bad") == 0

    json_ready_value = tree_search_impl._json_ready({"k": (1, "x"), "v": [True, object()]})
    assert isinstance(json_ready_value, dict)
    assert json_ready_value["k"] == [1, "x"]
    assert isinstance(json_ready_value["v"], list)
    assert isinstance(json_ready_value["v"][1], str)

    assert tree_search_impl._is_agent_like(_StaticGeneratorAgent(output={}))
    assert not tree_search_impl._is_agent_like(object())


def test_tree_search_handles_delegate_failures_and_invalid_config() -> None:
    with pytest.raises(ValueError, match="max_depth"):
        TreeSearchPattern(
            generator_delegate=lambda context: [],
            evaluator_delegate=lambda context: 0.0,
            max_depth=0,
        )
    with pytest.raises(ValueError, match="branch_factor"):
        TreeSearchPattern(
            generator_delegate=lambda context: [],
            evaluator_delegate=lambda context: 0.0,
            branch_factor=0,
        )
    with pytest.raises(ValueError, match="beam_width"):
        TreeSearchPattern(
            generator_delegate=lambda context: [],
            evaluator_delegate=lambda context: 0.0,
            beam_width=0,
        )

    failing_pattern = TreeSearchPattern(
        generator_delegate=_StaticGeneratorAgent(output={"candidates": []}, success=False),
        evaluator_delegate=_StaticEvaluatorAgent(output={"score": 1.0}),
        max_depth=2,
        branch_factor=2,
        beam_width=1,
    )
    result = failing_pattern.run("No children should be generated.")
    assert result.success
    assert result.output["explored_nodes"] == 0
    assert result.output["best_score"] == 0.0
    assert result.output["frontier_trace"] == []

    evaluator_failure_pattern = TreeSearchPattern(
        generator_delegate=lambda context: [{"name": "one"}],
        evaluator_delegate=_StaticEvaluatorAgent(output={"score": 10}, success=False),
        max_depth=1,
        branch_factor=1,
        beam_width=1,
    )
    evaluator_result = evaluator_failure_pattern.run("Evaluator failure should zero score.")
    assert evaluator_result.output["best_score"] == 0.0


def test_rag_reasoning_pattern_injects_retrieved_context_and_writes_back(
    tmp_path,
) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [
            MemoryWriteRecord(
                content="Safety requirement: include fail-safe shutdown.",
                metadata={"kind": "requirement"},
            )
        ],
        namespace="design",
    )

    reasoning_agent = _CaptureReasoningAgent()
    pattern = RagReasoningPattern(
        reasoning_delegate=reasoning_agent,
        memory_store=store,
        memory_namespace="design",
        memory_top_k=3,
        write_back=True,
    )

    result = pattern.run("Draft a design brief.")

    assert result.success
    assert pattern.workflow is not None
    assert result.output["retrieval"]["count"] >= 1
    assert result.output["write_back"]["written"] == 1
    assert reasoning_agent.last_prompt is not None
    assert "Retrieved context (JSON):" in reasoning_agent.last_prompt
    assert "fail-safe shutdown" in reasoning_agent.last_prompt

    write_back_matches = store.search(
        MemorySearchQuery(
            text="Draft a design brief",
            namespace="design",
            metadata_filters={"kind": "rag_reasoning"},
            top_k=5,
        )
    )
    store.close()

    assert len(write_back_matches) >= 1


def test_rag_reasoning_helpers_cover_edge_cases(tmp_path) -> None:
    with pytest.raises(ValueError, match="memory_top_k"):
        RagReasoningPattern(
            reasoning_delegate=_CaptureReasoningAgent(),
            memory_store=None,
            memory_top_k=0,
        )

    dependency_output = rag_reasoning_impl._extract_dependency_output(
        context={"dependency_results": {"memory_read": {"output": {"count": 3}}}},
        dependency_id="memory_read",
    )
    assert dependency_output == {"count": 3}
    assert rag_reasoning_impl._extract_dependency_output(context={}, dependency_id="x") == {}
    assert (
        rag_reasoning_impl._extract_dependency_output(
            context={"dependency_results": {"x": {"output": "bad"}}},
            dependency_id="x",
        )
        == {}
    )

    prompt = rag_reasoning_impl._build_reasoning_prompt(
        task_prompt="Draft",
        memory_read_step_output={
            "namespace": "ns",
            "count": "2",
            "matches": [
                {"item_id": "a", "score": 0.9, "content": "Keep redundancy."},
                {"item_id": "b", "content": ""},
                "invalid",
            ],
        },
    )
    assert "Retrieved context (JSON):" in prompt
    assert "Keep redundancy." in prompt
    assert "[a] score=0.9" in prompt
    assert "[b]" not in prompt

    records = rag_reasoning_impl._build_write_back_records(
        task_prompt="Task",
        reason_step_output={"output": "not-a-mapping"},
        memory_read_step_output={"matches": ["a", "b"]},
    )
    assert len(records) == 1
    assert records[0]["metadata"]["retrieved_count"] == 2
    assert rag_reasoning_impl._safe_int(True) == 1
    assert rag_reasoning_impl._safe_int(2.4) == 2
    assert rag_reasoning_impl._safe_int("5") == 5
    assert rag_reasoning_impl._safe_int("bad") == 0

    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write([MemoryWriteRecord(content="Context item")], namespace="ns")
    pattern = RagReasoningPattern(
        reasoning_delegate=_CaptureReasoningAgent(),
        memory_store=store,
        memory_namespace="ns",
        write_back=False,
    )
    result = pattern.run("Use context.")
    store.close()

    assert result.success
