"""WorkflowRuntime tests for memory, artifacts, and tracing semantics."""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents._contracts._memory import MemoryWriteRecord
from design_research_agents._contracts._workflow import (
    LogicStep,
    MemoryReadStep,
    MemoryWriteStep,
    WorkflowArtifact,
    WorkflowStepResult,
)
from design_research_agents._memory._stores._sqlite_store import SQLiteMemoryStore
from design_research_agents._runtime._workflow._engine import WorkflowRuntime
from design_research_agents._tracing import Tracer


def test_workflow_runtime_memory_steps_fail_without_memory_store_binding() -> None:
    workflow = WorkflowRuntime()
    steps = [
        MemoryReadStep(
            step_id="read_memory",
            query_builder=lambda context: "design brief",
        ),
        MemoryWriteStep(
            step_id="write_memory",
            dependencies=("read_memory",),
            records_builder=lambda context: [{"content": "artifact"}],
        ),
    ]

    result = workflow.run(
        steps,
        execution_mode="sequential",
        failure_policy="propagate_failed_state",
    )

    assert not result.success
    assert result.step_results["read_memory"].status == "failed"
    assert result.step_results["read_memory"].metadata["stage"] == "memory_binding"
    assert result.step_results["write_memory"].status == "failed"
    assert result.step_results["write_memory"].metadata["stage"] == "memory_binding"


def test_workflow_runtime_memory_steps_succeed_and_emit_standardized_outputs(
    tmp_path,
) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    workflow = WorkflowRuntime(memory_store=store)
    steps = [
        MemoryWriteStep(
            step_id="write_memory",
            records_builder=lambda context: [{"content": "alpha design note", "metadata": {"kind": "note"}}],
            namespace="research",
        ),
        MemoryReadStep(
            step_id="read_memory",
            dependencies=("write_memory",),
            query_builder=lambda context: {
                "text": "alpha design",
                "metadata_filters": {"kind": "note"},
            },
            namespace="research",
            top_k=3,
        ),
    ]

    result = workflow.run(steps, execution_mode="sequential")
    store.close()

    assert result.success
    write_output = result.step_results["write_memory"].output
    read_output = result.step_results["read_memory"].output

    assert write_output["written"] == 1
    assert write_output["namespace"] == "research"
    assert isinstance(write_output["ids"], list)
    assert read_output["count"] >= 1
    assert read_output["namespace"] == "research"
    assert isinstance(read_output["matches"], list)
    assert read_output["query"]["namespace"] == "research"


def test_workflow_runtime_memory_steps_participate_in_dag_dependencies(
    tmp_path,
) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")
    store.write(
        [MemoryWriteRecord(content="preloaded design context", metadata={"kind": "context"})],
        namespace="workspace",
    )
    workflow = WorkflowRuntime(memory_store=store)
    steps = [
        MemoryReadStep(
            step_id="read_memory",
            query_builder=lambda context: "design context",
            namespace="workspace",
            top_k=1,
        ),
        LogicStep(
            step_id="postprocess",
            dependencies=("read_memory",),
            handler=lambda context: {"count": context["dependency_results"]["read_memory"]["output"]["count"]},
        ),
    ]

    result = workflow.run(steps, execution_mode="dag")
    store.close()

    assert result.success
    assert result.execution_order == ["read_memory", "postprocess"]
    assert result.step_results["postprocess"].output["count"] == 1


def test_workflow_runtime_artifact_builder_failure_preserves_output_artifacts() -> None:
    workflow = WorkflowRuntime()
    result = workflow.run(
        [
            LogicStep(
                step_id="make_artifact",
                handler=lambda context: {
                    "artifacts": [{"path": "from-output.txt", "mime": "text/plain"}],
                    "value": 1,
                },
                artifacts_builder=lambda context: (_ for _ in ()).throw(RuntimeError("broken manifest")),
            )
        ],
        execution_mode="sequential",
    )
    step_result = result.step_results["make_artifact"]

    assert result.success is False
    assert step_result.status == "failed"
    assert step_result.error == "Artifact builder failed: broken manifest"
    assert step_result.metadata["stage"] == "artifact_builder"
    assert [artifact.path for artifact in step_result.artifacts] == ["from-output.txt"]


def test_workflow_runtime_normalizes_nested_and_typed_artifact_entries() -> None:
    workflow = WorkflowRuntime()
    typed_artifact = WorkflowArtifact(path="typed.txt", mime="text/plain")

    result = workflow.run(
        [
            LogicStep(
                step_id="make_artifacts",
                handler=lambda context: {
                    "artifacts": [
                        typed_artifact,
                        object(),
                        {"path": "  "},
                        {"path": "outer.txt", "mime": "text/plain", "metadata": {"kind": "outer"}},
                    ],
                    "result": {
                        "artifacts": [
                            {
                                "path": "nested.json",
                                "mime": "application/json",
                                "producer_step_id": "nested_producer",
                                "source_field": "result.artifacts",
                            }
                        ]
                    },
                    "output": {"artifacts": [{"path": "deep.bin"}]},
                },
            )
        ],
        execution_mode="sequential",
    )
    artifacts = result.step_results["make_artifacts"].artifacts

    assert result.success is True
    assert [artifact.path for artifact in artifacts] == [
        "typed.txt",
        "outer.txt",
        "nested.json",
        "deep.bin",
    ]
    assert artifacts[0].producer_step_id == "make_artifacts"
    assert artifacts[0].sources[0].step_id == "make_artifacts"
    assert artifacts[1].metadata == {"kind": "outer"}
    assert artifacts[2].producer_step_id == "nested_producer"
    assert artifacts[2].sources[0].field == "result.artifacts"
    assert result.metadata["artifact_count"] == 4


def test_workflow_runtime_private_artifact_collection_and_final_output_helpers() -> None:
    workflow = WorkflowRuntime()
    artifact = WorkflowArtifact(path="kept.txt", mime="text/plain")
    completed = WorkflowStepResult(
        step_id="completed",
        status="completed",
        success=True,
        output={"final_output": {"answer": 42}},
        artifacts=(artifact,),
    )
    fallback_completed = WorkflowStepResult(
        step_id="fallback",
        status="completed",
        success=True,
        output={"value": "fallback"},
    )
    failed = WorkflowStepResult(
        step_id="failed",
        status="failed",
        success=False,
        output={"final_output": "ignored"},
    )

    collected = workflow._collect_artifacts(
        step_results={"completed": completed},
        execution_order=("missing", "completed"),
    )

    assert collected == [artifact]
    assert workflow._resolve_final_output(
        step_results={"failed": failed, "fallback": fallback_completed, "completed": completed},
        execution_order=("failed", "fallback", "completed"),
    ) == {"answer": 42}
    assert workflow._resolve_final_output(
        step_results={"fallback": fallback_completed},
        execution_order=("fallback", "missing"),
    ) == {"value": "fallback"}
    assert workflow._resolve_final_output(step_results={"failed": failed}, execution_order=("failed",)) == {}


def test_workflow_runtime_emits_step_context_and_result_events(tmp_path: Path) -> None:
    tracer = Tracer(
        trace_dir=tmp_path / "traces",
        enable_jsonl=True,
        enable_console=False,
    )
    workflow = WorkflowRuntime(tracer=tracer)
    result = workflow.run(
        [LogicStep(step_id="a", handler=lambda ctx: {"value": 1})],
        execution_mode="sequential",
        request_id="workflow-trace-test",
    )
    assert result.success

    trace_files = sorted((tmp_path / "traces").glob("run_*_workflow-trace-test.jsonl"))
    assert trace_files
    events = [json.loads(line) for line in trace_files[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
    event_types = [str(event.get("event_type")) for event in events]
    assert "WorkflowStepContextObserved" in event_types
    assert "WorkflowStepResultObserved" in event_types
