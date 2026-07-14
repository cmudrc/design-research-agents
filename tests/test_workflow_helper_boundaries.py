from __future__ import annotations

import pytest

from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._workflow import (
    DelegateStep,
    LogicStep,
    ToolStep,
    WorkflowArtifact,
    WorkflowStepResult,
)
from design_research_agents._runtime._patterns._batch_results import (
    extract_call_error,
    extract_call_model_response,
    extract_call_output,
    extract_delegate_batch_call_result,
    extract_delegate_batch_call_result_from_context,
    is_call_success,
)
from design_research_agents._runtime._workflow._engine import WorkflowRuntime
from design_research_agents._runtime._workflow._step_context import (
    build_invocation_dependencies,
    build_step_context,
    has_upstream_failure,
    normalize_route_map,
    resolve_delegate_prompt,
    resolve_tool_input,
    route_deactivations,
)


@pytest.mark.parametrize(
    "context",
    [
        {},
        {"dependency_results": "invalid"},
        {"dependency_results": {"batch": "invalid"}},
        {"dependency_results": {"batch": {"output": "invalid"}}},
        {"dependency_results": {"batch": {"output": {"results": "invalid"}}}},
    ],
)
def test_context_batch_result_extraction_rejects_malformed_shapes(context: dict[str, object]) -> None:
    assert (
        extract_delegate_batch_call_result_from_context(
            context=context,
            dependency_step_id="batch",
            call_id="target",
        )
        is None
    )


def test_batch_result_helpers_normalize_success_output_response_and_errors() -> None:
    response = LLMResponse(text="answer")
    context = {
        "dependency_results": {
            "batch": {
                "output": {
                    "results": [
                        "invalid",
                        {"call_id": "other"},
                        {
                            "call_id": "target",
                            "success": True,
                            "output": {"value": 3},
                            "model_response": {"text": "serialized"},
                        },
                    ]
                }
            }
        }
    }

    call_result = extract_delegate_batch_call_result_from_context(
        context=context,
        dependency_step_id="batch",
        call_id="target",
    )
    assert call_result is not None
    assert (
        extract_delegate_batch_call_result(results=[{"call_id": "other"}, call_result], call_id="target") == call_result
    )
    assert extract_delegate_batch_call_result(results=[{"call_id": "other"}], call_id="target") is None
    assert extract_call_output(call_result) == {"value": 3}
    assert extract_call_output(None) == {}
    assert extract_call_output({"output": "invalid"}) == {}
    assert extract_call_model_response(call_result) == LLMResponse(text="serialized")
    assert extract_call_model_response({"model_response": response}) is response
    assert extract_call_model_response({"model_response": {"unknown": True}}) is None
    assert extract_call_model_response(None) is None
    assert is_call_success(call_result)
    assert not is_call_success(None)
    assert extract_call_error(None, fallback_message="fallback") == "fallback"
    assert extract_call_error({"error": " explicit "}, fallback_message="fallback") == "explicit"
    assert extract_call_error({"output": {"error": " nested "}}, fallback_message="fallback") == "nested"
    assert extract_call_error({"error": " ", "output": {}}, fallback_message="fallback") == "fallback"


def test_step_context_and_invocation_dependencies_copy_normalized_results() -> None:
    result = WorkflowStepResult(
        step_id="prepare",
        status="completed",
        success=True,
        output={"value": 7},
        metadata={"source": "test"},
    )
    context = build_step_context(
        base_context={"prompt": "base"},
        step_id="finish",
        step_dependencies=("missing", "prepare"),
        step_results={"prepare": result},
        request_id="request-1",
        execution_mode="parallel",
        failure_policy="continue",
        is_terminal_step=True,
        output_schema={"type": "object"},
    )

    assert context["dependency_results"] == {
        "prepare": {
            "status": "completed",
            "success": True,
            "output": {"value": 7},
            "error": None,
            "metadata": {"source": "test"},
        }
    }
    assert context["_workflow"] == {
        "request_id": "request-1",
        "step_id": "finish",
        "execution_mode": "parallel",
        "failure_policy": "continue",
        "dependency_count": 2,
        "is_terminal_step": True,
        "output_schema": {"type": "object"},
    }
    dependencies = build_invocation_dependencies(
        base_dependencies={"token": "value"},
        step_id="finish",
        request_id="request-1",
        execution_mode="parallel",
        failure_policy="continue",
        step_context=context,
    )
    assert dependencies["token"] == "value"
    assert dependencies["_workflow"]["dependency_results"] == context["dependency_results"]
    assert (
        build_invocation_dependencies(
            base_dependencies={},
            step_id="finish",
            request_id="request-1",
            execution_mode="sequential",
            failure_policy="skip_dependents",
            step_context={"dependency_results": "invalid"},
        )["_workflow"]["dependency_results"]
        == {}
    )


def test_tool_input_resolution_covers_static_builder_empty_and_invalid_paths() -> None:
    assert resolve_tool_input(step=ToolStep(step_id="empty", tool_name="tool"), step_context={}) == {}
    assert resolve_tool_input(
        step=ToolStep(step_id="static", tool_name="tool", input_data={"value": 1}),
        step_context={},
    ) == {"value": 1}
    assert resolve_tool_input(
        step=ToolStep(step_id="built", tool_name="tool", input_builder=lambda context: context),
        step_context={"value": 2},
    ) == {"value": 2}
    with pytest.raises(TypeError, match="input_builder must return a mapping"):
        resolve_tool_input(
            step=ToolStep(step_id="invalid", tool_name="tool", input_builder=lambda _context: "invalid"),  # type: ignore[arg-type,return-value]
            step_context={},
        )


def test_delegate_prompt_resolution_covers_all_sources_and_failures() -> None:
    assert (
        resolve_delegate_prompt(
            step=DelegateStep(step_id="built", delegate=object(), prompt_builder=lambda _context: " built "),
            step_context={},
        )
        == "built"
    )
    assert (
        resolve_delegate_prompt(
            step=DelegateStep(step_id="static", delegate=object(), prompt=" static "),
            step_context={},
        )
        == "static"
    )
    assert (
        resolve_delegate_prompt(
            step=DelegateStep(step_id="fallback", delegate=object()),
            step_context={"prompt": " fallback "},
        )
        == "fallback"
    )
    with pytest.raises(TypeError, match="prompt_builder must return a string"):
        resolve_delegate_prompt(
            step=DelegateStep(step_id="invalid", delegate=object(), prompt_builder=lambda _context: 3),  # type: ignore[arg-type,return-value]
            step_context={},
        )
    with pytest.raises(ValueError, match="returned an empty prompt"):
        resolve_delegate_prompt(
            step=DelegateStep(step_id="empty", delegate=object(), prompt_builder=lambda _context: " "),
            step_context={},
        )
    with pytest.raises(ValueError, match="requires a non-empty prompt"):
        resolve_delegate_prompt(step=DelegateStep(step_id="missing", delegate=object()), step_context={})


def test_failure_and_route_helpers_cover_invalid_and_transitive_paths() -> None:
    successful = WorkflowStepResult(step_id="ok", status="completed", success=True)
    failed = WorkflowStepResult(step_id="failed", status="failed", success=False)
    assert not has_upstream_failure(dependencies=("missing", "ok"), step_results={"ok": successful})
    assert has_upstream_failure(dependencies=("ok", "failed"), step_results={"ok": successful, "failed": failed})

    handler = lambda _context: {}  # noqa: E731
    assert route_deactivations(
        step=LogicStep(step_id="plain", handler=handler),
        step_output={},
        dependents={},
    ) == (set(), None)
    assert (
        route_deactivations(
            step=LogicStep(step_id="invalid-map", handler=handler, route_map={" ": ("target",)}),
            step_output={},
            dependents={},
        )[1]
        == "Step 'invalid-map' declared route_map but no valid routes were configured."
    )
    routed_step = LogicStep(
        step_id="route",
        handler=handler,
        route_map={" left ": (" left-step ",), "right": ("right-step",), 1: ("ignored",)},  # type: ignore[dict-item]
    )
    assert "non-empty 'route'" in (route_deactivations(step=routed_step, step_output={}, dependents={})[1] or "")
    assert "no targets were configured" in (
        route_deactivations(step=routed_step, step_output={"route": "unknown"}, dependents={})[1] or ""
    )
    deactivated, error = route_deactivations(
        step=routed_step,
        step_output={"route": "left"},
        dependents={"right-step": ("shared", "right-leaf"), "shared": ("leaf",), "leaf": ("right-step",)},
    )
    assert error is None
    assert deactivated == {"right-step", "shared", "right-leaf", "leaf"}
    assert normalize_route_map({" valid ": (" target ", "", 2), "": ("ignored",)}) == {  # type: ignore[dict-item]
        "valid": ("target",)
    }


def test_artifact_normalization_backfills_provenance_and_reads_nested_results() -> None:
    runtime = WorkflowRuntime()
    nested = runtime._coerce_artifacts_from_output(
        step_id="build",
        step_output={
            "result": {
                "artifacts": [
                    WorkflowArtifact(path="report.md", mime="text/markdown"),
                    {"path": "data.json", "mime": "application/json", "metadata": {"rows": 2}},
                    {"path": " "},
                    "invalid",
                ]
            }
        },
    )

    assert [artifact.path for artifact in nested] == ["report.md", "data.json"]
    assert nested[0].producer_step_id == "build"
    assert nested[0].sources[0].field == "artifacts"
    assert nested[1].metadata == {"rows": 2}
