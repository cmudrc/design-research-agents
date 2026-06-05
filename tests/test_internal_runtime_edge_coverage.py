from __future__ import annotations

import asyncio
import builtins
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMResponse
from design_research_agents._contracts._tools import (
    ToolCostHints,
    ToolResult,
    ToolRuntime,
    ToolSpec,
)
from design_research_agents._implementations._patterns import (
    _router_delegate_pattern as _pattern_import_anchor,
)
from design_research_agents._mcp_server import _server as mcp_server
from design_research_agents._runtime._common._prompt_inputs import normalize_prompt_like_input
from design_research_agents._runtime._patterns import _batch_results as batch_results
from design_research_agents._runtime._patterns._run_context import (
    WorkflowBudgetTracker,
    attach_pattern_workflow,
    build_pattern_failure_result,
    build_workflow_output_payload,
    execute_pattern_with_trace,
    resolve_pattern_run_context,
)


class _RuntimeWithSchema(ToolRuntime):
    def __init__(self, result: ToolResult | None = None) -> None:
        self.invocations: list[tuple[str, dict[str, object]]] = []
        self._result = result

    def list_tools(self) -> Sequence[ToolSpec]:
        return (
            ToolSpec(
                name="odd tool",
                description="Exercise MCP schema conversion.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "required_text": {"type": "string"},
                        "optional-count": {"type": ["integer", "null"]},
                        "payload": {"type": "object"},
                    },
                    "required": ["required_text"],
                },
                output_schema={"type": "object"},
            ),
        )

    def invoke(
        self,
        tool_name: str,
        input: Mapping[str, object],
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ToolResult:
        del request_id, dependencies
        payload = dict(input)
        self.invocations.append((tool_name, payload))
        return self._result or ToolResult(tool_name=tool_name, ok=True, result={"payload": payload})

    def close(self) -> None:
        return None

    def __enter__(self) -> _RuntimeWithSchema:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()


def test_delegate_batch_result_helpers_cover_malformed_and_serialized_payloads() -> None:
    assert _pattern_import_anchor.__name__.endswith("_router_delegate_pattern")
    assert (
        batch_results.extract_delegate_batch_call_result_from_context(
            context={},
            dependency_step_id="dep",
            call_id="c",
        )
        is None
    )
    assert (
        batch_results.extract_delegate_batch_call_result_from_context(
            context={"dependency_results": {"dep": {"output": {"results": [object()]}}}},
            dependency_step_id="dep",
            call_id="c",
        )
        is None
    )

    serialized = {"text": "ok", "latency_ms": 12}
    context = {
        "dependency_results": {
            "dep": {
                "output": {
                    "results": [
                        {"call_id": "other", "output": {"unused": True}},
                        {"call_id": "c", "success": True, "output": {"value": 1}, "model_response": serialized},
                    ]
                }
            }
        }
    }
    call_result = batch_results.extract_delegate_batch_call_result_from_context(
        context=context,
        dependency_step_id="dep",
        call_id="c",
    )

    assert batch_results.extract_delegate_batch_call_result(results=[{"call_id": "c"}], call_id="c") == {"call_id": "c"}
    assert batch_results.extract_delegate_batch_call_result(results=[{"call_id": "x"}], call_id="c") is None
    assert batch_results.extract_call_output(call_result) == {"value": 1}
    assert batch_results.extract_call_output(None) == {}
    assert batch_results.extract_call_model_response(call_result) == LLMResponse(text="ok", latency_ms=12)
    assert batch_results.extract_call_model_response({"model_response": {"bad": object()}}) is None
    response = LLMResponse(text="hydrated")
    assert batch_results.extract_call_model_response({"model_response": response}) is response
    assert batch_results.extract_call_model_response(None) is None
    assert batch_results.is_call_success(call_result) is True
    assert batch_results.is_call_success(None) is False
    assert batch_results.extract_call_error({"error": " boom "}, fallback_message="fallback") == "boom"
    assert batch_results.extract_call_error({"output": {"error": " nested "}}, fallback_message="fallback") == "nested"
    assert batch_results.extract_call_error(None, fallback_message="fallback") == "fallback"


def test_prompt_like_input_normalizes_problem_metadata_and_fallbacks() -> None:
    class _BadRenderer:
        def render_brief(self, unexpected: object) -> str:
            return "unreachable"

        statement_markdown = " Statement "
        metadata = SimpleNamespace(problem_id=Path("problem.toml"), title=SimpleNamespace(value="T"), kind="decision")
        candidate_kind = SimpleNamespace(value=7)
        family = Path("families/demo")

    normalized = normalize_prompt_like_input(_BadRenderer())

    assert normalized["prompt"] == "Statement"
    assert normalized["problem_metadata"] == {
        "problem_id": "problem.toml",
        "title": "T",
        "kind": "decision",
        "candidate_kind": 7,
        "family": "families/demo",
    }

    class _ExplodingRenderer:
        def render_brief(self) -> str:
            raise RuntimeError("ignore")

        brief = " Brief "

    assert normalize_prompt_like_input(_ExplodingRenderer())["prompt"] == "Brief"
    assert normalize_prompt_like_input(type("PromptOnly", (), {"prompt": " Prompt "})())["prompt"] == "Prompt"

    with pytest.raises(TypeError, match="string prompt or a problem-like object"):
        normalize_prompt_like_input({"prompt": "mapping disallowed here"})


def test_pattern_run_context_budget_and_trace_helpers_cover_edges() -> None:
    tracker = WorkflowBudgetTracker()
    tracker.add_model_response(None)
    tracker.add_model_response(LLMResponse(text="slow", latency_ms=15))
    tracker.add_model_response(LLMResponse(text="unknown", latency_ms=-1))
    tracker.add_tool_results(
        tool_results=[
            ToolResult(tool_name="priced", ok=True),
            ToolResult(tool_name="unknown", ok=True),
        ],
        tool_specs={
            "priced": ToolSpec(
                name="priced",
                description="Costs money.",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                cost_hints=ToolCostHints(usd_cost_estimate=0.125),
            )
        },
    )
    assert tracker.as_metadata() == {
        "observed_latency_ms": 15,
        "observed_model_calls": 2,
        "observed_tool_calls": 2,
        "observed_estimated_usd": 0.125,
    }

    run_context = resolve_pattern_run_context(
        prompt="hello",
        default_request_id_prefix="pattern",
        default_dependencies={"a": 1},
        request_id=None,
        dependencies={"b": 2},
    )
    assert run_context.prompt == "hello"
    assert run_context.dependencies == {"a": 1, "b": 2}
    assert run_context.request_id.startswith("pattern:")

    result = execute_pattern_with_trace(
        agent_name="agent",
        request_id="req",
        input_payload={"prompt": "p"},
        dependencies={},
        tracer=None,
        runner=lambda: ExecutionResult(success=True, output={"final_output": {"ok": True}}),
    )
    assert result.success is True

    with pytest.raises(RuntimeError, match="boom"):
        execute_pattern_with_trace(
            agent_name="agent",
            request_id="req",
            input_payload={},
            dependencies={},
            tracer=None,
            runner=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    pattern = SimpleNamespace()
    workflow = object()
    assert attach_pattern_workflow(pattern, workflow) is workflow
    assert pattern.workflow is workflow

    workflow_result = ExecutionResult(success=True, output={"artifacts": [{"path": "a"}]})
    assert build_workflow_output_payload(workflow_result)["artifacts"] == [{"path": "a"}]

    failure = build_pattern_failure_result(
        error="failed",
        model_response=None,
        request_id="req",
        dependencies={"dep": True},
        metadata={"stage": "test"},
        output={"terminated_reason": "bad"},
    )
    assert failure.success is False
    assert failure.output["error"] == "failed"
    assert failure.metadata["stage"] == "test"


def test_mcp_server_helpers_cover_schema_runtime_and_jsonable_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RuntimeWithSchema()
    runtime_tool = mcp_server._build_runtime_tool(runtime=runtime, spec=runtime.list_tools()[0])
    signature = runtime_tool.__signature__

    assert "optional_count" in signature.parameters
    assert runtime_tool.__name__ == "tool_odd_tool"
    payload = asyncio.run(runtime_tool(required_text="hello", optional_count=None, payload={"x": 1}))
    assert payload["result"] == {"payload": {"required_text": "hello", "payload": {"x": 1}}}
    assert runtime.invocations == [("odd tool", {"required_text": "hello", "payload": {"x": 1}})]

    failing_tool = mcp_server._build_runtime_tool(
        runtime=_RuntimeWithSchema(ToolResult(tool_name="odd tool", ok=False, error="nope")),
        spec=runtime.list_tools()[0],
    )
    with pytest.raises(ValueError, match="nope"):
        asyncio.run(failing_tool(required_text="hello"))

    assert mcp_server._signature_from_schema(tool_name="plain", input_schema={"type": "array"}).parameters == {}
    assert (
        mcp_server._signature_from_schema(
            tool_name="plain",
            input_schema={"type": "object", "properties": []},
        ).parameters
        == {}
    )
    assert (
        mcp_server._signature_from_schema(
            tool_name="plain",
            input_schema={"properties": {1: {"type": "string"}, "x": object()}, "required": "x"},
        )
        .parameters["x"]
        .default
        is None
    )
    with pytest.raises(ValueError, match="Cannot expose tool"):
        mcp_server._signature_from_schema(
            tool_name="bad",
            input_schema={"properties": {"a-b": {"type": "string"}, "a_b": {"type": "string"}}},
        )

    assert mcp_server._annotation_from_schema({"type": ["null", "number"]}) is float
    assert mcp_server._annotation_from_schema({"type": []}) is object
    assert mcp_server._annotation_from_schema({"type": "boolean"}) is bool
    assert mcp_server._annotation_from_schema({"type": "array"}) == list[object]
    assert mcp_server._annotation_from_schema({"type": "object"}) == dict[str, object]
    assert mcp_server._annotation_from_schema(object()) is object

    @dataclass
    class _Payload:
        path: Path

    sentinel = object()
    converted = mcp_server._to_jsonable(
        {
            "path": Path("x.txt"),
            "dataclass": _Payload(Path("nested.txt")),
            "sequence": (b"hi", sentinel),
        }
    )
    assert converted == {
        "path": "x.txt",
        "dataclass": {"path": "nested.txt"},
        "sequence": ["hi", str(sentinel)],
    }
    assert mcp_server._safe_identifier("1 bad-name") == "tool_1_bad_name"

    server_wrapper = object.__new__(mcp_server.StdioMcpServer)
    with pytest.raises(RuntimeError, match="process stdin/stdout"):
        mcp_server.StdioMcpServer.serve(server_wrapper, stdin=object(), stdout=sys.stdout)

    real_import = builtins.__import__

    def _missing_mcp(name: str, *args: object, **kwargs: object) -> object:
        if name == "mcp.server.fastmcp":
            raise ImportError("missing mcp")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_mcp)
    with pytest.raises(mcp_server.McpServerDependencyError, match="official MCP Python SDK"):
        mcp_server._import_fastmcp()
