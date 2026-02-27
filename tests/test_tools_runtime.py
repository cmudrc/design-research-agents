from __future__ import annotations

from pathlib import Path

import pytest

import design_research_agents.tools._registry as tool_registry
from design_research_agents.tools import CallableToolConfig, Toolbox


def _bashkit_available() -> bool:
    try:
        import bashkit  # noqa: F401
    except Exception:
        return False
    return True


def test_unified_runtime_lists_expected_core_tools() -> None:
    runtime = Toolbox()
    names = {spec.name for spec in runtime.list_tools()}

    assert "python.sandbox" in names
    assert "text.word_count" in names
    assert "bash.exec" in names
    assert "fs.read_text" in names
    assert "memory.search" in names
    assert "memory.write" in names
    assert "memory.stats" in names
    assert "eval.decision_matrix" in names
    assert "eval.pairwise_rank" in names
    assert "run.command" not in names
    assert "fs.read_json" not in names
    assert "fs.write_json" not in names
    assert "data.aggregate" not in names
    assert "eval.confidence_fuse" not in names


def test_text_word_count_invocation() -> None:
    runtime = Toolbox()

    result = runtime.invoke(
        "text.word_count",
        {"text": "design research agents"},
        request_id="unit-test",
        dependencies={},
    )

    assert result.ok is True
    assert isinstance(result.result, dict)
    assert result.result["word_count"] == 3


def test_invoke_dict_returns_mapping_payload() -> None:
    runtime = Toolbox()
    payload = runtime.invoke_dict(
        "text.word_count",
        {"text": "design research agents"},
        request_id="invoke-dict-success",
        dependencies={},
    )
    assert payload["word_count"] == 3


def test_invoke_dict_raises_on_failed_tool_result() -> None:
    runtime = Toolbox()
    with pytest.raises(RuntimeError, match="failed"):
        runtime.invoke_dict(
            "fs.write_text",
            {"path": "outside.txt", "content": "blocked"},
            request_id="invoke-dict-failure",
            dependencies={},
        )


def test_invoke_dict_raises_on_non_mapping_tool_result() -> None:
    runtime = Toolbox(
        enable_core_tools=False,
        callable_tools=(
            CallableToolConfig(
                name="custom.scalar",
                description="Return scalar",
                handler=lambda _payload: 7,
                output_schema={"type": "integer"},
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="non-dict result"):
        runtime.invoke_dict(
            "custom.scalar",
            {},
            request_id="invoke-dict-non-dict",
            dependencies={},
        )


def test_tool_registry_emits_observation_events(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = Toolbox()
    captured: dict[str, list[dict[str, object]]] = {"invocations": [], "results": []}

    monkeypatch.setattr(
        tool_registry,
        "emit_tool_invocation_observed",
        lambda **kwargs: captured["invocations"].append(dict(kwargs)),
    )
    monkeypatch.setattr(
        tool_registry,
        "emit_tool_result_observed",
        lambda **kwargs: captured["results"].append(dict(kwargs)),
    )

    result = runtime.invoke(
        "text.word_count",
        {"text": "design research agents"},
        request_id="unit-test-observed",
        dependencies={"trace_id": "abc"},
    )

    assert result.ok is True
    assert captured["invocations"]
    assert captured["results"]
    assert captured["invocations"][-1]["tool_name"] == "text.word_count"
    assert captured["results"][-1]["ok"] is True


def test_fs_write_is_restricted_to_artifacts_by_default(tmp_path: Path) -> None:
    runtime = Toolbox(workspace_root=str(tmp_path))

    blocked = runtime.invoke(
        "fs.write_text",
        {"path": "outside.txt", "content": "nope"},
        request_id="unit-test",
        dependencies={},
    )
    assert blocked.ok is False
    assert blocked.error is not None
    assert "artifacts" in blocked.error.message

    allowed = runtime.invoke(
        "fs.write_text",
        {"path": "artifacts/ok.txt", "content": "yes", "overwrite": True},
        request_id="unit-test",
        dependencies={},
    )
    assert allowed.ok is True
    assert (tmp_path / "artifacts" / "ok.txt").read_text(encoding="utf-8") == "yes"


def test_bash_exec_enforces_invocation_allowlist() -> None:
    runtime = Toolbox()
    result = runtime.invoke(
        "bash.exec",
        {
            "script": "python3 -c \"print('hello')\"",
            "allowed_commands": ["git"],
        },
        request_id="unit-test",
        dependencies={},
    )

    assert result.ok is False
    assert result.error is not None
    assert "allowed_commands" in result.error.message
    assert "python3" in result.error.message


def test_python_sandbox_invocation() -> None:
    runtime = Toolbox()
    result = runtime.invoke(
        "python.sandbox",
        {
            "code": (
                "values = context['values']\nresult = {'sum': sum(values), 'mean': mean(values)}\nprint('done')\n"
            ),
            "context": {"values": [1, 2, 3]},
        },
        request_id="unit-test",
        dependencies={},
    )
    assert result.ok is True
    assert isinstance(result.result, dict)
    payload = result.result
    assert payload["result"]["sum"] == 6
    assert payload["stdout"] == "done\n"
    assert payload["truncated"] is False


def test_memory_tools_invocation_round_trip(tmp_path: Path) -> None:
    runtime = Toolbox(workspace_root=str(tmp_path))
    db_path = "artifacts/memory/runtime_tools.sqlite3"

    write_result = runtime.invoke(
        "memory.write",
        {
            "db_path": db_path,
            "namespace": "unit",
            "records": [{"content": "alpha design note", "metadata": {"kind": "note"}}],
        },
        request_id="unit-test",
        dependencies={},
    )
    assert write_result.ok is True
    assert isinstance(write_result.result, dict)
    assert write_result.result["written"] == 1

    search_result = runtime.invoke(
        "memory.search",
        {
            "db_path": db_path,
            "namespace": "unit",
            "text": "alpha",
            "top_k": 3,
        },
        request_id="unit-test",
        dependencies={},
    )
    assert search_result.ok is True
    assert isinstance(search_result.result, dict)
    assert search_result.result["count"] == 1
    assert search_result.result["retrieval_mode"] == "lexical"

    stats_result = runtime.invoke(
        "memory.stats",
        {"db_path": db_path, "namespace": "unit"},
        request_id="unit-test",
        dependencies={},
    )
    assert stats_result.ok is True
    assert isinstance(stats_result.result, dict)
    assert stats_result.result["record_count"] == 1


def test_evaluation_tools_invocation() -> None:
    runtime = Toolbox()

    decision = runtime.invoke(
        "eval.decision_matrix",
        {
            "alternatives": [
                {"id": "A", "scores": {"cost": 5, "quality": 8}},
                {"id": "B", "scores": {"cost": 8, "quality": 6}},
            ],
            "criteria": [
                {"name": "cost", "goal": "min", "weight": 0.4},
                {"name": "quality", "goal": "max", "weight": 0.6},
            ],
            "normalize": True,
        },
        request_id="unit-test",
        dependencies={},
    )
    assert decision.ok is True
    assert isinstance(decision.result, dict)
    assert decision.result["ranked"][0]["alternative"] == "A"

    pairwise = runtime.invoke(
        "eval.pairwise_rank",
        {
            "alternatives": ["A", "B", "C"],
            "comparisons": [
                {"a": "A", "b": "B", "outcome": "a"},
                {"a": "A", "b": "C", "outcome": "a"},
                {"a": "B", "b": "C", "outcome": "b"},
            ],
        },
        request_id="unit-test",
        dependencies={},
    )
    assert pairwise.ok is True
    assert isinstance(pairwise.result, dict)
    assert pairwise.result["ranking"][0]["alternative"] == "A"


@pytest.mark.skipif(
    not _bashkit_available(),
    reason="bashkit dependency is not available in this environment",
)
def test_bash_exec_returns_structured_result_and_stays_sandboxed(
    tmp_path: Path,
) -> None:
    runtime = Toolbox(workspace_root=str(tmp_path))

    unique_host_path = tmp_path / "host_should_not_exist.txt"
    if unique_host_path.exists():
        unique_host_path.unlink()

    success_result = runtime.invoke(
        "bash.exec",
        {"script": "echo 'hello from bashkit'", "allowed_commands": ["echo"]},
        request_id="unit-test",
        dependencies={},
    )
    assert success_result.ok is True
    assert isinstance(success_result.result, dict)
    success_payload = success_result.result
    assert success_payload["success"] is True
    assert "hello from bashkit" in str(success_payload["stdout"])

    sandbox_result = runtime.invoke(
        "bash.exec",
        {
            "script": f"echo 'virtual write' > {unique_host_path}",
            "allowed_commands": ["echo"],
        },
        request_id="unit-test",
        dependencies={},
    )

    assert sandbox_result.ok is True
    assert isinstance(sandbox_result.result, dict)
    payload = sandbox_result.result
    assert "success" in payload
    assert unique_host_path.exists() is False
