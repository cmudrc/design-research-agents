from __future__ import annotations

from pathlib import Path

import pytest

from design_research_agents.tools.core import _helpers, evaluation_tools, memory_tools, python_tools
from design_research_agents.tools.policy import ToolPolicy, ToolPolicyConfig


def _policy(tmp_path: Path) -> ToolPolicy:
    return ToolPolicy(ToolPolicyConfig(workspace_root=str(tmp_path)))


class _CaptureSource:
    def __init__(self) -> None:
        self.specs: list[object] = []
        self.handlers: list[object] = []

    def register_tool(self, *, spec: object, handler: object) -> None:
        self.specs.append(spec)
        self.handlers.append(handler)


class _StoreLike:
    def __init__(self, *, db_path: object = None) -> None:
        self.db_path = db_path
        self.closed = False

    def write(self, records: object, *, namespace: str) -> list[object]:
        del records, namespace
        return []

    def search(self, query: object) -> list[object]:
        del query
        return []

    def close(self) -> None:
        self.closed = True


def test_core_helpers_cover_success_and_error_paths() -> None:
    assert _helpers.get_str({"x": 7}, "x") == "7"
    assert _helpers.get_int({"x": " 12 "}, "x", default=1) == 12
    assert _helpers.get_int({"x": -5}, "x", default=1) == -5
    with pytest.raises(ValueError, match="'x' must be an integer"):
        _helpers.get_int({"x": True}, "x", default=1)
    with pytest.raises(ValueError, match="'x' must be an integer"):
        _helpers.get_int({"x": "bad"}, "x", default=1)

    assert _helpers.get_bool({"x": "YES"}, "x") is True
    assert _helpers.get_bool({"x": "n"}, "x") is False
    with pytest.raises(ValueError, match="'x' must be a boolean"):
        _helpers.get_bool({"x": "maybe"}, "x")


def test_register_python_tools_and_python_guard_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _CaptureSource()
    python_tools.register_python_tools(source)
    names = [spec.name for spec in source.specs]
    assert "python.sandbox" in names

    with pytest.raises(ValueError, match="code is required"):
        python_tools._python_sandbox_handler(
            {"code": "   "},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="context must be an object"):
        python_tools._python_sandbox_handler(
            {"code": "result = 1", "context": []},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="timeout_s must be within"):
        python_tools._python_sandbox_handler(
            {"code": "result = 1", "timeout_s": 99},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="result_key must be a non-empty string"):
        python_tools._python_sandbox_handler(
            {"code": "result = 1", "result_key": "   "},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="Use of banned name"):
        python_tools._python_sandbox_handler(
            {"code": "result = eval('1')"},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="Dunder attribute access is not allowed"):
        python_tools._python_sandbox_handler(
            {"code": "result = (1).__class__"},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="Calling dunder names is not allowed"):
        python_tools._python_sandbox_handler(
            {"code": "__safe__()\nresult = 1"},
            request_id="req",
            dependencies={},
        )

    with pytest.raises(ValueError, match="result value must be JSON serializable"):
        python_tools._coerce_json_object(object())
    assert python_tools._truncate_stdout("abcdef", max_bytes=3) == ("abc", True)

    # No SIGALRM branch.
    monkeypatch.delattr(python_tools.signal, "SIGALRM", raising=False)
    with python_tools.execution_timeout(seconds=1):
        pass


@pytest.mark.skipif(not hasattr(python_tools.signal, "SIGALRM"), reason="POSIX-only signal branch")
def test_execution_timeout_signal_registration_error_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        python_tools.signal,
        "signal",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not-main-thread")),
    )
    with python_tools.execution_timeout(seconds=1):
        pass


def test_memory_tools_resolve_and_validation_branches(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    with pytest.raises(ValueError, match="text is required"):
        memory_tools._memory_search({"text": "   "}, dependencies={}, policy=policy)
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        memory_tools._memory_search({"text": "a", "top_k": 0}, dependencies={}, policy=policy)
    with pytest.raises(ValueError, match="metadata_filters must be an object"):
        memory_tools._extract_metadata_filters([])
    with pytest.raises(ValueError, match="min_score must be a number"):
        memory_tools._extract_optional_float(True, key="min_score")
    with pytest.raises(ValueError, match="min_score must be a number"):
        memory_tools._extract_optional_float("bad", key="min_score")

    with pytest.raises(ValueError, match="records must be a list"):
        memory_tools._normalize_write_records({})
    with pytest.raises(ValueError, match="records\\[0\\] must be an object"):
        memory_tools._normalize_write_records(["bad"])
    with pytest.raises(ValueError, match="records\\[0\\]\\.content must be a non-empty string"):
        memory_tools._normalize_write_records([{"content": "  "}])
    with pytest.raises(ValueError, match="records\\[0\\]\\.metadata must be an object"):
        memory_tools._normalize_write_records([{"content": "x", "metadata": []}])

    assert memory_tools._is_memory_store_like(object()) is False
    store = _StoreLike()
    memory_tools._close_store(store)
    assert store.closed is True

    with pytest.raises(ValueError, match="dependencies\\['memory_store'\\] must expose"):
        memory_tools._resolve_store(
            input_dict={},
            dependencies={"memory_store": object()},
            policy=policy,
        )

    path_store = _StoreLike(db_path=tmp_path / "artifacts" / "memory" / "from-path.sqlite3")
    resolved_path_store = memory_tools._resolve_store(
        input_dict={},
        dependencies={"memory_store": path_store},
        policy=policy,
    )
    assert resolved_path_store.close_after is False
    assert resolved_path_store.db_path.name == "from-path.sqlite3"

    str_store = _StoreLike(db_path=str(tmp_path / "artifacts" / "memory" / "from-str.sqlite3"))
    resolved_str_store = memory_tools._resolve_store(
        input_dict={},
        dependencies={"memory_store": str_store},
        policy=policy,
    )
    assert resolved_str_store.db_path.name == "from-str.sqlite3"

    fallback_store = _StoreLike(db_path=None)
    resolved_fallback = memory_tools._resolve_store(
        input_dict={},
        dependencies={"memory_store": fallback_store},
        policy=policy,
    )
    assert resolved_fallback.db_path.name == "memory.sqlite3"

    explicit_stats = memory_tools._resolve_stats_db_path(
        input_dict={"db_path": "artifacts/memory/custom.sqlite3"},
        dependencies={},
        policy=policy,
    )
    assert explicit_stats.name == "custom.sqlite3"

    dep_stats = memory_tools._resolve_stats_db_path(
        input_dict={},
        dependencies={"memory_store": _StoreLike(db_path="artifacts/memory/dep.sqlite3")},
        policy=policy,
    )
    assert dep_stats.name == "dep.sqlite3"


def test_evaluation_tools_error_and_edge_branches() -> None:
    source = _CaptureSource()
    evaluation_tools.register_evaluation_tools(source)
    names = [spec.name for spec in source.specs]
    assert "eval.decision_matrix" in names
    assert "eval.pairwise_rank" in names

    with pytest.raises(ValueError, match="Only method='copeland' is supported"):
        evaluation_tools._pairwise_rank_handler(
            {"alternatives": ["A", "B"], "comparisons": [], "method": "elo"},
            request_id="req",
            dependencies={},
        )
    with pytest.raises(ValueError, match="comparisons must be a list"):
        evaluation_tools._pairwise_rank_handler(
            {"alternatives": ["A", "B"], "comparisons": {}},
            request_id="req",
            dependencies={},
        )

    scores, matrix = evaluation_tools._initialize_pairwise_trackers(["A", "B"])
    evaluation_tools._apply_pairwise_outcome(
        scores=scores,
        matrix=matrix,
        a_name="A",
        b_name="B",
        outcome="tie",
        index=0,
    )
    assert scores["A"]["ties"] == 1.0
    assert matrix[("A", "B")]["ties"] == 1
    with pytest.raises(ValueError, match="unknown alternatives"):
        evaluation_tools._apply_pairwise_outcome(
            scores=scores,
            matrix=matrix,
            a_name="A",
            b_name="C",
            outcome="a",
            index=1,
        )
    with pytest.raises(ValueError, match="distinct alternatives"):
        evaluation_tools._apply_pairwise_outcome(
            scores=scores,
            matrix=matrix,
            a_name="A",
            b_name="A",
            outcome="a",
            index=2,
        )

    assert (
        evaluation_tools._decision_value(
            alternatives=[evaluation_tools._Alternative("A", {"cost": 2.0})],
            criterion=evaluation_tools._Criterion(name="cost", goal="min", weight=1.0),
            raw_value=2.0,
            normalize=False,
        )
        == -2.0
    )
    assert (
        evaluation_tools._decision_value(
            alternatives=[evaluation_tools._Alternative("A", {"same": 1.0})],
            criterion=evaluation_tools._Criterion(name="same", goal="max", weight=1.0),
            raw_value=1.0,
            normalize=True,
        )
        == 1.0
    )
    with pytest.raises(ValueError, match="missing criterion"):
        evaluation_tools._criterion_values(
            alternatives=[evaluation_tools._Alternative("A", {"x": 1.0})],
            criterion_name="y",
        )

    with pytest.raises(ValueError, match="unique names"):
        evaluation_tools._parse_pairwise_alternatives(["A", "A"])
    with pytest.raises(ValueError, match="must be a string or object"):
        evaluation_tools._parse_pairwise_alternatives([1])

    assert evaluation_tools._parse_comparison_entry(
        {"winner": "A", "loser": "B"},
        index=0,
    ) == ("A", "B", "a")
    with pytest.raises(ValueError, match="winner/loser must be non-empty"):
        evaluation_tools._parse_comparison_entry({"winner": "A", "loser": " "}, index=1)
    with pytest.raises(ValueError, match="requires non-empty a and b"):
        evaluation_tools._parse_comparison_entry({"a": "A", "b": " ", "outcome": "a"}, index=2)
    with pytest.raises(ValueError, match="outcome must be 'a', 'b', or 'tie'"):
        evaluation_tools._parse_comparison_entry({"a": "A", "b": "B", "outcome": "x"}, index=3)

    with pytest.raises(ValueError, match="criteria must be a non-empty list"):
        evaluation_tools._parse_criteria({})
    with pytest.raises(ValueError, match="alternatives must be a non-empty list"):
        evaluation_tools._parse_alternatives({})
    with pytest.raises(ValueError, match="must be a number"):
        evaluation_tools._coerce_number(True, label="field")
    with pytest.raises(ValueError, match="must be a number"):
        evaluation_tools._coerce_number("not-a-number", label="field")
