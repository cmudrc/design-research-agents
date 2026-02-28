from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from design_research_agents._contracts._memory import MemoryWriteRecord
from design_research_agents._memory import EmbeddingProvider, SQLiteMemoryStore
from design_research_agents.tools._core import _bash_tools as bash_tools
from design_research_agents.tools._core import _evaluation_tools as evaluation_tools
from design_research_agents.tools._core import _fs_tools as fs_tools
from design_research_agents.tools._core import _git_tools as git_tools
from design_research_agents.tools._core import _memory_tools as memory_tools
from design_research_agents.tools._core import _python_tools as python_tools
from design_research_agents.tools._core import _search_tools as search_tools
from design_research_agents.tools._core import _text_tools as text_tools
from design_research_agents.tools._policy import ToolPolicy, ToolPolicyConfig


def _policy(tmp_path: Path) -> ToolPolicy:
    return ToolPolicy(ToolPolicyConfig(workspace_root=str(tmp_path)))


class _StaticEmbeddingProvider(EmbeddingProvider):
    def __init__(self, *, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    @property
    def model_name(self) -> str:
        return "test-embeddings"

    def embed(self, texts: list[str] | tuple[str, ...]) -> list[list[float]] | None:
        return [list(self._vectors_by_text.get(text, [0.0, 0.0])) for text in texts]


def test_search_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        search_tools._search({"query": "   ", "root": "."}, policy=_policy(tmp_path))


def test_search_prefers_rg_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")

    called: list[str] = []

    def _fake_rg(**kwargs: object) -> dict[str, object]:
        called.append("rg")
        assert kwargs["query"] == "alpha"
        return {"engine": "rg", "count": 1}

    def _fake_py(**kwargs: object) -> dict[str, object]:
        called.append("python")
        return {"engine": "python", "count": 0}

    monkeypatch.setattr(search_tools, "which", lambda _name: "/usr/bin/rg")
    monkeypatch.setattr(search_tools, "_search_with_rg", _fake_rg)
    monkeypatch.setattr(search_tools, "_search_with_python", _fake_py)

    result = search_tools._search({"query": "alpha", "root": "."}, policy=_policy(tmp_path))
    assert result["engine"] == "rg"
    assert called == ["rg"]


def test_search_uses_python_fallback_when_rg_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    monkeypatch.setattr(search_tools, "which", lambda _name: None)
    result = search_tools._search(
        {"query": "alpha", "root": ".", "max_matches": 2},
        policy=_policy(tmp_path),
    )
    assert result["engine"] == "python"
    assert result["count"] == 1


def test_search_with_rg_parses_matches_and_keeps_context_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        return SimpleNamespace(
            stdout="\n".join(
                [
                    "src/a.py:10:2:print('x')",
                    "not-a-match-line",
                    "src/b.py:3:1:hello",
                ]
            ),
            stderr="warning",
        )

    monkeypatch.setattr(search_tools.subprocess, "run", _fake_run)
    result = search_tools._search_with_rg(
        rg_binary="/usr/bin/rg",
        root=tmp_path,
        query="x",
        globs=["*.py"],
        max_matches=1,
        context_lines=2,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "-C" in command and "2" in command
    assert "-g" in command and "*.py" in command
    assert result["count"] == 1
    assert result["matches"][0]["ref"] == "src/a.py:10:2"
    assert result["stderr"] == "warning"


def test_search_with_python_skips_unreadable_and_limits_results(tmp_path: Path) -> None:
    (tmp_path / "ok.txt").write_text("first alpha\nsecond alpha\n", encoding="utf-8")
    (tmp_path / "bad.bin").write_bytes(b"\xff\xfe\x00\x01")
    result = search_tools._search_with_python(root=tmp_path, query="alpha", max_matches=1)
    assert result["engine"] == "python"
    assert result["count"] == 1
    assert result["matches"][0]["line"] == 1


def test_text_word_count_and_diff_handlers() -> None:
    stats = text_tools._word_count_handler(
        {"text": "Hello, hello!\nworld"},
        request_id="r",
        dependencies={},
    )
    assert stats == {
        "char_count": len("Hello, hello!\nworld"),
        "word_count": 3,
        "line_count": 2,
        "unique_word_count": 2,
    }

    diff = text_tools._diff_tool_handler(
        {"a": "a\nb\n", "b": "a\nc\n"},
        request_id="r",
        dependencies={},
    )["diff"]
    assert "--- a" in diff
    assert "+++ b" in diff
    assert "-b" in diff
    assert "+c" in diff


def test_text_extract_json_direct_and_embedded() -> None:
    direct = text_tools._extract_json_tool_handler(
        {"text": '{"a": 1}'},
        request_id="r",
        dependencies={},
    )
    assert direct["json"] == {"a": 1}

    embedded = text_tools._extract_json_tool_handler(
        {"text": 'prefix {"k": true} suffix'},
        request_id="r",
        dependencies={},
    )
    assert embedded["json"] == {"k": True}


def test_text_extract_json_rejects_ambiguous_or_invalid() -> None:
    with pytest.raises(ValueError, match="exactly one JSON object"):
        text_tools._extract_json_tool_handler(
            {"text": '{"a":1} {"b":2}'},
            request_id="r",
            dependencies={},
        )

    with pytest.raises(ValueError, match="exactly one JSON object"):
        text_tools._extract_json_tool_handler(
            {"text": "[1,2,3]"},
            request_id="r",
            dependencies={},
        )


def test_fs_helpers_cover_read_write_stat_hash_and_glob(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "a.txt").write_text("abcdef", encoding="utf-8")
    (tmp_path / "dir" / "b.log").write_text("log", encoding="utf-8")

    listed = fs_tools._list_dir(
        {"path": "dir", "pattern": "*.txt", "max_entries": 10},
        policy=policy,
    )
    assert listed["count"] == 1
    assert listed["entries"][0]["name"] == "a.txt"

    read_truncated = fs_tools._read_text({"path": "dir/a.txt", "max_bytes": 3}, policy=policy)
    assert read_truncated["text"] == "abc"
    assert read_truncated["truncated"] is True
    assert read_truncated["size_bytes"] == 3

    write_target = fs_tools._write_text(
        {"path": "artifacts/out.txt", "content": "hello"},
        policy=policy,
    )
    assert write_target["bytes_written"] == 5
    with pytest.raises(ValueError, match="overwrite=true"):
        fs_tools._write_text({"path": "artifacts/out.txt", "content": "x"}, policy=policy)
    fs_tools._write_text(
        {"path": "artifacts/out.txt", "content": "x", "overwrite": True},
        policy=policy,
    )

    globbed = fs_tools._glob({"path": "dir", "pattern": "*.txt"}, policy=policy)
    assert globbed["count"] == 1
    assert globbed["matches"][0].endswith("a.txt")

    stat = fs_tools._stat({"path": "dir/a.txt"}, policy=policy)
    assert stat["exists"] is True
    assert stat["is_file"] is True

    digest = fs_tools._hash({"path": "dir/a.txt", "algo": "sha256"}, policy=policy)
    assert digest["algo"] == "sha256"
    assert len(str(digest["digest"])) == 64

    with pytest.raises(ValueError, match="Unsupported hash algorithm"):
        fs_tools._hash({"path": "dir/a.txt", "algo": "__invalid_algo__"}, policy=policy)


def test_fs_read_text_rejects_blank_paths_and_directories(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    (tmp_path / "dir").mkdir()

    with pytest.raises(ValueError, match="'path' is required"):
        fs_tools._read_text({}, policy=policy)

    with pytest.raises(ValueError, match="Path is not a file"):
        fs_tools._read_text({"path": "dir"}, policy=policy)


def test_git_helpers_build_expected_arguments_and_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    policy = _policy(tmp_path)
    calls: list[list[str]] = []

    def _fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(git_tools.subprocess, "run", _fake_run)

    assert git_tools._git_status({"repo": "repo"}, policy=policy)["status"] == "ok"
    assert git_tools._git_diff({"repo": "repo", "staged": True, "pathspec": "a.py"}, policy=policy)["diff"] == "ok"
    assert git_tools._git_log({"repo": "repo", "max_commits": 7}, policy=policy)["log"] == "ok"
    assert git_tools._git_show({"repo": "repo", "rev": "HEAD~1"}, policy=policy)["show"] == "ok"

    assert any("--staged" in cmd for cmd in calls)
    assert any(cmd[-2:] == ["--", "a.py"] for cmd in calls)
    assert any(cmd[-4:] == ["log", "--oneline", "-n", "7"] for cmd in calls)

    with pytest.raises(ValueError, match="rev is required"):
        git_tools._git_show({"repo": "repo", "rev": "   "}, policy=policy)


def test_run_git_merges_stderr_and_appends_truncated_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    policy = ToolPolicy(
        ToolPolicyConfig(workspace_root=str(tmp_path), default_max_output_bytes=5),
    )

    def _fake_run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="abcdef", stderr="err")

    monkeypatch.setattr(git_tools.subprocess, "run", _fake_run)
    output = git_tools._run_git(policy=policy, repo=str(repo), args=["status"])
    assert output.endswith("[truncated]")


def test_bash_allowed_commands_validation_and_normalization() -> None:
    assert bash_tools._get_allowed_commands(None) is None
    assert bash_tools._get_allowed_commands([" git ", "/usr/bin/rg"]) == {"git", "rg"}

    with pytest.raises(ValueError, match="list of command names"):
        bash_tools._get_allowed_commands("git")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="list of command names"):
        bash_tools._get_allowed_commands([1])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="non-empty strings"):
        bash_tools._get_allowed_commands(["   "])


def test_bash_extract_command_names_skips_control_words_and_env_assignments() -> None:
    script = "\n".join(
        [
            "FOO=bar",
            "if true; then",
            "  /usr/bin/git status > out.txt",
            "fi",
            "env PATH=/tmp rg hello .",
            "time python3 -V",
        ]
    )
    commands = bash_tools._extract_command_names(script)
    assert commands == ("true", "git", "rg", "python3")


def test_bash_exec_handler_blocks_disallowed_commands() -> None:
    with pytest.raises(ValueError, match="blocked commands"):
        bash_tools._bash_exec_handler(
            {"script": "git status\nrm -rf /tmp/x", "allowed_commands": ["git"]},
            request_id="req",
            dependencies={},
        )


def test_bash_exec_handler_import_error_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object):
        if name == "bashkit":
            raise ImportError("missing bashkit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(RuntimeError, match="bashkit is required"):
        bash_tools._bash_exec_handler({"script": "echo hi"}, request_id="req", dependencies={})


def test_bash_exec_handler_success_with_fake_bashkit(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = ModuleType("bashkit")

    class _FakeExecResult:
        stdout = "ok"
        stderr = ""
        exit_code = 0
        success = True
        error = None

    class _FakeBashTool:
        def __init__(
            self,
            *,
            username: str | None = None,
            hostname: str | None = None,
            max_commands: int = 0,
            max_loop_iterations: int = 0,
        ) -> None:
            self.username = username
            self.hostname = hostname
            self.max_commands = max_commands
            self.max_loop_iterations = max_loop_iterations

        def execute_sync(self, _script: str) -> _FakeExecResult:
            return _FakeExecResult()

    fake_module.BashTool = _FakeBashTool
    monkeypatch.setitem(sys.modules, "bashkit", fake_module)

    output = bash_tools._bash_exec_handler(
        {
            "script": "echo hi",
            "username": "unit",
            "hostname": "local",
            "max_commands": 7,
            "max_loop_iterations": 11,
        },
        request_id="req",
        dependencies={},
    )
    assert output["stdout"] == "ok"
    assert output["success"] is True


def test_python_sandbox_handler_success_and_stdout() -> None:
    output = python_tools._python_sandbox_handler(
        {
            "code": "values = context['values']\nresult = {'sum': sum(values)}\nprint('ok')\n",
            "context": {"values": [1, 2, 3]},
            "timeout_s": 2,
        },
        request_id="req",
        dependencies={},
    )
    assert output["result"] == {"sum": 6}
    assert output["stdout"] == "ok\n"
    assert output["truncated"] is False
    assert isinstance(output["execution_ms"], int)


def test_python_sandbox_handler_rejects_import() -> None:
    with pytest.raises(ValueError, match="Unsupported syntax node: Import"):
        python_tools._python_sandbox_handler(
            {"code": "import os\nresult = 1"},
            request_id="req",
            dependencies={},
        )


@pytest.mark.skipif(not hasattr(python_tools.signal, "SIGALRM"), reason="POSIX-only timeout")
def test_python_sandbox_handler_times_out() -> None:
    with pytest.raises(TimeoutError, match="Execution exceeded timeout"):
        python_tools._python_sandbox_handler(
            {"code": "while True:\n    pass\n", "timeout_s": 1},
            request_id="req",
            dependencies={},
        )


def test_memory_tools_write_search_stats_round_trip(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    db_path = "artifacts/memory/core_tools.sqlite3"

    write_result = memory_tools._memory_write(
        {
            "db_path": db_path,
            "namespace": "unit",
            "records": [{"content": "alpha design note", "metadata": {"kind": "note"}}],
        },
        dependencies={},
        policy=policy,
    )
    assert write_result["written"] == 1
    assert write_result["namespace"] == "unit"

    search_result = memory_tools._memory_search(
        {
            "db_path": db_path,
            "namespace": "unit",
            "text": "alpha",
            "top_k": 5,
        },
        dependencies={},
        policy=policy,
    )
    assert search_result["count"] == 1
    assert search_result["retrieval_mode"] == "lexical"

    stats_result = memory_tools._memory_stats(
        {"db_path": db_path, "namespace": "unit"},
        dependencies={},
        policy=policy,
    )
    assert stats_result["record_count"] == 1
    assert stats_result["embedding_count"] == 0


def test_memory_tools_search_reports_hybrid_vector_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "artifacts" / "memory" / "memory.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteMemoryStore(
        db_path=db_path,
        embedding_provider=_StaticEmbeddingProvider(
            vectors_by_text={
                "query": [1.0, 0.0],
                "mostly lexical": [0.0, 1.0],
                "mostly vector": [1.0, 0.0],
            }
        ),
    )
    store.write(
        [
            MemoryWriteRecord(content="mostly lexical"),
            MemoryWriteRecord(content="mostly vector"),
        ],
        namespace="default",
    )

    result = memory_tools._memory_search(
        {"text": "query", "namespace": "default", "top_k": 2},
        dependencies={"memory_store": store},
        policy=_policy(tmp_path),
    )
    store.close()

    assert result["retrieval_mode"] == "hybrid_vector"
    assert result["matches"][0]["content"] == "mostly vector"


def test_eval_decision_matrix_ranks_alternatives() -> None:
    result = evaluation_tools._decision_matrix_handler(
        {
            "alternatives": [
                {"id": "A", "scores": {"cost": 5, "quality": 8}},
                {"id": "B", "scores": {"cost": 9, "quality": 6}},
            ],
            "criteria": [
                {"name": "cost", "goal": "min", "weight": 0.4},
                {"name": "quality", "goal": "max", "weight": 0.6},
            ],
            "normalize": True,
        },
        request_id="req",
        dependencies={},
    )
    assert result["ranked"][0]["alternative"] == "A"
    assert result["ranked"][0]["rank"] == 1


def test_eval_pairwise_rank_copeland() -> None:
    result = evaluation_tools._pairwise_rank_handler(
        {
            "alternatives": ["A", "B", "C"],
            "comparisons": [
                {"a": "A", "b": "B", "outcome": "a"},
                {"a": "A", "b": "C", "outcome": "a"},
                {"a": "B", "b": "C", "outcome": "b"},
            ],
        },
        request_id="req",
        dependencies={},
    )
    assert result["method"] == "copeland"
    assert result["ranking"][0]["alternative"] == "A"
    assert result["ranking"][0]["copeland_score"] > result["ranking"][1]["copeland_score"]
