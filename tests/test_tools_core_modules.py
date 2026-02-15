from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from design_research_agents.tools.core import fs_tools, git_tools, search_tools, text_tools
from design_research_agents.tools.policy import ToolPolicy, ToolPolicyConfig


def _policy(tmp_path: Path) -> ToolPolicy:
    return ToolPolicy(ToolPolicyConfig(workspace_root=str(tmp_path)))


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


def test_search_uses_python_fallback_when_rg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    monkeypatch.setattr(search_tools, "which", lambda _name: None)
    result = search_tools._search(
        {"query": "alpha", "root": ".", "max_matches": 2},
        policy=_policy(tmp_path),
    )
    assert result["engine"] == "python"
    assert result["count"] == 1


def test_search_with_rg_parses_matches_and_keeps_context_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_git_helpers_build_expected_arguments_and_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    policy = _policy(tmp_path)
    calls: list[list[str]] = []

    def _fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(git_tools.subprocess, "run", _fake_run)

    assert git_tools._git_status({"repo": "repo"}, policy=policy)["status"] == "ok"
    assert (
        git_tools._git_diff({"repo": "repo", "staged": True, "pathspec": "a.py"}, policy=policy)[
            "diff"
        ]
        == "ok"
    )
    assert git_tools._git_log({"repo": "repo", "max_commits": 7}, policy=policy)["log"] == "ok"
    assert git_tools._git_show({"repo": "repo", "rev": "HEAD~1"}, policy=policy)["show"] == "ok"

    assert any("--staged" in cmd for cmd in calls)
    assert any(cmd[-2:] == ["--", "a.py"] for cmd in calls)
    assert any(cmd[-4:] == ["log", "--oneline", "-n", "7"] for cmd in calls)

    with pytest.raises(ValueError, match="rev is required"):
        git_tools._git_show({"repo": "repo", "rev": "   "}, policy=policy)


def test_run_git_merges_stderr_and_appends_truncated_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
