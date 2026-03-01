"""Read-focused git tools."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._policy import ToolPolicy
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_bool, get_int, get_str


def register_git_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register read-oriented git inspection tools.

    Args:
        source: Value supplied for ``source``.
        policy: Value supplied for ``policy``.
    """
    metadata = ToolMetadata(
        source="core",
        side_effects=ToolSideEffects(filesystem_read=True, commands=("git",)),
        timeout_s=20,
        max_output_bytes=131_072,
        risky=True,
    )

    source.register_tool(
        spec=ToolSpec(
            name="git.status",
            description="Read git status for a repository.",
            input_schema={
                "type": "object",
                "properties": {"repo": {"type": "string"}},
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=lambda i, r, d: _git_status(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="git.diff",
            description="Read git diff output.",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "staged": {"type": "boolean"},
                    "pathspec": {"type": "string"},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=lambda i, r, d: _git_diff(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="git.log",
            description="Read concise git commit history.",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "max_commits": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=lambda i, r, d: _git_log(i, policy=policy),
    )
    source.register_tool(
        spec=ToolSpec(
            name="git.show",
            description="Show details for one revision.",
            input_schema={
                "type": "object",
                "properties": {
                    "repo": {"type": "string"},
                    "rev": {"type": "string"},
                },
                "required": ["rev"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=metadata,
        ),
        handler=lambda i, r, d: _git_show(i, policy=policy),
    )


def _git_status(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Git status.

    Args:
        input_dict: Value supplied for ``input_dict``.
        policy: Value supplied for ``policy``.

    Returns:
        Result produced by this call.
    """
    repo = policy.resolve_read_path(get_str(input_dict, "repo", default="."))
    return {
        "repo": str(repo),
        "status": _run_git(policy=policy, repo=str(repo), args=["status", "--short", "--branch"]),
    }


def _git_diff(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Git diff.

    Args:
        input_dict: Value supplied for ``input_dict``.
        policy: Value supplied for ``policy``.

    Returns:
        Result produced by this call.
    """
    repo = policy.resolve_read_path(get_str(input_dict, "repo", default="."))
    staged = get_bool(input_dict, "staged", default=False)
    pathspec = get_str(input_dict, "pathspec", default="").strip()
    args = ["diff"]
    if staged:
        args.append("--staged")
    if pathspec:
        args.extend(["--", pathspec])
    return {
        "repo": str(repo),
        "diff": _run_git(policy=policy, repo=str(repo), args=args),
    }


def _git_log(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Git log.

    Args:
        input_dict: Value supplied for ``input_dict``.
        policy: Value supplied for ``policy``.

    Returns:
        Result produced by this call.
    """
    repo = policy.resolve_read_path(get_str(input_dict, "repo", default="."))
    max_commits = get_int(input_dict, "max_commits", default=20)
    return {
        "repo": str(repo),
        "log": _run_git(
            policy=policy,
            repo=str(repo),
            args=["log", "--oneline", "-n", str(max_commits)],
        ),
    }


def _git_show(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Git show.

    Args:
        input_dict: Value supplied for ``input_dict``.
        policy: Value supplied for ``policy``.

    Returns:
        Result produced by this call.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    repo = policy.resolve_read_path(get_str(input_dict, "repo", default="."))
    rev = _validate_git_rev(get_str(input_dict, "rev").strip())
    return {
        "repo": str(repo),
        "rev": rev,
        "show": _run_git(policy=policy, repo=str(repo), args=["show", rev]),
    }


def _validate_git_rev(rev: str) -> str:
    """Validate one revision string before passing it to ``git show``."""
    if not rev:
        raise ValueError("rev is required.")
    if rev.startswith("-"):
        raise ValueError("rev must not start with '-'.")
    if any(char.isspace() for char in rev):
        raise ValueError("rev must not contain whitespace.")
    if any(ord(char) < 32 or ord(char) == 127 for char in rev):
        raise ValueError("rev must not contain control characters.")
    return rev


def _run_git(*, policy: ToolPolicy, repo: str, args: list[str]) -> str:
    """Execute one ``git`` command and return normalized output.

    Args:
        policy: Tool policy used to validate command execution and clamp output.
        repo: Repository path passed to ``git -C``.
        args: Additional ``git`` arguments to execute.

    Returns:
        Command output with stderr appended when present.
    """
    policy.validate_command("git")
    timeout_s = policy.config.default_timeout_s
    try:
        completed = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"git command timed out after {timeout_s}s.") from exc
    output = completed.stdout
    if completed.stderr.strip():
        output = f"{output}\n{completed.stderr.strip()}".strip()
    clipped, truncated = policy.clamp_output(output)
    if truncated:
        return f"{clipped}\n[truncated]"
    return clipped
