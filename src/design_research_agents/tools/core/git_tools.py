"""Read-focused git tools."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping

from design_research_agents.contracts.tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools.policy import ToolPolicy
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from ._helpers import get_bool, get_int, get_str


def register_git_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register read-oriented git inspection tools.

    Args:
        source: Parameter value.
        policy: Parameter value.
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
    """Run git status.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
    """
    repo = policy.resolve_read_path(get_str(input_dict, "repo", default="."))
    return {
        "repo": str(repo),
        "status": _run_git(policy=policy, repo=str(repo), args=["status", "--short", "--branch"]),
    }


def _git_diff(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    """Run git diff.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
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
    """Run git log.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.
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
    """Run git show.

    Args:
        input_dict: Parameter value.
        policy: Parameter value.

    Returns:
        The resulting value.

    Raises:
        Exception: Raised when execution fails.
    """
    repo = policy.resolve_read_path(get_str(input_dict, "repo", default="."))
    rev = get_str(input_dict, "rev").strip()
    if not rev:
        raise ValueError("rev is required.")
    return {
        "repo": str(repo),
        "rev": rev,
        "show": _run_git(policy=policy, repo=str(repo), args=["show", rev]),
    }


def _run_git(*, policy: ToolPolicy, repo: str, args: list[str]) -> str:
    """Run run git.

    Args:
        policy: Parameter value.
        repo: Parameter value.
        args: Parameter value.

    Returns:
        The resulting value.
    """
    policy.validate_command("git")
    completed = subprocess.run(
        ["git", "-C", repo, *args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout
    if completed.stderr.strip():
        output = f"{output}\n{completed.stderr.strip()}".strip()
    clipped, truncated = policy.clamp_output(output)
    if truncated:
        return f"{clipped}\n[truncated]"
    return clipped
