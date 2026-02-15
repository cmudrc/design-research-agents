"""Guarded subprocess execution tools."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping

from design_research_agents.contracts.tools import ToolMetadata, ToolSideEffects, ToolSpec
from design_research_agents.tools.policy import ToolPolicy
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str


def register_run_tools(source: InProcessToolSource, *, policy: ToolPolicy) -> None:
    """Register guarded direct subprocess execution tooling."""
    source.register_tool(
        spec=ToolSpec(
            name="run.command",
            description=(
                "Execute one allowlisted command without a shell, with timeout and output caps. "
                "This tool is risky and disabled by policy unless commands are allowlisted."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout_s": {"type": "integer"},
                    "max_output_bytes": {"type": "integer"},
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            metadata=ToolMetadata(
                source="core",
                side_effects=ToolSideEffects(commands=("git", "rg", "python", "python3", "uv")),
                timeout_s=30,
                max_output_bytes=131_072,
                risky=True,
            ),
        ),
        handler=lambda i, r, d: _run_command(i, policy=policy),
    )


def _run_command(input_dict: Mapping[str, object], *, policy: ToolPolicy) -> Mapping[str, object]:
    argv_raw = input_dict.get("argv")
    if not isinstance(argv_raw, list) or not argv_raw:
        raise ValueError("argv must be a non-empty list of strings.")

    argv = [str(item) for item in argv_raw]
    command = argv[0].strip()
    policy.validate_command(command)

    cwd = policy.resolve_read_path(get_str(input_dict, "cwd", default="."))
    timeout_s = get_int(input_dict, "timeout_s", default=policy.config.default_timeout_s)
    max_output_bytes = get_int(
        input_dict,
        "max_output_bytes",
        default=policy.config.default_max_output_bytes,
    )

    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=False,
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        out_clipped, out_truncated = policy.clamp_output(stdout, max_output_bytes=max_output_bytes)
        err_clipped, err_truncated = policy.clamp_output(stderr, max_output_bytes=max_output_bytes)
        return {
            "ok": False,
            "returncode": None,
            "stdout": out_clipped,
            "stderr": err_clipped,
            "timed_out": True,
            "stdout_truncated": out_truncated,
            "stderr_truncated": err_truncated,
        }

    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")
    out_clipped, out_truncated = policy.clamp_output(stdout, max_output_bytes=max_output_bytes)
    err_clipped, err_truncated = policy.clamp_output(stderr, max_output_bytes=max_output_bytes)

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": out_clipped,
        "stderr": err_clipped,
        "timed_out": False,
        "stdout_truncated": out_truncated,
        "stderr_truncated": err_truncated,
    }
