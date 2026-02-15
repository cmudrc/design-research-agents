"""BashKit-backed core bash execution tools."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts.tools import ToolMetadata, ToolSideEffects, ToolSpec
from design_research_agents.tools.sources.inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str


def register_bash_tools(source: InProcessToolSource) -> None:
    """Register BashKit-backed execution tooling in the in-process source."""
    source.register_tool(
        spec=ToolSpec(
            name="bash.exec",
            description=(
                "Execute bash script text in BashKit's virtual sandbox. "
                "No host filesystem access by default."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "script": {"type": "string"},
                    "username": {"type": "string"},
                    "hostname": {"type": "string"},
                    "max_commands": {"type": "integer"},
                    "max_loop_iterations": {"type": "integer"},
                },
                "required": ["script"],
            },
            output_schema={"type": "object"},
            metadata=ToolMetadata(
                source="core",
                side_effects=ToolSideEffects(filesystem_read=False, filesystem_write=False),
                timeout_s=30,
                max_output_bytes=131_072,
                risky=True,
            ),
        ),
        handler=_bash_exec_handler,
    )


def _bash_exec_handler(
    input_dict: Mapping[str, object],
    request_id: str,
    dependencies: Mapping[str, object],
) -> Mapping[str, object]:
    del request_id, dependencies
    script = get_str(input_dict, "script").strip()
    if not script:
        raise ValueError("script is required.")

    try:
        from bashkit import BashTool
    except ImportError as exc:  # pragma: no cover - dependency wiring check.
        raise RuntimeError(
            "bashkit is required for bash.exec. Install dependencies with `pip install -e .`."
        ) from exc

    username = get_str(input_dict, "username", default="").strip() or None
    hostname = get_str(input_dict, "hostname", default="").strip() or None
    max_commands = get_int(input_dict, "max_commands", default=10_000)
    max_loop_iterations = get_int(input_dict, "max_loop_iterations", default=100_000)

    tool = BashTool(
        username=username,
        hostname=hostname,
        max_commands=max_commands,
        max_loop_iterations=max_loop_iterations,
    )
    exec_result = tool.execute_sync(script)
    return {
        "stdout": exec_result.stdout,
        "stderr": exec_result.stderr,
        "exit_code": exec_result.exit_code,
        "success": bool(exec_result.success),
        "error": exec_result.error,
    }
