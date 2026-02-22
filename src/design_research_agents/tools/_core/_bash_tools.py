"""BashKit-backed core bash execution tools."""

from __future__ import annotations

import shlex
from collections.abc import Mapping

from design_research_agents._contracts._tools import (
    ToolMetadata,
    ToolSideEffects,
    ToolSpec,
)
from design_research_agents.tools._sources._inprocess_source import InProcessToolSource

from ._helpers import get_int, get_str

_COMMAND_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")"})
_SHELL_CONTROL_WORDS = frozenset(
    {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "until",
        "do",
        "done",
        "case",
        "in",
        "esac",
    }
)
_COMMAND_PREFIXES = frozenset({"command", "builtin", "env", "sudo", "time", "nohup"})


def register_bash_tools(source: InProcessToolSource) -> None:
    """Register BashKit-backed execution tooling in the in-process source.

    Args:
        source: Input value for this parameter.
    """
    source.register_tool(
        spec=ToolSpec(
            name="bash.exec",
            description=(
                "Execute bash script text in BashKit's virtual sandbox. No host filesystem access by default."
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
                    "allowed_commands": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
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
    """Run bash exec handler.

    Args:
        input_dict: Input value for this parameter.
        request_id: Input value for this parameter.
        dependencies: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    del request_id, dependencies
    script = get_str(input_dict, "script").strip()
    if not script:
        raise ValueError("script is required.")

    allowed_commands = _get_allowed_commands(input_dict.get("allowed_commands"))
    if allowed_commands is not None:
        observed_commands = _extract_command_names(script)
        disallowed = sorted({name for name in observed_commands if name not in allowed_commands})
        if disallowed:
            raise ValueError("bash.exec blocked commands outside 'allowed_commands': " + ", ".join(disallowed))

    try:
        from bashkit import BashTool
    except ImportError as exc:  # pragma: no cover - dependency wiring check.
        raise RuntimeError("bashkit is required for bash.exec. Install dependencies with `pip install -e .`.") from exc

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


def _get_allowed_commands(value: object) -> set[str] | None:
    """Run get allowed commands.

    Args:
        value: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("'allowed_commands' must be a list of command names.")
    allowed: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError("'allowed_commands' must be a list of command names.")
        normalized = _normalize_command_name(item)
        if not normalized:
            raise ValueError("'allowed_commands' entries must be non-empty strings.")
        allowed.add(normalized)
    return allowed


def _extract_command_names(script: str) -> tuple[str, ...]:
    """Run extract command names.

    Args:
        script: Input value for this parameter.

    Returns:
        Computed return value.
    """
    commands: list[str] = []
    for raw_line in script.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        line_commands = _extract_line_command_names(stripped)
        commands.extend(line_commands)
    return tuple(commands)


def _extract_line_command_names(line: str) -> tuple[str, ...]:
    """Run extract line command names.

    Args:
        line: Input value for this parameter.

    Returns:
        Computed return value.
    """
    lexer = shlex.shlex(line, posix=True, punctuation_chars="();|&<>")
    lexer.whitespace_split = True
    lexer.commenters = "#"

    names: list[str] = []
    expect_command = True
    for token in lexer:
        normalized = token.strip()
        if not normalized:
            continue
        if normalized in _COMMAND_SEPARATORS:
            expect_command = True
            continue
        if _is_redirection_token(normalized):
            continue
        if not expect_command:
            continue
        if _is_env_assignment(normalized):
            continue
        lowered = normalized.lower()
        if lowered in _SHELL_CONTROL_WORDS:
            continue
        if lowered in _COMMAND_PREFIXES:
            continue

        command_name = _normalize_command_name(normalized)
        if command_name:
            names.append(command_name)
            expect_command = False
    return tuple(names)


def _normalize_command_name(token: str) -> str:
    """Run normalize command name.

    Args:
        token: Input value for this parameter.

    Returns:
        Computed return value.
    """
    candidate = token.strip()
    if not candidate:
        return ""
    if "=" in candidate and candidate.count("=") == 1 and candidate.index("=") > 0:
        return ""
    if "/" in candidate:
        candidate = candidate.rsplit("/", maxsplit=1)[-1]
    return candidate


def _is_env_assignment(token: str) -> bool:
    """Run is env assignment.

    Args:
        token: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if "=" not in token:
        return False
    key, _, value = token.partition("=")
    if not key or key[0].isdigit():
        return False
    if not key.replace("_", "").isalnum():
        return False
    return bool(value)


def _is_redirection_token(token: str) -> bool:
    """Run is redirection token.

    Args:
        token: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if token in {">", ">>", "<", "<<", "<<<", "<>", "&>", "&>>"}:
        return True
    return (token.endswith(">") and token[:-1].isdigit()) or (token.endswith(">>") and token[:-2].isdigit())
