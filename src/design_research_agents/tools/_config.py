"""Configuration and typed descriptors for toolbox sources."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CallableToolHandler = Callable[[Mapping[str, object]], object]


@dataclass(slots=True, frozen=True)
class CoreToolsConfig:
    """Configuration for built-in core tools."""

    enabled: bool = True
    """Whether built-in core tools are registered."""
    allow_network: bool = False
    """Whether core tools may use outbound network access."""
    allow_writes_outside_artifacts: bool = False
    """Whether core tools may write outside the artifacts directory."""
    allowed_commands: tuple[str, ...] = (
        "git",
        "rg",
        "python",
        "python3",
        "uv",
        "ruff",
        "pytest",
    )
    """Command allowlist enforced by shell-executing core tools."""
    artifacts_dir: str = "artifacts"
    """Directory used for tool-generated artifacts."""
    workspace_root: str = "."
    """Workspace root path exposed to filesystem-aware tools."""


@dataclass(slots=True, frozen=True)
class McpServer:
    """External MCP server definition."""

    id: str
    """Unique identifier for the server. This is used to reference the server in 
    tool definitions and logs."""

    type: Literal["stdio"] = "stdio"
    """Communication protocol to use with the server. Currently, only 'stdio' 
    is supported, which means the server will be launched as a subprocess and 
    communicated with via its standard input and output streams."""

    command: tuple[str, ...] = ()
    """Command to launch the server, specified as a tuple of strings. The first 
    element should be the executable, and the subsequent elements are its arguments."""

    timeout_s: int = 20
    """Timeout in seconds for server responses before treating it as unresponsive."""

    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )
    """Allowlist of environment variable names that will be passed to the server process. 
    Only variables in this list will be included in the server's environment, which 
    helps to limit exposure of sensitive  information and reduce the attack surface."""

    env: dict[str, str] = field(default_factory=dict)
    """Explicit environment variables to set for the server process. This is a mapping of variable 
    names to their desired values. These variables will be included in the server's environment in 
    addition to any variables from the allowlist that are present in the parent process."""


@dataclass(slots=True, frozen=True)
class McpConfig:
    """Configuration for attached MCP servers."""

    enabled: bool = False
    """Whether MCP-backed tools are enabled."""
    servers: tuple[McpServer, ...] = ()
    """Configured external MCP servers to attach."""


@dataclass(slots=True, frozen=True)
class ScriptTool:
    """One explicit script-backed tool definition."""

    name: str
    """Unique name of the tool. This is used to reference the tool in prompts and logs."""

    path: str
    """Filesystem path to the script that implements the tool's behavior. This should be an 
    absolute path or a path relative to the configured workspace root. The script will be
    executed as a subprocess when the tool is invoked, and communicated with via its 
    standard input and output streams."""

    description: str
    """Short description of the tool's behavior. This should be a concise summary of what 
    the tool does, suitable for inclusion in prompts and documentation."""

    input_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
            "required": [],
        }
    )
    """JSON Schema describing the expected input structure for the tool. This is used for
    validation and documentation purposes. The tool will receive its input as a JSON-encoded
    string on its standard input, and it should produce its output as a JSON-encoded string on
    its standard output. The input schema should describe the structure of the JSON object that
    the tool expects to receive, including any required properties and their types."""

    output_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
        }
    )
    """JSON Schema describing the structure of the tool's output. This is used for validation and
    documentation purposes. The tool's output should be a JSON-encoded string written to its
    standard output, and the output schema should describe the structure of the JSON object that
    the tool produces, including any properties and their types."""

    filesystem_read: bool = False
    """Flag indicating whether the tool needs read access to the filesystem. If True, the tool will
    be granted read access to the workspace root and artifacts directory. If False, the tool will
    not be granted any filesystem access. This is used to enforce security constraints and limit 
    the tool's capabilities."""

    filesystem_write: bool = False
    """Flag indicating whether the tool needs write access to the filesystem. If True, 
    the tool will be granted write access to the workspace root and artifacts directory. 
    If False, the tool will not be granted any filesystem access. This is used to enforce 
    security constraints and limit the tool's capabilities."""

    network: bool = False
    """Flag indicating whether the tool needs access to the network. If True, the tool 
    will be granted access to the network. If False, the tool will not be granted any 
    network access. This is used to enforce security constraints and limit the 
    tool's capabilities."""

    commands: tuple[str, ...] = ()
    """Optional tuple of allowed shell commands that the tool is permitted to execute. If 
    non-empty,  the tool will only be allowed to execute commands in this list, and 
    attempts to execute any  other commands will be blocked. This is used to enforce 
    security constraints and limit the tool's capabilities."""

    timeout_s: int = 30
    """Timeout in seconds for the tool's execution. If the tool does not produce output within 
    this time frame, it will be considered unresponsive, and appropriate error handling will 
    be triggered."""

    permissions: tuple[str, ...] = ()
    """Optional tuple of permission strings that the tool requires. This can be used to 
    enforce security constraints or to inform users about the tool's capabilities. The 
    specific permission strings and their meanings are not defined by this configuration
    and should be interpreted by the tool runtime or the user interface accordingly."""

    risky: bool | None = None
    """Optional boolean flag indicating whether the tool performs potentially risky operations, 
    such as executing shell commands, accessing the filesystem, or making network requests. 
    This can be used to inform users about the tool's capabilities and potential risks."""


@dataclass(slots=True, frozen=True)
class ScriptToolsConfig:
    """Configuration for explicitly declared script tools."""

    enabled: bool = False
    """Whether script-backed tools are enabled."""
    tools: tuple[ScriptTool, ...] = ()
    """Explicit script-backed tool definitions."""


@dataclass(slots=True, frozen=True)
class CallableTool:
    """Simple in-process callable tool wrapper descriptor."""

    name: str
    """Unique name of the tool."""

    description: str
    """Short description of the tool's behavior."""

    handler: CallableToolHandler
    """Python callable that implements the tool's behavior. It should accept a single argument of 
    type Mapping[str, object] and return an arbitrary JSON-serializable object."""

    input_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
            "required": [],
        }
    )
    """JSON Schema describing the expected input structure for the tool. This is used for 
    validation and documentation purposes."""

    output_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
        }
    )
    """JSON Schema describing the structure of the tool's output. This is used for validation and 
    documentation purposes."""

    permissions: tuple[str, ...] = ()
    """Optional tuple of permission strings that the tool requires. This can be used to enforce 
    security constraints or to inform users about the tool's capabilities."""

    risky: bool | None = None
    """Whether the tool performs potentially risky operations."""


@dataclass(slots=True, frozen=True)
class ToolRuntimeConfig:
    """Top-level configuration for source-enabled toolbox runtime."""

    core_tools: CoreToolsConfig = field(default_factory=CoreToolsConfig)
    """Configuration block for built-in core tools."""
    mcp: McpConfig = field(default_factory=McpConfig)
    """Configuration block for MCP-backed tools."""
    script_tools: ScriptToolsConfig = field(default_factory=ScriptToolsConfig)
    """Configuration block for explicit script tools."""


def load_tool_runtime_config(path: str) -> ToolRuntimeConfig:
    """Load toolbox runtime config from JSON.

    Args:
        path: Path to JSON configuration file.

    Returns:
        Parsed runtime configuration.

    Raises:
        ValueError: If configuration shape or values are invalid.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tool runtime config must be valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Tool runtime config root must be a mapping.")
    if "lazy_tools" in payload:
        raise ValueError("Use 'script_tools' instead of the removed 'lazy_tools' key.")

    core_raw = payload.get("core_tools", {})
    mcp_raw = payload.get("mcp", {})
    script_raw = payload.get("script_tools", {})

    # Parse sections independently so malformed optional blocks fail with targeted messages.
    core_cfg = _parse_core_config(core_raw)
    mcp_cfg = _parse_mcp_config(mcp_raw)
    script_cfg = _parse_script_config(script_raw)

    return ToolRuntimeConfig(core_tools=core_cfg, mcp=mcp_cfg, script_tools=script_cfg)


def _parse_core_config(raw: object) -> CoreToolsConfig:
    """Parse the ``core_tools`` configuration section.

    Args:
        raw: Raw section payload from parsed configuration.

    Returns:
        Normalized core-tools configuration.
    """
    if not isinstance(raw, dict):
        return CoreToolsConfig()
    defaults = CoreToolsConfig()
    # Fall back to constructor defaults for omitted keys to keep config files concise.
    return CoreToolsConfig(
        enabled=bool(raw.get("enabled", True)),
        allow_network=bool(raw.get("allow_network", False)),
        allow_writes_outside_artifacts=bool(raw.get("allow_writes_outside_artifacts", False)),
        allowed_commands=_parse_str_list(raw.get("allowed_commands")) or defaults.allowed_commands,
        artifacts_dir=_parse_str(raw.get("artifacts_dir")) or "artifacts",
        workspace_root=_parse_str(raw.get("workspace_root")) or ".",
    )


def _parse_mcp_config(raw: object) -> McpConfig:
    """Parse the ``mcp`` configuration section.

    Args:
        raw: Raw section payload from parsed configuration.

    Returns:
        Normalized MCP configuration.

    Raises:
        ValueError: If server definitions are malformed.
    """
    if not isinstance(raw, dict):
        return McpConfig()
    enabled = bool(raw.get("enabled", False))
    defaults = McpServer(id="__defaults__")
    servers_raw = raw.get("servers", [])
    parsed_servers: list[McpServer] = []
    if isinstance(servers_raw, list):
        for index, item in enumerate(servers_raw):
            if not isinstance(item, dict):
                raise ValueError(f"mcp.servers[{index}] must be a mapping.")
            server_id = _parse_str(item.get("id"))
            if server_id is None:
                raise ValueError(f"mcp.servers[{index}].id is required.")
            server_type = _parse_str(item.get("type")) or "stdio"
            if server_type != "stdio":
                raise ValueError(f"mcp.servers[{index}].type '{server_type}' is not supported.")
            command = _parse_str_list(item.get("command"))
            if not command:
                raise ValueError(f"mcp.servers[{index}].command must be a non-empty string list.")
            timeout_s = _parse_int(item.get("timeout_s"), default=20)
            env_allowlist = _parse_str_list(item.get("env_allowlist")) or defaults.env_allowlist
            env = _parse_env(item.get("env"))
            parsed_servers.append(
                McpServer(
                    id=server_id,
                    type="stdio",
                    command=command,
                    timeout_s=timeout_s,
                    env_allowlist=env_allowlist,
                    env=env,
                )
            )

    return McpConfig(enabled=enabled, servers=tuple(parsed_servers))


def _parse_script_config(raw: object) -> ScriptToolsConfig:
    """Parse the ``script_tools`` configuration section.

    Args:
        raw: Raw section payload from parsed configuration.

    Returns:
        Normalized script-tools configuration.

    Raises:
        ValueError: If tool definitions are malformed.
    """
    if not isinstance(raw, dict):
        return ScriptToolsConfig()
    enabled = bool(raw.get("enabled", False))
    tools_raw = raw.get("tools", [])
    parsed_tools: list[ScriptTool] = []
    if isinstance(tools_raw, list):
        for index, item in enumerate(tools_raw):
            if not isinstance(item, dict):
                raise ValueError(f"script_tools.tools[{index}] must be a mapping.")
            name = _parse_str(item.get("name"))
            path = _parse_str(item.get("path"))
            if name is None:
                raise ValueError(f"script_tools.tools[{index}].name is required.")
            if path is None:
                raise ValueError(f"script_tools.tools[{index}].path is required.")
            description = _parse_str(item.get("description")) or f"Script tool {name}"
            input_schema = _parse_mapping(
                item.get("input_schema"),
                default={
                    "type": "object",
                    "additionalProperties": True,
                    "properties": {},
                    "required": [],
                },
            )
            output_schema = _parse_mapping(item.get("output_schema"), default={"type": "object"})
            permissions = _parse_str_list(item.get("permissions"))
            risky = _parse_optional_bool(item.get("risky"))
            commands = _parse_str_list(item.get("commands"))
            timeout_s = _parse_int(item.get("timeout_s"), default=30)
            parsed_tools.append(
                ScriptTool(
                    name=name,
                    path=str(Path(path).expanduser()),
                    description=description,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    filesystem_read=bool(item.get("filesystem_read", False)),
                    filesystem_write=bool(item.get("filesystem_write", False)),
                    network=bool(item.get("network", False)),
                    commands=commands,
                    timeout_s=timeout_s,
                    permissions=permissions,
                    risky=risky,
                )
            )

    return ScriptToolsConfig(enabled=enabled, tools=tuple(parsed_tools))


def _parse_str(value: object) -> str | None:
    """Normalize an optional string value.

    Args:
        value: Raw value to normalize.

    Returns:
        Stripped string, or ``None`` when value is empty/non-string.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _parse_str_list(value: object) -> tuple[str, ...]:
    """Parse a list of strings into a normalized tuple.

    Args:
        value: Raw value expected to be a list of strings.

    Returns:
        Tuple of non-empty stripped strings.

    Raises:
        ValueError: If ``value`` is not a list of strings.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("Expected list of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("Expected list of strings.")
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return tuple(normalized)


def _parse_int(value: object, *, default: int) -> int:
    """Parse an optional integer with default fallback.

    Args:
        value: Raw value to parse.
        default: Default integer when ``value`` is ``None``.

    Returns:
        Parsed integer value.

    Raises:
        ValueError: If ``value`` is not an integer.
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected integer value.")
    return value


def _parse_optional_bool(value: object) -> bool | None:
    """Parse an optional boolean value.

    Args:
        value: Raw value to parse.

    Returns:
        Parsed boolean value, or ``None`` when value is absent.

    Raises:
        ValueError: If ``value`` is not a boolean.
    """
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected bool value.")
    return value


def _parse_mapping(value: object, *, default: dict[str, object]) -> dict[str, object]:
    """Parse an optional mapping with default fallback.

    Args:
        value: Raw value to parse.
        default: Default mapping when ``value`` is ``None``.

    Returns:
        Parsed mapping copied into a dictionary.

    Raises:
        ValueError: If ``value`` is not a mapping.
    """
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ValueError("Expected mapping value.")
    return dict(value)


def _parse_env(value: object) -> dict[str, str]:
    """Parse environment-variable mapping for MCP server execution.

    Args:
        value: Raw value expected to be a string-keyed mapping.

    Returns:
        Parsed environment mapping.

    Raises:
        ValueError: If ``value`` is not a valid mapping.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected env mapping.")
    parsed: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Environment variable names must be non-empty strings.")
        parsed[key.strip()] = str(item)
    return parsed


__all__ = [
    "CallableTool",
    "CallableToolHandler",
    "CoreToolsConfig",
    "McpConfig",
    "McpServer",
    "ScriptTool",
    "ScriptToolsConfig",
    "ToolRuntimeConfig",
    "load_tool_runtime_config",
]
