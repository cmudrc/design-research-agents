"""Configuration and typed descriptors for toolbox sources."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import yaml
except ImportError:  # pragma: no cover - optional during very small installs.
    yaml = None


CallableToolHandler = Callable[[Mapping[str, object]], object]


@dataclass(slots=True, frozen=True)
class CoreToolsConfig:
    """Configuration for built-in core tools."""

    enabled: bool = True
    allow_network: bool = False
    allow_writes_outside_artifacts: bool = False
    allowed_commands: tuple[str, ...] = (
        "git",
        "rg",
        "python",
        "python3",
        "uv",
        "ruff",
        "pytest",
    )
    artifacts_dir: str = "artifacts"
    workspace_root: str = "."


@dataclass(slots=True, frozen=True)
class McpServer:
    """External MCP server definition."""

    id: str
    type: Literal["stdio"] = "stdio"
    command: tuple[str, ...] = ()
    timeout_s: int = 20
    env_allowlist: tuple[str, ...] = (
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "PYTHONPATH",
        "VIRTUAL_ENV",
    )
    env: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class McpConfig:
    """Configuration for attached MCP servers."""

    enabled: bool = False
    servers: tuple[McpServer, ...] = ()


@dataclass(slots=True, frozen=True)
class ScriptTool:
    """One explicit script-backed tool definition."""

    name: str
    path: str
    description: str
    input_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
            "required": [],
        }
    )
    output_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
        }
    )
    filesystem_read: bool = False
    filesystem_write: bool = False
    network: bool = False
    commands: tuple[str, ...] = ()
    timeout_s: int = 30
    permissions: tuple[str, ...] = ()
    risky: bool | None = None


@dataclass(slots=True, frozen=True)
class ScriptToolsConfig:
    """Configuration for explicitly declared script tools."""

    enabled: bool = False
    tools: tuple[ScriptTool, ...] = ()


@dataclass(slots=True, frozen=True)
class CallableTool:
    """Simple in-process callable tool wrapper descriptor."""

    name: str
    description: str
    handler: CallableToolHandler
    input_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
            "required": [],
        }
    )
    output_schema: dict[str, object] = field(
        default_factory=lambda: {
            "type": "object",
        }
    )
    permissions: tuple[str, ...] = ()
    risky: bool | None = None


@dataclass(slots=True, frozen=True)
class ToolRuntimeConfig:
    """Top-level configuration for source-enabled toolbox runtime."""

    core_tools: CoreToolsConfig = field(default_factory=CoreToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    script_tools: ScriptToolsConfig = field(default_factory=ScriptToolsConfig)


def load_tool_runtime_config(path: str) -> ToolRuntimeConfig:
    """Load toolbox runtime config from YAML."""
    if yaml is None:
        raise RuntimeError("YAML support requires PyYAML. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Tool runtime config root must be a mapping.")
    if "lazy_tools" in payload:
        raise ValueError("Use 'script_tools' instead of the removed 'lazy_tools' key.")

    core_raw = payload.get("core_tools", {})
    mcp_raw = payload.get("mcp", {})
    script_raw = payload.get("script_tools", {})

    core_cfg = _parse_core_config(core_raw)
    mcp_cfg = _parse_mcp_config(mcp_raw)
    script_cfg = _parse_script_config(script_raw)

    return ToolRuntimeConfig(core_tools=core_cfg, mcp=mcp_cfg, script_tools=script_cfg)


def _parse_core_config(raw: object) -> CoreToolsConfig:
    if not isinstance(raw, dict):
        return CoreToolsConfig()
    defaults = CoreToolsConfig()
    return CoreToolsConfig(
        enabled=bool(raw.get("enabled", True)),
        allow_network=bool(raw.get("allow_network", False)),
        allow_writes_outside_artifacts=bool(raw.get("allow_writes_outside_artifacts", False)),
        allowed_commands=_parse_str_list(raw.get("allowed_commands")) or defaults.allowed_commands,
        artifacts_dir=_parse_str(raw.get("artifacts_dir")) or "artifacts",
        workspace_root=_parse_str(raw.get("workspace_root")) or ".",
    )


def _parse_mcp_config(raw: object) -> McpConfig:
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
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _parse_str_list(value: object) -> tuple[str, ...]:
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
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Expected integer value.")
    return value


def _parse_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Expected bool value.")
    return value


def _parse_mapping(value: object, *, default: dict[str, object]) -> dict[str, object]:
    if value is None:
        return dict(default)
    if not isinstance(value, Mapping):
        raise ValueError("Expected mapping value.")
    return dict(value)


def _parse_env(value: object) -> dict[str, str]:
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
