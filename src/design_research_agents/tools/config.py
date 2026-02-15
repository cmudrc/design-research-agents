"""Configuration models for the unified tool runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

try:
    import yaml
except ImportError:  # pragma: no cover - optional during very small installs.
    yaml = None


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
class McpServerConfig:
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
    servers: tuple[McpServerConfig, ...] = ()


@dataclass(slots=True, frozen=True)
class LazyToolsConfig:
    """Configuration for manifest-less lazy tools."""

    enabled: bool = False
    search_paths: tuple[str, ...] = ("./.dra/tools", "~/.dra/tools")
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
    timeout_s_default: int = 30


@dataclass(slots=True, frozen=True)
class ToolRuntimeConfig:
    """Top-level configuration for source-enabled tool runtime."""

    core_tools: CoreToolsConfig = field(default_factory=CoreToolsConfig)
    mcp: McpConfig = field(default_factory=McpConfig)
    lazy_tools: LazyToolsConfig = field(default_factory=LazyToolsConfig)


def load_tool_runtime_config(path: str) -> ToolRuntimeConfig:
    """Load tool runtime config from YAML.

    Args:
        path: YAML file path.

    Returns:
        Parsed runtime config.
    """
    if yaml is None:
        raise RuntimeError("YAML support requires PyYAML. Install with: pip install pyyaml")
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Tool runtime config root must be a mapping.")

    core_raw = payload.get("core_tools", {})
    mcp_raw = payload.get("mcp", {})
    lazy_raw = payload.get("lazy_tools", {})

    core_cfg = _parse_core_config(core_raw)
    mcp_cfg = _parse_mcp_config(mcp_raw)
    lazy_cfg = _parse_lazy_config(lazy_raw)

    return ToolRuntimeConfig(core_tools=core_cfg, mcp=mcp_cfg, lazy_tools=lazy_cfg)


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
    defaults = McpServerConfig(id="__defaults__")
    servers_raw = raw.get("servers", [])
    parsed_servers: list[McpServerConfig] = []
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
                McpServerConfig(
                    id=server_id,
                    type="stdio",
                    command=command,
                    timeout_s=timeout_s,
                    env_allowlist=env_allowlist,
                    env=env,
                )
            )

    return McpConfig(enabled=enabled, servers=tuple(parsed_servers))


def _parse_lazy_config(raw: object) -> LazyToolsConfig:
    if not isinstance(raw, dict):
        return LazyToolsConfig()
    defaults = LazyToolsConfig()
    return LazyToolsConfig(
        enabled=bool(raw.get("enabled", False)),
        search_paths=_parse_str_list(raw.get("search_paths")) or defaults.search_paths,
        allow_network=bool(raw.get("allow_network", False)),
        allow_writes_outside_artifacts=bool(raw.get("allow_writes_outside_artifacts", False)),
        allowed_commands=_parse_str_list(raw.get("allowed_commands")) or defaults.allowed_commands,
        timeout_s_default=_parse_int(raw.get("timeout_s_default"), default=30),
    )


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
    "CoreToolsConfig",
    "LazyToolsConfig",
    "McpConfig",
    "McpServerConfig",
    "ToolRuntimeConfig",
    "load_tool_runtime_config",
]
