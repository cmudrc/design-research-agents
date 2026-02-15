"""Command-line interface for tool runtime, MCP, and lazy tools."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from design_research_agents.mcp_server import serve_stdio
from design_research_agents.tools import (
    ToolRuntimeConfig,
    UnifiedToolRuntime,
    load_tool_runtime_config,
)
from design_research_agents.tools.config import McpConfig
from design_research_agents.tools.lazy.discovery import discover_lazy_tools


def main(argv: list[str] | None = None) -> int:
    """Run CLI entrypoint and return process exit code."""
    parser = argparse.ArgumentParser(prog="dra")
    subparsers = parser.add_subparsers(dest="command")

    mcp_parser = subparsers.add_parser("mcp", help="MCP server/client commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    mcp_subparsers.add_parser("serve", help="Serve built-in MCP tools over stdio")

    ping_parser = mcp_subparsers.add_parser("ping", help="Connect to one MCP server and list tools")
    ping_parser.add_argument("--server", required=True, help="Configured MCP server id")
    ping_parser.add_argument("--config", help="Tool runtime YAML config path")

    call_parser = mcp_subparsers.add_parser("call", help="Call one tool through unified runtime")
    call_parser.add_argument("tool", help="Tool name to invoke")
    call_parser.add_argument("--json", required=True, help="JSON object with tool input")
    call_parser.add_argument("--config", help="Tool runtime YAML config path")

    lazy_parser = subparsers.add_parser("lazy", help="Lazy tool commands")
    lazy_subparsers = lazy_parser.add_subparsers(dest="lazy_command")

    lint_parser = lazy_subparsers.add_parser("lint", help="Lint lazy tool headers")
    lint_parser.add_argument("target", help="Directory or file to lint")

    list_parser = lazy_subparsers.add_parser("list", help="List discovered lazy tools")
    list_parser.add_argument("--config", help="Tool runtime YAML config path")

    run_parser = lazy_subparsers.add_parser("run", help="Run one lazy tool")
    run_parser.add_argument("tool_name", help="Lazy tool canonical or short name")
    run_parser.add_argument("--json", required=True, help="JSON object tool input")
    run_parser.add_argument("--config", help="Tool runtime YAML config path")

    args = parser.parse_args(argv)

    if args.command == "mcp":
        return _handle_mcp(args)
    if args.command == "lazy":
        return _handle_lazy(args)

    parser.print_help()
    return 1


def _handle_mcp(args: argparse.Namespace) -> int:
    if args.mcp_command == "serve":
        serve_stdio()
        return 0

    if args.mcp_command == "ping":
        config = _load_config(args.config)
        server_id = str(args.server).strip()
        if not _server_exists(config.mcp, server_id):
            print(f"Server '{server_id}' is not configured.")
            return 1
        runtime = UnifiedToolRuntime(config=config)
        tools = [
            spec.name for spec in runtime.list_tools() if spec.name.startswith(f"{server_id}::")
        ]
        print(json.dumps({"server": server_id, "tools": tools}, ensure_ascii=True, indent=2))
        return 0

    if args.mcp_command == "call":
        config = _load_config(args.config)
        runtime = UnifiedToolRuntime(config=config)
        payload = _parse_json_object(args.json)
        if payload is None:
            print("--json must be a valid JSON object.")
            return 1
        result = runtime.invoke(args.tool, payload, request_id="cli", dependencies={})
        print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
        return 0 if result.ok else 2

    print("Unknown mcp command.")
    return 1


def _handle_lazy(args: argparse.Namespace) -> int:
    if args.lazy_command == "lint":
        target = Path(str(args.target)).expanduser()
        search_paths = (str(target),)
        _, diagnostics = discover_lazy_tools(search_paths)
        if not diagnostics:
            print(json.dumps({"ok": True, "diagnostics": []}, ensure_ascii=True, indent=2))
            return 0
        print(
            json.dumps(
                {
                    "ok": False,
                    "diagnostics": [asdict(diagnostic) for diagnostic in diagnostics],
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 2

    if args.lazy_command == "list":
        config = _load_config(args.config)
        runtime = UnifiedToolRuntime(config=config)
        lazy_tools = [spec.name for spec in runtime.list_tools() if spec.metadata.source == "lazy"]
        print(json.dumps({"tools": lazy_tools}, ensure_ascii=True, indent=2))
        return 0

    if args.lazy_command == "run":
        config = _load_config(args.config)
        runtime = UnifiedToolRuntime(config=config)
        payload = _parse_json_object(args.json)
        if payload is None:
            print("--json must be a valid JSON object.")
            return 1

        requested_name = str(args.tool_name).strip()
        resolved_name = requested_name
        if not requested_name.startswith("lazy::"):
            resolved_name = f"lazy::{requested_name}"

        result = runtime.invoke(resolved_name, payload, request_id="cli", dependencies={})
        print(json.dumps(asdict(result), ensure_ascii=True, indent=2))
        return 0 if result.ok else 2

    print("Unknown lazy command.")
    return 1


def _load_config(path: str | None) -> ToolRuntimeConfig:
    if path:
        return load_tool_runtime_config(path)
    return ToolRuntimeConfig()


def _parse_json_object(raw: str) -> dict[str, object] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return {str(key): value for key, value in parsed.items()}


def _server_exists(mcp_config: McpConfig, server_id: str) -> bool:
    return any(server.id == server_id for server in mcp_config.servers)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
