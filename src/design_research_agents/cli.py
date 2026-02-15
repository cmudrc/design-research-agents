"""Command-line interface for tool runtime, MCP, and lazy tools."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from design_research_agents.mcp_server import serve_stdio
from design_research_agents.tools import UnifiedToolRuntime
from design_research_agents.tools.config import McpServerConfig
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

    parsed_args = parser.parse_args(argv)

    if parsed_args.command == "mcp":
        return _handle_mcp(parsed_args)
    if parsed_args.command == "lazy":
        return _handle_lazy(parsed_args)

    parser.print_help()
    return 1


def _handle_mcp(cli_args: argparse.Namespace) -> int:
    if cli_args.mcp_command == "serve":
        serve_stdio()
        return 0

    if cli_args.mcp_command == "ping":
        tool_runtime = _build_runtime(cli_args.config)
        server_id = str(cli_args.server).strip()
        try:
            if not _server_exists(tool_runtime.config.mcp.servers, server_id):
                print(f"Server '{server_id}' is not configured.")
                return 1
            server_tool_names = [
                spec.name
                for spec in tool_runtime.list_tools()
                if spec.name.startswith(f"{server_id}::")
            ]
            print(
                json.dumps(
                    {"server": server_id, "tools": server_tool_names},
                    ensure_ascii=True,
                    indent=2,
                )
            )
            return 0
        finally:
            tool_runtime.close()

    if cli_args.mcp_command == "call":
        tool_runtime = _build_runtime(cli_args.config)
        try:
            tool_input_payload = _parse_json_object(cli_args.json)
            if tool_input_payload is None:
                print("--json must be a valid JSON object.")
                return 1
            tool_result = tool_runtime.invoke(
                cli_args.tool,
                tool_input_payload,
                request_id="cli",
                dependencies={},
            )
            print(json.dumps(asdict(tool_result), ensure_ascii=True, indent=2))
            return 0 if tool_result.ok else 2
        finally:
            tool_runtime.close()

    print("Unknown mcp command.")
    return 1


def _handle_lazy(cli_args: argparse.Namespace) -> int:
    if cli_args.lazy_command == "lint":
        target_path = Path(str(cli_args.target)).expanduser()
        search_paths = (str(target_path),)
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

    if cli_args.lazy_command == "list":
        tool_runtime = _build_runtime(cli_args.config)
        try:
            lazy_tools = [
                spec.name for spec in tool_runtime.list_tools() if spec.metadata.source == "lazy"
            ]
            print(json.dumps({"tools": lazy_tools}, ensure_ascii=True, indent=2))
            return 0
        finally:
            tool_runtime.close()

    if cli_args.lazy_command == "run":
        tool_runtime = _build_runtime(cli_args.config)
        try:
            tool_input_payload = _parse_json_object(cli_args.json)
            if tool_input_payload is None:
                print("--json must be a valid JSON object.")
                return 1

            requested_name = str(cli_args.tool_name).strip()
            resolved_name = requested_name
            if not requested_name.startswith("lazy::"):
                resolved_name = f"lazy::{requested_name}"

            tool_result = tool_runtime.invoke(
                resolved_name,
                tool_input_payload,
                request_id="cli",
                dependencies={},
            )
            print(json.dumps(asdict(tool_result), ensure_ascii=True, indent=2))
            return 0 if tool_result.ok else 2
        finally:
            tool_runtime.close()

    print("Unknown lazy command.")
    return 1


def _build_runtime(path: str | None) -> UnifiedToolRuntime:
    if path:
        return UnifiedToolRuntime.from_yaml(path)
    return UnifiedToolRuntime()


def _parse_json_object(raw_json_text: str) -> dict[str, object] | None:
    try:
        parsed_payload = json.loads(raw_json_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_payload, dict):
        return None
    return {str(key): value for key, value in parsed_payload.items()}


def _server_exists(servers: tuple[McpServerConfig, ...], server_id: str) -> bool:
    return any(server.id == server_id for server in servers)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
