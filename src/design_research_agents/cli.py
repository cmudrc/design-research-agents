"""Command-line interface for toolbox, MCP, and script tools."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from design_research_agents.mcp_server.server import _serve_stdio
from design_research_agents.tools import Toolbox
from design_research_agents.tools.config import McpServer, load_tool_runtime_config


def main(argv: list[str] | None = None) -> int:
    """Run CLI entrypoint and return process exit code.

    Args:
        argv: Parameter value.

    Returns:
        The resulting value.
    """
    parser = argparse.ArgumentParser(prog="dra")
    subparsers = parser.add_subparsers(dest="command")

    mcp_parser = subparsers.add_parser("mcp", help="MCP server/client commands")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")

    mcp_subparsers.add_parser("serve", help="Serve built-in MCP tools over stdio")

    ping_parser = mcp_subparsers.add_parser("ping", help="Connect to one MCP server and list tools")
    ping_parser.add_argument("--server", required=True, help="Configured MCP server id")
    ping_parser.add_argument("--config", help="Tool runtime YAML config path")

    call_parser = mcp_subparsers.add_parser("call", help="Call one tool through toolbox")
    call_parser.add_argument("tool", help="Tool name to invoke")
    call_parser.add_argument("--json", required=True, help="JSON object with tool input")
    call_parser.add_argument("--config", help="Tool runtime YAML config path")

    script_parser = subparsers.add_parser("script", help="Script tool commands")
    script_subparsers = script_parser.add_subparsers(dest="script_command")

    lint_parser = script_subparsers.add_parser("lint", help="Lint script-tool files")
    lint_parser.add_argument("target", help="Directory or file to lint")

    list_parser = script_subparsers.add_parser("list", help="List configured script tools")
    list_parser.add_argument("--config", help="Tool runtime YAML config path")

    run_parser = script_subparsers.add_parser("run", help="Run one script tool")
    run_parser.add_argument("tool_name", help="Script tool canonical or short name")
    run_parser.add_argument("--json", required=True, help="JSON object tool input")
    run_parser.add_argument("--config", help="Tool runtime YAML config path")

    parsed_args = parser.parse_args(argv)

    if parsed_args.command == "mcp":
        return _handle_mcp(parsed_args)
    if parsed_args.command == "script":
        return _handle_script(parsed_args)

    parser.print_help()
    return 1


def _handle_mcp(cli_args: argparse.Namespace) -> int:
    """Run handle mcp.

    Args:
        cli_args: Parameter value.

    Returns:
        The resulting value.
    """
    if cli_args.mcp_command == "serve":
        _serve_stdio()
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


def _handle_script(cli_args: argparse.Namespace) -> int:
    """Run handle script.

    Args:
        cli_args: Parameter value.

    Returns:
        The resulting value.
    """
    if cli_args.script_command == "lint":
        diagnostics = _lint_script_target(Path(str(cli_args.target)).expanduser())
        if not diagnostics:
            print(json.dumps({"ok": True, "diagnostics": []}, ensure_ascii=True, indent=2))
            return 0
        print(
            json.dumps(
                {
                    "ok": False,
                    "diagnostics": diagnostics,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 2

    if cli_args.script_command == "list":
        tool_runtime = _build_runtime(cli_args.config)
        try:
            script_tools = [
                spec.name for spec in tool_runtime.list_tools() if spec.metadata.source == "script"
            ]
            print(json.dumps({"tools": script_tools}, ensure_ascii=True, indent=2))
            return 0
        finally:
            tool_runtime.close()

    if cli_args.script_command == "run":
        tool_runtime = _build_runtime(cli_args.config)
        try:
            tool_input_payload = _parse_json_object(cli_args.json)
            if tool_input_payload is None:
                print("--json must be a valid JSON object.")
                return 1

            requested_name = str(cli_args.tool_name).strip()
            resolved_name = requested_name
            if not requested_name.startswith("script::"):
                resolved_name = f"script::{requested_name}"

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

    print("Unknown script command.")
    return 1


def _build_runtime(path: str | None) -> Toolbox:
    """Run build runtime.

    Args:
        path: Parameter value.

    Returns:
        The resulting value.
    """
    if not path:
        return Toolbox()

    runtime_config = load_tool_runtime_config(path)
    mcp_servers = runtime_config.mcp.servers if runtime_config.mcp.enabled else None
    script_tools = (
        runtime_config.script_tools.tools if runtime_config.script_tools.enabled else None
    )
    return Toolbox(
        workspace_root=runtime_config.core_tools.workspace_root,
        enable_core_tools=runtime_config.core_tools.enabled,
        script_tools=script_tools,
        mcp_servers=mcp_servers,
    )


def _parse_json_object(raw_json_text: str) -> dict[str, object] | None:
    """Run parse json object.

    Args:
        raw_json_text: Parameter value.

    Returns:
        The resulting value.
    """
    try:
        parsed_payload = json.loads(raw_json_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed_payload, dict):
        return None
    return {str(key): value for key, value in parsed_payload.items()}


def _server_exists(servers: tuple[McpServer, ...], server_id: str) -> bool:
    """Run server exists.

    Args:
        servers: Parameter value.
        server_id: Parameter value.

    Returns:
        The resulting value.
    """
    return any(server.id == server_id for server in servers)


def _lint_script_target(target_path: Path) -> list[dict[str, str]]:
    """Run lint script target.

    Args:
        target_path: Parameter value.

    Returns:
        The resulting value.
    """
    diagnostics: list[dict[str, str]] = []
    if not target_path.exists():
        return [{"path": str(target_path), "error": "Target path does not exist."}]

    candidates: list[Path] = []
    if target_path.is_file():
        candidates.append(target_path)
    else:
        candidates.extend(target_path.rglob("*.py"))
        candidates.extend(target_path.rglob("*.sh"))

    if not candidates:
        return [{"path": str(target_path), "error": "No .py or .sh scripts found."}]

    for candidate in sorted(candidates):
        if candidate.suffix not in {".py", ".sh"}:
            continue
        if not candidate.is_file():
            diagnostics.append({"path": str(candidate), "error": "Not a regular file."})
            continue
        if not candidate.exists():
            diagnostics.append({"path": str(candidate), "error": "Script file is missing."})

    return diagnostics


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
