"""CLI helpers for built-in MCP server."""

from __future__ import annotations

from .server import serve_stdio


def main() -> int:
    """Run stdio MCP server for CLI entrypoints."""
    serve_stdio()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
