"""CLI helpers for built-in MCP server."""

from __future__ import annotations

from ._server import _serve_stdio


def main() -> int:
    """Stdio MCP server for CLI entrypoints.

    Returns:
        Process exit code (``0`` on normal shutdown).
    """
    _serve_stdio()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
