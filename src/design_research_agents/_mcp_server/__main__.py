"""Module entrypoint for built-in stdio MCP server."""

from ._server import _serve_stdio

if __name__ == "__main__":
    _serve_stdio()
