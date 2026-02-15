"""Module entrypoint for built-in stdio MCP server."""

from .server import serve_stdio

if __name__ == "__main__":
    serve_stdio()
