"""Built-in MCP server package."""

from .server import StdioMcpServer, serve_stdio

__all__ = ["StdioMcpServer", "serve_stdio"]
