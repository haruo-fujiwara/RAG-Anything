from fastmcp import FastMCP

mcp = FastMCP("RAG-Anything MCP")


@mcp.tool()
def ping(name: str = "world") -> str:
    """Simple tool to validate MCP connectivity."""
    return f"pong, {name}"


if __name__ == "__main__":
    mcp.run()
