"""``python -m market_analyst.mcp_server`` entry point.

Runs the MCP sidecar in stdio mode by default (suitable for use as a child
process of the worker). When ``MCP_TRANSPORT=http`` is set, runs as an HTTP
server on ``MCP_PORT`` (default 8765) so the docker-compose sidecar service
can be reached over the bridge network.
"""

import os

from market_analyst.mcp_server.server import run


def main() -> None:
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    port = int(os.getenv("MCP_PORT", "8765"))
    run(transport=transport, port=port)


if __name__ == "__main__":
    main()
