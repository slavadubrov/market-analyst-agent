"""MCP sidecar that exposes the stock/news tools to the worker.

Per the article's "Reference Stack: Docker Compose" and "Secret broker" sections:

> For Git, we use each repository's access token to clone the repo during
> sandbox initialization and wire it into the local git remote. … For custom
> tools, we support MCP and store OAuth tokens in a secure vault. Claude calls
> MCP tools via a dedicated proxy; this proxy takes in a token associated with
> the session. … The harness is never made aware of any credentials.

In the reference stack we approximate the same pattern: the sidecar reads
``TAVILY_API_KEY`` / ``ANTHROPIC_API_KEY`` from its environment (a Docker
secret in production, ``.env`` in dev) and re-exposes the tool surface to the
LangGraph worker over MCP. The worker container has no API keys in its
environment.
"""

from market_analyst.mcp_server.server import build_server, run

__all__ = ["build_server", "run"]
