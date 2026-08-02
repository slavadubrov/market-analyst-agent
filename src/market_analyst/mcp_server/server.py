"""MCP server that wraps the project's tools.

Design notes
------------

The sidecar exposes a curated subset of the worker's tools — the ones that
need API credentials (Tavily, brokerage, etc.). Tools that are pure
computations (``execute_python_analysis``) stay in-process; there is no
secret to broker for those.

The MCP protocol is implemented by the ``mcp`` package. Two transports are
supported:

- **stdio**: the canonical MCP transport. The worker spawns the sidecar as a
  child process and reads/writes JSON-RPC over the child's stdio. This is the
  shape used by Claude Desktop and the OpenAI Agents SDK MCP integration.
- **http**: an HTTP-bound variant used by the docker-compose sidecar so the
  worker container can talk to the sidecar over the bridge network.

Why we don't pull the worker over to MCP today: the rest of this repo uses
LangChain ``@tool`` functions directly. Migrating the worker to consume
tools via an MCP client is a separate refactor (the article describes the
shape but doesn't mandate the migration). What this sidecar does is *make
that migration possible* without rewriting the tools — same wrapped
implementations, just one extra hop.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# The MCP package's public entrypoints. Imported lazily inside ``build_server``
# so this module can be imported (for testing) even when ``mcp`` is missing.

_SERVER_NAME = "market-analyst-tools"


def _tool_get_stock_snapshot(ticker: str) -> str:
    """Return a JSON-serialized stock snapshot."""
    from market_analyst.tools.stock import get_stock_snapshot

    result = get_stock_snapshot.invoke({"ticker": ticker})
    return str(result.model_dump_json())


def _tool_get_price_history(ticker: str, period: str = "1mo") -> str:
    from market_analyst.tools.stock import get_price_history

    result = get_price_history.invoke({"ticker": ticker, "period": period})
    return str(result.model_dump_json())


def _tool_get_financials(ticker: str, statement_type: str = "income") -> str:
    from market_analyst.tools.stock import get_financials

    result = get_financials.invoke({"ticker": ticker, "statement_type": statement_type})
    return str(result.model_dump_json())


def _tool_search_news(query: str, max_results: int = 5) -> str:
    from market_analyst.tools.search import search_news

    result = search_news.invoke({"query": query, "max_results": max_results})
    return str(result.model_dump_json())


def _tool_search_competitors(ticker: str, max_results: int = 3) -> str:
    from market_analyst.tools.search import search_competitors

    result = search_competitors.invoke({"ticker": ticker, "max_results": max_results})
    return str(result.model_dump_json())


TOOL_TABLE: dict[str, Callable[..., str]] = {
    "get_stock_snapshot": _tool_get_stock_snapshot,
    "get_price_history": _tool_get_price_history,
    "get_financials": _tool_get_financials,
    "search_news": _tool_search_news,
    "search_competitors": _tool_search_competitors,
}


def build_server() -> Any:
    """Construct an MCP ``Server`` instance with all tools registered.

    Returns the live server object so callers can plug it into whichever
    transport they want (stdio for child-process use, HTTP for the sidecar
    container). Lazy-imports ``mcp`` so unit tests don't need the package.
    """
    from mcp.server.fastmcp import FastMCP

    app = FastMCP(_SERVER_NAME)

    @app.tool()
    def get_stock_snapshot(ticker: str) -> str:
        """Get a comprehensive stock snapshot (price, volume, market cap, P/E)."""
        return _tool_get_stock_snapshot(ticker)

    @app.tool()
    def get_price_history(ticker: str, period: str = "1mo") -> str:
        """Historical price data over the given period."""
        return _tool_get_price_history(ticker, period)

    @app.tool()
    def get_financials(ticker: str, statement_type: str = "income") -> str:
        """Income statement, balance sheet, or cash flow for a ticker."""
        return _tool_get_financials(ticker, statement_type)

    @app.tool()
    def search_news(query: str, max_results: int = 5) -> str:
        """Search recent market news via Tavily."""
        return _tool_search_news(query, max_results)

    @app.tool()
    def search_competitors(ticker: str, max_results: int = 3) -> str:
        """Find competitor analysis for a ticker."""
        return _tool_search_competitors(ticker, max_results)

    return app


def invoke_tool(name: str, arguments: dict[str, Any]) -> str:
    """Invoke a registered tool by name without going through the MCP runtime.

    Used by tests and by direct in-process callers. Returns the JSON-serialized
    tool result or a JSON error envelope.
    """
    fn = TOOL_TABLE.get(name)
    if fn is None:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        return fn(**arguments)
    except Exception as exc:  # noqa: BLE001 — surface any tool failure
        return json.dumps({"error": str(exc), "type": type(exc).__name__})


def run(*, transport: str = "stdio", port: int = 8765) -> None:
    """Run the sidecar with the requested transport. Blocks the caller.

    Args:
        transport: ``"stdio"`` or ``"http"``. Stdio is the default because it
            matches the canonical MCP client behavior; HTTP is for the
            docker-compose sidecar where the worker is in a different
            container.
        port: HTTP port (ignored for stdio).
    """
    app = build_server()
    if transport == "stdio":
        logger.info("Starting MCP sidecar on stdio")
        app.run()
    elif transport == "http":
        logger.info("Starting MCP sidecar on http://0.0.0.0:%d", port)
        # FastMCP exposes both an SSE and an HTTP transport; HTTP is what the
        # sidecar service in docker-compose expects.
        app.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        raise ValueError(f"Unsupported MCP transport: {transport}")
