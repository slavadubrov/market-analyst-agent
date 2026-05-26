"""Tests for the MCP sidecar (Part 5: secret-broker / tool proxy)."""

from __future__ import annotations

import json
from typing import Any


def test_invoke_tool_returns_json_error_for_unknown_tool():
    from market_analyst.mcp_server.server import invoke_tool

    out = invoke_tool("does_not_exist", {})
    payload = json.loads(out)
    assert "error" in payload
    assert "does_not_exist" in payload["error"]


def test_invoke_tool_get_stock_snapshot_returns_json(mocker):
    """The sidecar's tool entry point should produce the same JSON envelope
    callers would get from the LangChain tool directly."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")
    mock_ticker = mock_ticker_cls.return_value
    mock_ticker.info = {
        "currentPrice": 100.0,
        "regularMarketChangePercent": 1.0,
        "regularMarketVolume": 1_000_000,
        "marketCap": 1e12,
        "trailingPE": 20.0,
    }
    mock_ticker.fast_info = {"lastPrice": 100.0}

    from market_analyst.mcp_server.server import invoke_tool

    out = invoke_tool("get_stock_snapshot", {"ticker": "AAPL"})
    payload = json.loads(out)
    assert payload["ticker"] == "AAPL"
    assert payload["price"] == 100.0


def test_invoke_tool_propagates_validation_errors(mocker):
    """Pydantic input validation errors should land in the JSON envelope, not crash."""
    from market_analyst.mcp_server.server import invoke_tool

    out = invoke_tool("get_stock_snapshot", {"ticker": "123BAD"})
    payload: dict[str, Any] = json.loads(out)
    assert "error" in payload


def test_tool_table_has_expected_entries():
    """Document the curated MCP surface so regressions are loud."""
    from market_analyst.mcp_server.server import TOOL_TABLE

    assert set(TOOL_TABLE.keys()) == {
        "get_stock_snapshot",
        "get_price_history",
        "get_financials",
        "search_news",
        "search_competitors",
    }
