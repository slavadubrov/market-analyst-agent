"""Tools module for Market Analyst Agent.

Five consolidated tools following ACI design principles:
- get_stock_snapshot: Price, metrics, and key data in one call
- get_price_history: Historical prices with volume data
- get_financials: Unified financial statements
- search_news: Recent news with extracted key points
- search_competitors: Competitor analysis with relative metrics
"""

from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.stock import (
    get_financials,
    get_price_history,
    get_stock_snapshot,
)
from market_analyst.tools.trade import execute_trade, parse_trade_request

__all__ = [
    "get_stock_snapshot",
    "get_price_history",
    "get_financials",
    "search_news",
    "search_competitors",
    "execute_trade",
    "parse_trade_request",
]
