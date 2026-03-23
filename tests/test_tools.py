import json
import os

import pandas as pd
import pytest

from market_analyst.tools.search import (
    NewsItem,
    NewsSearchResult,
    search_competitors,
    search_news,
)
from market_analyst.tools.stock import (
    FinancialsResult,
    PriceHistoryResult,
    StockQuery,
    StockSnapshot,
    get_financials,
    get_price_history,
    get_stock_snapshot,
)
from market_analyst.tools.trade import (
    TradeAction,
    TradeRequest,
    execute_trade,
    parse_trade_request,
)

# --- Test Input Validation ---


def test_stock_query_valid_ticker():
    """Test StockQuery validates and uppercases ticker."""
    q = StockQuery(ticker="nvda")
    assert q.ticker == "NVDA"


def test_stock_query_invalid_ticker():
    """Test StockQuery rejects invalid tickers."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        StockQuery(ticker="123ABC")

    with pytest.raises(ValueError, match="Invalid ticker"):
        StockQuery(ticker="TOOLONG")


def test_stock_query_invalid_period():
    """Test period validation rejects bad values."""
    from market_analyst.tools.stock import StockHistoryQuery

    with pytest.raises(ValueError, match="Invalid period"):
        StockHistoryQuery(ticker="AAPL", period="2w")


# --- Test Stock Tools ---


def test_get_stock_snapshot(mocker):
    """Test get_stock_snapshot consolidated tool."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")

    mock_ticker = mock_ticker_cls.return_value
    mock_ticker.info = {
        "currentPrice": 150.0,
        "regularMarketChangePercent": 2.5,
        "regularMarketVolume": 50000000,
        "marketCap": 2500000000000,
        "trailingPE": 25.5,
    }
    mock_ticker.fast_info = {"lastPrice": 155.0}

    result = get_stock_snapshot.invoke({"ticker": "AAPL"})

    assert isinstance(result, StockSnapshot)
    assert result.ticker == "AAPL"
    assert result.price == 155.0
    assert result.change_pct == 2.5
    assert result.volume == 50000000
    assert result.market_cap_b == 2500.0
    assert result.pe_ratio == 25.5
    assert "AAPL" in result.summary
    assert "$155.00" in result.summary


def test_get_price_history(mocker):
    """Test get_price_history tool."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")

    mock_ticker = mock_ticker_cls.return_value
    data = {
        "Close": [100.0, 110.0],
        "High": [112.0, 115.0],
        "Low": [98.0, 105.0],
        "Volume": [1000000, 1200000],
    }
    mock_ticker.history.return_value = pd.DataFrame(data)

    result = get_price_history.invoke({"ticker": "AAPL"})

    assert isinstance(result, PriceHistoryResult)
    assert result.ticker == "AAPL"
    assert result.start_price == 100.0
    assert result.end_price == 110.0
    assert result.change_percent == 10.0
    assert result.avg_volume == 1100000


def test_get_financials(mocker):
    """Test get_financials tool with income statement."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")

    mock_ticker = mock_ticker_cls.return_value

    # Create mock income statement DataFrame
    dates = pd.to_datetime(["2024-12-31", "2023-12-31"])
    income_data = pd.DataFrame(
        {
            dates[0]: [100e9, 25e9, 40e9, 60e9, 45e9],
            dates[1]: [90e9, 20e9, 35e9, 55e9, 40e9],
        },
        index=["Total Revenue", "Net Income", "Operating Income", "Gross Profit", "EBITDA"],
    )
    mock_ticker.income_stmt = income_data
    mock_ticker.balance_sheet = pd.DataFrame()
    mock_ticker.cashflow = pd.DataFrame()

    result = get_financials.invoke({"ticker": "AAPL", "statement_type": "income"})

    assert isinstance(result, FinancialsResult)
    assert result.ticker == "AAPL"
    assert result.statement_type == "income"
    assert "Total Revenue" in result.data
    assert len(result.periods) == 2
    assert "AAPL" in result.summary


# --- Test Search Tools ---


def test_search_news(mocker):
    """Test search_news tool returns NewsItem objects."""
    mock_client_cls = mocker.patch("market_analyst.tools.search.TavilyClient")

    mocker.patch.dict(os.environ, {"TAVILY_API_KEY": "test"})

    mock_client = mock_client_cls.return_value
    mock_client.search.return_value = {
        "results": [
            {
                "title": "NVIDIA Reports Record Earnings",
                "url": "http://finance.example.com/nvidia-earnings",
                "content": "NVIDIA reported record quarterly revenue. The company exceeded analyst expectations. AI chip demand continues to grow.",
                "score": 0.9,
                "published_date": "2024-01-15",
            }
        ],
        "answer": "Test summary",
    }

    result = search_news.invoke({"query": "NVDA earnings"})

    assert isinstance(result, NewsSearchResult)
    assert result.summary == "Test summary"
    assert len(result.results) == 1
    assert isinstance(result.results[0], NewsItem)
    assert result.results[0].headline == "NVIDIA Reports Record Earnings"
    assert result.results[0].source == "finance.example.com"
    assert len(result.results[0].key_points) > 0


def test_search_competitors(mocker):
    """Test search_competitors tool."""
    mock_client_cls = mocker.patch("market_analyst.tools.search.TavilyClient")

    mocker.patch.dict(os.environ, {"TAVILY_API_KEY": "test"})

    mock_client = mock_client_cls.return_value
    mock_client.search.return_value = {
        "results": [],
        "answer": "Competitors summary",
    }

    result = search_competitors.invoke({"ticker": "AAPL"})

    mock_client.search.assert_called_once()
    call_args = mock_client.search.call_args
    assert "AAPL" in call_args.kwargs["query"]
    assert "competitors" in call_args.kwargs["query"]
    assert isinstance(result, NewsSearchResult)


# --- Test Trade Tool ---


def test_execute_trade_returns_valid_request():
    """Test execute_trade returns formatted string."""
    result = execute_trade.invoke(
        {
            "action": "buy",
            "ticker": "NVDA",
            "amount_usd": 5000,
            "reason": "Strong earnings",
        }
    )

    assert result.startswith("TRADE_REQUEST:")

    json_str = result.replace("TRADE_REQUEST:", "")
    data = json.loads(json_str)

    assert data["action"] == "buy"
    assert data["ticker"] == "NVDA"
    assert data["amount_usd"] == 5000


def test_parse_trade_request_valid():
    """Test parsing a valid trade request string."""
    request_str = 'TRADE_REQUEST:{"action": "sell", "ticker": "TSLA", "amount_usd": 1000, "reason": "Overvalued"}'

    req = parse_trade_request(request_str)

    assert isinstance(req, TradeRequest)
    assert req.action == TradeAction.SELL
    assert req.ticker == "TSLA"


def test_parse_trade_request_invalid():
    """Test parsing invalid string returns None."""
    assert parse_trade_request("Not a trade request") is None
