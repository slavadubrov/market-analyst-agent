import json
import os

from market_analyst.tools.search import (
    NewsSearchResult,
    search_competitors,
    search_news,
)
from market_analyst.tools.stock import (
    get_company_metrics,
    get_price_history,
    get_stock_price,
)
from market_analyst.tools.trade import (
    TradeAction,
    TradeRequest,
    execute_trade,
    parse_trade_request,
)

# --- Test Search Tool (search.py) ---


def test_search_news(mocker):
    """Test search_news tool."""
    mock_client_cls = mocker.patch("market_analyst.tools.search.TavilyClient")

    mocker.patch.dict(os.environ, {"TAVILY_API_KEY": "test"})

    mock_client = mock_client_cls.return_value
    mock_client.search.return_value = {
        "results": [
            {
                "title": "Test News",
                "url": "http://test.com",
                "content": "Test content",
                "score": 0.9,
            }
        ],
        "answer": "Test summary",
    }

    result = search_news.invoke({"query": "stock market"})

    assert isinstance(result, NewsSearchResult)
    assert result.summary == "Test summary"
    assert len(result.results) == 1
    assert result.results[0]["title"] == "Test News"


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


# --- Test Stock Tool (stock.py) ---


def test_get_stock_price(mocker):
    """Test get_stock_price tool."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")

    mock_ticker = mock_ticker_cls.return_value
    mock_ticker.info = {"currency": "USD", "currentPrice": 150.0}
    mock_ticker.fast_info = {"lastPrice": 155.0}

    result = get_stock_price.invoke({"ticker": "AAPL"})

    assert result.ticker == "AAPL"
    assert result.current_price == 155.0
    assert result.currency == "USD"


def test_get_company_metrics_concise(mocker):
    """Test get_company_metrics in concise mode."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")

    mock_ticker = mock_ticker_cls.return_value
    mock_ticker.info = {
        "shortName": "Apple Inc",
        "marketCap": 2000000000,
        "trailingPE": 25.5,
    }

    result = get_company_metrics.invoke({"ticker": "AAPL", "mode": "concise"})

    assert result.ticker == "AAPL"
    assert result.name == "Apple Inc"
    assert result.pe_ratio == 25.5
    assert result.description is None  # Should be hidden in concise mode


def test_get_price_history(mocker):
    """Test get_price_history tool."""
    mock_ticker_cls = mocker.patch("market_analyst.tools.stock.yf.Ticker")

    mock_ticker = mock_ticker_cls.return_value
    # Mock history dataframe
    import pandas as pd

    data = {"Close": [100.0, 110.0], "High": [112.0, 115.0], "Low": [98.0, 105.0]}
    mock_ticker.history.return_value = pd.DataFrame(data)

    result = get_price_history.invoke({"ticker": "AAPL"})

    assert result.ticker == "AAPL"
    assert result.start_price == 100.0
    assert result.end_price == 110.0
    assert result.change_percent == 10.0


# --- Test Trade Tool (trade.py) ---


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

    # Parse back the json
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
