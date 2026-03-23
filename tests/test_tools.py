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


# --- Test Skills (Modality 2) ---


def test_skill_frontmatter_parsing():
    """Test SKILL.md frontmatter parsing."""
    from market_analyst.tools.skills import _parse_frontmatter

    text = """---
name: test_skill
description: A test skill for unit testing
---

# Body content here
"""
    meta = _parse_frontmatter(text)
    assert meta["name"] == "test_skill"
    assert meta["description"] == "A test skill for unit testing"


def test_skill_body_extraction():
    """Test extracting body from SKILL.md."""
    from market_analyst.tools.skills import _get_body

    text = """---
name: test
description: test
---

# The Body

Some content here."""
    body = _get_body(text)
    assert body.startswith("# The Body")
    assert "Some content here." in body


def test_load_skill_metadata():
    """Test loading skill metadata from the skills directory."""
    from market_analyst.tools.skills import load_skill_metadata

    skills = load_skill_metadata()
    assert len(skills) >= 2
    names = {s.name for s in skills}
    assert "earnings_analysis" in names
    assert "sector_comparison" in names


def test_use_skill_valid():
    """Test activating a valid skill returns its body."""
    from market_analyst.tools.skills import use_skill

    result = use_skill.invoke({"skill_name": "earnings_analysis"})
    assert "Revenue Assessment" in result
    assert "Profitability Metrics" in result


def test_use_skill_invalid():
    """Test activating an unknown skill returns error."""
    from market_analyst.tools.skills import use_skill

    result = use_skill.invoke({"skill_name": "nonexistent_skill"})
    assert "Unknown skill" in result
    assert "earnings_analysis" in result


# --- Test Code Execution (Modality 4) ---


def test_code_exec_simple_calculation():
    """Test executing a simple Python calculation."""
    from market_analyst.tools.code_exec import execute_python_analysis

    result = execute_python_analysis.invoke({"code": "print(2 + 2)"})
    assert "4" in result


def test_code_exec_financial_calculation():
    """Test executing a financial calculation."""
    from market_analyst.tools.code_exec import execute_python_analysis

    code = """
import math
cagr = (150 / 100) ** (1 / 3) - 1
print(f"CAGR: {cagr:.2%}")
"""
    result = execute_python_analysis.invoke({"code": code})
    assert "CAGR:" in result
    assert "14" in result  # ~14.47%


def test_code_exec_blocks_dangerous_imports():
    """Test that dangerous imports are blocked."""
    from market_analyst.tools.code_exec import execute_python_analysis

    result = execute_python_analysis.invoke({"code": "import os; os.listdir('.')"})
    assert "Blocked" in result

    result = execute_python_analysis.invoke({"code": "import subprocess"})
    assert "Blocked" in result


def test_code_exec_blocks_eval():
    """Test that eval/exec calls are blocked."""
    from market_analyst.tools.code_exec import execute_python_analysis

    result = execute_python_analysis.invoke({"code": "eval('2+2')"})
    assert "Blocked" in result


# --- Test CLI-as-Tool (Modality 3) ---


def test_cli_list_reports_command_construction(mocker):
    """Test that cli_list_reports constructs the correct CLI command."""
    mock_run = mocker.patch("market_analyst.tools.cli_tools.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "[]"

    from market_analyst.tools.cli_tools import cli_list_reports

    cli_list_reports.invoke({})

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == ["uv", "run", "market-analyst", "--list-reports", "--json"]


def test_cli_show_report_command_construction(mocker):
    """Test that cli_show_report constructs the correct CLI command."""
    mock_run = mocker.patch("market_analyst.tools.cli_tools.subprocess.run")
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = '{"key": "test", "content": "report"}'

    from market_analyst.tools.cli_tools import cli_show_report

    cli_show_report.invoke({"report_key": "research_NVDA_20260322"})

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd == ["uv", "run", "market-analyst", "--show-report", "research_NVDA_20260322", "--json"]


def test_cli_timeout_handling(mocker):
    """Test that CLI command timeout is handled gracefully."""
    import subprocess as sp

    mock_run = mocker.patch("market_analyst.tools.cli_tools.subprocess.run")
    mock_run.side_effect = sp.TimeoutExpired(cmd="test", timeout=30)

    from market_analyst.tools.cli_tools import cli_list_reports

    result = cli_list_reports.invoke({})
    assert "timed out" in result
