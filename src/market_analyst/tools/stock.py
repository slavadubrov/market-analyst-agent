"""Stock market tools using YFinance.

These tools demonstrate good ACI (Agent-Computer Interface) design:
- Clear, semantic parameter names (ticker, not company_name)
- Pydantic validation for type safety
- Concise vs detailed modes to optimize token usage
- Error handling with meaningful messages
"""

from typing import Literal

import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel


class StockPriceResult(BaseModel):
    """Result from get_stock_price tool."""

    ticker: str
    current_price: float
    currency: str
    change_percent: float
    market_state: str  # "REGULAR", "PRE", "POST", "CLOSED"


class CompanyMetricsResult(BaseModel):
    """Result from get_company_metrics tool."""

    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: float | None
    pe_ratio: float | None
    forward_pe: float | None
    dividend_yield: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    avg_volume: int | None
    # Only included in detailed mode
    description: str | None = None
    full_time_employees: int | None = None
    revenue: float | None = None
    profit_margin: float | None = None


class PriceHistoryResult(BaseModel):
    """Result from get_price_history tool."""

    ticker: str
    period: str
    data_points: int
    start_price: float
    end_price: float
    high: float
    low: float
    change_percent: float
    summary: str


@tool
def get_stock_price(ticker: str) -> StockPriceResult:
    """Get the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA", "MSFT").
                Must be a valid ticker symbol, NOT a company name.

    Returns:
        Current price, change percentage, and market state.

    Example:
        >>> get_stock_price("NVDA")
        StockPriceResult(ticker="NVDA", current_price=875.32, ...)
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        # Get fast info for current price
        fast_info = stock.fast_info

        return StockPriceResult(
            ticker=ticker.upper(),
            current_price=fast_info.get("lastPrice", info.get("currentPrice", 0)),
            currency=info.get("currency", "USD"),
            change_percent=info.get("regularMarketChangePercent", 0) or 0,
            market_state=info.get("marketState", "UNKNOWN"),
        )
    except Exception as e:
        raise ValueError(f"Failed to get price for {ticker}: {str(e)}")


@tool
def get_company_metrics(ticker: str, mode: Literal["concise", "detailed"] = "concise") -> CompanyMetricsResult:
    """Get company financial metrics and information.

    This tool returns key financial metrics for investment analysis.
    Use 'concise' mode for quick lookups (saves tokens).
    Use 'detailed' mode when you need full company information.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA").
        mode: "concise" for key metrics only, "detailed" for full info.

    Returns:
        Company metrics including P/E ratio, market cap, sector, etc.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info

        result = CompanyMetricsResult(
            ticker=ticker.upper(),
            name=info.get("shortName", info.get("longName", ticker)),
            sector=info.get("sector", "Unknown"),
            industry=info.get("industry", "Unknown"),
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            dividend_yield=info.get("dividendYield"),
            fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=info.get("fiftyTwoWeekLow"),
            avg_volume=info.get("averageVolume"),
        )

        # Add extra fields in detailed mode
        if mode == "detailed":
            result.description = info.get("longBusinessSummary")
            result.full_time_employees = info.get("fullTimeEmployees")
            result.revenue = info.get("totalRevenue")
            result.profit_margin = info.get("profitMargins")

        return result
    except Exception as e:
        raise ValueError(f"Failed to get metrics for {ticker}: {str(e)}")


@tool
def get_price_history(ticker: str, period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y"] = "1mo") -> PriceHistoryResult:
    """Get historical price data and summary statistics.

    Returns a summary of price movement over the specified period.
    For detailed raw data, use period="1d" for intraday or longer periods
    for daily data.

    Args:
        ticker: Stock ticker symbol.
        period: Time period - "1d", "5d", "1mo", "3mo", "6mo", or "1y".

    Returns:
        Summary statistics including start/end price, high/low, and change.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        hist = stock.history(period=period)

        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        high = float(hist["High"].max())
        low = float(hist["Low"].min())
        change_pct = ((end_price - start_price) / start_price) * 100

        # Generate human-readable summary
        direction = "up" if change_pct > 0 else "down"
        summary = (
            f"{ticker.upper()} is {direction} {abs(change_pct):.1f}% over the last {period}. "
            f"Price ranged from ${low:.2f} to ${high:.2f}. "
            f"Currently at ${end_price:.2f}."
        )

        return PriceHistoryResult(
            ticker=ticker.upper(),
            period=period,
            data_points=len(hist),
            start_price=start_price,
            end_price=end_price,
            high=high,
            low=low,
            change_percent=change_pct,
            summary=summary,
        )
    except Exception as e:
        raise ValueError(f"Failed to get history for {ticker}: {str(e)}")
