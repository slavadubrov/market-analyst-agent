"""Stock market tools using YFinance.

These tools demonstrate ACI (Agent-Computer Interface) design principles:
- Tool consolidation: 5 high-level tools instead of 10+ granular ones
- Pydantic input validation with field_validator guardrails
- Structured outputs with human-readable summary fields
- Retry with exponential backoff for transient failures
- Concise, high-signal responses optimized for token efficiency
"""

import yfinance as yf
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Input validation models (ACI guardrails — catch errors before API calls)
# ---------------------------------------------------------------------------


class StockQuery(BaseModel):
    """Validated input for stock queries.

    Pydantic catches malformed tickers before the API call,
    preventing error propagation through the reasoning loop.
    """

    ticker: str = Field(description="Stock ticker symbol (e.g., NVDA)")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not v.isalpha() or len(v) > 5:
            raise ValueError(f"Invalid ticker format: {v}")
        return v


class StockHistoryQuery(StockQuery):
    """Validated input for price history queries."""

    period: str = Field(default="1mo", description="Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y")

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        valid = {"1d", "5d", "1mo", "3mo", "6mo", "1y"}
        if v not in valid:
            raise ValueError(f"Invalid period: {v}. Must be one of {valid}")
        return v


class FinancialsQuery(StockQuery):
    """Validated input for financial statement queries."""

    statement_type: str = Field(default="income", description="Statement type: income, balance_sheet, cash_flow, all")

    @field_validator("statement_type")
    @classmethod
    def validate_statement_type(cls, v: str) -> str:
        valid = {"income", "balance_sheet", "cash_flow", "all"}
        if v not in valid:
            raise ValueError(f"Invalid statement_type: {v}. Must be one of {valid}")
        return v


# ---------------------------------------------------------------------------
# Output models (structured, high-signal responses)
# ---------------------------------------------------------------------------


class StockSnapshot(BaseModel):
    """Consolidated stock snapshot — the agent never sees raw API noise.

    Replaces separate get_stock_price + get_company_metrics calls.
    The summary field provides a ready-to-use string for reports.
    """

    ticker: str
    price: float
    change_pct: float
    volume: int
    market_cap_b: float
    pe_ratio: float | None
    summary: str


class PriceHistoryResult(BaseModel):
    """Result from get_price_history tool."""

    ticker: str
    period: str
    data_points: int
    start_price: float
    end_price: float
    high: float
    low: float
    avg_volume: int
    change_percent: float
    summary: str


class FinancialsResult(BaseModel):
    """Result from get_financials tool."""

    ticker: str
    statement_type: str
    periods: list[str]
    data: dict[str, dict[str, float | None]]
    summary: str


# ---------------------------------------------------------------------------
# Retry decorator for transient API failures
# ---------------------------------------------------------------------------

_yfinance_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)


@_yfinance_retry
def _fetch_ticker_info(ticker: str) -> tuple:
    """Fetch ticker info and fast_info with retry on transient failures."""
    stock = yf.Ticker(ticker)
    return stock, stock.info, stock.fast_info


@_yfinance_retry
def _fetch_ticker_history(ticker: str, period: str) -> tuple:
    """Fetch ticker history with retry on transient failures."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return stock, hist


@_yfinance_retry
def _fetch_ticker_financials(ticker: str) -> tuple:
    """Fetch ticker financial statements with retry on transient failures."""
    stock = yf.Ticker(ticker)
    return stock, stock.income_stmt, stock.balance_sheet, stock.cashflow


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(args_schema=StockQuery)
def get_stock_snapshot(ticker: str) -> StockSnapshot:
    """Get a comprehensive stock snapshot including price, volume, market cap, and P/E ratio.

    Returns everything needed for basic stock analysis in a single call. Use this
    as your first tool when analyzing any stock. The summary field provides a
    human-readable one-liner suitable for direct use in reports. For historical
    price trends, use get_price_history instead. For detailed financial statements,
    use get_financials.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA", "MSFT").
                Must be a valid ticker symbol, NOT a company name.

    Returns:
        StockSnapshot with price, change_pct, volume, market_cap_b, pe_ratio, and summary.

    Example:
        >>> get_stock_snapshot("NVDA")
        StockSnapshot(ticker="NVDA", price=875.32, change_pct=2.1, ...)
    """
    try:
        _stock, info, fast_info = _fetch_ticker_info(ticker)

        price = fast_info.get("lastPrice", info.get("currentPrice", 0))
        change_pct = round(info.get("regularMarketChangePercent", 0) or 0, 2)
        volume = info.get("regularMarketVolume", 0) or 0
        market_cap = info.get("marketCap", 0) or 0
        market_cap_b = round(market_cap / 1e9, 1)
        pe_ratio = info.get("trailingPE")

        direction = "up" if change_pct > 0 else "down"
        summary = f"{ticker} at ${price:.2f} ({direction} {abs(change_pct):.1f}%), market cap ${market_cap_b}B, P/E {pe_ratio or 'N/A'}"

        return StockSnapshot(
            ticker=ticker,
            price=price,
            change_pct=change_pct,
            volume=volume,
            market_cap_b=market_cap_b,
            pe_ratio=pe_ratio,
            summary=summary,
        )
    except Exception as e:
        raise ValueError(f"Failed to get snapshot for {ticker}: {e!s}") from e


@tool(args_schema=StockHistoryQuery)
def get_price_history(ticker: str, period: str = "1mo") -> PriceHistoryResult:
    """Get historical price data and summary statistics for a stock.

    Returns price movement summary over the specified period including start/end
    prices, high/low range, average volume, and percentage change. Use period='1d'
    for intraday data or longer periods like '3mo' or '1y' for trend analysis.
    The summary field provides a human-readable description of price action.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA").
        period: Time period — "1d", "5d", "1mo", "3mo", "6mo", or "1y".

    Returns:
        PriceHistoryResult with start/end prices, high/low, avg_volume, change, and summary.
    """
    try:
        _stock, hist = _fetch_ticker_history(ticker, period)

        if hist.empty:
            raise ValueError(f"No historical data available for {ticker}")

        start_price = float(hist["Close"].iloc[0])
        end_price = float(hist["Close"].iloc[-1])
        high = float(hist["High"].max())
        low = float(hist["Low"].min())
        avg_volume = int(hist["Volume"].mean()) if "Volume" in hist.columns else 0
        change_pct = ((end_price - start_price) / start_price) * 100

        direction = "up" if change_pct > 0 else "down"
        summary = (
            f"{ticker} is {direction} {abs(change_pct):.1f}% over the last {period}. "
            f"Price ranged from ${low:.2f} to ${high:.2f}. "
            f"Currently at ${end_price:.2f}."
        )

        return PriceHistoryResult(
            ticker=ticker,
            period=period,
            data_points=len(hist),
            start_price=start_price,
            end_price=end_price,
            high=high,
            low=low,
            avg_volume=avg_volume,
            change_percent=change_pct,
            summary=summary,
        )
    except Exception as e:
        raise ValueError(f"Failed to get history for {ticker}: {e!s}") from e


_KEY_ROWS = {
    "income": ["Total Revenue", "Net Income", "Operating Income", "Gross Profit", "EBITDA"],
    "balance_sheet": ["Total Assets", "Total Liabilities Net Minority Interest", "Stockholders Equity", "Total Debt", "Cash And Cash Equivalents"],
    "cash_flow": ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure", "Investing Cash Flow", "Financing Cash Flow"],
}


def _extract_statement_data(df, stype: str, prefix: str, all_data: dict, all_periods: set) -> None:
    """Extract key rows from a financial statement DataFrame."""
    if df is None or df.empty:
        return
    periods = [col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col) for col in df.columns[:4]]
    all_periods.update(periods)
    for row_name in _KEY_ROWS[stype]:
        if row_name in df.index:
            row_data = {}
            for col, period_str in zip(df.columns[:4], periods, strict=False):
                val = df.loc[row_name, col]
                row_data[period_str] = float(val) if val is not None and str(val) != "nan" else None
            label = f"{stype}/{row_name}" if prefix else row_name
            all_data[label] = row_data


def _build_financials_summary(ticker: str, statement_type: str, data: dict) -> str:
    """Build a human-readable summary of key financial metrics."""
    parts = [f"{ticker} financials ({statement_type}):"]
    for metric, values in list(data.items())[:5]:
        latest = next((v for v in values.values() if v is not None), None)
        if latest is not None:
            if abs(latest) >= 1e9:
                parts.append(f"  {metric}: ${latest / 1e9:.1f}B")
            elif abs(latest) >= 1e6:
                parts.append(f"  {metric}: ${latest / 1e6:.1f}M")
            else:
                parts.append(f"  {metric}: ${latest:,.0f}")
    return "\n".join(parts)


@tool(args_schema=FinancialsQuery)
def get_financials(ticker: str, statement_type: str = "income") -> FinancialsResult:
    """Get financial statements for a company: income statement, balance sheet, or cash flow.

    Returns key financial metrics from the selected statement type. Use
    statement_type='all' to get a summary across all three statements. Data covers
    the most recent annual periods available. For quick valuation metrics like P/E
    ratio and market cap, use get_stock_snapshot instead.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "NVDA").
        statement_type: One of "income", "balance_sheet", "cash_flow", or "all".

    Returns:
        FinancialsResult with periods, data dict, and summary.
    """
    try:
        _stock, income_stmt, balance_sheet, cashflow = _fetch_ticker_financials(ticker)

        stmt_map = {"income": income_stmt, "balance_sheet": balance_sheet, "cash_flow": cashflow}
        types_to_process = list(_KEY_ROWS.keys()) if statement_type == "all" else [statement_type]
        use_prefix = statement_type == "all"

        all_data: dict[str, dict[str, float | None]] = {}
        all_periods: set[str] = set()

        for stype in types_to_process:
            _extract_statement_data(stmt_map[stype], stype, use_prefix, all_data, all_periods)

        sorted_periods = sorted(all_periods, reverse=True)

        return FinancialsResult(
            ticker=ticker,
            statement_type=statement_type,
            periods=sorted_periods,
            data=all_data,
            summary=_build_financials_summary(ticker, statement_type, all_data),
        )
    except Exception as e:
        raise ValueError(f"Failed to get financials for {ticker}: {e!s}") from e
