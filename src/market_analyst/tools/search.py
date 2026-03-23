"""Web search tools using Tavily API.

Tavily is optimized for AI agents, providing clean, relevant search results
without the noise of traditional search engines. Tools follow ACI principles:
structured outputs, concise feedback, and retry-based error handling.
"""

import os
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator
from tavily import TavilyClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Retry decorator for transient API failures
# ---------------------------------------------------------------------------

_tavily_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    reraise=True,
)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class CompetitorQuery(BaseModel):
    """Validated input for competitor search."""

    ticker: str = Field(description="Stock ticker symbol to find competitors for")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        v = v.upper().strip()
        if not v.isalpha() or len(v) > 5:
            raise ValueError(f"Invalid ticker format: {v}")
        return v


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------


class NewsItem(BaseModel):
    """Individual news result with extracted key points.

    Each news item is pre-processed for agent consumption.
    The key_points field saves the agent from spending inference tokens
    parsing article bodies.
    """

    headline: str
    source: str
    date: str
    relevance_score: float
    key_points: list[str]


class NewsSearchResult(BaseModel):
    """Result from news search with structured items."""

    query: str
    results: list[NewsItem]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_key_points(content: str) -> list[str]:
    """Extract key points by splitting content into sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", content.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 20][:3]


def _extract_source(url: str) -> str:
    """Extract domain name from URL as source."""
    try:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc
        # Remove www. prefix
        return domain.removeprefix("www.")
    except Exception:
        return "unknown"


def _search_news_impl(
    query: str,
    max_results: int = 5,
) -> NewsSearchResult:
    """Implementation of news search (without tool decorator)."""
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable not set")

        client = TavilyClient(api_key=api_key)

        @_tavily_retry
        def _do_search():
            return client.search(
                query=query,
                search_depth="advanced",
                max_results=min(max_results, 10),
                include_answer=True,
            )

        response = _do_search()

        results = []
        for item in response.get("results", []):
            content = item.get("content", "")[:500]
            results.append(
                NewsItem(
                    headline=item.get("title", ""),
                    source=_extract_source(item.get("url", "")),
                    date=item.get("published_date", ""),
                    relevance_score=item.get("score", 0),
                    key_points=_extract_key_points(content),
                )
            )

        summary = response.get("answer", "No summary available.")

        return NewsSearchResult(
            query=query,
            results=results,
            summary=summary,
        )

    except ImportError as e:
        raise ValueError("tavily-python not installed. Run: pip install tavily-python") from e
    except Exception as e:
        raise ValueError(f"Search failed: {e!s}") from e


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_news(
    query: str,
    max_results: int = 5,
) -> NewsSearchResult:
    """Search for recent news and articles about a stock or market topic.

    Use this tool when the user asks about recent events, earnings,
    announcements, or market-moving news for a specific ticker. Returns
    up to 10 articles sorted by relevance with extracted key points.
    For broad market news (not ticker-specific), use a general query.

    Args:
        query: Search query (e.g., "NVIDIA earnings report 2024",
               "semiconductor industry outlook").
        max_results: Maximum number of results to return (1-10).

    Returns:
        NewsSearchResult with structured NewsItem objects and a summary.

    Example:
        >>> search_news("NVDA stock analysis")
        NewsSearchResult(query="NVDA stock analysis", results=[...], summary="...")
    """
    return _search_news_impl(query=query, max_results=max_results)


@tool(args_schema=CompetitorQuery)
def search_competitors(
    ticker: str,
    max_results: int = 3,
) -> NewsSearchResult:
    """Search for competitor analysis and relative performance comparison.

    Finds news and analysis comparing the given company to its competitors,
    including relative metrics like market share, growth rates, and valuation
    multiples. Use this after get_stock_snapshot to understand competitive
    positioning within the sector.

    Args:
        ticker: Stock ticker symbol to find competitors for.
        max_results: Maximum number of results per competitor.

    Returns:
        Competitor comparison news and analysis with relative metrics.
    """
    query = f"{ticker} competitors comparison stock analysis market share"
    return _search_news_impl(query=query, max_results=max_results)
