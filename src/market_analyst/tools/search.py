"""Web search tools using Tavily API.

Tavily is optimized for AI agents, providing clean, relevant search results
without the noise of traditional search engines.
"""

import os

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class NewsSearchResult(BaseModel):
    """Result from news search."""

    query: str
    results: list[dict]
    summary: str


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

        # Search with Tavily
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=min(max_results, 10),
            include_answer=True,
        )

        # Extract clean results
        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", "")[
                        :500
                    ],  # Truncate for token efficiency
                    "score": item.get("score", 0),
                }
            )

        # Use Tavily's AI-generated answer as summary
        summary = response.get("answer", "No summary available.")

        return NewsSearchResult(
            query=query,
            results=results,
            summary=summary,
        )

    except ImportError:
        raise ValueError("tavily-python not installed. Run: pip install tavily-python")
    except Exception as e:
        raise ValueError(f"Search failed: {str(e)}")


@tool
def search_news(
    query: str,
    max_results: int = 5,
) -> NewsSearchResult:
    """Search for recent news and articles about a topic.

    Use this tool to find recent news, announcements, and analysis
    about companies, market trends, or economic events.

    Args:
        query: Search query (e.g., "NVIDIA earnings report 2024",
               "semiconductor industry outlook").
        max_results: Maximum number of results to return (1-10).

    Returns:
        Search results with titles, snippets, and a summary.

    Example:
        >>> search_news("NVDA stock analysis")
        NewsSearchResult(query="NVDA stock analysis", results=[...], summary="...")
    """
    return _search_news_impl(query=query, max_results=max_results)


@tool
def search_competitors(
    ticker: str,
    max_results: int = 3,
) -> NewsSearchResult:
    """Search for competitor analysis and comparison.

    Finds news and analysis comparing the given company to its competitors.

    Args:
        ticker: Stock ticker symbol to find competitors for.
        max_results: Maximum number of results per competitor.

    Returns:
        Competitor comparison news and analysis.
    """
    # Build a competitor-focused search query
    query = f"{ticker} competitors comparison stock analysis"

    # Call the underlying search function directly (not the tool wrapper)
    return _search_news_impl(query=query, max_results=max_results)
