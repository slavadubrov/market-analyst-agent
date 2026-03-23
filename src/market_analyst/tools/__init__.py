"""Tools module for Market Analyst Agent.

Demonstrates five tool modalities following ACI design principles:

1. JSON Tool Calling (baseline):
   - get_stock_snapshot, get_price_history, get_financials, search_news, search_competitors

2. Skills (SKILL.md — expertise, not execution):
   - use_skill: Activates domain playbooks (earnings analysis, sector comparison)

3. CLI-as-Tool (Unix interface, near-zero schema overhead):
   - cli_list_reports, cli_show_report: Agent calls its own CLI with --json output

4. Code Execution / PTC (agent writes and runs Python):
   - execute_python_analysis: Financial calculations, ratio analysis, portfolio math
"""

from market_analyst.tools.cli_tools import cli_list_reports, cli_show_report
from market_analyst.tools.code_exec import execute_python_analysis
from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.skills import use_skill
from market_analyst.tools.stock import (
    get_financials,
    get_price_history,
    get_stock_snapshot,
)
from market_analyst.tools.trade import execute_trade, parse_trade_request

__all__ = [
    # JSON Tool Calling (modality 1)
    "get_stock_snapshot",
    "get_price_history",
    "get_financials",
    "search_news",
    "search_competitors",
    "execute_trade",
    "parse_trade_request",
    # Skills (modality 2)
    "use_skill",
    # CLI-as-Tool (modality 3)
    "cli_list_reports",
    "cli_show_report",
    # Code Execution (modality 4)
    "execute_python_analysis",
]
