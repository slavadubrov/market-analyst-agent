"""The research tool surface shared by both loops.

Only explicitly supplied tools are callable. Add a validated LangChain tool here
or pass a replacement list to the workflow; trade effects have a separate gate.
"""

from langchain_core.tools import BaseTool

from market_analyst.tools.cli_tools import cli_list_reports, cli_show_report
from market_analyst.tools.code_exec import execute_python_analysis
from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.skills import use_skill
from market_analyst.tools.stock import get_financials, get_price_history, get_stock_snapshot

RESEARCH_TOOLS: tuple[BaseTool, ...] = (
    get_stock_snapshot,
    get_price_history,
    get_financials,
    search_news,
    search_competitors,
    use_skill,
    cli_list_reports,
    cli_show_report,
    execute_python_analysis,
)


def tool_registry(tools=None) -> dict[str, BaseTool]:
    selected = RESEARCH_TOOLS if tools is None else tools
    registry = {tool.name: tool for tool in selected}
    if len(registry) != len(selected):
        raise ValueError("Tool names must be unique")
    return registry
