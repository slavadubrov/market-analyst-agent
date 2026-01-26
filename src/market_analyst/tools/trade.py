"""Trade tool for executing stock trades.

This tool demonstrates the Guardian pattern - all trade requests
are validated against policies before execution.
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from market_analyst.schemas import TradeAction, TradeRequest


class TradeInput(BaseModel):
    """Input schema for the trade tool.

    Pydantic validation acts as a first-line "firewall" ensuring
    the agent provides well-formed requests.
    """

    action: str = Field(description="Trade action: 'buy', 'sell', 'delete_portfolio', or 'delete_logs'")
    ticker: str = Field(description="Stock ticker symbol (e.g., 'NVDA', 'AAPL')")
    amount_usd: float = Field(ge=0, description="Trade amount in USD")
    reason: str = Field(description="Reasoning for the trade decision")


@tool(args_schema=TradeInput)
def execute_trade(action: str, ticker: str, amount_usd: float, reason: str) -> str:
    """Execute a stock trade (buy/sell) or portfolio action.

    IMPORTANT: All trades are validated by the Guardian before execution.
    High-value trades (>$10,000) require human approval.
    Destructive actions (delete_*) are blocked automatically.

    Args:
        action: Trade action type
        ticker: Stock ticker symbol
        amount_usd: Amount in USD
        reason: Reasoning for this action

    Returns:
        Trade execution result or error message
    """
    # This tool creates a TradeRequest that gets routed through the Guardian
    # The actual execution happens only after policy checks pass

    try:
        trade_action = TradeAction(action)
    except ValueError:
        return f"Error: Invalid action '{action}'. Valid actions: buy, sell"

    # Create the trade request (will be processed by Guardian)
    trade_request = TradeRequest(
        action=trade_action,
        ticker=ticker.upper(),
        amount_usd=amount_usd,
        reason=reason,
    )

    # Return a marker that triggers Guardian processing
    # The actual execution is handled by the graph flow
    return f"TRADE_REQUEST:{trade_request.model_dump_json()}"


def parse_trade_request(tool_output: str) -> TradeRequest | None:
    """Parse a trade request from tool output.

    Returns None if the output is not a trade request marker.
    """
    if not tool_output.startswith("TRADE_REQUEST:"):
        return None

    import json

    json_str = tool_output[len("TRADE_REQUEST:") :]
    data = json.loads(json_str)
    return TradeRequest(**data)
