"""Trade executor node - executes approved trades.

This node runs after Guardian approval (either auto-approved
or human-approved via the compliance officer flow).
"""

from market_analyst.schemas import AgentState, GuardianDecision


def trade_executor_node(state: AgentState) -> dict:
    """Execute an approved trade.

    This node only runs after:
    1. Guardian auto-approved (safe path), OR
    2. Human approved via compliance officer

    The trade is simulated since this is an educational demo.

    Args:
        state: Current agent state with approved trade

    Returns:
        Updated state with execution result
    """
    trade = state.pending_trade
    guardian = state.guardian_result

    if trade is None:
        print("  ⚠️  Trade Executor: No trade to execute")
        return {"error": "No pending trade"}

    if not state.trade_approved:
        # Check if this was rejected by guardian
        if guardian and guardian.decision == GuardianDecision.REJECT:
            print(f"\n  ❌ Trade blocked by Guardian: {guardian.reason}")
            return {
                "trade_executed": False,
                "error": f"Trade rejected: {guardian.reason}",
            }
        print("  ⚠️  Trade Executor: Trade not approved")
        return {"error": "Trade not approved"}

    # Simulate trade execution
    print("\n💰 Executing Trade:")
    print(f"   Action: {trade.action.value.upper()}")
    print(f"   Ticker: {trade.ticker}")
    print(f"   Amount: ${trade.amount_usd:,.2f}")

    # In a real system, this would call a brokerage API
    # For demo purposes, we just simulate success
    execution_id = f"TXN-{trade.ticker}-{int(trade.amount_usd)}"

    print("\n   ✅ Trade executed successfully!")
    print(f"   Execution ID: {execution_id}")

    return {
        "trade_executed": True,
        "pending_trade": None,  # Clear the pending trade
        "guardian_result": None,  # Clear guardian result
    }
