"""Trade executor node - executes approved trades.

This node runs after Guardian approval (either auto-approved
or human-approved via the compliance officer flow).

Idempotency: the trade is a side-effecting tool call, so per the article's
"Idempotency is not optional" rule we look the call up by
``(thread_id, trade_call_id)`` in the idempotency store before executing.
A retry from a crashed worker hands back the original execution_id instead
of double-charging.
"""

from langchain_core.runnables import RunnableConfig

from market_analyst.runtime import get_idempotency_store
from market_analyst.schemas import AgentState, GuardianDecision


def _trade_call_id(state: AgentState) -> str:
    """Derive a stable key for the trade so the idempotency store can dedupe.

    Includes the action, ticker, and amount so a planner that produces two
    different trades on the same thread still gets separate keys, but a worker
    that crashes between "execute" and "checkpoint" hits the same key on
    resume and replays the prior result.
    """
    trade = state.pending_trade
    assert trade is not None  # caller checked
    return f"trade:{trade.action.value}:{trade.ticker}:{int(trade.amount_usd)}"


def trade_executor_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Execute an approved trade.

    This node only runs after:
    1. Guardian auto-approved (safe path), OR
    2. Human approved via compliance officer

    The trade is simulated since this is an educational demo.

    Args:
        state: Current agent state with approved trade
        config: LangGraph runtime config; used to pull ``thread_id`` for the
            idempotency key.

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

    # Idempotency check: short-circuit if this trade has already executed for
    # this thread (e.g. after a crash-and-resume in the middle of a super-step).
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if thread_id:
        store = get_idempotency_store()
        seen, prior = store.reserve_or_replay(thread_id, _trade_call_id(state))
        if seen and prior:
            print(f"\n♻️  Trade Executor: replaying prior execution for {trade.ticker}")
            print(f"   Execution ID: {prior.get('execution_id', 'unknown')}")
            return {
                "trade_executed": True,
                "pending_trade": None,
                "guardian_result": None,
            }

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

    # Persist the result against the idempotency key so a future retry replays.
    if thread_id:
        get_idempotency_store().store(
            thread_id,
            _trade_call_id(state),
            {"executed": True, "execution_id": execution_id},
        )

    return {
        "trade_executed": True,
        "pending_trade": None,  # Clear the pending trade
        "guardian_result": None,  # Clear guardian result
    }
