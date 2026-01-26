"""Trade workflow with Guardian policy validation.

This module defines a separate graph for trade execution that
demonstrates the Guardian pattern:

    Trade Request → Guardian → [Approve/Escalate/Reject] → Execute
                                    ↓
                              Human Review (if escalated)
"""

from typing import Literal

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from market_analyst.nodes.guardian import guardian_node
from market_analyst.nodes.trade_executor import trade_executor_node
from market_analyst.schemas import (
    AgentState,
    GuardianDecision,
    TradeAction,
    TradeRequest,
)


def route_after_guardian(state: AgentState) -> Literal["execute", "escalate", "end"]:
    """Route based on Guardian's decision.

    - APPROVE: Proceed directly to execution
    - ESCALATE: Pause for human review (interrupt)
    - REJECT: End with error
    """
    if state.guardian_result is None:
        return "end"

    decision = state.guardian_result.decision

    if decision == GuardianDecision.APPROVE:
        return "execute"
    elif decision == GuardianDecision.ESCALATE:
        return "escalate"
    else:  # REJECT
        return "end"


def compliance_officer_node(state: AgentState) -> dict:
    """Compliance Officer node - handles escalated trades.

    This node runs when the Guardian escalates a trade for review.
    The graph will interrupt before this node, allowing human input.
    When resumed with approval, it marks the trade as approved.
    """
    # This node is called after human approval (via update_state)
    # The trade_approved flag should already be set by the human

    if state.trade_approved:
        print("\n👔 Compliance Officer: Trade approved by human reviewer")
        return {}
    else:
        print("\n👔 Compliance Officer: Awaiting human decision...")
        return {}


def create_trade_graph(checkpointer: PostgresSaver | None = None) -> StateGraph:
    """Create the trade execution graph with Guardian.

    Graph Structure:
    ```
    START → guardian ─┬─→ compliance_officer (HITL) → execute → END
                      ├─→ execute ──────────────────────────────→ END
                      └─→ END (rejected)
    ```
    """
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("guardian", guardian_node)
    builder.add_node("compliance_officer", compliance_officer_node)
    builder.add_node("execute", trade_executor_node)

    # Define edges
    builder.add_edge(START, "guardian")

    builder.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {
            "execute": "execute",
            "escalate": "compliance_officer",
            "end": END,
        },
    )

    # After compliance officer approval, execute
    builder.add_edge("compliance_officer", "execute")
    builder.add_edge("execute", END)

    # Compile with interrupt before compliance_officer for HITL
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["compliance_officer"],
    )


def run_trade(
    action: str,
    ticker: str,
    amount_usd: float,
    reason: str,
    checkpointer: PostgresSaver | None = None,
    thread_id: str | None = None,
) -> dict:
    """Execute a trade through the Guardian workflow.

    Args:
        action: Trade action (buy, sell)
        ticker: Stock ticker symbol
        amount_usd: Trade amount in USD
        reason: Reasoning for the trade
        checkpointer: Optional PostgresSaver for persistence
        thread_id: Optional thread ID for resuming

    Returns:
        Result dict with trade status and thread_id
    """
    import uuid

    # Validate action
    try:
        trade_action = TradeAction(action)
    except ValueError:
        return {"error": f"Invalid action: {action}", "executed": False}

    # Create trade request
    trade_request = TradeRequest(
        action=trade_action,
        ticker=ticker.upper(),
        amount_usd=amount_usd,
        reason=reason,
    )

    # Create initial state
    initial_state = AgentState(
        pending_trade=trade_request,
    )

    # Create graph
    graph = create_trade_graph(checkpointer=checkpointer)

    # Configure thread
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    result = graph.invoke(initial_state, config)

    # Check if we hit the HITL interrupt (only works with checkpointer)
    requires_approval = False
    if checkpointer:
        state = graph.get_state(config)
        next_nodes = state.next if state else []
        requires_approval = "compliance_officer" in next_nodes

    return {
        "thread_id": thread_id,
        "state": result,
        "executed": result.get("trade_executed", False),
        "requires_approval": requires_approval,
        "guardian_result": result.get("guardian_result"),
    }


def approve_trade(
    thread_id: str,
    checkpointer: PostgresSaver,
    approve: bool = True,
    modified_amount: float | None = None,
) -> dict:
    """Approve or reject a pending trade.

    Args:
        thread_id: Thread ID of the pending trade
        checkpointer: PostgresSaver with the state
        approve: Whether to approve (True) or reject (False)
        modified_amount: Optional modified trade amount

    Returns:
        Result dict with execution status
    """
    graph = create_trade_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state
    current_state = graph.get_state(config)

    if not current_state or not current_state.values:
        raise ValueError(f"No state found for thread {thread_id}")

    if not approve:
        # Reject the trade
        print("\n❌ Trade rejected by human reviewer")
        graph.update_state(
            config, {"trade_approved": False, "error": "Rejected by reviewer"}
        )
        return {"thread_id": thread_id, "executed": False, "rejected": True}

    # Apply modifications if any
    update_values = {"trade_approved": True}

    if modified_amount is not None:
        pending = current_state.values.get("pending_trade")
        if pending:
            pending.amount_usd = modified_amount
            update_values["pending_trade"] = pending
            print(f"\n📝 Trade amount modified to: ${modified_amount:,.2f}")

    # Update state and resume
    graph.update_state(config, update_values)

    # Resume execution
    result = graph.invoke(None, config)

    return {
        "thread_id": thread_id,
        "state": result,
        "executed": result.get("trade_executed", False),
    }
