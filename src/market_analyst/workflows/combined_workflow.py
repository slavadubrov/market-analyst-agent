"""Combined workflow: Analysis → Guardian → Trade execution.

This module chains the analysis and trade workflows into a single demo
that demonstrates the full architecture:

    START → Router → [Deep/Flash Analysis] → Reporter
                                              ↓
                               (HITL: Approve Report)
                                              ↓
                                 Create Trade Request
                                              ↓
                                    Guardian (Policy)
                                  /       |        \
                             REJECT   ESCALATE   APPROVE
                                ↓         ↓          ↓
                               END     (HITL)     Execute
                                          ↓          ↓
                                      Execute       END
                                          ↓
                                         END

This keeps existing separate workflows intact while providing
an integrated demo that matches the architecture diagram.
"""

import uuid
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from market_analyst.llm import ModelSettings
from market_analyst.memory import load_user_profile
from market_analyst.nodes.guardian import guardian_node
from market_analyst.nodes.trade_executor import trade_executor_node
from market_analyst.runtime.ownership import serialized_run
from market_analyst.schemas import (
    AgentState,
    ExecutionMode,
    GuardianDecision,
    TradeAction,
    TradeRequest,
)
from market_analyst.workflows.analysis_workflow import publish_node
from market_analyst.workflows.research import add_research_nodes

# Default trade amount for the combined demo
DEFAULT_TRADE_AMOUNT = 1000.0


def create_trade_from_report_node(state: AgentState) -> dict:
    """Create a trade request based on the analysis report.

    This node bridges the analysis and trade workflows by:
    1. Extracting the recommendation from the draft report
    2. Creating a TradeRequest with the configured amount
    3. Setting up state for the Guardian to evaluate
    """
    report = state.draft_report
    trade_amount = state.trade_amount if state.trade_amount is not None else DEFAULT_TRADE_AMOUNT

    if not report:
        print("  ⚠️  No report available to create trade from")
        return {"error": "No report to trade on"}

    # Map recommendation to trade action
    recommendation = report.recommendation
    if recommendation in ("strong_buy", "buy"):
        action = TradeAction.BUY
    elif recommendation in ("strong_sell", "sell"):
        action = TradeAction.SELL
    else:
        # Hold recommendation - no trade
        print(f"\n📊 Report recommendation is '{recommendation}' - no trade action")
        return {"pending_trade": None}

    # Create trade request
    trade_request = TradeRequest(
        action=action,
        ticker=report.ticker,
        amount_usd=trade_amount,
        reason=f"Based on analysis: {report.summary[:200]}...",
    )

    print("\n📈 Creating trade from report:")
    print(f"   Recommendation: {recommendation.upper()}")
    print(f"   Action: {action.value.upper()} {report.ticker}")
    print(f"   Amount: ${trade_amount:,.2f}")

    return {
        "pending_trade": trade_request,
        "trade_approved": False,
    }


def compliance_officer_node(state: AgentState) -> dict:
    """Compliance Officer node - handles escalated trades.

    This node runs when the Guardian escalates a trade for review.
    The graph will interrupt before this node, allowing human input.
    """
    if state.trade_approved:
        print("\n👔 Compliance Officer: Trade approved by human reviewer")
        return {}
    else:
        print("\n👔 Compliance Officer: Awaiting human decision...")
        return {}


def route_after_guardian(state: AgentState) -> Literal["execute", "escalate", "end"]:
    """Route based on Guardian's decision."""
    if state.guardian_result is None:
        # No trade to evaluate (hold recommendation)
        return "end"

    decision = state.guardian_result.decision

    if decision == GuardianDecision.APPROVE:
        return "execute"
    elif decision == GuardianDecision.ESCALATE:
        return "escalate"
    else:  # REJECT
        return "end"


def skip_trade_check(state: AgentState) -> Literal["guardian", "end"]:
    """Check if we should skip trading (hold recommendation)."""
    if state.pending_trade is None:
        return "end"
    return "guardian"


def create_combined_graph(
    checkpointer: BaseCheckpointSaver | None = None,
    force_mode: ExecutionMode | None = None,
) -> CompiledStateGraph:
    """Create the combined Analysis → Guardian → Trade graph.

    Graph Structure:
    ```
    START → router ─┬─→ planner → executor ─┬─→ executor (loop)
                    │                       └─→ reporter ───────────┐
                    │                                               │
                    └─→ rewoo_planner → rewoo_worker → rewoo_solver─┤
                                                                    │
                                              [HITL: Approve Report]│
                                                                    ↓
                                                   publish → create_trade
                                                                    │
                                                    ┌───────────────┘
                                                    ↓
                                                guardian ─┬─→ execute → END
                                                          ├─→ compliance_officer → execute → END
                                                          └─→ END (rejected/hold)
    ```
    """
    builder = StateGraph(AgentState)

    add_research_nodes(builder)
    builder.add_node("publish", publish_node)

    # === Trade nodes ===
    builder.add_node("create_trade", create_trade_from_report_node)
    builder.add_node("guardian", guardian_node)
    builder.add_node("compliance_officer", compliance_officer_node)
    builder.add_node("execute", trade_executor_node)

    # === Bridge: Analysis → Trade ===
    builder.add_edge("publish", "create_trade")

    # Check if we should trade or skip (hold recommendation)
    builder.add_conditional_edges(
        "create_trade",
        skip_trade_check,
        {
            "guardian": "guardian",
            "end": END,
        },
    )

    # === Trade edges ===
    builder.add_conditional_edges(
        "guardian",
        route_after_guardian,
        {
            "execute": "execute",
            "escalate": "compliance_officer",
            "end": END,
        },
    )

    builder.add_edge("compliance_officer", "execute")
    builder.add_edge("execute", END)

    # Compile with interrupt points:
    # 1. Before publish - for report approval (HITL)
    # 2. Before compliance_officer - for trade approval (HITL)
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["publish", "compliance_officer"],
    )


@serialized_run
def run_combined_analysis(
    query: str,
    user_id: str = "default",
    thread_id: str | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    force_mode: ExecutionMode | None = None,
    trade_amount: float = DEFAULT_TRADE_AMOUNT,
    *,
    model_settings: ModelSettings | None = None,
) -> dict:
    """Run the combined analysis-to-trade workflow.

    This is the main entry point for the combined demo.

    Args:
        query: User's analysis request (e.g., "Analyze NVDA stock")
        user_id: User identifier for profile lookup
        thread_id: Optional thread ID for resuming
        checkpointer: Optional checkpointer for persistence
        force_mode: Optional execution mode override
        trade_amount: Amount in USD for the trade (default: $1000)

    Returns:
        Result dict with state, report, and trade info
    """
    if checkpointer is None:
        raise ValueError("Combined workflow requires persistence for its approval steps")

    # Load user profile
    user_profile = load_user_profile(user_id)

    # Create initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        user_profile=user_profile,
        user_id=user_id,
        execution_mode=force_mode,
        trade_amount=trade_amount,
        model_settings=(model_settings or ModelSettings.from_env()).model_dump(),
    )

    # Create graph
    graph = create_combined_graph(checkpointer=checkpointer)

    # Configure thread
    thread_id = thread_id or str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    result = graph.invoke(initial_state, config, durability="sync" if checkpointer else None)
    if error := result.get("error"):
        raise RuntimeError(error)

    # Check what state we're in
    requires_report_approval = False
    requires_trade_approval = False

    if checkpointer:
        state = graph.get_state(config)
        next_nodes = state.next if state else []
        requires_report_approval = "publish" in next_nodes
        requires_trade_approval = "compliance_officer" in next_nodes

    return {
        "thread_id": thread_id,
        "state": result,
        "draft_report": result.get("draft_report"),
        "execution_mode": result.get("execution_mode"),
        "requires_report_approval": requires_report_approval,
        "requires_trade_approval": requires_trade_approval,
        "trade_executed": result.get("trade_executed", False),
        "guardian_result": result.get("guardian_result"),
    }


@serialized_run
def approve_combined_report(
    thread_id: str,
    checkpointer: BaseCheckpointSaver,
) -> dict:
    """Approve the report in the combined workflow and continue.

    After approval, the workflow continues to create_trade → guardian → execute.

    Returns:
        Result dict - may require trade approval if Guardian escalates
    """
    graph = create_combined_graph(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Get current state
    current_state = graph.get_state(config)

    if not current_state or not current_state.values:
        raise ValueError(f"No state found for thread {thread_id}")
    if "publish" not in current_state.next:
        raise ValueError("Workflow is not awaiting report approval")

    if current_state.values.get("evaluator_verdict") not in {"pass", "needs_human"} or current_state.values.get("error"):
        raise ValueError("Draft has not passed evaluation; regenerate with sufficient evidence")
    # Approve the report
    graph.update_state(config, {"report_approved": True})

    # Resume execution
    result = graph.invoke(None, config, durability="sync" if checkpointer else None)

    # Check if trade needs approval
    requires_trade_approval = False
    state = graph.get_state(config)
    if state and state.next:
        requires_trade_approval = "compliance_officer" in state.next

    return {
        "thread_id": thread_id,
        "state": result,
        "requires_trade_approval": requires_trade_approval,
        "trade_executed": result.get("trade_executed", False),
        "guardian_result": result.get("guardian_result"),
    }


@serialized_run
def approve_combined_trade(
    thread_id: str,
    checkpointer: BaseCheckpointSaver,
    approve: bool = True,
    modified_amount: float | None = None,
) -> dict:
    """Approve or reject the trade in the combined workflow.

    Args:
        thread_id: Thread ID of the pending trade
        checkpointer: PostgresSaver with the state
        approve: Whether to approve (True) or reject (False)
        modified_amount: Optional modified trade amount

    Returns:
        Result dict with execution status
    """
    graph = create_combined_graph(checkpointer=checkpointer)
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    current_state = graph.get_state(config)

    if not current_state or not current_state.values:
        raise ValueError(f"No state found for thread {thread_id}")
    if "compliance_officer" not in current_state.next:
        raise ValueError("Workflow is not awaiting trade approval")

    if not approve:
        print("\n❌ Trade rejected by human reviewer")
        graph.update_state(config, {"trade_approved": False, "error": "Rejected by reviewer"})
        return {"thread_id": thread_id, "executed": False, "rejected": True}

    # Apply modifications if any
    update_values: dict[str, object] = {"trade_approved": True}

    if modified_amount is not None:
        pending = current_state.values.get("pending_trade")
        if pending:
            pending = TradeRequest.model_validate(pending)
            pending = TradeRequest.model_validate({**pending.model_dump(), "amount_usd": modified_amount})
            update_values["pending_trade"] = pending
            print(f"\n📝 Trade amount modified to: ${modified_amount:,.2f}")

    # Update state and resume
    graph.update_state(config, update_values)

    # Resume execution
    result = graph.invoke(None, config, durability="sync" if checkpointer else None)

    return {
        "thread_id": thread_id,
        "state": result,
        "executed": result.get("trade_executed", False),
    }
