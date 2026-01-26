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
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from market_analyst.memory import load_user_profile
from market_analyst.nodes.executor import executor_node
from market_analyst.nodes.guardian import guardian_node
from market_analyst.nodes.planner import planner_node
from market_analyst.nodes.reporter import reporter_node
from market_analyst.nodes.rewoo_planner import rewoo_planner_node
from market_analyst.nodes.rewoo_solver import rewoo_solver_node
from market_analyst.nodes.rewoo_worker import rewoo_worker_node
from market_analyst.nodes.router import router_node
from market_analyst.nodes.trade_executor import trade_executor_node
from market_analyst.schemas import (
    AgentState,
    DraftReport,
    ExecutionMode,
    GuardianDecision,
    TradeAction,
    TradeRequest,
)

# Default trade amount for the combined demo
DEFAULT_TRADE_AMOUNT = 1000.0


def route_after_router(state: AgentState) -> Literal["planner", "rewoo_planner"]:
    """Route based on execution mode set by router."""
    if state.execution_mode == ExecutionMode.FLASH_BRIEFING:
        return "rewoo_planner"
    return "planner"


def route_after_executor(state: AgentState) -> Literal["executor", "reporter"]:
    """Route based on plan completion status."""
    if state.current_step_index >= len(state.plan):
        return "reporter"
    return "executor"


def create_trade_from_report_node(state: AgentState) -> dict:
    """Create a trade request based on the analysis report.

    This node bridges the analysis and trade workflows by:
    1. Extracting the recommendation from the draft report
    2. Creating a TradeRequest with the configured amount
    3. Setting up state for the Guardian to evaluate
    """
    report = state.draft_report
    trade_amount = getattr(state, "_trade_amount", DEFAULT_TRADE_AMOUNT)

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

    print(f"\n📈 Creating trade from report:")
    print(f"   Recommendation: {recommendation.upper()}")
    print(f"   Action: {action.value.upper()} {report.ticker}")
    print(f"   Amount: ${trade_amount:,.2f}")

    return {
        "pending_trade": trade_request,
        "trade_approved": False,
    }


def publish_node(state: AgentState) -> dict:
    """Publish the approved report.

    This is a simplified version for the combined workflow.
    The report is marked as approved and we proceed to trade creation.
    """
    from datetime import datetime
    from pathlib import Path

    if not state.report_approved:
        return {"error": "Report was not approved"}

    if not state.draft_report:
        return {"error": "No report to publish"}

    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Generate filename
    report = state.draft_report
    mode_suffix = (
        "flash" if state.execution_mode == ExecutionMode.FLASH_BRIEFING else "deep"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{report.ticker}_{mode_suffix}_{timestamp}.md"
    filepath = reports_dir / filename

    # Format report as markdown
    risk_factors = "\n".join(f"- {r}" for r in report.risk_factors)

    content = f"""# {report.title}

**Ticker:** {report.ticker}  
**Mode:** {mode_suffix.capitalize()} {"(ReWOO)" if mode_suffix == "flash" else "(ReAct)"}  
**Recommendation:** {report.recommendation.upper().replace("_", " ")}  
**Confidence:** {report.confidence:.0%}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Summary

{report.summary}

## Analysis

{report.analysis}

## Risk Factors

{risk_factors}

---

*This report was generated by Market Analyst Agent and approved for publication.*
"""

    filepath.write_text(content)
    print(f"\n📄 Report saved to: {filepath}")

    return {"report_approved": True}


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
    checkpointer: PostgresSaver | None = None,
    force_mode: ExecutionMode | None = None,
) -> StateGraph:
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

    # === Analysis nodes (reused from agent.py) ===
    builder.add_node("router", router_node)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reporter", reporter_node)
    builder.add_node("rewoo_planner", rewoo_planner_node)
    builder.add_node("rewoo_worker", rewoo_worker_node)
    builder.add_node("rewoo_solver", rewoo_solver_node)
    builder.add_node("publish", publish_node)

    # === Trade nodes ===
    builder.add_node("create_trade", create_trade_from_report_node)
    builder.add_node("guardian", guardian_node)
    builder.add_node("compliance_officer", compliance_officer_node)
    builder.add_node("execute", trade_executor_node)

    # === Analysis edges ===
    builder.add_edge(START, "router")

    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "planner": "planner",
            "rewoo_planner": "rewoo_planner",
        },
    )

    # Deep Research path
    builder.add_edge("planner", "executor")
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {
            "executor": "executor",
            "reporter": "reporter",
        },
    )
    builder.add_edge("reporter", "publish")

    # Flash Briefing path
    builder.add_edge("rewoo_planner", "rewoo_worker")
    builder.add_edge("rewoo_worker", "rewoo_solver")
    builder.add_edge("rewoo_solver", "publish")

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


def run_combined_analysis(
    query: str,
    user_id: str = "default",
    thread_id: str | None = None,
    checkpointer: PostgresSaver | None = None,
    force_mode: ExecutionMode | None = None,
    trade_amount: float = DEFAULT_TRADE_AMOUNT,
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
    # Load user profile
    user_profile = load_user_profile(user_id)

    # Create initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        user_profile=user_profile,
        user_id=user_id,
        execution_mode=force_mode,
    )

    # Store trade amount in state (accessed by create_trade_from_report_node)
    initial_state._trade_amount = trade_amount

    # Create graph
    graph = create_combined_graph(checkpointer=checkpointer, force_mode=force_mode)

    # Configure thread
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph
    result = graph.invoke(initial_state, config)

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


def approve_combined_report(
    thread_id: str,
    checkpointer: PostgresSaver,
) -> dict:
    """Approve the report in the combined workflow and continue.

    After approval, the workflow continues to create_trade → guardian → execute.

    Returns:
        Result dict - may require trade approval if Guardian escalates
    """
    graph = create_combined_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state
    current_state = graph.get_state(config)

    if not current_state or not current_state.values:
        raise ValueError(f"No state found for thread {thread_id}")

    # Approve the report
    graph.update_state(config, {"report_approved": True})

    # Resume execution
    result = graph.invoke(None, config)

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


def approve_combined_trade(
    thread_id: str,
    checkpointer: PostgresSaver,
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
    config = {"configurable": {"thread_id": thread_id}}

    current_state = graph.get_state(config)

    if not current_state or not current_state.values:
        raise ValueError(f"No state found for thread {thread_id}")

    if not approve:
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
