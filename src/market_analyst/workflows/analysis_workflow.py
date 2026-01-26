"""Analysis workflow: Deep Research and Flash Briefing.

This module defines the complete Market Analyst Agent graph with:
- Router for intent classification (deep research vs flash briefing)
- Plan-and-Execute + ReAct path (thorough analysis)
- ReWOO path (fast, token-efficient snapshots)
- PostgreSQL checkpointing for state persistence
- Human-in-the-loop interrupts for report approval
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from market_analyst.memory import load_user_profile
from market_analyst.nodes.executor import executor_node
from market_analyst.nodes.planner import planner_node
from market_analyst.nodes.reporter import reporter_node
from market_analyst.nodes.rewoo_planner import rewoo_planner_node
from market_analyst.nodes.rewoo_solver import rewoo_solver_node
from market_analyst.nodes.rewoo_worker import rewoo_worker_node
from market_analyst.nodes.router import router_node
from market_analyst.schemas import AgentState, ExecutionMode


def route_after_router(state: AgentState) -> Literal["planner", "rewoo_planner"]:
    """Route based on execution mode set by router.

    Routes to:
    - planner: For deep research (Plan-and-Execute + ReAct)
    - rewoo_planner: For flash briefing (ReWOO)
    """
    if state.execution_mode == ExecutionMode.FLASH_BRIEFING:
        return "rewoo_planner"
    return "planner"


def route_after_executor(state: AgentState) -> Literal["executor", "reporter"]:
    """Route based on plan completion status.

    If there are more steps to execute, continue to executor.
    If all steps complete, move to reporter.
    """
    if state.current_step_index >= len(state.plan):
        return "reporter"
    return "executor"


def create_graph(
    checkpointer: PostgresSaver | None = None,
    force_mode: ExecutionMode | None = None,
) -> StateGraph:
    """Create the Market Analyst Agent graph.

    Graph Structure:
    ```
    START → router ─┬─→ planner → executor ─┬─→ executor (loop)
                    │                       └─→ reporter ──────────┐
                    │                                              ├─→ publish → END
                    └─→ rewoo_planner → rewoo_worker → rewoo_solver─┘
    ```

    Args:
        checkpointer: Optional PostgresSaver for state persistence.
        force_mode: Optional mode override (bypasses router classification).

    Returns:
        Compiled StateGraph ready for invocation
    """
    # Build the graph
    builder = StateGraph(AgentState)

    # Add all nodes
    # Router (entry point)
    builder.add_node("router", router_node)

    # Deep Research path (Plan-and-Execute + ReAct)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("reporter", reporter_node)

    # Flash Briefing path (ReWOO)
    builder.add_node("rewoo_planner", rewoo_planner_node)
    builder.add_node("rewoo_worker", rewoo_worker_node)
    builder.add_node("rewoo_solver", rewoo_solver_node)

    # Publish (shared by both paths)
    builder.add_node("publish", publish_node)

    # Define edges
    # Entry: Router classifies intent
    builder.add_edge(START, "router")

    # Router branches to appropriate path
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "planner": "planner",
            "rewoo_planner": "rewoo_planner",
        },
    )

    # Deep Research path: planner → executor (loop) → reporter → publish
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

    # Flash Briefing path: rewoo_planner → rewoo_worker → rewoo_solver → publish
    builder.add_edge("rewoo_planner", "rewoo_worker")
    builder.add_edge("rewoo_worker", "rewoo_solver")
    builder.add_edge("rewoo_solver", "publish")

    # End
    builder.add_edge("publish", END)

    # Compile with checkpointer and interrupt
    # interrupt_before=["publish"] pauses for human approval
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["publish"],
    )


def publish_node(state: AgentState) -> dict:
    """Final node that publishes the approved report.

    This node runs after human approval (via interrupt_before).
    Saves the report to the reports/ directory as a markdown file.
    """
    if not state.report_approved:
        return {"error": "Report was not approved"}

    if not state.draft_report:
        return {"error": "No report to publish"}

    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Generate filename with ticker, mode, and timestamp
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

    # Save the file
    filepath.write_text(content)
    print(f"\n📄 Report saved to: {filepath}")

    return {
        "report_approved": True,
    }


def run_analysis(
    query: str,
    user_id: str = "default",
    thread_id: str | None = None,
    checkpointer: PostgresSaver | None = None,
    force_mode: ExecutionMode | None = None,
) -> dict:
    """Run a complete stock analysis.

    This is the main entry point for running the agent.

    Args:
        query: User's analysis request (e.g., "Analyze NVDA stock")
        user_id: User identifier for profile lookup
        thread_id: Optional thread ID for resuming a conversation
        checkpointer: Optional checkpointer for persistence
        force_mode: Optional execution mode override

    Returns:
        Final state with draft report
    """
    # Load user profile from Qdrant
    user_profile = load_user_profile(user_id)

    # Create initial state
    initial_state = AgentState(
        messages=[HumanMessage(content=query)],
        user_profile=user_profile,
        user_id=user_id,
        execution_mode=force_mode,  # Will be overwritten by router if None
    )

    # Create graph
    graph = create_graph(checkpointer=checkpointer, force_mode=force_mode)

    # Configure thread
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Run the graph (will pause at publish node for approval)
    result = graph.invoke(initial_state, config)

    return {
        "thread_id": thread_id,
        "state": result,
        "draft_report": result.get("draft_report"),
        "execution_mode": result.get("execution_mode"),
        "requires_approval": not result.get("report_approved", False),
    }


def approve_and_publish(
    thread_id: str,
    checkpointer: PostgresSaver,
    edits: dict | None = None,
) -> dict:
    """Approve the draft report and continue to publish.

    This function implements the HITL approval flow:
    1. Optionally edit the draft report
    2. Mark as approved
    3. Resume the graph to complete

    Args:
        thread_id: The thread ID of the paused analysis
        checkpointer: Checkpointer with the saved state
        edits: Optional dict of report field edits

    Returns:
        Final state after publishing
    """
    graph = create_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Get current state
    current_state = graph.get_state(config)

    if not current_state or not current_state.values:
        raise ValueError(
            f"No state found for thread {thread_id}. Make sure persistence is enabled when running the analysis."
        )

    # Check if we're at the right interrupt point
    next_nodes = current_state.next
    if not next_nodes:
        # Already completed or no pending nodes
        print("   ⚠️  Analysis already completed or not at interrupt point")
        return {
            "thread_id": thread_id,
            "state": current_state.values,
            "published": True,
        }

    print(f"   📍 Resuming from interrupt (next: {next_nodes})")

    # Apply edits if provided
    update_values = {"report_approved": True}

    if edits and current_state.values.get("draft_report"):
        draft = current_state.values["draft_report"]
        for key, value in edits.items():
            if hasattr(draft, key):
                setattr(draft, key, value)
        update_values["draft_report"] = draft

    # Update state and resume
    graph.update_state(config, update_values)

    # Resume execution from the interrupt
    result = graph.invoke(None, config)

    return {
        "thread_id": thread_id,
        "state": result,
        "published": True,
    }
