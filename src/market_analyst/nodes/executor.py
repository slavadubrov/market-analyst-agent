"""Executor node implementing the ReAct pattern.

The executor takes one step from the plan and executes it using available tools.
This creates the classic Thought-Action-Observation loop, but guided by the
pre-generated plan (combining Plan-and-Execute with ReAct).
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.schemas import AgentState, PlanStep
from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.stock import (
    get_company_metrics,
    get_price_history,
    get_stock_price,
)

EXECUTOR_SYSTEM_PROMPT = """You are a research analyst executing a specific step in an investment analysis plan.

You have access to the following tools:
- get_stock_price: Get current price for a ticker
- get_company_metrics: Get financial metrics (use mode="concise" unless you need details)
- get_price_history: Get historical price data
- search_news: Search for recent news about a topic
- search_competitors: Find competitor analysis

Execute the assigned step thoroughly but efficiently. Use tools as needed to gather the required information.
After gathering data, provide a clear summary of your findings for this step.

Be concise but comprehensive. Token efficiency matters."""


# All available tools for the executor
TOOLS = [
    get_stock_price,
    get_company_metrics,
    get_price_history,
    search_news,
    search_competitors,
]


def executor_node(state: AgentState) -> dict:
    """Execute the current step in the research plan.

    This node:
    1. Gets the current step from the plan
    2. Uses a ReAct agent to execute it with tools
    3. Records the result and advances to the next step

    Args:
        state: Current agent state with plan and current_step_index

    Returns:
        Updated state with step results and incremented index
    """
    # Check if we have a plan to execute
    if not state.plan:
        return {"error": "No plan to execute"}

    # Check if we've completed all steps
    if state.current_step_index >= len(state.plan):
        return {}  # All steps complete, will route to reporter

    current_step = state.plan[state.current_step_index]

    # Build context from previous steps
    previous_context = ""
    for i, step in enumerate(state.plan[: state.current_step_index]):
        if step.result:
            previous_context += f"\nStep {step.step_number} ({step.description}): {step.result}\n"

    # Create the ReAct agent for this step
    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
    )

    react_agent = create_react_agent(
        model=llm,
        tools=TOOLS,
    )

    # Prepare the task for this step
    ticker = state.research_data.ticker if state.research_data else "UNKNOWN"

    task_message = f"""Execute Step {current_step.step_number}: {current_step.description}

Ticker being analyzed: {ticker}
{f"Suggested tool: {current_step.tool_hint}" if current_step.tool_hint else ""}

Previous research findings:
{previous_context if previous_context else "This is the first step."}

Complete this step and summarize your findings concisely."""

    # Progress indicator
    print(f"\n🔄 Executing step {current_step.step_number}/{len(state.plan)}: {current_step.description}")

    # Run the ReAct agent
    try:
        result = react_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
                    HumanMessage(content=task_message),
                ]
            }
        )

        # Extract the final response
        final_message = result["messages"][-1]
        step_result = final_message.content if hasattr(final_message, "content") else str(final_message)

        # Update the step with its result
        updated_plan = list(state.plan)
        updated_step = PlanStep(
            step_number=current_step.step_number,
            description=current_step.description,
            tool_hint=current_step.tool_hint,
            completed=True,
            result=step_result,
        )
        updated_plan[state.current_step_index] = updated_step

        # Update research data based on tool calls
        research_data = state.research_data
        if research_data:
            # Try to extract key data from tool results
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls"):
                    for tool_call in msg.tool_calls:
                        # Store raw results for later synthesis
                        if research_data.raw_data is None:
                            research_data.raw_data = {}
                        research_data.raw_data[f"step_{current_step.step_number}"] = step_result

        print(f"   ✅ Step {current_step.step_number} complete")

        return {
            "plan": updated_plan,
            "current_step_index": state.current_step_index + 1,
            "research_data": research_data,
            "messages": [AIMessage(content=f"Completed step {current_step.step_number}: {step_result[:200]}...")],
        }

    except Exception as e:
        # Mark step as failed but continue
        print(f"\n❌ Step {current_step.step_number} failed: {str(e)}")
        updated_plan = list(state.plan)
        updated_step = PlanStep(
            step_number=current_step.step_number,
            description=current_step.description,
            tool_hint=current_step.tool_hint,
            completed=True,
            result=f"Error: {str(e)}",
        )
        updated_plan[state.current_step_index] = updated_step

        return {
            "plan": updated_plan,
            "current_step_index": state.current_step_index + 1,
            "error": f"Step {current_step.step_number} failed: {str(e)}",
        }


def should_continue_executing(state: AgentState) -> str:
    """Routing function to determine if we should continue executing or move to reporting.

    Returns:
        "executor" to continue with next step
        "reporter" when all steps are complete
        "error" if there's a critical error
    """
    if state.error and "critical" in state.error.lower():
        return "error"

    if state.current_step_index >= len(state.plan):
        return "reporter"

    return "executor"
