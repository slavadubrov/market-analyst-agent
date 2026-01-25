"""ReWOO Planner node.

Generates a complete plan of tool calls upfront with variable placeholders.
This is the key difference from ReAct - no intermediate LLM calls during execution.

Example output:
    #E1 = get_stock_price(ticker="NVDA")
    #E2 = search_news(query="NVDA recent news")
    #E3 = get_company_metrics(ticker="NVDA", mode="concise")
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.schemas import AgentState, ReWOOPlanStep

REWOO_PLANNER_PROMPT = """You are a research analyst creating an efficient data gathering plan.

Your task is to plan ALL tool calls upfront for a quick stock briefing. 
You will NOT see the outputs until the end - plan everything now.

Available tools:
- get_stock_price(ticker: str) -> Current price and basic info
- get_company_metrics(ticker: str, mode: str = "concise") -> Financial metrics
- get_price_history(ticker: str, period: str = "1mo") -> Historical prices  
- search_news(query: str) -> Recent news articles
- search_competitors(ticker: str) -> Competitor analysis

Create 3-5 tool calls maximum for a quick briefing. Use variable placeholders (#E1, #E2, etc.) to reference outputs.

For a flash briefing, focus on:
1. Current price
2. Key metrics
3. Recent news

Be efficient - this is for a QUICK snapshot, not deep research."""


class ReWOOPlanOutput(BaseModel):
    """Structured output for ReWOO planner."""

    steps: list[ReWOOPlanStep] = Field(description="Planned tool calls with variables")


def rewoo_planner_node(state: AgentState) -> dict:
    """Generate a complete plan of tool calls upfront.

    Unlike the regular planner, this creates the full execution plan
    with specific tool calls, not just step descriptions.

    Args:
        state: Current agent state with research_data.ticker

    Returns:
        Updated state with rewoo_plan
    """
    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(ReWOOPlanOutput)

    ticker = state.research_data.ticker if state.research_data else "UNKNOWN"

    # Get user's original query for context
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        user_messages = [
            m for m in state.messages if hasattr(m, "type") and m.type == "human"
        ]
    query = (
        user_messages[-1].content if user_messages else f"Quick analysis of {ticker}"
    )

    messages = [
        SystemMessage(content=REWOO_PLANNER_PROMPT),
        HumanMessage(
            content=f"""Create a ReWOO plan for this request: "{query}"

Ticker: {ticker}

Output a list of tool calls with:
- step_id: Variable name (#E1, #E2, etc.)
- description: What this step accomplishes
- tool_name: Exact tool name from the list
- tool_args: Dictionary of arguments
- depends_on: List of step_ids this depends on (usually empty for parallel execution)"""
        ),
    ]

    try:
        result: ReWOOPlanOutput = structured_llm.invoke(messages)

        print(f"\n⚡ ReWOO Plan created with {len(result.steps)} parallel tool calls:")
        for step in result.steps:
            deps = (
                f" (depends on: {', '.join(step.depends_on)})"
                if step.depends_on
                else ""
            )
            print(f"   {step.step_id} = {step.tool_name}({step.tool_args}){deps}")

        return {
            "rewoo_plan": result.steps,
        }

    except Exception as e:
        print(f"\n❌ ReWOO planning failed: {e}")
        return {
            "error": f"ReWOO planning failed: {e}",
            "rewoo_plan": [],
        }
