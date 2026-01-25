"""Planner node for the Plan-and-Execute architecture.

The planner breaks down user requests into a sequence of research steps.
This demonstrates the ReWOO-style "plan upfront" pattern, which is more
token-efficient than pure ReAct for structured tasks.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.schemas import AgentState, PlanStep, ResearchData

PLANNER_SYSTEM_PROMPT = """You are a senior investment research analyst at an institutional fund.
Your role is to break down stock analysis requests into clear, actionable research steps.

When given a stock to analyze, create a research plan with 4-6 steps covering:
1. Current price and basic metrics
2. Recent news and announcements  
3. Competitor analysis (if relevant)
4. Financial health assessment
5. Risk factors
6. Investment thesis synthesis

Output your plan as a JSON array of steps. Each step should have:
- step_number: Integer starting from 1
- description: What to research
- tool_hint: Suggested tool ("get_stock_price", "get_company_metrics", "get_price_history", "search_news", "search_competitors", or null for synthesis steps)

Be specific and actionable. The executor will follow these steps exactly."""


class PlanOutput(BaseModel):
    """Structured output for the planner."""

    steps: list[PlanStep] = Field(description="Research steps to execute")
    ticker: str = Field(description="The stock ticker being analyzed")


def planner_node(state: AgentState) -> dict:
    """Generate a research plan from the user's request.

    This node:
    1. Analyzes the user's message to understand the research request
    2. Creates a structured plan with specific steps
    3. Returns the plan for the executor to follow

    Args:
        state: Current agent state with messages

    Returns:
        Updated state with plan and research_data initialized
    """
    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
    )

    # Get structured output
    structured_llm = llm.with_structured_output(PlanOutput)

    # Get the last user message
    user_messages = [
        m for m in state.messages if hasattr(m, "type") and m.type == "human"
    ]
    if not user_messages:
        # Check for HumanMessage instances
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]

    last_user_message = (
        user_messages[-1].content if user_messages else "Analyze the market"
    )

    # Include user profile context if available
    profile_context = ""
    if state.user_profile:
        profile_context = f"""
        
User Profile:
- Risk Tolerance: {state.user_profile.risk_tolerance}
- Investment Horizon: {state.user_profile.investment_horizon}
- Preferred Sectors: {", ".join(state.user_profile.preferred_sectors) if state.user_profile.preferred_sectors else "None specified"}
- Notes: {state.user_profile.notes if state.user_profile.notes else "None"}

Consider this profile when planning the analysis."""

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT + profile_context),
        HumanMessage(content=f"Create a research plan for: {last_user_message}"),
    ]

    try:
        result: PlanOutput = structured_llm.invoke(messages)

        print(f"\n📋 Research plan created with {len(result.steps)} steps:")
        for step in result.steps:
            print(f"   {step.step_number}. {step.description}")

        # Initialize research data with the ticker
        research_data = ResearchData(ticker=result.ticker)

        return {
            "plan": result.steps,
            "current_step_index": 0,
            "research_data": research_data,
        }
    except Exception as e:
        print(f"\n❌ Planning failed: {str(e)}")
        return {
            "error": f"Planning failed: {str(e)}",
            "plan": [],
        }
