"""Router node for intent classification.

Classifies user requests to route between:
- DEEP_RESEARCH: Plan-and-Execute + ReAct (thorough analysis)
- FLASH_BRIEFING: ReWOO (fast, token-efficient snapshot)
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.schemas import AgentState, ExecutionMode, ResearchData

ROUTER_SYSTEM_PROMPT = """You are an intent classifier for a stock analysis agent.

Classify the user's request into one of two modes:

1. **deep_research**: For complex, strategic analysis requiring multiple data sources and synthesis.
   - Examples: "Analyze strategic risks", "deep dive into fundamentals", "investment thesis", "competitive analysis"
   
2. **flash_briefing**: For quick snapshots and simple data retrieval.
   - Examples: "quick snapshot", "current price", "brief update", "what's the latest on", "quick look at"

Consider:
- Explicit speed/depth keywords
- Complexity of the question
- Whether synthesis or just data retrieval is needed

Default to deep_research if unclear."""


class RouterOutput(BaseModel):
    """Structured output for the router."""

    mode: ExecutionMode = Field(description="The execution mode to use")
    ticker: str = Field(description="The stock ticker extracted from the query")
    reasoning: str = Field(description="Brief reasoning for the classification")


def router_node(state: AgentState) -> dict:
    """Classify the user's intent and route to appropriate execution path.

    Args:
        state: Current agent state with messages

    Returns:
        Updated state with execution_mode and research_data initialized
    """
    # Check for explicit mode override in state (set by CLI --mode flag)
    # If already set, skip LLM classification to save tokens
    if state.execution_mode is not None:
        mode_emoji = "⚡" if state.execution_mode == ExecutionMode.FLASH_BRIEFING else "🔬"
        mode_name = "Flash Briefing (ReWOO)" if state.execution_mode == ExecutionMode.FLASH_BRIEFING else "Deep Research (ReAct)"
        print(f"\n{mode_emoji} Mode: {mode_name} (forced via CLI)")

        # Still need to extract ticker from query
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        if not user_messages:
            user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
        query = user_messages[-1].content if user_messages else ""

        # Simple ticker extraction (look for uppercase 1-5 letter words)
        import re

        ticker_match = re.search(r"\b([A-Z]{1,5})\b", query)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        print(f"   Ticker: {ticker}")

        return {
            "research_data": ResearchData(ticker=ticker),
        }

    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(RouterOutput)

    # Get the user's query
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]

    query = user_messages[-1].content if user_messages else "Analyze the market"

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=f"Classify this request: {query}"),
    ]

    try:
        result: RouterOutput = structured_llm.invoke(messages)

        mode_emoji = "⚡" if result.mode == ExecutionMode.FLASH_BRIEFING else "🔬"
        mode_name = "Flash Briefing (ReWOO)" if result.mode == ExecutionMode.FLASH_BRIEFING else "Deep Research (ReAct)"

        print(f"\n{mode_emoji} Mode: {mode_name}")
        print(f"   Ticker: {result.ticker}")
        print(f"   Reasoning: {result.reasoning}")

        # Initialize research data with the ticker
        research_data = ResearchData(ticker=result.ticker)

        return {
            "execution_mode": result.mode,
            "research_data": research_data,
        }

    except Exception as e:
        print(f"\n⚠️  Router failed, defaulting to deep research: {e}")
        return {
            "execution_mode": ExecutionMode.DEEP_RESEARCH,
            "research_data": ResearchData(ticker="UNKNOWN"),
        }
