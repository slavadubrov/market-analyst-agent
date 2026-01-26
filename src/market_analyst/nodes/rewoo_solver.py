"""ReWOO Solver node.

Takes the collected tool results and synthesizes a final answer in ONE LLM call.
This is the second key efficiency gain - only one synthesis call after all tools complete.
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.schemas import AgentState, DraftReport

REWOO_SOLVER_PROMPT = """You are a senior investment analyst creating a quick briefing.

You have gathered the following data through our research tools. 
Synthesize this into a concise, actionable flash briefing.

Be concise but comprehensive. This is a QUICK snapshot, not a deep dive.
Focus on the most important takeaways."""


class FlashBriefingOutput(BaseModel):
    """Structured output for the flash briefing."""

    ticker: str = Field(description="Stock ticker")
    title: str = Field(description="Briefing title")
    summary: str = Field(description="2-3 sentence executive summary")
    analysis: str = Field(description="Key findings from the data, 3-5 bullet points")
    recommendation: str = Field(description="One of: strong_buy, buy, hold, sell, strong_sell")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendation")
    risk_factors: list[str] = Field(description="Top 2-3 risk factors")


def rewoo_solver_node(state: AgentState) -> dict:
    """Synthesize all tool results into a flash briefing.

    This node makes ONE LLM call to create the final output,
    completing the ReWOO pattern's efficiency advantage.

    Args:
        state: Current state with rewoo_plan containing tool results

    Returns:
        Updated state with draft_report
    """
    if not state.rewoo_plan:
        return {"error": "No ReWOO plan results to synthesize"}

    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(
        model=model_name,
        temperature=0,
    )

    structured_llm = llm.with_structured_output(FlashBriefingOutput)

    ticker = state.research_data.ticker if state.research_data else "UNKNOWN"

    # Build context from all tool results
    tool_results = []
    for step in state.rewoo_plan:
        if step.result:
            tool_results.append(f"### {step.description}\n{step.result}")

    context = "\n\n".join(tool_results)

    # Get original query for context
    user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
    if not user_messages:
        user_messages = [m for m in state.messages if hasattr(m, "type") and m.type == "human"]
    query = user_messages[-1].content if user_messages else f"Quick analysis of {ticker}"

    messages = [
        SystemMessage(content=REWOO_SOLVER_PROMPT),
        HumanMessage(
            content=f"""Original request: "{query}"

## Collected Research Data

{context}

---

Create a flash briefing from this data. Be concise and actionable."""
        ),
    ]

    try:
        print("\n📝 Synthesizing flash briefing...")
        result: FlashBriefingOutput = structured_llm.invoke(messages)

        # Convert to DraftReport format for consistency with the publish flow
        draft_report = DraftReport(
            ticker=result.ticker or ticker,
            title=result.title,
            summary=result.summary,
            analysis=result.analysis,
            recommendation=result.recommendation,
            confidence=result.confidence,
            risk_factors=result.risk_factors,
        )

        print(f"   ✅ Flash briefing complete: {draft_report.title}")

        return {
            "draft_report": draft_report,
        }

    except Exception as e:
        print(f"\n❌ Solver failed: {e}")
        return {
            "error": f"ReWOO solver failed: {e}",
        }
