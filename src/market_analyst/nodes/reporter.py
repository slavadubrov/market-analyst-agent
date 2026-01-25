"""Reporter node for generating investment reports.

This node synthesizes all research into a draft report that requires
human approval before being finalized (HITL pattern).
"""

import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.schemas import AgentState, DraftReport

REPORTER_SYSTEM_PROMPT = """You are a senior investment analyst writing a research report.

Based on the research findings provided, create a comprehensive but concise investment report.
Your report should include:

1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Investment Thesis
4. Risk Factors (list key risks)
5. Recommendation (strong_buy, buy, hold, sell, strong_sell)

Consider the user's risk profile when making recommendations.
Be objective and data-driven. Acknowledge uncertainties.

Output your report as structured JSON matching the DraftReport schema."""


def reporter_node(state: AgentState) -> dict:
    """Generate a draft investment report from research findings.

    This node:
    1. Collects all research from completed plan steps
    2. Synthesizes findings into a structured report
    3. Creates a draft that will require human approval

    The graph will interrupt after this node to allow human review.

    Args:
        state: Current agent state with completed research

    Returns:
        Updated state with draft_report
    """
    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(
        model=model_name,
        temperature=0.3,  # Slightly more creative for report writing
    )

    structured_llm = llm.with_structured_output(DraftReport)

    # Compile research findings
    research_summary = ""
    for step in state.plan:
        if step.completed and step.result:
            research_summary += (
                f"\n### Step {step.step_number}: {step.description}\n{step.result}\n"
            )

    # User profile context
    profile_context = ""
    if state.user_profile:
        profile_context = f"""
User Profile:
- Risk Tolerance: {state.user_profile.risk_tolerance}
- Investment Horizon: {state.user_profile.investment_horizon}
- Preferred Sectors: {", ".join(state.user_profile.preferred_sectors) if state.user_profile.preferred_sectors else "Not specified"}

Tailor your recommendation to this profile."""

    ticker = state.research_data.ticker if state.research_data else "UNKNOWN"

    messages = [
        SystemMessage(content=REPORTER_SYSTEM_PROMPT),
        HumanMessage(
            content=f"""Create an investment report for {ticker}.

{profile_context}

Research Findings:
{research_summary}

Generate a complete DraftReport with your analysis and recommendation."""
        ),
    ]

    print(f"\n📝 Generating investment report for {ticker}...")

    try:
        report: DraftReport = structured_llm.invoke(messages)

        return {
            "draft_report": report,
            "report_approved": False,  # Requires human approval
        }

    except Exception as e:
        print(f"\n❌ Report generation failed: {str(e)}")
        return {
            "error": f"Report generation failed: {str(e)}",
            "draft_report": None,
        }


def format_report_for_display(report: DraftReport) -> str:
    """Format the draft report for human review."""
    if not report:
        return "No report generated."

    risk_factors = (
        "\n".join(f"  - {r}" for r in report.risk_factors)
        if report.risk_factors
        else "  None identified"
    )

    return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    DRAFT INVESTMENT REPORT                        ║
╠══════════════════════════════════════════════════════════════════╣

📊 {report.title}
   Ticker: {report.ticker}

📋 SUMMARY
{report.summary}

📈 ANALYSIS
{report.analysis}

⚠️  RISK FACTORS
{risk_factors}

🎯 RECOMMENDATION: {report.recommendation.upper().replace("_", " ")}
   Confidence: {report.confidence:.0%}

╚══════════════════════════════════════════════════════════════════╝

[This report requires your approval before being finalized]
"""
