"""Pydantic models and state schemas for the Market Analyst Agent."""

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """User investment profile stored in Redis for cross-thread memory."""

    risk_tolerance: Literal["conservative", "moderate", "aggressive"] = Field(
        default="moderate", description="User's risk tolerance level"
    )
    investment_horizon: Literal["short", "medium", "long"] = Field(
        default="medium", description="Investment time horizon"
    )
    preferred_sectors: list[str] = Field(
        default_factory=list, description="Preferred industry sectors"
    )
    notes: str = Field(default="", description="Additional user notes/preferences")


class PlanStep(BaseModel):
    """A single step in the research plan."""

    step_number: int
    description: str
    tool_hint: str | None = Field(
        default=None, description="Suggested tool to use for this step"
    )
    completed: bool = False
    result: str | None = None


class ResearchData(BaseModel):
    """Collected research data from tool executions."""

    ticker: str
    current_price: float | None = None
    price_change_pct: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    news_summary: str | None = None
    competitor_analysis: str | None = None
    raw_data: dict = Field(default_factory=dict)


class DraftReport(BaseModel):
    """Generated investment report awaiting approval."""

    ticker: str
    title: str
    summary: str
    analysis: str
    recommendation: Literal["strong_buy", "buy", "hold", "sell", "strong_sell"]
    confidence: float = Field(ge=0.0, le=1.0)
    risk_factors: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Main state for the Market Analyst Agent graph.

    This state is persisted via PostgresSaver for checkpointing,
    allowing the agent to pause and resume mid-analysis.
    """

    # Message history with LangGraph's add_messages reducer
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # Plan-and-Execute state
    plan: list[PlanStep] = Field(default_factory=list)
    current_step_index: int = 0

    # Research results
    research_data: ResearchData | None = None

    # Draft report for HITL approval
    draft_report: DraftReport | None = None
    report_approved: bool = False

    # User context (loaded from Redis at start)
    user_profile: UserProfile = Field(default_factory=UserProfile)
    user_id: str = "default"

    # Workflow control
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
