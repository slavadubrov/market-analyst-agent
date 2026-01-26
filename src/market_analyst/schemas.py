"""Pydantic models and state schemas for the Market Analyst Agent."""

from enum import Enum
from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Execution mode for the agent."""

    DEEP_RESEARCH = "deep_research"  # Plan-and-Execute + ReAct (thorough)
    FLASH_BRIEFING = "flash_briefing"  # ReWOO (fast, token-efficient)


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


class ReWOOPlanStep(BaseModel):
    """A step in the ReWOO plan with variable placeholders.

    ReWOO plans all tool calls upfront with variable references (e.g., #E1, #E2).
    This allows parallel execution without intermediate LLM calls.
    """

    step_id: str = Field(description="Variable ID, e.g., '#E1'")
    description: str = Field(description="What this step accomplishes")
    tool_name: str = Field(description="Tool to call (required)")
    tool_args: dict = Field(
        default_factory=dict,
        description="Tool arguments, may contain variable refs like '#E1'",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of step_ids this step depends on",
    )
    result: str | None = Field(default=None, description="Tool execution result")


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


class TradeAction(str, Enum):
    """Supported trade actions."""

    BUY = "buy"
    SELL = "sell"
    DELETE_PORTFOLIO = "delete_portfolio"
    DELETE_LOGS = "delete_logs"


class TradeRequest(BaseModel):
    """A trade request that requires policy validation.

    This represents an action the agent wants to take that may
    need human approval based on configured policies.
    """

    action: TradeAction = Field(description="The type of trade action")
    ticker: str = Field(description="Stock ticker symbol")
    amount_usd: float = Field(ge=0, description="Trade amount in USD")
    reason: str = Field(description="Agent's reasoning for the trade")


class GuardianDecision(str, Enum):
    """Decisions the Guardian can make."""

    APPROVE = "approve"  # Auto-approve (safe path)
    ESCALATE = "escalate"  # Needs human review
    REJECT = "reject"  # Auto-reject (policy violation)


class GuardianResult(BaseModel):
    """Result from the Guardian policy check.

    The Guardian is an automated, deterministic policy layer that
    inspects actions before they execute.
    """

    decision: GuardianDecision
    policy_name: str = Field(description="Which policy triggered the decision")
    reason: str = Field(description="Human-readable explanation")
    original_request: TradeRequest | None = None


class AgentState(BaseModel):
    """Main state for the Market Analyst Agent graph.

    This state is persisted via PostgresSaver for checkpointing,
    allowing the agent to pause and resume mid-analysis.
    """

    # Message history with LangGraph's add_messages reducer
    messages: Annotated[list, add_messages] = Field(default_factory=list)

    # Execution mode (set by router)
    execution_mode: ExecutionMode | None = None

    # Plan-and-Execute state (for DEEP_RESEARCH mode)
    plan: list[PlanStep] = Field(default_factory=list)
    current_step_index: int = 0

    # ReWOO state (for FLASH_BRIEFING mode)
    rewoo_plan: list[ReWOOPlanStep] = Field(default_factory=list)

    # Research results
    research_data: ResearchData | None = None

    # Draft report for HITL approval
    draft_report: DraftReport | None = None
    report_approved: bool = False

    # User context (loaded from Redis at start)
    user_profile: UserProfile = Field(default_factory=UserProfile)
    user_id: str = "default"

    # Guardian workflow (for trade actions with policy checks)
    pending_trade: TradeRequest | None = None
    guardian_result: GuardianResult | None = None
    trade_approved: bool = False
    trade_executed: bool = False

    # Workflow control
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True
