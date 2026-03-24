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
from market_analyst.tools.cli_tools import cli_list_reports, cli_show_report
from market_analyst.tools.code_exec import execute_python_analysis
from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.skills import get_skill_descriptions, use_skill
from market_analyst.tools.stock import (
    get_financials,
    get_price_history,
    get_stock_snapshot,
)

_SKILL_DESCRIPTIONS = get_skill_descriptions()

EXECUTOR_SYSTEM_PROMPT = f"""You are a research analyst executing a specific step in an investment analysis plan.

You have access to tools across multiple modalities:

**Data Tools (JSON tool calling):**
- get_stock_snapshot: Get price, volume, market cap, P/E ratio, and a summary in one call
- get_price_history: Get historical price data with volume over a configurable period
- get_financials: Get income statement, balance sheet, or cash flow data
- search_news: Search for recent news with extracted key points
- search_competitors: Find competitor analysis with relative metrics

**Expertise Tools (Skills):**
- use_skill: Activate a domain playbook for structured analysis methodology
{_SKILL_DESCRIPTIONS}

**Computation Tools (Code Execution):**
- execute_python_analysis: Write and run Python code for financial calculations, ratio analysis, \
growth rate computations, or data transformations. Use this when you need loops, conditionals, or math \
that would be tedious to do manually.

**Memory Tools (CLI):**
- cli_list_reports: List previously saved analysis reports from document memory
- cli_show_report: Retrieve a specific saved report by key

**Guidelines:**
- Use data tools to gather information
- Use skills when you need a structured methodology (e.g., earnings analysis playbook)
- Use code execution for calculations on data you already have
- Use CLI tools to reference past analyses
- Be concise but comprehensive. Token efficiency matters."""


# All available tools across modalities
TOOLS = [
    # JSON Tool Calling (modality 1)
    get_stock_snapshot,
    get_price_history,
    get_financials,
    search_news,
    search_competitors,
    # Skills (modality 2)
    use_skill,
    # CLI-as-Tool (modality 3)
    cli_list_reports,
    cli_show_report,
    # Code Execution / PTC (modality 4)
    execute_python_analysis,
]


def _build_previous_context(plan, current_step_index):
    """Build context string from previously completed steps."""
    parts = []
    for step in plan[:current_step_index]:
        if step.result:
            parts.append(f"\nStep {step.step_number} ({step.description}): {step.result}\n")
    return "".join(parts)


def _create_updated_plan(state, current_step, result_text):
    """Create a copy of the plan with the current step marked complete."""
    updated_plan = list(state.plan)
    updated_plan[state.current_step_index] = PlanStep(
        step_number=current_step.step_number,
        description=current_step.description,
        tool_hint=current_step.tool_hint,
        completed=True,
        result=result_text,
    )
    return updated_plan


def _update_research_data(state, result, current_step, step_result):
    """Update research data with raw results from tool calls."""
    research_data = state.research_data
    if not research_data:
        return research_data
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            if research_data.raw_data is None:
                research_data.raw_data = {}
            research_data.raw_data[f"step_{current_step.step_number}"] = step_result
            break
    return research_data


def executor_node(state: AgentState) -> dict:
    """Execute the current step in the research plan.

    This node:
    1. Gets the current step from the plan
    2. Uses a ReAct agent to execute it with tools
    3. Records the result and advances to the next step
    """
    if not state.plan:
        return {"error": "No plan to execute"}

    if state.current_step_index >= len(state.plan):
        return {}

    current_step = state.plan[state.current_step_index]
    previous_context = _build_previous_context(state.plan, state.current_step_index)

    model_name = os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(model=model_name, temperature=0)
    react_agent = create_react_agent(model=llm, tools=TOOLS)

    ticker = state.research_data.ticker if state.research_data else "UNKNOWN"
    task_message = f"""Execute Step {current_step.step_number}: {current_step.description}

Ticker being analyzed: {ticker}
{f"Suggested tool: {current_step.tool_hint}" if current_step.tool_hint else ""}

Previous research findings:
{previous_context if previous_context else "This is the first step."}

Complete this step and summarize your findings concisely."""

    print(f"\n🔄 Executing step {current_step.step_number}/{len(state.plan)}: {current_step.description}")

    try:
        result = react_agent.invoke(
            {
                "messages": [
                    SystemMessage(content=EXECUTOR_SYSTEM_PROMPT),
                    HumanMessage(content=task_message),
                ]
            }
        )

        final_message = result["messages"][-1]
        step_result = final_message.content if hasattr(final_message, "content") else str(final_message)
        updated_plan = _create_updated_plan(state, current_step, step_result)
        research_data = _update_research_data(state, result, current_step, step_result)

        print(f"   ✅ Step {current_step.step_number} complete")

        return {
            "plan": updated_plan,
            "current_step_index": state.current_step_index + 1,
            "research_data": research_data,
            "messages": [AIMessage(content=f"Completed step {current_step.step_number}: {step_result[:200]}...")],
        }

    except Exception as e:
        print(f"\n❌ Step {current_step.step_number} failed: {str(e)}")
        updated_plan = _create_updated_plan(state, current_step, f"Error: {str(e)}")

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
