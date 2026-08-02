"""ReWOO Worker node.

Executes the planned tool calls in parallel (or with dependency ordering).
This is where the token efficiency comes from - NO LLM calls between tool executions.
"""

import concurrent.futures

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from market_analyst.nodes._telemetry import AGENT_NS, get_conversation_id
from market_analyst.observability import tool_span
from market_analyst.schemas import AgentState, ReWOOPlanStep
from market_analyst.tools.cli_tools import cli_list_reports, cli_show_report
from market_analyst.tools.code_exec import execute_python_analysis
from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.skills import use_skill
from market_analyst.tools.stock import (
    get_financials,
    get_price_history,
    get_stock_snapshot,
)

TOOL_REGISTRY: dict[str, BaseTool] = {
    "get_stock_snapshot": get_stock_snapshot,
    "get_price_history": get_price_history,
    "get_financials": get_financials,
    "search_news": search_news,
    "search_competitors": search_competitors,
    "use_skill": use_skill,
    "cli_list_reports": cli_list_reports,
    "cli_show_report": cli_show_report,
    "execute_python_analysis": execute_python_analysis,
}


def execute_tool(
    step: ReWOOPlanStep,
    results: dict[str, str],
    *,
    conversation_id: str | None = None,
) -> str:
    """Execute a single tool call, substituting variable references."""
    tool_fn = TOOL_REGISTRY.get(step.tool_name)
    if not tool_fn:
        return f"Error: Unknown tool '{step.tool_name}'"

    resolved_args = {key: results.get(value, value) if isinstance(value, str) and value.startswith("#E") else value for key, value in step.tool_args.items()}

    try:
        with tool_span(
            tool_name=step.tool_name,
            agent_name=f"{AGENT_NS}.rewoo_worker",
            conversation_id=conversation_id,
            tool_call_id=step.step_id,
        ):
            result = tool_fn.invoke(resolved_args)
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)
    except Exception as e:
        return f"Error executing {step.tool_name}: {e}"


def rewoo_worker_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Execute dependency-ready tool calls in parallel until the plan is complete."""
    if not state.rewoo_plan:
        return {"error": state.error or "No ReWOO plan to execute"}

    print(f"\n🔧 Executing {len(state.rewoo_plan)} tool calls...")

    conversation_id = get_conversation_id(config)
    results: dict[str, str] = {}
    updated_by_id: dict[str, ReWOOPlanStep] = {}
    pending = {step.step_id: step for step in state.rewoo_plan}

    while pending:
        ready = [step for step in pending.values() if all(dep in results for dep in step.depends_on)]
        if not ready:
            unresolved = ", ".join(pending)
            return {"error": f"Unresolvable ReWOO dependencies: {unresolved}"}

        print(f"   ⚡ Executing {len(ready)} ready tool(s) in parallel...")
        trace_args = {"conversation_id": conversation_id} if conversation_id else {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(ready))) as executor:
            future_to_step = {executor.submit(execute_tool, step, results, **trace_args): step for step in ready}
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                result = future.result()
                results[step.step_id] = result
                updated_by_id[step.step_id] = step.model_copy(update={"result": result[:500]})
                del pending[step.step_id]
                print(f"   ✅ {step.step_id}: {step.tool_name} complete")

    updated_steps = [updated_by_id[step.step_id] for step in state.rewoo_plan]
    print(f"   🎯 All {len(updated_steps)} tools executed")
    return {"rewoo_plan": updated_steps}
