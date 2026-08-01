"""ReWOO Worker node.

Executes the planned tool calls in parallel (or with dependency ordering).
This is where the token efficiency comes from - NO LLM calls between tool executions.
"""

import concurrent.futures

from langchain_core.tools import BaseTool
from pydantic import BaseModel

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

# Tool registry mapping names to functions for ReWOO parallel execution
TOOL_REGISTRY: dict[str, BaseTool] = {
    # JSON Tool Calling (modality 1)
    "get_stock_snapshot": get_stock_snapshot,
    "get_price_history": get_price_history,
    "get_financials": get_financials,
    "search_news": search_news,
    "search_competitors": search_competitors,
    # Skills (modality 2)
    "use_skill": use_skill,
    # CLI-as-Tool (modality 3)
    "cli_list_reports": cli_list_reports,
    "cli_show_report": cli_show_report,
    # Code Execution / PTC (modality 4)
    "execute_python_analysis": execute_python_analysis,
}


def execute_tool(step: ReWOOPlanStep, results: dict[str, str]) -> str:
    """Execute a single tool call, substituting variable references.

    Args:
        step: The planned step to execute
        results: Already-computed results keyed by step_id

    Returns:
        Tool execution result as string
    """
    tool_fn = TOOL_REGISTRY.get(step.tool_name)
    if not tool_fn:
        return f"Error: Unknown tool '{step.tool_name}'"

    # Substitute variable references in arguments
    resolved_args = {}
    for key, value in step.tool_args.items():
        if isinstance(value, str) and value.startswith("#E"):
            # This is a variable reference, substitute with actual result
            resolved_args[key] = results.get(value, value)
        else:
            resolved_args[key] = value

    try:
        result = tool_fn.invoke(resolved_args)
        # Handle Pydantic model results
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        return str(result)
    except Exception as e:
        return f"Error executing {step.tool_name}: {e}"


def rewoo_worker_node(state: AgentState) -> dict:
    """Execute all planned tool calls with parallel execution where possible.

    This node:
    1. Builds a dependency graph from the plan
    2. Executes independent tools in parallel
    3. Stores results for the solver

    Args:
        state: Current state with rewoo_plan

    Returns:
        Updated state with tool results stored in rewoo_plan steps
    """
    if not state.rewoo_plan:
        return {"error": state.error or "No ReWOO plan to execute"}

    print(f"\n🔧 Executing {len(state.rewoo_plan)} tool calls...")

    results: dict[str, str] = {}
    updated_by_id: dict[str, ReWOOPlanStep] = {}
    pending = {step.step_id: step for step in state.rewoo_plan}

    while pending:
        ready = [step for step in pending.values() if all(dep in results for dep in step.depends_on)]
        if not ready:
            unresolved = ", ".join(pending)
            return {"error": f"Unresolvable ReWOO dependencies: {unresolved}"}

        print(f"   ⚡ Executing {len(ready)} ready tool(s) in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(ready))) as executor:
            future_to_step = {executor.submit(execute_tool, step, results): step for step in ready}
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                result = future.result()
                results[step.step_id] = result
                updated_by_id[step.step_id] = step.model_copy(update={"result": result[:500]})
                del pending[step.step_id]
                print(f"   ✅ {step.step_id}: {step.tool_name} complete")

    updated_steps = [updated_by_id[step.step_id] for step in state.rewoo_plan]

    print(f"   🎯 All {len(updated_steps)} tools executed")

    return {
        "rewoo_plan": updated_steps,
    }
