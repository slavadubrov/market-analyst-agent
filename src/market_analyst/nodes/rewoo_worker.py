"""ReWOO Worker node.

Executes the planned tool calls in parallel (or with dependency ordering).
This is where the token efficiency comes from - NO LLM calls between tool executions.
"""

import concurrent.futures
from collections.abc import Callable
from typing import Any

from market_analyst.schemas import AgentState, ReWOOPlanStep
from market_analyst.tools.search import search_competitors, search_news
from market_analyst.tools.stock import (
    get_company_metrics,
    get_price_history,
    get_stock_price,
)

# Tool registry mapping names to functions for ReWOO parallel execution
TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    "get_stock_price": get_stock_price,
    "get_company_metrics": get_company_metrics,
    "get_price_history": get_price_history,
    "search_news": search_news,
    "search_competitors": search_competitors,
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
        if hasattr(result, "model_dump_json"):
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
        return {"error": "No ReWOO plan to execute"}

    print(f"\n🔧 Executing {len(state.rewoo_plan)} tool calls...")

    # Separate steps into those with and without dependencies
    results: dict[str, str] = {}
    updated_steps: list[ReWOOPlanStep] = []

    # First pass: execute all independent steps in parallel
    independent_steps = [s for s in state.rewoo_plan if not s.depends_on]
    dependent_steps = [s for s in state.rewoo_plan if s.depends_on]

    # Execute independent steps in parallel using ThreadPoolExecutor
    if independent_steps:
        print(f"   ⚡ Executing {len(independent_steps)} independent tools in parallel...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_step = {executor.submit(execute_tool, step, results): step for step in independent_steps}

            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                try:
                    result = future.result()
                    results[step.step_id] = result

                    # Create updated step with result
                    updated_step = ReWOOPlanStep(
                        step_id=step.step_id,
                        description=step.description,
                        tool_name=step.tool_name,
                        tool_args=step.tool_args,
                        depends_on=step.depends_on,
                        result=result[:500] if len(result) > 500 else result,  # Truncate for display
                    )
                    updated_steps.append(updated_step)
                    print(f"   ✅ {step.step_id}: {step.tool_name} complete")
                except Exception as e:
                    print(f"   ❌ {step.step_id}: {step.tool_name} failed: {e}")
                    results[step.step_id] = f"Error: {e}"

    # Second pass: execute dependent steps sequentially
    for step in dependent_steps:
        # Wait for dependencies (they should already be done from parallel phase)
        missing_deps = [dep for dep in step.depends_on if dep not in results]
        if missing_deps:
            print(f"   ⚠️  {step.step_id}: Missing dependencies {missing_deps}")

        result = execute_tool(step, results)
        results[step.step_id] = result

        updated_step = ReWOOPlanStep(
            step_id=step.step_id,
            description=step.description,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            depends_on=step.depends_on,
            result=result[:500] if len(result) > 500 else result,
        )
        updated_steps.append(updated_step)
        print(f"   ✅ {step.step_id}: {step.tool_name} complete")

    # Sort by step_id to maintain order
    updated_steps.sort(key=lambda s: s.step_id)

    print(f"   🎯 All {len(updated_steps)} tools executed")

    return {
        "rewoo_plan": updated_steps,
    }
