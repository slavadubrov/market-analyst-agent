"""ReWOO Worker node.

Executes the planned tool calls in parallel (or with dependency ordering).
This is where the token efficiency comes from - NO LLM calls between tool executions.
"""

import concurrent.futures
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from market_analyst.nodes._telemetry import AGENT_NS, get_conversation_id
from market_analyst.observability import tool_span
from market_analyst.runtime.evidence import record_evidence
from market_analyst.schemas import AgentState, ReWOOPlanStep
from market_analyst.tools.registry import tool_registry

TOOL_REGISTRY = tool_registry()


def references(value):
    """Find exact #E references recursively; embedded prose is not a reference."""
    if isinstance(value, str):
        return {value} if value.startswith("#E") else set()
    if isinstance(value, dict):
        return set().union(*(references(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(references(v) for v in value))
    return set()


def resolve_arguments(value, results):
    if isinstance(value, str) and value.startswith("#E"):
        return results[value]
    if isinstance(value, dict):
        return {k: resolve_arguments(v, results) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(v, results) for v in value]
    return value


def validate_plan(steps):
    """Reject an invalid graph before any tool is dispatched."""
    import graphlib
    import re

    ids = [step.step_id for step in steps]
    if not 1 <= len(ids) <= 20 or len(set(ids)) != len(ids):
        raise ValueError("ReWOO requires 1-20 unique step IDs")
    if any(not re.fullmatch(r"#E[1-9][0-9]*", name) for name in ids):
        raise ValueError("Invalid ReWOO step ID")
    dependencies = {}
    for step in steps:
        deps = set(step.depends_on) | references(step.tool_args)
        if not deps <= set(ids):
            raise ValueError("Missing ReWOO dependency")
        dependencies[step.step_id] = deps
    try:
        tuple(graphlib.TopologicalSorter(dependencies).static_order())
    except graphlib.CycleError as exc:
        raise ValueError("Cyclic ReWOO dependencies") from exc
    return dependencies


def execute_tool(
    step: ReWOOPlanStep,
    results: dict[str, str],
    *,
    conversation_id: str | None = None,
    registry: dict[str, BaseTool] | None = None,
) -> str:
    """Execute a single tool call, substituting variable references."""
    tool_fn = (TOOL_REGISTRY if registry is None else registry).get(step.tool_name)
    if not tool_fn:
        return f"Error: Unknown tool '{step.tool_name}'"

    resolved_args = resolve_arguments(step.tool_args, results)

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
        if isinstance(result, (dict, list)):
            import json

            return json.dumps(result)
        return str(result)
    except Exception as e:
        return f"Error executing {step.tool_name}: {e}"


def rewoo_worker_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    """Execute dependency-ready tool calls in parallel until the plan is complete."""
    if state.error:
        return {"error": state.error}
    if not state.rewoo_plan:
        return {"error": state.error or "No ReWOO plan to execute"}

    try:
        dependencies = validate_plan(state.rewoo_plan)
    except ValueError as exc:
        return {"error": str(exc)}

    print(f"\n🔧 Executing {len(state.rewoo_plan)} tool calls...")

    available = tool_registry((config or {}).get("configurable", {}).get("tools"))
    if any(step.tool_name not in available for step in state.rewoo_plan):
        return {"error": "Plan references a tool outside the configured registry"}
    conversation_id = get_conversation_id(config)
    results: dict[str, str] = {}
    updated_by_id: dict[str, ReWOOPlanStep] = {}
    pending = {step.step_id: step for step in state.rewoo_plan}

    while pending:
        ready = [step for step in pending.values() if all(dep in results for dep in dependencies[step.step_id])]
        if not ready:
            unresolved = ", ".join(pending)
            return {"error": f"Unresolvable ReWOO dependencies: {unresolved}"}

        print(f"   ⚡ Executing {len(ready)} ready tool(s) in parallel...")
        trace_args: dict[str, Any] = {"conversation_id": conversation_id} if conversation_id else {}
        selected = (config or {}).get("configurable", {}).get("tools")
        if selected is not None:
            trace_args["registry"] = tool_registry(selected)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(ready))) as executor:
            future_to_step = {executor.submit(execute_tool, step, results, **trace_args): step for step in ready}
            for future in concurrent.futures.as_completed(future_to_step):
                step = future_to_step[future]
                result = future.result()
                results[step.step_id] = result
                updated_by_id[step.step_id] = step.model_copy(update={"result": result})
                del pending[step.step_id]
                print(f"   ✅ {step.step_id}: {step.tool_name} complete")
        failed = [step for step in updated_by_id.values() if (step.result or "").startswith(("Error", "Blocked"))]
        if failed:
            return {
                "error": str(failed[0].result),
                "rewoo_plan": [updated_by_id.get(step.step_id, step) for step in state.rewoo_plan],
                "evidence": state.evidence + [record_evidence(step.tool_name, step.result) for step in updated_by_id.values()],
            }

    updated_steps = [updated_by_id[step.step_id] for step in state.rewoo_plan]
    print(f"   🎯 All {len(updated_steps)} tools executed")
    evidence = state.evidence + [record_evidence(step.tool_name, step.result) for step in updated_steps]
    errors = [str(step.result) for step in updated_steps if (step.result or "").startswith(("Error", "Blocked"))]
    return {"rewoo_plan": updated_steps, "evidence": evidence, **({"error": "; ".join(errors)} if errors else {})}
