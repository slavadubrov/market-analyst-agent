"""Shared research graph wiring. Keep the two reasoning loops visible to readers.

Pass node replacements for experiments; both analysis and combined workflows use
this exact graph, so failure handling and evaluator placement cannot drift.
"""

from langgraph.graph import END, START

from market_analyst.nodes.executor import executor_node
from market_analyst.nodes.planner import planner_node
from market_analyst.nodes.reporter import reporter_node
from market_analyst.nodes.rewoo_planner import rewoo_planner_node
from market_analyst.nodes.rewoo_solver import rewoo_solver_node
from market_analyst.nodes.rewoo_worker import rewoo_worker_node
from market_analyst.nodes.router import router_node
from market_analyst.runtime.evaluator import evaluator_node
from market_analyst.schemas import AgentState, ExecutionMode


def route_after_router(state: AgentState):
    return "rewoo_planner" if state.execution_mode == ExecutionMode.FLASH_BRIEFING else "planner"


def route_after_executor(state: AgentState):
    if state.error:
        return END
    return "reporter" if state.current_step_index >= len(state.plan) else "executor"


def add_research_nodes(builder, *, nodes=None):
    implementations = {
        "router": router_node,
        "planner": planner_node,
        "executor": executor_node,
        "reporter": reporter_node,
        "rewoo_planner": rewoo_planner_node,
        "rewoo_worker": rewoo_worker_node,
        "rewoo_solver": rewoo_solver_node,
        "evaluator": evaluator_node,
    }
    if nodes:
        if not nodes.keys() <= implementations.keys():
            raise ValueError("Unknown research node replacement")
        implementations.update(nodes)
    for name, implementation in implementations.items():
        builder.add_node(name, implementation)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_after_router)
    builder.add_conditional_edges("executor", route_after_executor)
    for source, target in (
        ("planner", "executor"),
        ("reporter", "evaluator"),
        ("rewoo_planner", "rewoo_worker"),
        ("rewoo_worker", "rewoo_solver"),
        ("rewoo_solver", "evaluator"),
        ("evaluator", "publish"),
    ):

        def next_node(state: AgentState, destination=target):
            return END if state.error else destination

        builder.add_conditional_edges(source, next_node, [target, END])
