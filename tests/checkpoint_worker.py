"""Subprocess fixture: crash only after PostgreSQL committed a completed step."""

import os
import sys
from functools import partial
from pathlib import Path

from market_analyst.llm import ModelSettings
from market_analyst.memory import get_checkpointer
from market_analyst.schemas import DraftReport, ExecutionMode, PlanStep, UserProfile
from market_analyst.workflows.analysis_workflow import create_graph, run_analysis

phase, evidence_path = sys.argv[1:]


def executor(state):
    if state.current_step_index == 1 and phase == "crash":
        os._exit(23)
    with Path(evidence_path).open("a") as file:
        file.write(f"step-{state.current_step_index}\n")
    plan = list(state.plan)
    plan[state.current_step_index] = plan[state.current_step_index].model_copy(update={"completed": True, "result": "checkpointed"})
    return {"plan": plan, "current_step_index": state.current_step_index + 1}


factory = partial(
    create_graph,
    nodes={
        "planner": lambda state: {"plan": [PlanStep(step_number=1, description="first"), PlanStep(step_number=2, description="second")]},
        "executor": executor,
        "reporter": lambda state: {
            "draft_report": DraftReport(ticker="NVDA", title="Fixture", summary="Fixture", analysis="Fixture", recommendation="hold", confidence=0)
        },
        "evaluator": lambda state: {"evaluator_verdict": "pass"},
    },
)
result = run_analysis(
    "NVDA",
    thread_id="postgres-crash",
    checkpointer=get_checkpointer(),
    force_mode=ExecutionMode.DEEP_RESEARCH,
    model_settings=ModelSettings(provider="openai", model="fixture"),
    profile_loader=lambda _: UserProfile(),
    graph_factory=factory,
    retry_delivery=phase != "crash",
)
assert result["requires_approval"]
assert result["state"]["plan"][0].result == "checkpointed"
