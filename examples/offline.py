"""A deterministic teaching fixture: no API keys, network, or databases.

The real ReWOO worker executes an injected tool. Only planning, writing, and
judgment are fixtures; this does not measure a model's reasoning quality.
"""

import uuid
from functools import partial

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from market_analyst.harness import AgentHarness
from market_analyst.llm import ModelSettings
from market_analyst.memory.encryption import checkpoint_serializer
from market_analyst.runtime.evidence import validate_evidence
from market_analyst.schemas import DraftReport, ExecutionMode, ReWOOPlanStep
from market_analyst.workflows.analysis_workflow import create_graph


@tool
def fixture_quote(ticker: str) -> dict:
    """Return a synthetic quote for the offline lesson, never a real market price."""
    return {"ticker": ticker, "price": 100, "source": "synthetic fixture"}


def plan(state):
    return {"rewoo_plan": [ReWOOPlanStep(step_id="#E1", description="Read a fixture quote", tool_name="fixture_quote", tool_args={"ticker": "DEMO"})]}


def write_report(state):
    return {
        "draft_report": DraftReport(
            ticker="DEMO",
            title="Synthetic quote lesson",
            summary="The fixture price is 100.",
            analysis="The configured tool returned synthetic data, not a live market observation.",
            recommendation="hold",
            confidence=0,
            risk_factors=["Synthetic fixture; cannot support investment decisions."],
        )
    }


def evaluate(state):
    issues = validate_evidence(state.evidence)
    return {"evaluator_verdict": "fail" if issues else "pass", "evaluator_reasons": issues or ["Fixture evidence exists"]}


def main():
    agent = AgentHarness(
        model=ModelSettings(provider="openai", model="unused-offline-fixture"),
        tools=(fixture_quote,),
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        graph_factory=partial(create_graph, nodes={"rewoo_planner": plan, "rewoo_solver": write_report, "evaluator": evaluate}),
    )
    run_id = "offline-" + uuid.uuid4().hex
    first = agent.run("Read the DEMO fixture", thread_id=run_id, mode=ExecutionMode.FLASH_BRIEFING)
    replay = agent.run("Read the DEMO fixture", thread_id=run_id, retry=True)
    assert first["state"]["evidence"] == replay["state"]["evidence"]
    assert replay["requires_approval"]
    print("OFFLINE PASS: injected tool ran; checkpoint replay retained its result; publication awaits approval.")


if __name__ == "__main__":
    main()
