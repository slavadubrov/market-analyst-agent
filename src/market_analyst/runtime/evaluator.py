"""Fresh-context evaluator subagent ("default-FAIL").

Per the article's failure-modes table (row 1, "Premature completion"): a
generator/evaluator split with a fresh-context evaluator that reads files
(not chat) and votes "done" or "not done." Default-FAIL on every acceptance
check.

The evaluator is intentionally a separate LangChain ``ChatAnthropic`` call
with its own conversation: no shared message history with the generator, no
write tools, a deterministic verdict schema. Anthropic's
``cwc-long-running-agents`` quick-start ships exactly this pattern.

We expose two surfaces:

- :func:`evaluate_draft_report` -- the function the workflow calls when the
  reporter produces a ``DraftReport``.
- :func:`evaluator_node` -- a LangGraph node form for graphs that want to
  splice the evaluator inline between ``reporter`` and ``publish``.

The verdict is one of ``pass`` / ``fail`` / ``needs_human``. Anything that
isn't a clear ``pass`` should default to ``needs_human`` so the HITL gate
stays meaningful.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from market_analyst.constants import DEFAULT_MODEL, MODEL_ENV_VAR
from market_analyst.nodes._telemetry import node_callbacks
from market_analyst.schemas import AgentState, DraftReport

EVALUATOR_SYSTEM_PROMPT = """You are a fresh, skeptical investment review analyst.

You were NOT involved in writing the draft below. Your job is to read it and
return a verdict on whether it is ready to publish.

Rules:
- Default to FAIL when in doubt. A draft that "could be improved" is FAIL.
- If the draft is well-supported, internally consistent, and ready to publish,
  return PASS.
- If you cannot make a confident call without a human (e.g. tone is off,
  recommendation conflicts with the analysis), return NEEDS_HUMAN.

Acceptance criteria:
1. The summary is two to three sentences and matches the recommendation.
2. The analysis cites at least one concrete data point.
3. Risk factors are non-empty and specific (not generic boilerplate).
4. The recommendation is consistent with the analysis.

Return your verdict, with one short sentence per reason.
"""


class EvaluatorVerdict(BaseModel):
    """Structured output for the evaluator subagent."""

    verdict: Literal["pass", "fail", "needs_human"] = Field(
        description="The evaluator's decision. Default to fail or needs_human when in doubt."
    )
    reasons: list[str] = Field(
        description="One-line reasons backing the verdict (1-4 items).",
        max_length=4,
    )


def _build_evaluator_input(report: DraftReport) -> str:
    return f"""Draft Investment Report (read-only, fresh context):

Ticker: {report.ticker}
Title: {report.title}
Recommendation: {report.recommendation}
Confidence: {report.confidence:.0%}

Summary:
{report.summary}

Analysis:
{report.analysis}

Risk Factors:
{chr(10).join(f"- {r}" for r in report.risk_factors) if report.risk_factors else "(none provided)"}
"""


def evaluate_draft_report(
    report: DraftReport,
    *,
    conversation_id: str | None = None,
    model_name: str | None = None,
) -> EvaluatorVerdict:
    """Run the evaluator subagent against a draft report.

    Args:
        report: The draft to evaluate.
        conversation_id: ``thread_id`` for span correlation. Optional.
        model_name: Override the model. Defaults to the same model the rest of
            the pipeline uses (via ``MARKET_ANALYST_MODEL``).

    Returns:
        Structured verdict. Callers typically promote ``fail`` to a re-run or
        ``needs_human`` to a HITL interrupt; ``pass`` clears the gate.
    """
    model = model_name or os.getenv(MODEL_ENV_VAR, DEFAULT_MODEL)
    llm = ChatAnthropic(model=model, temperature=0)
    structured = llm.with_structured_output(EvaluatorVerdict)

    return structured.invoke(
        [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=_build_evaluator_input(report)),
        ],
        config={
            "callbacks": node_callbacks(
                node_name="evaluator",
                config={"configurable": {"thread_id": conversation_id}} if conversation_id else None,
            )
        },
    )


def evaluator_node(
    state: AgentState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """LangGraph-compatible node form of the evaluator.

    Returns:
        ``{"evaluator_verdict": ..., "evaluator_reasons": [...]}``. Workflows
        that want to interrupt on a failed verdict should branch on
        ``state.evaluator_verdict`` after this node.
    """
    if state.draft_report is None:
        return {"evaluator_verdict": "fail", "evaluator_reasons": ["No draft report to evaluate"]}

    try:
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        verdict = evaluate_draft_report(state.draft_report, conversation_id=thread_id)
        print(f"\n🧪 Evaluator verdict: {verdict.verdict.upper()}")
        for reason in verdict.reasons:
            print(f"   - {reason}")
        return {
            "evaluator_verdict": verdict.verdict,
            "evaluator_reasons": list(verdict.reasons),
        }
    except Exception as exc:
        # Default-FAIL: if the evaluator itself errors, treat the draft as
        # needs_human rather than silently passing.
        print(f"\n⚠️  Evaluator failed; defaulting to needs_human: {exc}")
        return {
            "evaluator_verdict": "needs_human",
            "evaluator_reasons": [f"Evaluator error: {exc}"],
        }
