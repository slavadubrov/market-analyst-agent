from pathlib import Path

import pytest

from market_analyst.app import _handle_combined_result
from market_analyst.nodes.rewoo_worker import rewoo_worker_node
from market_analyst.schemas import (
    AgentState,
    DraftReport,
    GuardianDecision,
    GuardianResult,
    ReWOOPlanStep,
    TradeAction,
    UserProfile,
)
from market_analyst.workflows.analysis_workflow import (
    ExecutionMode,
    create_graph,
    publish_node,
    route_after_executor,
    route_after_router,
    run_analysis,
)
from market_analyst.workflows.combined_workflow import (
    create_combined_graph,
    create_trade_from_report_node,
    run_combined_analysis,
)
from market_analyst.workflows.trade_workflow import (
    create_trade_graph,
    route_after_guardian,
)

# --- Analysis Workflow Tests ---


def test_route_after_router(mocker):
    """Test routing based on execution mode."""
    # Test Deep Research
    state = mocker.MagicMock(spec=AgentState)
    state.execution_mode = ExecutionMode.DEEP_RESEARCH
    assert route_after_router(state) == "planner"

    # Test Flash Briefing
    state.execution_mode = ExecutionMode.FLASH_BRIEFING
    assert route_after_router(state) == "rewoo_planner"


def test_route_after_executor(mocker):
    """Test routing based on plan completion."""
    state = mocker.MagicMock(spec=AgentState)

    # More steps needed
    state.error = None
    state.current_step_index = 0
    state.plan = ["step1", "step2"]
    assert route_after_executor(state) == "executor"

    # Plan completed
    state.current_step_index = 2
    state.plan = ["step1", "step2"]
    assert route_after_executor(state) == "reporter"


def test_create_graph_compilation_smoke(mocker):
    """Smoke test for graph creation."""
    mock_graph_cls = mocker.patch("market_analyst.workflows.analysis_workflow.StateGraph")
    mock_graph = mock_graph_cls.return_value
    mock_graph.compile.return_value = "compiled_graph"

    graph = create_graph(checkpointer=None)
    assert graph == "compiled_graph"


def test_run_analysis_raises_workflow_errors(mocker):
    graph = mocker.MagicMock()
    graph.invoke.return_value = {"error": "planning failed"}
    mocker.patch(
        "market_analyst.workflows.analysis_workflow.load_user_profile",
        return_value=UserProfile(),
    )
    mocker.patch("market_analyst.workflows.analysis_workflow.create_graph", return_value=graph)

    with pytest.raises(RuntimeError, match="planning failed"):
        run_analysis("Analyze NVDA", force_mode=ExecutionMode.FLASH_BRIEFING)


def test_publish_node(mocker):
    """Test publish node saves report to document memory and legacy directory."""
    # Mock DocumentMemory to avoid filesystem side effects
    mock_doc_memory = mocker.MagicMock()
    mock_doc_memory.write_doc.return_value = Path("memory/documents/research/test.json")
    mocker.patch(
        "market_analyst.workflows.analysis_workflow.get_document_memory",
        return_value=mock_doc_memory,
    )

    # Mock legacy reports/ directory writes
    mocker.patch("pathlib.Path.mkdir")
    mocker.patch("pathlib.Path.write_text")

    state = mocker.MagicMock(spec=AgentState)
    state.error = None
    state.evaluator_verdict = "pass"
    state.report_approved = True
    state.draft_report = DraftReport(
        title="Test Report",
        ticker="AAPL",
        recommendation="buy",
        confidence=0.9,
        summary="Summary",
        analysis="Analysis",
        risk_factors=["Risk 1"],
    )
    state.execution_mode = ExecutionMode.DEEP_RESEARCH
    state.user_id = "test-user"

    result = publish_node(state)

    assert result["report_approved"] is True
    # Document memory should be called once to save the report
    mock_doc_memory.write_doc.assert_called_once()
    call_kwargs = mock_doc_memory.write_doc.call_args
    assert call_kwargs.kwargs["namespace"] == "research"
    assert "AAPL" in call_kwargs.kwargs["key"]
    # Legacy directory should also be created and written


# --- Trade Workflow Tests ---


def test_route_after_guardian(mocker):
    """Test routing based on guardian decision."""
    state = mocker.MagicMock(spec=AgentState)

    # Approve
    state.guardian_result = GuardianResult(decision=GuardianDecision.APPROVE, reason="Ok", policy_name="safe_trade")
    assert route_after_guardian(state) == "execute"

    # Escalate
    state.guardian_result = GuardianResult(
        decision=GuardianDecision.ESCALATE,
        reason="Check",
        policy_name="high_value_trade",
    )
    assert route_after_guardian(state) == "escalate"

    # Reject
    state.guardian_result = GuardianResult(decision=GuardianDecision.REJECT, reason="Bad", policy_name="dangerous_trade")
    assert route_after_guardian(state) == "end"

    # None (should end)
    state.guardian_result = None
    assert route_after_guardian(state) == "end"


def test_create_trade_graph_smoke(mocker):
    """Smoke test for trade graph."""
    mock_graph_cls = mocker.patch("market_analyst.workflows.trade_workflow.StateGraph")
    mock_graph = mock_graph_cls.return_value
    mock_graph.compile.return_value = "compiled_graph"
    create_trade_graph()
    mock_graph.compile.assert_called()


# --- Combined Workflow Tests ---


def test_create_trade_from_report_node(mocker):
    """Test trade creation from report recommendation uses state.trade_amount."""
    state = mocker.MagicMock(spec=AgentState)
    state.draft_report = DraftReport(
        title="Test Report",
        ticker="NVDA",
        recommendation="strong_buy",
        confidence=0.8,
        summary="Buy now",
        analysis="...",
        risk_factors=[],
    )
    state.trade_amount = 2000.0

    result = create_trade_from_report_node(state)

    assert result["trade_approved"] is False
    trade = result["pending_trade"]
    assert trade.action == TradeAction.BUY
    assert trade.ticker == "NVDA"
    assert trade.amount_usd == 2000.0


def test_create_trade_from_report_node_defaults_when_unset(mocker):
    """When state.trade_amount is None, the bridge falls back to the default."""
    from market_analyst.workflows.combined_workflow import DEFAULT_TRADE_AMOUNT

    state = mocker.MagicMock(spec=AgentState)
    state.draft_report = DraftReport(
        title="Test Report",
        ticker="MSFT",
        recommendation="buy",
        confidence=0.7,
        summary="Buy",
        analysis="...",
        risk_factors=[],
    )
    state.trade_amount = None

    result = create_trade_from_report_node(state)

    assert result["pending_trade"].amount_usd == DEFAULT_TRADE_AMOUNT


def test_create_combined_graph_smoke(mocker):
    """Smoke test for combined graph."""
    mock_graph_cls = mocker.patch("market_analyst.workflows.combined_workflow.StateGraph")
    mock_graph = mock_graph_cls.return_value
    mock_graph.compile.return_value = "compiled_graph"
    create_combined_graph()
    mock_graph.compile.assert_called()


def test_run_combined_analysis_raises_workflow_errors(mocker):
    graph = mocker.MagicMock()
    graph.invoke.return_value = {"error": "planning failed"}
    mocker.patch(
        "market_analyst.workflows.combined_workflow.load_user_profile",
        return_value=UserProfile(),
    )
    mocker.patch("market_analyst.workflows.combined_workflow.create_combined_graph", return_value=graph)

    with pytest.raises(RuntimeError, match="planning failed"):
        run_combined_analysis(
            "Analyze NVDA",
            checkpointer=mocker.MagicMock(),
            force_mode=ExecutionMode.FLASH_BRIEFING,
        )


def test_rewoo_worker_respects_dependency_chains(mocker):
    """A dependent step never runs before its transitive dependency."""
    calls = []

    def fake_execute(step, results):
        calls.append((step.step_id, set(results)))
        return step.step_id

    mocker.patch("market_analyst.nodes.rewoo_worker.execute_tool", side_effect=fake_execute)
    state = AgentState(
        rewoo_plan=[
            ReWOOPlanStep(step_id="#E1", description="one", tool_name="get_stock_snapshot"),
            ReWOOPlanStep(step_id="#E2", description="two", tool_name="get_stock_snapshot", depends_on=["#E1"]),
            ReWOOPlanStep(step_id="#E3", description="three", tool_name="get_stock_snapshot", depends_on=["#E2"]),
        ]
    )

    result = rewoo_worker_node(state)

    assert calls == [("#E1", set()), ("#E2", {"#E1"}), ("#E3", {"#E1", "#E2"})]
    assert [step.result for step in result["rewoo_plan"]] == ["#E1", "#E2", "#E3"]


def test_combined_ui_returns_trade_info_as_its_own_output():
    """Trade details target the textbox, not the approval group."""
    outputs = _handle_combined_result(
        {
            "requires_trade_approval": True,
            "guardian_result": GuardianResult(
                decision=GuardianDecision.ESCALATE,
                policy_name="default_review",
                reason="Needs approval",
            ),
        },
        "started",
        "thread-1",
    )

    assert len(outputs) == 6
    assert outputs[4] == "Policy: default_review\nReason: Needs approval"
