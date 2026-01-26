from market_analyst.schemas import (
    AgentState,
    DraftReport,
    GuardianDecision,
    GuardianResult,
    TradeAction,
)
from market_analyst.workflows.analysis_workflow import (
    ExecutionMode,
    create_graph,
    publish_node,
    route_after_executor,
    route_after_router,
)
from market_analyst.workflows.combined_workflow import (
    create_combined_graph,
    create_trade_from_report_node,
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


def test_publish_node(mocker):
    """Test publish node writes report to file."""
    mock_mkdir = mocker.patch("pathlib.Path.mkdir")
    mock_write = mocker.patch("pathlib.Path.write_text")

    state = mocker.MagicMock(spec=AgentState)
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

    result = publish_node(state)

    assert result["report_approved"] is True
    mock_mkdir.assert_called_once()
    mock_write.assert_called_once()


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
    """Test trade creation from report recommendation."""
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

    # Hack to mock the _trade_amount attribute that is dynamically added
    state._trade_amount = 2000.0

    result = create_trade_from_report_node(state)

    assert result["trade_approved"] is False
    trade = result["pending_trade"]
    assert trade.action == TradeAction.BUY
    assert trade.ticker == "NVDA"
    assert trade.amount_usd == 2000.0


def test_create_combined_graph_smoke(mocker):
    """Smoke test for combined graph."""
    mock_graph_cls = mocker.patch("market_analyst.workflows.combined_workflow.StateGraph")
    mock_graph = mock_graph_cls.return_value
    mock_graph.compile.return_value = "compiled_graph"
    create_combined_graph()
    mock_graph.compile.assert_called()
