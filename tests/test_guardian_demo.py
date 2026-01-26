"""Guardian Demo Tests.

These tests demonstrate the Guardian policy layer in action:
1. Low-value trades are auto-approved (safe path)
2. High-value trades are escalated for human review
3. Restricted actions are auto-rejected

Run with: uv run pytest tests/test_guardian_demo.py -v
"""

import pytest

from market_analyst.nodes.guardian import (
    AUTO_APPROVE_THRESHOLD,
    HIGH_VALUE_THRESHOLD,
    check_policies,
)
from market_analyst.schemas import (
    AgentState,
    GuardianDecision,
    TradeAction,
    TradeRequest,
)
from market_analyst.trade_workflow import create_trade_graph


class TestGuardianPolicies:
    """Test the Guardian policy layer directly."""

    def test_low_value_trade_auto_approved(self):
        """Trades under $500 should be auto-approved (safe path)."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="NVDA",
            amount_usd=300.00,
            reason="Test low-value trade",
        )

        result = check_policies(trade)

        assert result.decision == GuardianDecision.APPROVE
        assert result.policy_name == "safe_path"
        print(f"\n✅ Low-value trade ($300): {result.decision.value}")
        print(f"   Policy: {result.policy_name}")
        print(f"   Reason: {result.reason}")

    def test_medium_value_trade_escalated(self):
        """Trades between $500 and $10,000 should be escalated for review."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="AAPL",
            amount_usd=5000.00,
            reason="Test medium-value trade",
        )

        result = check_policies(trade)

        assert result.decision == GuardianDecision.ESCALATE
        assert result.policy_name == "default_review"
        print(f"\n⏸️  Medium-value trade ($5,000): {result.decision.value}")
        print(f"   Policy: {result.policy_name}")
        print(f"   Reason: {result.reason}")

    def test_high_value_trade_escalated(self):
        """Trades over $10,000 should be escalated with high_value_threshold."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="TSLA",
            amount_usd=50000.00,
            reason="Test high-value trade",
        )

        result = check_policies(trade)

        assert result.decision == GuardianDecision.ESCALATE
        assert result.policy_name == "high_value_threshold"
        print(f"\n⏸️  High-value trade ($50,000): {result.decision.value}")
        print(f"   Policy: {result.policy_name}")
        print(f"   Reason: {result.reason}")

    def test_delete_portfolio_rejected(self):
        """delete_portfolio action should be auto-rejected."""
        trade = TradeRequest(
            action=TradeAction.DELETE_PORTFOLIO,
            ticker="NVDA",
            amount_usd=0,
            reason="Agent attempting to delete portfolio",
        )

        result = check_policies(trade)

        assert result.decision == GuardianDecision.REJECT
        assert result.policy_name == "restricted_action"
        print(f"\n❌ Delete portfolio: {result.decision.value}")
        print(f"   Policy: {result.policy_name}")
        print(f"   Reason: {result.reason}")

    def test_delete_logs_rejected(self):
        """delete_logs action should be auto-rejected."""
        trade = TradeRequest(
            action=TradeAction.DELETE_LOGS,
            ticker="NVDA",
            amount_usd=0,
            reason="Agent attempting to delete logs",
        )

        result = check_policies(trade)

        assert result.decision == GuardianDecision.REJECT
        assert result.policy_name == "restricted_action"
        print(f"\n❌ Delete logs: {result.decision.value}")
        print(f"   Policy: {result.policy_name}")
        print(f"   Reason: {result.reason}")


class TestGuardianWorkflow:
    """Test the complete Guardian workflow graph."""

    def test_auto_approve_workflow(self):
        """Low-value trade should flow through to execution."""
        # Create graph without checkpointer (no persistence)
        graph = create_trade_graph(checkpointer=None)

        # Create initial state with low-value trade
        initial_state = AgentState(
            pending_trade=TradeRequest(
                action=TradeAction.BUY,
                ticker="NVDA",
                amount_usd=300.00,
                reason="Low-value test",
            )
        )

        # Run the graph
        result = graph.invoke(initial_state, {"configurable": {"thread_id": "test-1"}})

        # Should have executed
        assert result.get("trade_executed") is True
        assert result.get("trade_approved") is True
        print("\n✅ Workflow test: Low-value trade executed successfully")

    def test_reject_workflow(self):
        """Restricted action should be rejected without execution."""
        graph = create_trade_graph(checkpointer=None)

        initial_state = AgentState(
            pending_trade=TradeRequest(
                action=TradeAction.DELETE_LOGS,
                ticker="NVDA",
                amount_usd=0,
                reason="Malicious attempt",
            )
        )

        result = graph.invoke(initial_state, {"configurable": {"thread_id": "test-2"}})

        # Should NOT have executed
        assert result.get("trade_executed") is not True
        assert result.get("guardian_result").decision == GuardianDecision.REJECT
        print("\n❌ Workflow test: Restricted action blocked successfully")


class TestPolicyThresholds:
    """Test boundary conditions of policy thresholds."""

    def test_exactly_at_auto_approve_threshold(self):
        """Trade at exactly $500 should be auto-approved."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="NVDA",
            amount_usd=AUTO_APPROVE_THRESHOLD,  # $500
            reason="Boundary test",
        )

        result = check_policies(trade)
        assert result.decision == GuardianDecision.APPROVE
        print(f"\n✅ Exactly ${AUTO_APPROVE_THRESHOLD}: auto-approved")

    def test_just_above_auto_approve_threshold(self):
        """Trade at $501 should be escalated."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="NVDA",
            amount_usd=501.00,
            reason="Boundary test",
        )

        result = check_policies(trade)
        assert result.decision == GuardianDecision.ESCALATE
        print(f"\n⏸️  $501: escalated for review")

    def test_exactly_at_high_value_threshold(self):
        """Trade at exactly $10,000 should be escalated (default_review)."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="NVDA",
            amount_usd=HIGH_VALUE_THRESHOLD,  # $10,000
            reason="Boundary test",
        )

        result = check_policies(trade)
        assert result.decision == GuardianDecision.ESCALATE
        # At exactly $10,000, it's <= so goes to default_review
        print(f"\n⏸️  Exactly ${HIGH_VALUE_THRESHOLD:,}: escalated (default_review)")

    def test_just_above_high_value_threshold(self):
        """Trade at $10,001 should trigger high_value_threshold policy."""
        trade = TradeRequest(
            action=TradeAction.BUY,
            ticker="NVDA",
            amount_usd=10001.00,
            reason="Boundary test",
        )

        result = check_policies(trade)
        assert result.decision == GuardianDecision.ESCALATE
        assert result.policy_name == "high_value_threshold"
        print(f"\n⏸️  $10,001: escalated (high_value_threshold)")


if __name__ == "__main__":
    # Run as standalone demo
    print("=" * 60)
    print("🛡️  GUARDIAN DEMO")
    print("=" * 60)

    print("\n--- Testing Policy Decisions ---")
    tests = TestGuardianPolicies()
    tests.test_low_value_trade_auto_approved()
    tests.test_medium_value_trade_escalated()
    tests.test_high_value_trade_escalated()
    tests.test_delete_portfolio_rejected()
    tests.test_delete_logs_rejected()

    print("\n--- Testing Workflow Integration ---")
    workflow_tests = TestGuardianWorkflow()
    workflow_tests.test_auto_approve_workflow()
    workflow_tests.test_reject_workflow()

    print("\n" + "=" * 60)
    print("✅ All Guardian demos completed!")
    print("=" * 60)
