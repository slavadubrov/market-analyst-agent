"""Guardian node - automated policy layer for action validation.

The Guardian is a deterministic (non-LLM) node that inspects all
tool call arguments BEFORE execution. It implements three policy layers:

1. Allowlist (Auto-Reject): Block restricted/dangerous actions
2. Thresholds (Escalate): Flag high-stakes actions for human review
3. Safe Path (Auto-Approve): Allow low-risk actions without interruption
"""

from market_analyst.schemas import (
    AgentState,
    GuardianDecision,
    GuardianResult,
    TradeAction,
    TradeRequest,
)

# Policy configuration
RESTRICTED_ACTIONS = {TradeAction.DELETE_PORTFOLIO, TradeAction.DELETE_LOGS}
HIGH_VALUE_THRESHOLD = 10_000  # USD - requires human approval
AUTO_APPROVE_THRESHOLD = 500  # USD - auto-approve without intervention


def check_policies(trade: TradeRequest) -> GuardianResult:
    """Apply policy checks to a trade request.

    Policy 1 (Allowlist): Block restricted actions immediately.
    Policy 2 (Thresholds): Escalate high-value trades to human.
    Policy 3 (Safe Path): Auto-approve low-value trades.
    """
    # Policy 1: Restricted actions (auto-reject)
    if trade.action in RESTRICTED_ACTIONS:
        return GuardianResult(
            decision=GuardianDecision.REJECT,
            policy_name="restricted_action",
            reason=f"Action '{trade.action.value}' is restricted and not allowed.",
            original_request=trade,
        )

    # Policy 2: High-value threshold (escalate to human)
    if trade.amount_usd > HIGH_VALUE_THRESHOLD:
        return GuardianResult(
            decision=GuardianDecision.ESCALATE,
            policy_name="high_value_threshold",
            reason=f"Trade value ${trade.amount_usd:,.2f} exceeds limit of ${HIGH_VALUE_THRESHOLD:,}. Requires human approval.",
            original_request=trade,
        )

    # Policy 3: Safe path (auto-approve)
    if trade.amount_usd <= AUTO_APPROVE_THRESHOLD:
        return GuardianResult(
            decision=GuardianDecision.APPROVE,
            policy_name="safe_path",
            reason=f"Trade value ${trade.amount_usd:,.2f} is within auto-approve limit.",
            original_request=trade,
        )

    # Default: escalate for review (middle ground between thresholds)
    return GuardianResult(
        decision=GuardianDecision.ESCALATE,
        policy_name="default_review",
        reason=f"Trade value ${trade.amount_usd:,.2f} requires review (between ${AUTO_APPROVE_THRESHOLD} and ${HIGH_VALUE_THRESHOLD:,}).",
        original_request=trade,
    )


def guardian_node(state: AgentState) -> dict:
    """Guardian node that validates pending trade requests.

    This is a deterministic node (no LLM calls) that applies
    configured policies to decide whether to:
    - APPROVE: Auto-approve and proceed to execution
    - ESCALATE: Pause for human review (HITL interrupt)
    - REJECT: Block the action with an error message

    Args:
        state: Current agent state with pending_trade

    Returns:
        Updated state with guardian_result
    """
    trade = state.pending_trade

    if trade is None:
        print("  ⚠️  Guardian: No pending trade to evaluate")
        return {}

    # Display the trade being evaluated
    print("\n🛡️  Guardian: Evaluating trade request")
    print(f"   Action: {trade.action.value.upper()}")
    print(f"   Ticker: {trade.ticker}")
    print(f"   Amount: ${trade.amount_usd:,.2f}")
    print(f"   Reason: {trade.reason[:80]}...")

    # Apply policy checks
    result = check_policies(trade)

    # Display decision
    if result.decision == GuardianDecision.REJECT:
        print(f"\n   ❌ REJECTED: {result.reason}")
        print(f"      Policy: {result.policy_name}")
    elif result.decision == GuardianDecision.ESCALATE:
        print(f"\n   ⏸️  ESCALATED: {result.reason}")
        print(f"      Policy: {result.policy_name}")
        print("      → Requires human approval")
    else:  # APPROVE
        print(f"\n   ✅ AUTO-APPROVED: {result.reason}")
        print(f"      Policy: {result.policy_name}")

    return {
        "guardian_result": result,
        "trade_approved": result.decision == GuardianDecision.APPROVE,
    }
