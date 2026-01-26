"""Command-line interface for Market Analyst Agent.

This CLI provides an interactive way to run the agent,
demonstrating all the key features from the blog series.
"""

import argparse
import os
import sys
import traceback
import uuid

from dotenv import load_dotenv

from market_analyst.constants import DEFAULT_MODEL_KEY, MODEL_ENV_VAR, MODEL_MAP
from market_analyst.memory.checkpointer import get_checkpointer
from market_analyst.memory.profile import get_profile_store
from market_analyst.nodes.reporter import format_report_for_display
from market_analyst.schemas import ExecutionMode
from market_analyst.workflows.analysis_workflow import (
    approve_and_publish,
    create_graph,
    run_analysis,
)
from market_analyst.workflows.combined_workflow import (
    approve_combined_report,
    approve_combined_trade,
    run_combined_analysis,
)
from market_analyst.workflows.trade_workflow import approve_trade, run_trade


def main():
    """Main entry point for the CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Market Analyst Agent - Institutional Investment Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a simple analysis
  market-analyst "Analyze NVDA stock"
  
  # Set user profile first
  market-analyst --set-profile --risk-tolerance conservative
  
  # Resume a previous analysis
  market-analyst --resume --thread-id abc123
  
  # Show help for profile settings
  market-analyst --set-profile --help
        """,
    )

    # Main arguments
    parser.add_argument(
        "query", nargs="?", help="Stock analysis query (e.g., 'Analyze NVDA')"
    )
    parser.add_argument("--thread-id", help="Thread ID for resuming a conversation")
    parser.add_argument(
        "--user-id", default="default", help="User ID for profile management"
    )

    # Profile management
    parser.add_argument(
        "--set-profile", action="store_true", help="Set user profile preferences"
    )
    parser.add_argument(
        "--risk-tolerance",
        choices=["conservative", "moderate", "aggressive"],
        help="Risk tolerance level",
    )
    parser.add_argument(
        "--horizon", choices=["short", "medium", "long"], help="Investment time horizon"
    )

    # Execution modes
    parser.add_argument(
        "--resume", action="store_true", help="Resume a paused analysis"
    )
    parser.add_argument(
        "--approve", action="store_true", help="Approve the draft report and publish"
    )
    parser.add_argument(
        "--no-persist", action="store_true", help="Run without database persistence"
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_MAP.keys()),
        default=DEFAULT_MODEL_KEY,
        help="Model to use: 'sonnet' (powerful, slower) or 'haiku' (fast, cheaper)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "deep", "flash"],
        default="auto",
        help="Execution mode: 'auto' (router decides), 'deep' (ReAct - thorough), 'flash' (ReWOO - fast)",
    )

    # Combined workflow (full demo)
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Run combined Analysis → Guardian → Trade workflow (full demo)",
    )
    parser.add_argument(
        "--trade-amount",
        type=float,
        default=1000.0,
        help="Trade amount in USD for combined workflow (default: $1000)",
    )

    # Trade commands (Guardian demo)
    parser.add_argument(
        "--trade",
        action="store_true",
        help="Execute a trade (demonstrates Guardian policy layer)",
    )
    parser.add_argument(
        "--action",
        choices=["buy", "sell", "delete_portfolio", "delete_logs"],
        help="Trade action type",
    )
    parser.add_argument(
        "--ticker",
        help="Stock ticker for trade (e.g., NVDA)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        help="Trade amount in USD",
    )
    parser.add_argument(
        "--reason",
        default="Agent recommendation based on analysis",
        help="Reason for the trade",
    )
    parser.add_argument(
        "--approve-trade",
        action="store_true",
        help="Approve a pending trade (after Guardian escalation)",
    )
    parser.add_argument(
        "--reject-trade",
        action="store_true",
        help="Reject a pending trade",
    )
    parser.add_argument(
        "--modify-amount",
        type=float,
        help="Modify trade amount when approving",
    )

    # Debugging
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Show the research plan before execution",
    )

    args = parser.parse_args()

    # Check for required env vars
    if not args.set_profile:
        if not os.getenv("ANTHROPIC_API_KEY"):
            print("❌ Error: ANTHROPIC_API_KEY environment variable not set")
            print("   Copy .env.example to .env and add your API key")
            sys.exit(1)

    # Handle profile setting
    if args.set_profile:
        set_user_profile(args)
        return

    # Handle approval
    if args.approve:
        if not args.thread_id:
            print("❌ Error: --thread-id required with --approve")
            sys.exit(1)
        approve_report(args)
        return

    # Handle trade approval
    if args.approve_trade or args.reject_trade:
        if not args.thread_id:
            print("❌ Error: --thread-id required with --approve-trade/--reject-trade")
            sys.exit(1)
        handle_trade_approval(args)
        return

    # Handle trade execution
    if args.trade:
        if not args.action or not args.ticker or args.amount is None:
            print("❌ Error: --trade requires --action, --ticker, and --amount")
            print("   Example: --trade --action buy --ticker NVDA --amount 5000")
            sys.exit(1)
        run_trade_command(args)
        return

    # Handle resume
    if args.resume:
        if not args.thread_id:
            print("❌ Error: --thread-id required with --resume")
            sys.exit(1)
        resume_analysis(args)
        return

    # Normal analysis
    if not args.query:
        parser.print_help()
        sys.exit(1)

    # Handle combined workflow
    if args.combined:
        run_combined_command(args)
        return

    run_new_analysis(args)


def set_user_profile(args):
    """Set user profile preferences in Redis."""

    try:
        store = get_profile_store()

        # Get existing profile or create new
        profile = store.get_profile(args.user_id)

        if args.risk_tolerance:
            profile.risk_tolerance = args.risk_tolerance
        if args.horizon:
            profile.investment_horizon = args.horizon

        store.save_profile(args.user_id, profile)

        print(f"✅ Profile updated for user: {args.user_id}")
        print(f"   Risk Tolerance: {profile.risk_tolerance}")
        print(f"   Investment Horizon: {profile.investment_horizon}")

    except Exception as e:
        print(f"⚠️  Could not save to Redis (might not be running): {e}")
        print("   Profile will use defaults")


def run_new_analysis(args):
    """Run a new stock analysis."""

    thread_id = args.thread_id or str(uuid.uuid4())

    print(f"\n🔬 Starting analysis: {args.query}")
    print(f"   Thread ID: {thread_id}")
    print(f"   User ID: {args.user_id}")
    print("-" * 60)

    # Get checkpointer if persistence is enabled
    checkpointer = None
    if not args.no_persist:
        try:
            checkpointer = get_checkpointer()
            print("   ✅ PostgreSQL checkpointing enabled")
        except Exception as e:
            print(f"   ⚠️  PostgreSQL not available: {e}")
            print("   Continuing without persistence...")

    # Set model selection for nodes to use
    os.environ[MODEL_ENV_VAR] = MODEL_MAP[args.model]
    print(f"   🤖 Using model: {args.model}")

    # Map CLI mode arg to ExecutionMode
    force_mode = None
    if args.mode == "deep":
        force_mode = ExecutionMode.DEEP_RESEARCH
        print("   📊 Mode: Deep Research (ReAct) - forced")
    elif args.mode == "flash":
        force_mode = ExecutionMode.FLASH_BRIEFING
        print("   ⚡ Mode: Flash Briefing (ReWOO) - forced")
    else:
        print("   🔀 Mode: Auto (router will classify intent)")

    try:
        result = run_analysis(
            query=args.query,
            user_id=args.user_id,
            thread_id=thread_id,
            checkpointer=checkpointer,
            force_mode=force_mode,
        )

        if args.show_plan:
            print("\n📋 Research Plan:")
            for step in result["state"].get("plan", []):
                status = "✅" if step.completed else "⏳"
                print(f"   {status} Step {step.step_number}: {step.description}")

        if result.get("requires_approval"):
            print("\n" + "=" * 60)
            print("⏸️  PAUSED - Awaiting your approval")
            print("=" * 60)

            if result.get("draft_report"):
                print(format_report_for_display(result["draft_report"]))

            if args.no_persist:
                # Can't approve without persistence - state is not saved
                print("\n⚠️  Running with --no-persist: approval workflow disabled")
                print("   (Run without --no-persist to enable save/approve workflow)")
                print("\n✅ Analysis complete (auto-approved in no-persist mode)")
            else:
                print("\nTo approve and publish:")
                print(f"  uv run market-analyst --approve --thread-id {thread_id}")
                print("\nTo edit and approve:")
                print(
                    f"  uv run market-analyst --approve --thread-id {thread_id} --edit-recommendation hold"
                )
        else:
            print("\n✅ Analysis complete!")
            if result.get("draft_report"):
                print(format_report_for_display(result["draft_report"]))

    except KeyboardInterrupt:
        print("\n\n⏸️  Analysis interrupted. Resume with:")
        print(f"   uv run market-analyst --resume --thread-id {thread_id}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def resume_analysis(args):
    """Resume a paused analysis."""

    print(f"\n🔄 Resuming analysis: {args.thread_id}")

    try:
        checkpointer = get_checkpointer()
        graph = create_graph(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": args.thread_id}}

        # Get current state
        state = graph.get_state(config)

        if not state:
            print(f"❌ No saved state found for thread {args.thread_id}")
            sys.exit(1)

        print(f"   Found state at step {state.values.get('current_step_index', 0)}")

        # Resume execution
        result = graph.invoke(None, config)

        if result.get("draft_report"):
            print(format_report_for_display(result["draft_report"]))

    except Exception as e:
        print(f"❌ Error resuming: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def approve_report(args):
    """Approve and publish a draft report."""

    print(f"\n✅ Approving report for thread: {args.thread_id}")

    try:
        checkpointer = get_checkpointer()

        result = approve_and_publish(
            thread_id=args.thread_id,
            checkpointer=checkpointer,
        )

        if result.get("published"):
            print("\n🎉 Report published successfully!")

            # Display the final report
            state = result.get("state", {})
            # Handle both dict and object access
            if hasattr(state, "draft_report"):
                draft_report = state.draft_report
            elif isinstance(state, dict):
                draft_report = state.get("draft_report")
            else:
                draft_report = None

            if draft_report:
                print(format_report_for_display(draft_report))
            else:
                print("\n(No report content to display)")
        else:
            print("\n⚠️  Report could not be published")

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def run_trade_command(args):
    """Execute a trade through the Guardian workflow."""

    print("\n💼 Executing trade via Guardian...")
    print(f"   Action: {args.action.upper()}")
    print(f"   Ticker: {args.ticker}")
    print(f"   Amount: ${args.amount:,.2f}")
    print("-" * 60)

    # Get checkpointer if persistence is enabled
    checkpointer = None
    if not args.no_persist:
        try:
            checkpointer = get_checkpointer()
            print("   ✅ PostgreSQL checkpointing enabled")
        except Exception as e:
            print(f"   ⚠️  PostgreSQL not available: {e}")
            print("   Continuing without persistence...")

    try:
        result = run_trade(
            action=args.action,
            ticker=args.ticker,
            amount_usd=args.amount,
            reason=args.reason,
            checkpointer=checkpointer,
        )

        if result.get("error"):
            print(f"\n❌ Error: {result['error']}")
            sys.exit(1)

        if result.get("executed"):
            print("\n🎉 Trade executed successfully!")
        elif result.get("requires_approval"):
            print("\n" + "=" * 60)
            print("⏸️  TRADE PAUSED - Awaiting human approval")
            print("=" * 60)

            guardian_result = result.get("guardian_result")
            if guardian_result:
                print(f"\n   Policy: {guardian_result.policy_name}")
                print(f"   Reason: {guardian_result.reason}")

            thread_id = result["thread_id"]
            if args.no_persist:
                print("\n⚠️  Running with --no-persist: approval workflow disabled")
            else:
                print("\nTo approve this trade:")
                print(
                    f"  uv run market-analyst --approve-trade --thread-id {thread_id}"
                )
                print("\nTo approve with modified amount:")
                print(
                    f"  uv run market-analyst --approve-trade --thread-id {thread_id} --modify-amount 9000"
                )
                print("\nTo reject this trade:")
                print(f"  uv run market-analyst --reject-trade --thread-id {thread_id}")
        else:
            # Trade was rejected by Guardian
            guardian_result = result.get("guardian_result")
            if guardian_result:
                print(f"\n❌ Trade blocked: {guardian_result.reason}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def handle_trade_approval(args):
    """Handle trade approval or rejection."""

    action = "Approving" if args.approve_trade else "Rejecting"
    print(f"\n{action} trade for thread: {args.thread_id}")

    try:
        checkpointer = get_checkpointer()

        result = approve_trade(
            thread_id=args.thread_id,
            checkpointer=checkpointer,
            approve=args.approve_trade,
            modified_amount=args.modify_amount,
        )

        if result.get("rejected"):
            print("\n❌ Trade rejected by reviewer")
        elif result.get("executed"):
            print("\n🎉 Trade approved and executed!")
        else:
            print("\n⚠️  Trade could not be processed")

    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def run_combined_command(args):
    """Run the combined Analysis → Guardian → Trade workflow."""

    thread_id = args.thread_id or str(uuid.uuid4())

    print("\n🔬 Starting combined Analysis → Guardian → Trade workflow")
    print(f"   Query: {args.query}")
    print(f"   Trade Amount: ${args.trade_amount:,.2f}")
    print(f"   Thread ID: {thread_id}")
    print(f"   User ID: {args.user_id}")
    print("-" * 60)

    # Get checkpointer if persistence is enabled
    checkpointer = None
    if not args.no_persist:
        try:
            checkpointer = get_checkpointer()
            print("   ✅ PostgreSQL checkpointing enabled")
        except Exception as e:
            print(f"   ⚠️  PostgreSQL not available: {e}")
            print("   Continuing without persistence...")

    # Set model selection
    os.environ[MODEL_ENV_VAR] = MODEL_MAP[args.model]
    print(f"   🤖 Using model: {args.model}")

    # Map CLI mode arg to ExecutionMode
    force_mode = None
    if args.mode == "deep":
        force_mode = ExecutionMode.DEEP_RESEARCH
        print("   📊 Mode: Deep Research (ReAct) - forced")
    elif args.mode == "flash":
        force_mode = ExecutionMode.FLASH_BRIEFING
        print("   ⚡ Mode: Flash Briefing (ReWOO) - forced")
    else:
        print("   🔀 Mode: Auto (router will classify intent)")

    try:
        result = run_combined_analysis(
            query=args.query,
            user_id=args.user_id,
            thread_id=thread_id,
            checkpointer=checkpointer,
            force_mode=force_mode,
            trade_amount=args.trade_amount,
        )

        # Check which approval point we stopped at
        if result.get("requires_report_approval"):
            print("\n" + "=" * 60)
            print("⏸️  PAUSED - Awaiting report approval")
            print("=" * 60)

            if result.get("draft_report"):
                print(format_report_for_display(result["draft_report"]))

            if args.no_persist:
                print("\n⚠️  Running with --no-persist: approval workflow disabled")
                print("   (Run without --no-persist to enable approval workflow)")
            else:
                print("\nTo approve the report and continue to trade:")
                print(
                    f"  uv run market-analyst --approve --combined --thread-id {thread_id}"
                )

        elif result.get("requires_trade_approval"):
            print("\n" + "=" * 60)
            print("⏸️  TRADE PAUSED - Awaiting human approval")
            print("=" * 60)

            guardian_result = result.get("guardian_result")
            if guardian_result:
                print(f"\n   Policy: {guardian_result.policy_name}")
                print(f"   Reason: {guardian_result.reason}")

            if args.no_persist:
                print("\n⚠️  Running with --no-persist: approval workflow disabled")
            else:
                print("\nTo approve this trade:")
                print(
                    f"  uv run market-analyst --approve-trade --thread-id {thread_id}"
                )
                print("\nTo reject this trade:")
                print(f"  uv run market-analyst --reject-trade --thread-id {thread_id}")

        elif result.get("trade_executed"):
            print("\n" + "=" * 60)
            print("🎉 Combined workflow complete!")
            print("=" * 60)
            print("   ✅ Report published")
            print("   ✅ Trade executed")

        else:
            # Either no trade (hold recommendation) or trade rejected
            guardian_result = result.get("guardian_result")
            if guardian_result and guardian_result.decision.value == "reject":
                print(f"\n❌ Trade blocked by Guardian: {guardian_result.reason}")
            else:
                print("\n✅ Analysis complete (no trade action - hold recommendation)")

            if result.get("draft_report"):
                print(format_report_for_display(result["draft_report"]))

    except KeyboardInterrupt:
        print("\n\n⏸️  Workflow interrupted. Resume with:")
        print(f"   uv run market-analyst --resume --thread-id {thread_id}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
