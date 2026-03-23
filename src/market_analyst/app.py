"""Gradio UI for Market Analyst Agent.

This file provides a web interface for the Market Analyst Agent,
offering similar functionality to the CLI but in a browser-based environment.
"""

import os
import uuid

import gradio as gr
from dotenv import load_dotenv

from market_analyst.constants import DEFAULT_MODEL_KEY, MODEL_ENV_VAR, MODEL_MAP
from market_analyst.memory import get_checkpointer, get_long_term_memory
from market_analyst.nodes.reporter import format_report_for_display
from market_analyst.schemas import ExecutionMode
from market_analyst.utils import get_state_attr
from market_analyst.workflows.analysis_workflow import (
    approve_and_publish,
    run_analysis,
)
from market_analyst.workflows.combined_workflow import (
    approve_combined_report,
    run_combined_analysis,
)
from market_analyst.workflows.trade_workflow import approve_trade, run_trade

# Load environment variables
load_dotenv()


def set_profile(user_id: str, risk_tolerance: str | None, horizon: str | None) -> str:
    """Set user profile preferences in Qdrant.

    Args:
        user_id: Unique user identifier.
        risk_tolerance: Risk appetite level (conservative/moderate/aggressive).
        horizon: Investment time horizon (short/medium/long).

    Returns:
        Status message confirming update or describing error.
    """
    try:
        store = get_long_term_memory()
        profile = store.get_profile(user_id)

        if risk_tolerance:
            profile.risk_tolerance = risk_tolerance
        if horizon:
            profile.investment_horizon = horizon

        store.save_profile(user_id, profile)
        return f"✅ Profile updated for user: {user_id}\nRisk Tolerance: {profile.risk_tolerance}\nInvestment Horizon: {profile.investment_horizon}"
    except Exception as e:
        return f"⚠️ Could not save to Qdrant: {str(e)}"


def format_report_markdown(report: dict | None) -> str:
    """Format report for Markdown display.

    Args:
        report: Report dictionary from analysis workflow.

    Returns:
        Formatted markdown string or placeholder if no report.
    """
    if not report:
        return "No report available."
    return format_report_for_display(report)


def run_analysis_ui(
    query: str,
    user_id: str,
    model: str,
    mode: str,
    thread_id_input: str,
    resume_thread_id: str,
) -> tuple:
    """Run analysis workflow from the UI.

    Args:
        query: Stock analysis query (e.g., 'Analyze NVDA').
        user_id: User identifier for profile lookup.
        model: Model selection key (e.g., 'sonnet', 'haiku').
        mode: Execution mode ('auto', 'deep', 'flash').
        thread_id_input: Optional thread ID for resuming.
        resume_thread_id: Alternative thread ID for resuming.

    Returns:
        Tuple of (report_markdown, status_message, thread_id, approval_group_visibility).
    """

    # Use provided thread ID or generate new ones
    if resume_thread_id:
        thread_id = resume_thread_id
    elif thread_id_input:
        thread_id = thread_id_input
    else:
        thread_id = str(uuid.uuid4())

    # Set model
    os.environ[MODEL_ENV_VAR] = MODEL_MAP[model]

    # Determine mode
    force_mode = None
    if mode == "deep":
        force_mode = ExecutionMode.DEEP_RESEARCH
    elif mode == "flash":
        force_mode = ExecutionMode.FLASH_BRIEFING

    status_msg = f"Started analysis with Thread ID: {thread_id}"

    try:
        checkpointer = get_checkpointer()
    except Exception:
        checkpointer = None
        status_msg += "\n(Running without persistence)"

    try:
        result = run_analysis(
            query=query,
            user_id=user_id,
            thread_id=thread_id,
            checkpointer=checkpointer,
            force_mode=force_mode,
        )

        report_md = ""
        if result.get("draft_report"):
            report_md = format_report_for_display(result["draft_report"])

        if result.get("requires_approval"):
            status_msg += "\n\n⏸️ PAUSED - Awaiting approval."
            return report_md, status_msg, thread_id, gr.update(visible=True)
        else:
            status_msg += "\n\n✅ Analysis complete!"
            return report_md, status_msg, thread_id, gr.update(visible=False)

    except Exception as e:
        return f"❌ Error: {str(e)}", status_msg, thread_id, gr.update(visible=False)


def approve_report_ui(thread_id):
    """Approve report workflow."""
    if not thread_id:
        return "❌ Error: No Thread ID provided.", ""

    try:
        checkpointer = get_checkpointer()
        result = approve_and_publish(
            thread_id=thread_id,
            checkpointer=checkpointer,
        )

        if result.get("published"):
            state = result.get("state", {})
            draft_report = get_state_attr(state, "draft_report")
            report_md = format_report_for_display(draft_report) if draft_report else "No report content."
            return f"🎉 Report published successfully!\n\n{report_md}", "Approved"
        else:
            return "⚠️ Report could not be published", "Failed"

    except Exception as e:
        return f"❌ Error: {str(e)}", "Error"


def run_trade_ui(action, ticker, amount, reason):
    """Run trade workflow."""
    try:
        checkpointer = get_checkpointer()
    except Exception:
        checkpointer = None

    try:
        result = run_trade(
            action=action.lower(),
            ticker=ticker,
            amount_usd=float(amount),
            reason=reason,
            checkpointer=checkpointer,
        )

        if result.get("error"):
            return f"❌ Error: {result['error']}", "", gr.update(visible=False)

        if result.get("executed"):
            return "🎉 Trade executed successfully!", "", gr.update(visible=False)
        elif result.get("requires_approval"):
            guardian_result = result.get("guardian_result")
            msg = "⏸️ TRADE PAUSED - Awaiting human approval\n"
            if guardian_result:
                msg += f"\nPolicy: {guardian_result.policy_name}"
                msg += f"\nReason: {guardian_result.reason}"
            return msg, result["thread_id"], gr.update(visible=True)
        else:
            guardian_result = result.get("guardian_result")
            reason_blocked = guardian_result.reason if guardian_result else "Unknown"
            return f"❌ Trade blocked: {reason_blocked}", "", gr.update(visible=False)

    except Exception as e:
        return f"❌ Error: {str(e)}", "", gr.update(visible=False)


def approve_trade_ui(thread_id, decision, modified_amount):
    """Approve or reject trade."""
    if not thread_id:
        return "❌ Error: No Thread ID."

    approve = decision == "Approve"

    try:
        checkpointer = get_checkpointer()
        result = approve_trade(
            thread_id=thread_id,
            checkpointer=checkpointer,
            approve=approve,
            modified_amount=float(modified_amount) if modified_amount else None,
        )

        if result.get("rejected"):
            return "❌ Trade rejected by reviewer"
        elif result.get("executed"):
            return "🎉 Trade approved and executed!"
        else:
            return "⚠️ Trade could not be processed"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def _parse_force_mode(mode):
    """Map mode string to ExecutionMode enum."""
    if mode == "deep":
        return ExecutionMode.DEEP_RESEARCH
    if mode == "flash":
        return ExecutionMode.FLASH_BRIEFING
    return None


def _handle_combined_result(result, status_log, thread_id):
    """Process combined workflow result into UI return values."""
    report_md = ""

    if result.get("requires_report_approval"):
        status_log += "\n⏸️ PAUSED - Awaiting report approval"
        if result.get("draft_report"):
            report_md = format_report_for_display(result["draft_report"])
        return status_log, report_md, thread_id, gr.update(visible=True), gr.update(visible=False)

    if result.get("requires_trade_approval"):
        status_log += "\n⏸️ TRADE PAUSED - Awaiting trade approval"
        guardian_result = result.get("guardian_result")
        trade_status = ""
        if guardian_result:
            trade_status = f"Policy: {guardian_result.policy_name}\nReason: {guardian_result.reason}"
        return status_log, report_md, thread_id, gr.update(visible=False), gr.update(visible=True, value=trade_status)

    if result.get("trade_executed"):
        status_log += "\n🎉 Combined workflow complete!\n✅ Report published\n✅ Trade executed"
        return status_log, report_md, thread_id, gr.update(visible=False), gr.update(visible=False)

    guardian_result = result.get("guardian_result")
    if guardian_result and guardian_result.decision.value == "reject":
        status_log += f"\n❌ Trade blocked by Guardian: {guardian_result.reason}"
    else:
        status_log += "\n✅ Analysis complete (no trade action - hold recommendation)"

    if result.get("draft_report"):
        report_md = format_report_for_display(result["draft_report"])

    return status_log, report_md, thread_id, gr.update(visible=False), gr.update(visible=False)


def run_combined_ui(query, user_id, model, mode, trade_amount):
    """Run combined workflow."""
    thread_id = str(uuid.uuid4())
    os.environ[MODEL_ENV_VAR] = MODEL_MAP[model]
    force_mode = _parse_force_mode(mode)

    try:
        checkpointer = get_checkpointer()
    except Exception:
        checkpointer = None

    status_log = f"Started combined workflow with Thread ID: {thread_id}\n"

    try:
        result = run_combined_analysis(
            query=query,
            user_id=user_id,
            thread_id=thread_id,
            checkpointer=checkpointer,
            force_mode=force_mode,
            trade_amount=float(trade_amount),
        )
        return _handle_combined_result(result, status_log, thread_id)

    except Exception as e:
        return f"❌ Error: {str(e)}", "", thread_id, gr.update(visible=False), gr.update(visible=False)


def approve_combined_report_ui(thread_id):
    """Approve report in combined workflow."""
    try:
        checkpointer = get_checkpointer()
        result = approve_combined_report(thread_id=thread_id, checkpointer=checkpointer)

        status_update = ""
        trade_vis = gr.update(visible=False)

        trade_info = ""

        if result.get("published") or result.get("state", {}).get("report_approved"):
            status_update = "🎉 Report published successfully!"

        if result.get("requires_trade_approval"):
            status_update += "\n⏸️ TRADE PAUSED - Awaiting human approval"
            guardian_result = result.get("guardian_result")
            if guardian_result:
                trade_info = f"Policy: {guardian_result.policy_name}\nReason: {guardian_result.reason}"
            trade_vis = gr.update(visible=True, value=trade_info)

        elif result.get("trade_executed"):
            status_update += "\n🎉 Combined workflow complete!\n✅ Trade executed"

        else:
            guardian_result = result.get("guardian_result")
            if guardian_result and guardian_result.decision.value == "reject":
                status_update += f"\n❌ Trade blocked by Guardian: {guardian_result.reason}"
            else:
                status_update += "\n✅ Analysis complete (no trade action)"

        return status_update, trade_vis

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(visible=False)


# Build the Interface
with gr.Blocks(title="Market Analyst Agent") as demo:
    gr.Markdown("# 🤖 Market Analyst Agent")

    with gr.Tabs():
        # --- Profile Tab ---
        with gr.Tab("👤 Profile Settings"):
            gr.Markdown("Configure your investment profile.")
            with gr.Row():
                p_user_id = gr.Textbox(label="User ID", value="default")
                p_risk = gr.Dropdown(choices=["conservative", "moderate", "aggressive"], label="Risk Tolerance")
                p_horizon = gr.Dropdown(choices=["short", "medium", "long"], label="Investment Horizon")
            p_btn = gr.Button("Update Profile")
            p_output = gr.Textbox(label="Status")

            p_btn.click(set_profile, inputs=[p_user_id, p_risk, p_horizon], outputs=p_output)

        # --- Analysis Tab ---
        with gr.Tab("🔬 Market Analysis"):
            gr.Markdown("Run a stock analysis and generate a report.")
            with gr.Row():
                with gr.Column():
                    a_query = gr.Textbox(label="Query", placeholder="Analyze NVDA stock...")
                    a_user_id = gr.Textbox(label="User ID", value="default")
                    a_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value=DEFAULT_MODEL_KEY, label="Model")
                    a_mode = gr.Dropdown(choices=["auto", "deep", "flash"], value="auto", label="Mode")
                    a_thread_id_input = gr.Textbox(label="Thread ID (Optional - for resuming)", placeholder="Leave empty for new analysis")
                    a_resume_thread = gr.Textbox(label="Resume Thread ID (Alternative)", visible=False)  # Helper for interruption

                    a_run_btn = gr.Button("Run Analysis", variant="primary")

                with gr.Column():
                    a_status = gr.Textbox(label="Status", interactive=False)
                    a_report = gr.Markdown(label="Report")
                    a_current_thread = gr.Textbox(label="Current Thread ID", interactive=False)

                    with gr.Group(visible=False) as a_approve_group:
                        gr.Markdown("### Actions Needed")
                        a_approve_btn = gr.Button("Approve & Publish Report", variant="secondary")
                        a_approval_status = gr.Textbox(label="Approval Status")

            a_run_btn.click(
                run_analysis_ui,
                inputs=[a_query, a_user_id, a_model, a_mode, a_thread_id_input, a_resume_thread],
                outputs=[a_report, a_status, a_current_thread, a_approve_group],
            )

            a_approve_btn.click(approve_report_ui, inputs=[a_current_thread], outputs=[a_report, a_approval_status])

        # --- Trade Tab ---
        with gr.Tab("💼 Trade Execution"):
            gr.Markdown("Execute trades with Guardian policy checks.")
            with gr.Row():
                with gr.Column():
                    t_action = gr.Dropdown(choices=["Buy", "Sell"], value="Buy", label="Action")
                    t_ticker = gr.Textbox(label="Ticker", placeholder="NVDA")
                    t_amount = gr.Number(label="Amount (USD)", value=1000)
                    t_reason = gr.Textbox(label="Reason", value="Agent recommendation")
                    t_btn = gr.Button("Execute Trade", variant="primary")

                with gr.Column():
                    t_status = gr.Textbox(label="Status")
                    t_thread_id = gr.Textbox(label="Thread ID", interactive=False)

                    with gr.Group(visible=False) as t_approve_group:
                        gr.Markdown("### Approval Required")
                        t_decision = gr.Radio(["Approve", "Reject"], label="Decision", value="Approve")
                        t_mod_amount = gr.Number(label="Modified Amount (Optional)")
                        t_submit_decision = gr.Button("Submit Decision")
                        t_decision_status = gr.Textbox(label="Decision Result")

            t_btn.click(run_trade_ui, inputs=[t_action, t_ticker, t_amount, t_reason], outputs=[t_status, t_thread_id, t_approve_group])

            t_submit_decision.click(approve_trade_ui, inputs=[t_thread_id, t_decision, t_mod_amount], outputs=[t_decision_status])

        # --- Combined Workflow Tab ---
        with gr.Tab("🔄 Combined Workflow"):
            gr.Markdown("Run Analysis → Guardian → Trade in one flow.")
            with gr.Row():
                with gr.Column():
                    c_query = gr.Textbox(label="Query", placeholder="Analyze output and trade...")
                    c_user_id = gr.Textbox(label="User ID", value="default")
                    c_trade_amt = gr.Number(label="Trade Amount (USD)", value=1000)
                    c_model = gr.Dropdown(choices=list(MODEL_MAP.keys()), value=DEFAULT_MODEL_KEY, label="Model")
                    c_mode = gr.Dropdown(choices=["auto", "deep", "flash"], value="auto", label="Mode")
                    c_run_btn = gr.Button("Start Workflow", variant="primary")

                with gr.Column():
                    c_status = gr.Textbox(label="Workflow Log", lines=5)
                    c_report = gr.Markdown(label="Report")
                    c_thread_id = gr.Textbox(label="Thread ID", interactive=False)

                    # Report Approval Group
                    with gr.Group(visible=False) as c_report_approve_group:
                        c_approve_report_btn = gr.Button("Approve Report & Continue")

                    # Trade Approval Group
                    with gr.Group(visible=False) as c_trade_approve_group:
                        gr.Markdown("### Trade Approval Required")
                        c_trade_info = gr.Textbox(label="Trade Info", interactive=False)
                        c_trade_decision = gr.Radio(["Approve", "Reject"], label="Decision", value="Approve")
                        c_trade_mod = gr.Number(label="Modified Amount")
                        c_submit_trade_btn = gr.Button("Submit Trade Decision")

            c_run_btn.click(
                run_combined_ui,
                inputs=[c_query, c_user_id, c_model, c_mode, c_trade_amt],
                outputs=[c_status, c_report, c_thread_id, c_report_approve_group, c_trade_approve_group],
            )

            c_approve_report_btn.click(approve_combined_report_ui, inputs=[c_thread_id], outputs=[c_status, c_trade_approve_group])

            c_submit_trade_btn.click(
                approve_trade_ui,
                inputs=[c_thread_id, c_trade_decision, c_trade_mod],
                outputs=[c_status],  # Reusing status box for result
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
