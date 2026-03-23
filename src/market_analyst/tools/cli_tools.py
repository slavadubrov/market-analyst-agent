"""CLI-as-Tool modality — agent invokes its own CLI commands.

Demonstrates the CLI tool pattern: the agent composes shell commands,
executes them via subprocess, and parses structured (JSON) output.
This is fundamentally different from JSON tool calling — it uses text I/O
through the Unix interface, with near-zero token overhead for tool schemas.

Only specific, allowlisted commands are exposed. The agent cannot run
arbitrary shell commands.
"""

import subprocess

from langchain_core.tools import tool
from pydantic import BaseModel, Field


def _run_cli(args: list[str]) -> str:
    """Run a market-analyst CLI command and return its output."""
    cmd = ["uv", "run", "market-analyst", *args]
    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return result.stderr.strip() or f"Command failed with exit code {result.returncode}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "Error: CLI command timed out after 30 seconds"


@tool
def cli_list_reports() -> str:
    """List all previously saved analysis reports from document memory.

    Uses the market-analyst CLI to retrieve reports stored in the
    document memory system. Returns JSON with report keys, tickers,
    execution modes, and creation dates.

    Use this when the user asks about past analyses, wants to review
    previous reports, or needs to find a specific report by ticker.
    """
    return _run_cli(["--list-reports", "--json"])


class ShowReportInput(BaseModel):
    """Input schema for showing a specific report."""

    report_key: str = Field(
        description="The report key to retrieve (e.g., 'research_NVDA_20260322')",
    )


@tool(args_schema=ShowReportInput)
def cli_show_report(report_key: str) -> str:
    """Retrieve and display a specific saved report by its key.

    Use this after cli_list_reports to fetch the full content of a
    specific report. Returns the report content and metadata as JSON.
    """
    return _run_cli(["--show-report", report_key, "--json"])
