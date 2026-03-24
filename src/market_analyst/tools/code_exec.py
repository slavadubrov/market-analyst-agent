"""Code execution tool — Programmatic Tool Calling (PTC) modality.

Allows the agent to write and execute Python code for computations that
are impossible with static tool calls: loops, conditionals, ratio calculations,
portfolio math. This demonstrates the biggest shift in agent tooling — letting
agents write code instead of calling schemas one at a time.

For production use, replace PythonAstREPLTool with a sandboxed environment
like E2B (e2b.dev) or LangSmith Sandboxes.
"""

import re

from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
from pydantic import BaseModel, Field

# Patterns that should be blocked for safety
_BLOCKED_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+shutil\b",
    r"\bfrom\s+os\b",
    r"\bfrom\s+subprocess\b",
    r"\bfrom\s+sys\b",
    r"\bfrom\s+shutil\b",
    r"\b__import__\s*\(",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\bopen\s*\(",
    r"\bcompile\s*\(",
]

_BLOCKED_RE = re.compile("|".join(_BLOCKED_PATTERNS))

# Pre-injected globals for the execution namespace
_GLOBALS: dict = {}
exec("import math, json, statistics", _GLOBALS)  # noqa: S102


class CodeInput(BaseModel):
    """Input schema for the code execution tool."""

    code: str = Field(
        description=(
            "Python code to execute. Use for financial calculations, "
            "ratio analysis, data transformations, or any multi-step "
            "computation. Common imports available: math, json, statistics. "
            "Print results to return them."
        ),
    )


def _check_safety(code: str) -> str | None:
    """Check code for blocked patterns. Returns error message or None."""
    match = _BLOCKED_RE.search(code)
    if match:
        return f"Blocked: '{match.group()}' is not allowed for safety reasons."
    return None


@tool(args_schema=CodeInput)
def execute_python_analysis(code: str) -> str:
    """Execute Python code for financial calculations and data analysis.

    Use this tool when you need to:
    - Compute financial ratios (P/E relative to sector, PEG, debt-to-equity)
    - Calculate growth rates, CAGR, or moving averages
    - Process and transform data with loops and conditionals
    - Perform portfolio math (position sizing, risk-adjusted returns)

    Do NOT use this tool for fetching data — use the data tools instead.
    This tool is for computation on data you already have.

    Available imports: math, json, statistics.
    Print your results to return them to the conversation.
    """
    error = _check_safety(code)
    if error:
        return error

    repl = PythonAstREPLTool(globals=_GLOBALS.copy())
    result = repl.invoke(code)
    if result is None or result == "":
        return "(Code executed successfully but produced no output. Use print() to return results.)"
    return str(result)
