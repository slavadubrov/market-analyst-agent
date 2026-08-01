"""Code execution tool — Programmatic Tool Calling (PTC) modality.

Allows the agent to write and execute Python code for computations that
are impossible with static tool calls: loops, conditionals, ratio calculations,
portfolio math. This demonstrates the biggest shift in agent tooling — letting
agents write code instead of calling schemas one at a time.

For production use, replace this restricted in-process evaluator with a
process or container sandbox that also enforces CPU and memory limits.
"""

import ast
import builtins
import io
import json
import math
import statistics
from contextlib import redirect_stdout

from langchain_core.tools import tool
from pydantic import BaseModel, Field

_ALLOWED_MODULES = {"json", "math", "statistics"}
_ALLOWED_CALLS = {
    "abs",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
_BLOCKED_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.FunctionDef,
    ast.Global,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
)

_GLOBALS = {"math": math, "json": json, "statistics": statistics}
_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_CALLS}
_BUILTINS["__import__"] = builtins.__import__


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


def _check_safety(code: str) -> str | None:  # noqa: C901
    """Allow calculations while rejecting filesystem, process, and introspection access."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"Blocked: invalid Python syntax ({exc.msg})."

    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_NODES):
            return f"Blocked: {type(node).__name__} is not allowed."
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return f"Blocked: name '{node.id}' is not allowed."
        if isinstance(node, ast.Import):
            if any(alias.name not in _ALLOWED_MODULES or alias.asname for alias in node.names):
                return "Blocked: only math, json, and statistics imports are allowed."
        if isinstance(node, ast.ImportFrom):
            return "Blocked: from-imports are not allowed."
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id not in _ALLOWED_MODULES or node.attr.startswith("_"):
                return "Blocked: attribute access is limited to approved modules."
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id not in _ALLOWED_CALLS:
                return f"Blocked: call to '{node.func.id}' is not allowed."
            if not isinstance(node.func, (ast.Name, ast.Attribute)):
                return "Blocked: dynamic calls are not allowed."
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

    output = io.StringIO()
    namespace = {"__builtins__": _BUILTINS, **_GLOBALS}
    try:
        with redirect_stdout(output):
            exec(compile(code, "<analysis>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"
    result = output.getvalue().rstrip()
    if not result:
        return "(Code executed successfully but produced no output. Use print() to return results.)"
    return result
