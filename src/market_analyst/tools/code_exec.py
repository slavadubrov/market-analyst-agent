"""Code execution tool — Programmatic Tool Calling (PTC) modality.

Allows the agent to write and execute Python code for computations that
are impossible with static tool calls: loops, conditionals, ratio calculations,
portfolio math. This demonstrates the biggest shift in agent tooling — letting
agents write code instead of calling schemas one at a time.

Calculations run in a disposable restricted Docker container. There is no
in-process fallback.
"""

import ast
import os
import selectors
import subprocess
import time
import uuid

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


class CodeInput(BaseModel):
    """Input schema for the code execution tool."""

    code: str = Field(
        max_length=32768,
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

    return run_isolated(code)


def run_isolated(code: str, *, timeout: float = 10) -> str:
    """Run in a disposable container with no mounts, network, or host credentials.

    AST checks improve feedback; the container is the isolation boundary. Missing
    Docker/image fails closed. stdout is capped inside the container and on disk.
    """
    name = f"market-calculation-{uuid.uuid4().hex}"
    image = os.getenv("MARKET_ANALYST_SANDBOX_IMAGE", "python:3.13-alpine")
    bootstrap = """import sys, resource
resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
class Output:
    remaining = 16384
    def write(self, text):
        if len(text) > self.remaining:
            raise RuntimeError('output limit exceeded')
        self.remaining -= len(text)
        return sys.__stdout__.write(text)
    def flush(self):
        sys.__stdout__.flush()
sys.stdout = sys.stderr = Output()
exec(compile(sys.stdin.read(), '<analysis>', 'exec'), {})
"""
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=128m",
        "--memory-swap=128m",
        "--cpus=0.5",
        "--pids-limit=16",
        "--user=65534:65534",
        "--log-driver=none",
        "-i",
        image,
        "python",
        "-I",
        "-B",
        "-c",
        bootstrap,
    ]
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write(code.encode())
            process.stdin.close()
            output = bytearray()
            deadline = time.monotonic() + timeout
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(command, timeout)
                    chunk = os.read(process.stdout.fileno(), 4096)
                    if not chunk:
                        break
                    output.extend(chunk)
                    if len(output) > 16384:
                        return "Error: output limit exceeded"
            process.wait(timeout=max(0.01, deadline - time.monotonic()))
            text = output.decode(errors="replace").strip()
            if process.returncode:
                return f"Error: isolated calculation failed: {text}"
            return text or "(Code executed successfully but produced no output.)"
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
    except subprocess.TimeoutExpired:
        return "Error: isolated calculation timed out"
    except OSError:
        return "Blocked: Docker sandbox is unavailable"
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
