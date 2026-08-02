"""Per-run budgets, kill-switch, and circuit breaker.

The article's "Cost-control failures" row in the failure-modes table says: per-
run token budget, per-tool budget, kill-switch tied to a Prometheus counter,
iteration cap per turn, exponential backoff, circuit breaker on tool error
rate. This module is the bookkeeping for those rules.

Design choice: the tracker holds counters in-process. Persistence across worker
restarts comes from the LangGraph checkpoint (``AgentState.tokens_used`` and
``AgentState.tool_calls_used`` are checkpointed fields). On boot the harness
rehydrates the tracker from those fields, so a crash-then-resume doesn't reset
the budget to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from market_analyst.observability import metrics


class BudgetExceeded(RuntimeError):
    """Raised when a per-run budget is exhausted.

    The harness catches this, marks the session as ``aborted_budget``, and
    drops the debug bundle so the failure is debuggable later.
    """


class CircuitBreakerOpen(RuntimeError):
    """Raised when the consecutive-error threshold has been crossed."""


@dataclass
class BudgetTracker:
    """In-process accounting for a single agent run.

    Attributes:
        token_budget: Hard cap on input+output tokens for the whole run.
            ``None`` disables the cap.
        tool_call_budget: Hard cap on tool invocations.
        consecutive_error_threshold: Open the circuit breaker after this many
            tool errors in a row.
        tokens_used / tool_calls_used: Running counters. Initialize from the
            checkpointed values on resume.
        consecutive_errors: Reset to zero on any successful tool call.
    """

    token_budget: int | None = None
    tool_call_budget: int | None = None
    consecutive_error_threshold: int = 5
    tokens_used: int = 0
    tool_calls_used: int = 0
    consecutive_errors: int = 0
    _aborted: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_state(cls, state: object) -> BudgetTracker:
        """Rehydrate from an ``AgentState``-like object's checkpointed fields."""
        return cls(
            token_budget=getattr(state, "token_budget", None),
            tool_call_budget=getattr(state, "tool_call_budget", None),
            tokens_used=getattr(state, "tokens_used", 0) or 0,
            tool_calls_used=getattr(state, "tool_calls_used", 0) or 0,
        )

    # --- Token accounting --------------------------------------------------

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Add tokens to the running total; raise if the budget is exceeded.

        Adds first, then checks: this matches the article's "kill switch tied
        to a Prometheus counter" pattern (counter goes up before the abort
        fires, so the alert has data to fire on).
        """
        self.tokens_used += max(0, int(input_tokens) + int(output_tokens))
        if self.token_budget is not None and self.tokens_used > self.token_budget:
            self._abort("token_budget")
            raise BudgetExceeded(f"Token budget exceeded: {self.tokens_used} > {self.token_budget}")

    # --- Tool-call accounting ---------------------------------------------

    def add_tool_call(self, *, errored: bool) -> None:
        """Account for one tool invocation. Raises on budget / breaker trip."""
        self.tool_calls_used += 1
        if errored:
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0

        if self.tool_call_budget is not None and self.tool_calls_used > self.tool_call_budget:
            self._abort("tool_call_budget")
            raise BudgetExceeded(f"Tool-call budget exceeded: {self.tool_calls_used} > {self.tool_call_budget}")

        if self.consecutive_errors >= self.consecutive_error_threshold:
            self._abort("circuit_breaker")
            raise CircuitBreakerOpen(f"Circuit breaker tripped after {self.consecutive_errors} consecutive tool errors")

    # --- State plumbing ---------------------------------------------------

    def to_state_dict(self) -> dict[str, int]:
        """Snapshot for writing back into ``AgentState`` on each super-step."""
        return {
            "tokens_used": self.tokens_used,
            "tool_calls_used": self.tool_calls_used,
        }

    @property
    def aborted(self) -> bool:
        return self._aborted

    def _abort(self, reason: str) -> None:
        if self._aborted:
            return
        self._aborted = True
        metrics.record_budget_kill(reason)
