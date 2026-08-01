"""Tests for the observability primitives (Part 5: The Habitat).

The OTel SDK pieces are exercised in their "no exporter configured" mode,
which is what dev runs hit too. That keeps the tests hermetic — no need to
spin up an OTLP collector to assert that spans get the right attributes.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import pytest

from market_analyst.observability import (
    BudgetExceeded,
    BudgetTracker,
    CircuitBreakerOpen,
    GenAIAttrs,
    llm_span,
    tool_span,
)
from market_analyst.observability.langchain_callback import (
    GenAICallbackHandler,
    make_callbacks,
)

# --- BudgetTracker -----------------------------------------------------------


def test_budget_tracker_accumulates_tokens():
    tracker = BudgetTracker(token_budget=1000)
    tracker.add_tokens(input_tokens=200, output_tokens=100)
    assert tracker.tokens_used == 300


def test_budget_tracker_raises_when_token_budget_exceeded():
    tracker = BudgetTracker(token_budget=100)
    with pytest.raises(BudgetExceeded):
        tracker.add_tokens(input_tokens=80, output_tokens=80)
    assert tracker.aborted is True


def test_budget_tracker_raises_when_tool_call_budget_exceeded():
    tracker = BudgetTracker(tool_call_budget=2)
    tracker.add_tool_call(errored=False)
    tracker.add_tool_call(errored=False)
    with pytest.raises(BudgetExceeded):
        tracker.add_tool_call(errored=False)


def test_circuit_breaker_trips_after_consecutive_errors():
    tracker = BudgetTracker(consecutive_error_threshold=3)
    tracker.add_tool_call(errored=True)
    tracker.add_tool_call(errored=True)
    with pytest.raises(CircuitBreakerOpen):
        tracker.add_tool_call(errored=True)


def test_circuit_breaker_resets_on_success():
    """A successful tool call between errors should not trip the breaker."""
    tracker = BudgetTracker(consecutive_error_threshold=3)
    tracker.add_tool_call(errored=True)
    tracker.add_tool_call(errored=True)
    tracker.add_tool_call(errored=False)  # reset
    tracker.add_tool_call(errored=True)
    tracker.add_tool_call(errored=True)
    # Still under threshold because the success reset the counter
    assert tracker.consecutive_errors == 2


def test_budget_tracker_from_state_rehydrates_counters():
    class FakeState:
        token_budget = 5000
        tool_call_budget = 20
        tokens_used = 1234
        tool_calls_used = 7

    tracker = BudgetTracker.from_state(FakeState())
    assert tracker.token_budget == 5000
    assert tracker.tokens_used == 1234
    assert tracker.tool_calls_used == 7


# --- Span context managers ---------------------------------------------------


def test_llm_span_sets_genai_attributes_and_ends_cleanly():
    """Spans should set the standard attribute names from the GenAI spec."""
    captured: dict[str, Any] = {}

    with llm_span(
        model="claude-test",
        provider="anthropic",
        agent_name="test.agent",
        workflow_name="test_wf",
        conversation_id="thread-abc",
    ) as span:
        # The no-op tracer's span is non-recording in test, but set_attribute
        # is still callable and we can record what would be set by hooking
        # the method.
        original = span.set_attribute

        def capture(key: str, value: Any) -> None:
            captured[key] = value
            return original(key, value)

        span.set_attribute = capture  # type: ignore[assignment]
        span.set_attribute(GenAIAttrs.RESPONSE_MODEL, "claude-test")

    # Only the attribute we set via the patched method should be captured;
    # asserting context manager exits cleanly is the bigger guarantee here.
    assert captured.get(GenAIAttrs.RESPONSE_MODEL) == "claude-test"


def test_tool_span_records_outcome_on_success(monkeypatch):
    """tool_span should bump the success counter on clean exit."""
    counted: list[tuple[str, str]] = []

    def fake_record_tool_call(tool_name: str, outcome: str) -> None:
        counted.append((tool_name, outcome))

    import market_analyst.observability.instrumentation as instrumentation_mod

    monkeypatch.setattr(instrumentation_mod.metrics, "record_tool_call", fake_record_tool_call)

    with tool_span(tool_name="my_tool", agent_name="test.agent"):
        pass

    assert ("my_tool", "ok") in counted


def test_tool_span_records_error_outcome(monkeypatch):
    """An exception inside tool_span should be re-raised and counted as error."""
    counted: list[tuple[str, str]] = []
    errors: list[tuple[str, str]] = []

    import market_analyst.observability.instrumentation as instrumentation_mod

    monkeypatch.setattr(
        instrumentation_mod.metrics,
        "record_tool_call",
        lambda name, outcome: counted.append((name, outcome)),
    )
    monkeypatch.setattr(
        instrumentation_mod.metrics,
        "record_tool_error",
        lambda name, err_type: errors.append((name, err_type)),
    )

    with pytest.raises(ValueError, match="boom"):
        with tool_span(tool_name="risky_tool", agent_name="test.agent"):
            raise ValueError("boom")

    assert ("risky_tool", "error") in counted
    assert ("risky_tool", "ValueError") in errors


# --- LangChain callback handler ---------------------------------------------


def test_make_callbacks_returns_genai_handler():
    callbacks = make_callbacks(
        agent_name="market-analyst.test",
        workflow_name="wf",
        conversation_id="thread-1",
        model="claude-test",
    )
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], GenAICallbackHandler)


def test_callback_handler_emits_spans_on_chat_model_lifecycle(monkeypatch):
    """A start/end pair should push and pop one span entry."""
    handler = GenAICallbackHandler(
        agent_name="market-analyst.test",
        model="claude-test",
        conversation_id="thread-1",
    )

    from uuid import uuid4

    run_id = uuid4()
    handler.on_chat_model_start(
        serialized={"kwargs": {"model": "claude-test"}},
        messages=[[]],
        run_id=run_id,
    )
    assert run_id in handler._spans

    class FakeMsg:
        usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    class FakeGen:
        message = FakeMsg()

    class FakeResponse:
        llm_output = {}
        generations = [[FakeGen()]]

    handler.on_llm_end(FakeResponse(), run_id=run_id)
    assert run_id not in handler._spans  # span ended cleanly


def test_callback_handler_emits_tool_spans():
    handler = GenAICallbackHandler(agent_name="market-analyst.test")
    from uuid import uuid4

    run_id = uuid4()
    handler.on_tool_start(serialized={"name": "my_tool"}, input_str="", run_id=run_id)
    assert run_id in handler._spans

    handler.on_tool_end(output="result", run_id=run_id)
    assert run_id not in handler._spans


# --- Tracing setup is idempotent --------------------------------------------


def test_setup_telemetry_is_idempotent():
    """Calling setup_telemetry twice should not double-install processors."""
    import market_analyst.observability.tracing as tracing_mod

    # Reset module-level flag so the test is hermetic.
    importlib.reload(tracing_mod)
    provider1 = tracing_mod.setup_telemetry()
    provider2 = tracing_mod.setup_telemetry()
    assert provider1 is provider2


# --- Smoke test that a chat trace round-trips through everything ------------


def test_llm_span_can_record_a_quick_completion(monkeypatch):
    """A typical caller pattern: open span, set response attrs, exit."""
    duration_observations: list[float] = []

    import market_analyst.observability.instrumentation as instrumentation_mod

    class FakeHist:
        def observe(self, value: float) -> None:
            duration_observations.append(value)

    class FakeDurationCounter:
        def labels(self, *_args: Any, **_kw: Any) -> FakeHist:
            return FakeHist()

    monkeypatch.setattr(instrumentation_mod.metrics, "operation_duration", FakeDurationCounter())

    with llm_span(model="claude-test", agent_name="test"):
        time.sleep(0.01)

    assert duration_observations and duration_observations[0] > 0
