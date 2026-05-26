"""Tests for the runtime primitives (Part 5: The Habitat)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from market_analyst.runtime import (
    IdempotencyStore,
    InitializerContext,
    ensure_thread_workspace,
    read_progress,
    workspace_path_for,
    write_debug_bundle,
    write_progress,
)
from market_analyst.runtime.initializer import write_feature_list

# --- Workspace --------------------------------------------------------------


def test_workspace_path_for_sanitizes_thread_id(tmp_path: Path):
    """Disallowed characters should not appear in the filesystem path."""
    path = workspace_path_for("../etc/passwd", root=tmp_path)
    assert "/etc/passwd" not in str(path)
    assert path.parent == tmp_path.resolve()


def test_ensure_thread_workspace_creates_debug_subdir(tmp_path: Path):
    workspace = ensure_thread_workspace("thread-1", root=tmp_path)
    assert workspace.is_dir()
    assert (workspace / "_debug").is_dir()


def test_ensure_thread_workspace_is_idempotent(tmp_path: Path):
    """Second call should not error even though the dir already exists."""
    ensure_thread_workspace("thread-1", root=tmp_path)
    ensure_thread_workspace("thread-1", root=tmp_path)


# --- Initializer ------------------------------------------------------------


def test_initializer_context_fresh_boot(tmp_path: Path):
    ctx = InitializerContext.load(tmp_path)
    assert ctx.is_resume() is False
    assert ctx.progress == ""
    assert ctx.feature_list == {}


def test_initializer_context_resume(tmp_path: Path):
    write_progress(tmp_path, line="started research", prepend_timestamp=False)
    write_feature_list(tmp_path, {"step": 2, "total": 5})
    ctx = InitializerContext.load(tmp_path)
    assert ctx.is_resume() is True
    assert "started research" in ctx.progress
    assert ctx.feature_list == {"step": 2, "total": 5}


def test_write_progress_appends_not_truncates(tmp_path: Path):
    write_progress(tmp_path, line="step 1", prepend_timestamp=False)
    write_progress(tmp_path, line="step 2", prepend_timestamp=False)
    content = read_progress(tmp_path)
    assert "step 1" in content
    assert "step 2" in content


def test_initializer_context_handles_corrupt_feature_list(tmp_path: Path):
    (tmp_path / "feature-list.json").write_text("not json {{{ }")
    ctx = InitializerContext.load(tmp_path)
    assert ctx.feature_list == {}
    # Progress file is missing, so still not a "resume" signal.
    assert ctx.is_resume() is False


# --- Idempotency ------------------------------------------------------------


def test_idempotency_store_round_trips_a_value(tmp_path: Path):
    store = IdempotencyStore(root=tmp_path)
    seen, _ = store.reserve_or_replay("t1", "call-1")
    assert seen is False
    store.store("t1", "call-1", {"executed": True, "ref": "T-001"})
    assert store.has("t1", "call-1")
    assert store.fetch("t1", "call-1") == {"executed": True, "ref": "T-001"}


def test_idempotency_second_call_replays_prior_result(tmp_path: Path):
    store = IdempotencyStore(root=tmp_path)
    store.reserve_or_replay("t1", "call-1")
    store.store("t1", "call-1", "done")

    seen, prior = store.reserve_or_replay("t1", "call-1")
    assert seen is True
    assert prior == "done"


def test_idempotency_distinct_threads_dont_collide(tmp_path: Path):
    store = IdempotencyStore(root=tmp_path)
    store.reserve_or_replay("thread-A", "call-1")
    store.store("thread-A", "call-1", "A")

    seen, prior = store.reserve_or_replay("thread-B", "call-1")
    assert seen is False
    assert prior is None


# --- Debug bundle -----------------------------------------------------------


def test_write_debug_bundle_creates_expected_files(tmp_path: Path):
    debug_dir = write_debug_bundle(
        thread_id="thread-fail",
        state={"current_step_index": 3, "error": "boom"},
        exception=RuntimeError("test failure"),
        tool_calls=[
            {"ts": "t1", "tool": "get_stock_snapshot", "input_hash": "x", "latency_ms": 120, "status": "ok"},
            {"ts": "t2", "tool": "search_news", "input_hash": "y", "latency_ms": 800, "status": "error", "error": "ConnectionError"},
        ],
        workspace_root=tmp_path,
    )
    assert debug_dir.name == "_debug"
    # Required artifacts present.
    assert (debug_dir / "last_state.json").exists()
    assert (debug_dir / "error.txt").exists()
    assert (debug_dir / "tool_calls.csv").exists()
    assert (debug_dir / "env.txt").exists()
    assert (debug_dir / "workspace.tar.gz").exists()
    assert (debug_dir / "_debug.ok").exists()

    # last_state.json round-trips through JSON cleanly.
    state_data = json.loads((debug_dir / "last_state.json").read_text())
    assert state_data["current_step_index"] == 3


def test_write_debug_bundle_handles_pydantic_state(tmp_path: Path):
    from market_analyst.schemas import AgentState

    state = AgentState(token_budget=1000, tokens_used=50)
    debug_dir = write_debug_bundle(
        thread_id="thread-pyd",
        state=state,
        workspace_root=tmp_path,
    )
    state_data = json.loads((debug_dir / "last_state.json").read_text())
    assert state_data["token_budget"] == 1000


def test_write_debug_bundle_excludes_debug_dir_from_tarball(tmp_path: Path):
    """The tarball must not contain ``_debug/`` itself (avoids recursion)."""
    import tarfile

    ws = ensure_thread_workspace("thread-tar", root=tmp_path)
    (ws / "PROGRESS.md").write_text("hi")
    debug_dir = write_debug_bundle(
        thread_id="thread-tar",
        state={},
        workspace_root=tmp_path,
    )
    with tarfile.open(debug_dir / "workspace.tar.gz") as tar:
        names = tar.getnames()
    assert all(not n.startswith("_debug") for n in names)
    assert any("PROGRESS.md" in n for n in names)


# --- Encryption helper ------------------------------------------------------


def test_get_encrypted_serializer_returns_none_without_key(monkeypatch):
    """No env var → no encryption (the dev default)."""
    from market_analyst.memory import encryption

    monkeypatch.delenv(encryption.ENCRYPTION_KEY_ENV, raising=False)
    assert encryption.get_encrypted_serializer() is None


def test_get_encrypted_serializer_handles_missing_optional_dep(monkeypatch):
    """If the LangGraph release doesn't ship encryption, we log + soft-fail."""
    from market_analyst.memory import encryption

    monkeypatch.setenv(encryption.ENCRYPTION_KEY_ENV, "some-key")
    # Force the import to fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "langgraph.checkpoint.serde.encrypted":
            raise ImportError("not present in this version")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert encryption.get_encrypted_serializer() is None


# --- Evaluator subagent -----------------------------------------------------


def test_evaluator_node_returns_fail_when_no_report(mocker):
    """When the state has no draft, the evaluator must not pass."""
    from market_analyst.runtime.evaluator import evaluator_node
    from market_analyst.schemas import AgentState

    state = mocker.MagicMock(spec=AgentState)
    state.draft_report = None
    result = evaluator_node(state, config=None)
    assert result["evaluator_verdict"] == "fail"


def test_evaluator_node_default_fails_on_llm_error(mocker):
    """LLM failure must default to needs_human, not silently pass."""
    from market_analyst.runtime import evaluator as evaluator_mod
    from market_analyst.schemas import AgentState, DraftReport

    mocker.patch.object(
        evaluator_mod,
        "evaluate_draft_report",
        side_effect=RuntimeError("model unreachable"),
    )

    state = mocker.MagicMock(spec=AgentState)
    state.draft_report = DraftReport(
        ticker="NVDA",
        title="Test",
        summary="x",
        analysis="y",
        recommendation="buy",
        confidence=0.9,
        risk_factors=["risk"],
    )
    result = evaluator_mod.evaluator_node(state, config={"configurable": {"thread_id": "t"}})
    assert result["evaluator_verdict"] == "needs_human"


def test_evaluator_returns_structured_verdict(mocker):
    """Happy path: structured output flows through to the node."""
    from market_analyst.runtime import evaluator as evaluator_mod
    from market_analyst.runtime.evaluator import EvaluatorVerdict
    from market_analyst.schemas import AgentState, DraftReport

    mocker.patch.object(
        evaluator_mod,
        "evaluate_draft_report",
        return_value=EvaluatorVerdict(verdict="pass", reasons=["looks good"]),
    )

    state = mocker.MagicMock(spec=AgentState)
    state.draft_report = DraftReport(
        ticker="NVDA",
        title="Test",
        summary="x",
        analysis="y",
        recommendation="buy",
        confidence=0.9,
        risk_factors=["risk"],
    )
    result = evaluator_mod.evaluator_node(state)
    assert result["evaluator_verdict"] == "pass"
    assert result["evaluator_reasons"] == ["looks good"]


@pytest.fixture(autouse=True)
def _isolate_idempotency_store(tmp_path: Path, monkeypatch):
    """Reset the process-singleton idempotency store between tests."""
    import market_analyst.runtime.idempotency as idempotency_mod

    monkeypatch.setattr(idempotency_mod, "_store", None)
    yield
