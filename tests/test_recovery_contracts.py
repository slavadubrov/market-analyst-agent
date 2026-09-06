"""Fault regressions use real ledgers/graphs rather than mocking their internals."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import partial

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient

from market_analyst.llm import ModelSettings, resolve_model
from market_analyst.memory.long import LongTermMemory
from market_analyst.nodes.rewoo_worker import resolve_arguments, rewoo_worker_node, validate_plan
from market_analyst.runtime.evaluator import evaluate_draft_report
from market_analyst.runtime.idempotency import IdempotencyStore, OperationUncertain
from market_analyst.schemas import AgentState, DraftReport, ExecutionMode, PlanStep, ReWOOPlanStep, UserProfile
from market_analyst.workflows.analysis_workflow import create_graph, run_analysis


def draft():
    return DraftReport(
        ticker="NVDA", title="Fixture", summary="Supported?", analysis="Price is 999999.", recommendation="buy", confidence=1, risk_factors=["Risk"]
    )


def test_concurrent_duplicate_effect_is_reserved_once(tmp_path):
    store = IdempotencyStore(tmp_path)

    def attempt(_):
        try:
            return store.reserve_or_replay("run", "order", {"amount": 10.01})[0]
        except OperationUncertain:
            return "pending"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(attempt, range(20)))
    assert outcomes.count(False) == 1
    assert outcomes.count("pending") == 19
    # Crash after effect, before settlement: reopening the DB cannot run it again.
    with pytest.raises(OperationUncertain):
        IdempotencyStore(tmp_path).reserve_or_replay("run", "order", {"amount": 10.01})
    store.mark_unknown("run", "order")
    with pytest.raises(OperationUncertain, match="unknown"):
        store.fetch("run", "order")
    store.store("run", "order", {"execution_id": "confirmed-by-lookup"})
    assert store.reserve_or_replay("run", "order", {"amount": 10.01}) == (True, {"execution_id": "confirmed-by-lookup"})
    with pytest.raises(ValueError, match="different parameters"):
        store.reserve_or_replay("run", "order", {"amount": 10.99})


def test_falsy_completed_result_is_still_replayed(tmp_path):
    store = IdempotencyStore(tmp_path)
    store.reserve_or_replay("r", "op")
    store.store("r", "op", None)
    assert store.reserve_or_replay("r", "op") == (True, None)


def step(name="#E1", **kwargs):
    return ReWOOPlanStep(step_id=name, description="fixture", tool_name="get_stock_snapshot", **kwargs)


@pytest.mark.parametrize(
    "steps",
    [
        [step(), step()],
        [step(depends_on=["#E2"])],
        [step(depends_on=["#E2"]), step("#E2", depends_on=["#E1"])],
        [step(tool_args={"nested": [{"value": "#E404"}]})],
        [step("not-an-id")],
        [step(f"#E{i}") for i in range(1, 22)],
    ],
)
def test_invalid_plan_dispatches_nothing(steps, mocker):
    dispatch = mocker.patch("market_analyst.nodes.rewoo_worker.execute_tool")
    assert "error" in rewoo_worker_node(AgentState(rewoo_plan=steps))
    dispatch.assert_not_called()


def test_nested_references_infer_dependencies():
    plan = [step(), step("#E2", tool_args={"nested": ["#E1"]})]
    assert validate_plan(plan)["#E2"] == {"#E1"}
    assert resolve_arguments(plan[1].tool_args, {"#E1": {"price": 42}}) == {"nested": [{"price": 42}]}


def test_long_json_evidence_is_not_truncated(mocker):
    payload = json.dumps({"prices": list(range(1000))})
    mocker.patch("market_analyst.nodes.rewoo_worker.execute_tool", return_value=payload)
    result = rewoo_worker_node(AgentState(rewoo_plan=[step()]))
    assert json.loads(result["rewoo_plan"][0].result)["prices"] == list(range(1000))
    assert result["evidence"][0]["result"] == payload


def test_recovery_keeps_committed_step_and_stops_at_approval():
    calls = []
    crash = [True]

    def planner(state):
        return {"plan": [PlanStep(step_number=1, description="first"), PlanStep(step_number=2, description="second")]}

    def executor(state):
        if state.current_step_index == 1 and crash[0]:
            crash[0] = False
            raise RuntimeError("worker died")
        calls.append(state.current_step_index)
        plan = list(state.plan)
        plan[state.current_step_index] = plan[state.current_step_index].model_copy(update={"completed": True, "result": "durable evidence"})
        return {"plan": plan, "current_step_index": state.current_step_index + 1}

    factory = partial(
        create_graph,
        nodes={
            "planner": planner,
            "executor": executor,
            "reporter": lambda state: {"draft_report": draft()},
            "evaluator": lambda state: {"evaluator_verdict": "pass"},
        },
    )
    saver = InMemorySaver()
    kwargs = {
        "thread_id": "recovery",
        "checkpointer": saver,
        "force_mode": ExecutionMode.DEEP_RESEARCH,
        "model_settings": ModelSettings(provider="openai", model="fixture"),
        "profile_loader": lambda _: UserProfile(),
        "graph_factory": factory,
    }
    with pytest.raises(RuntimeError, match="worker died"):
        run_analysis("NVDA", **kwargs)
    result = run_analysis("NVDA", retry_delivery=True, **kwargs)
    assert calls == [0, 1]
    assert result["state"]["plan"][0].result == "durable evidence"
    assert result["requires_approval"]
    run_analysis("NVDA", retry_delivery=True, **kwargs)
    assert calls == [0, 1]  # terminal redelivery neither re-runs nor auto-approves
    with pytest.raises(ValueError, match="already exists"):
        run_analysis("new user turn", **kwargs)
    with pytest.raises(ValueError, match="different principal"):
        run_analysis("NVDA", user_id="other", retry_delivery=True, **kwargs)


def test_no_persist_never_loads_remote_profile(mocker):
    loader = mocker.patch("market_analyst.workflows.analysis_workflow.load_user_profile", side_effect=AssertionError("remote access"))
    factory = partial(create_graph, nodes={"planner": lambda state: {"error": "stop fixture"}})
    with pytest.raises(RuntimeError, match="stop fixture"):
        run_analysis("NVDA", force_mode=ExecutionMode.DEEP_RESEARCH, graph_factory=factory, model_settings=ModelSettings(provider="openai", model="fixture"))
    loader.assert_not_called()


def test_unsupported_confident_report_fails_without_model(mocker):
    model = mocker.patch("market_analyst.runtime.evaluator.get_structured_model")
    assert evaluate_draft_report(draft(), evidence=[]).verdict == "fail"
    stale = [{"tool": "quote", "result": "999999", "observed_at": (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()}]
    assert evaluate_draft_report(draft(), evidence=stale).verdict == "fail"
    model.assert_not_called()


def test_principal_scope_deletion_and_expiry_with_real_qdrant():
    memory = LongTermMemory(client=QdrantClient(":memory:"))
    memory.save_profile("alice", UserProfile(notes="alice-only"))
    memory.save_profile("bob", UserProfile(notes="bob-only"))
    profiles = memory.search_profiles([1.0] * memory.vector_size, user_id="alice")
    assert [profile.notes for profile in profiles] == ["alice-only"]
    assert memory.search_profiles([1.0] * memory.vector_size, user_id="charlie") == []
    memory.delete_profile("alice")
    assert memory.get_profile("alice") == UserProfile()
    assert memory.get_profile("bob").notes == "bob-only"
    with pytest.raises(ValueError):
        memory.search_profiles([1.0] * memory.vector_size, user_id="")


def test_arbitrary_model_id_and_run_isolation(monkeypatch):
    monkeypatch.setenv("MARKET_ANALYST_PROVIDER", "openai")
    assert resolve_model("o3") == ("openai", "o3")
    a = {"configurable": {"model_settings": ModelSettings(provider="compatible", model="team/llama", base_url="http://localhost:4000/v1").model_dump()}}
    b = {"configurable": {"model_settings": ModelSettings(provider="anthropic", model="claude-fixture").model_dump()}}
    with ThreadPoolExecutor() as pool:
        assert list(pool.map(lambda c: resolve_model(config=c), [a, b])) == [("compatible", "team/llama"), ("anthropic", "claude-fixture")]


def test_expired_profile_is_not_retrieved_and_can_be_purged():
    from qdrant_client.http import models

    memory = LongTermMemory(client=QdrantClient(":memory:"))
    memory.save_profile("expired", UserProfile(notes="obsolete"))
    memory.client.set_payload(
        memory.collection_name,
        {"expires_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()},
        points=models.Filter(must=[models.FieldCondition(key="user_id", match=models.MatchValue(value="expired"))]),
    )
    assert memory.get_profile("expired") == UserProfile()
    assert memory.search_profiles([1.0] * memory.vector_size, user_id="expired") == []
    memory.purge_expired()
    assert memory.client.count(memory.collection_name).count == 0


def test_workspace_special_identifiers_cannot_escape(tmp_path):
    from market_analyst.runtime.workspace import workspace_path_for

    ids = [".", "..", "../outside", "x/y", "x_y", "", "z" * 500]
    paths = [workspace_path_for(name, tmp_path) for name in ids]
    assert all(path.parent == tmp_path for path in paths)
    assert len(set(paths)) == len(ids)


def test_two_writers_cannot_mutate_one_run():
    import threading

    from market_analyst.runtime.ownership import RunBusy, serialized_run

    entered, release = threading.Event(), threading.Event()

    @serialized_run
    def operation(thread_id):
        entered.set()
        release.wait(2)
        return thread_id

    with ThreadPoolExecutor() as pool:
        first = pool.submit(operation, "concurrent-writer-fixture")
        try:
            assert entered.wait(2)
            with pytest.raises(RunBusy):
                operation("concurrent-writer-fixture")
        finally:
            release.set()
        assert first.result() == "concurrent-writer-fixture"


def test_cli_archive_commands_need_no_model_key(monkeypatch, mocker, capsys, tmp_path):
    from market_analyst import cli
    from market_analyst.memory.document import DocumentMemory

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mocker.patch.object(cli, "load_dotenv")
    mocker.patch.object(cli, "get_document_memory", return_value=DocumentMemory(tmp_path))
    monkeypatch.setattr("sys.argv", ["market-analyst", "--list-reports", "--json"])
    cli.main()
    assert json.loads(capsys.readouterr().out) == []


def test_failed_evaluator_cannot_publish(mocker):
    from market_analyst.workflows.analysis_workflow import publish_node

    archive = mocker.patch("market_analyst.workflows.analysis_workflow.get_document_memory")
    with pytest.raises(ValueError, match="Evaluator rejected"):
        publish_node(AgentState(draft_report=draft(), report_approved=True, evaluator_verdict="fail"))
    archive.assert_not_called()
