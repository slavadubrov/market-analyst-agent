"""Model transport/context contracts, without live API credentials."""

import pytest
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

from market_analyst.llm import ModelSettings, get_chat_model
from market_analyst.runtime.evidence import ToolEvidenceCollector
from market_analyst.runtime.intervention import ProviderIntervention, raise_if_intervention


def test_compaction_preserves_recent_reasoning_and_tool_pair():
    model = FakeListChatModel(responses=["Earlier research found fixture evidence."])
    middleware = SummarizationMiddleware(model=model, trigger=("messages", 7), keep=("messages", 3), trim_tokens_to_summarize=None)
    middleware._summary_model = model
    reasoning = {"type": "reasoning", "id": "rs_fixture", "encrypted_content": "opaque-provider-payload"}
    recent = AIMessage(content=[reasoning], tool_calls=[{"name": "quote", "args": {}, "id": "call-fixture", "type": "tool_call"}])
    messages = [HumanMessage(content=f"old-{i}") for i in range(8)] + [
        recent,
        ToolMessage(content='{"price": 42}', tool_call_id="call-fixture"),
        HumanMessage(content="continue"),
    ]
    updated = middleware.before_model({"messages": messages}, None)
    compacted = add_messages(messages, updated["messages"])
    assert len(compacted) < len(messages)
    assert compacted[-3].content == [reasoning]
    assert compacted[-2].tool_call_id == "call-fixture"
    assert "Earlier research" in str(compacted[0].content)


def test_evidence_is_retained_separately_from_compacted_messages():
    collector = ToolEvidenceCollector()
    collector.on_tool_end(ToolMessage(content='{"price": 42}', name="quote", tool_call_id="call-fixture"))
    assert collector.records[0]["result"] == '{"price": 42}'
    assert collector.records[0]["observed_at"]


def test_provider_intervention_does_not_fall_back_to_another_model(mocker):
    from market_analyst.nodes.router import router_node
    from market_analyst.schemas import AgentState

    structured = mocker.patch("market_analyst.nodes.router.get_structured_model").return_value
    structured.invoke.side_effect = RuntimeError("misalignment_policy_violation after partial response")
    with pytest.raises(ProviderIntervention):
        router_node(AgentState(messages=[HumanMessage(content="NVDA")]))
    assert structured.invoke.call_count == 1


def test_gateway_credentials_are_separate(monkeypatch, mocker):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent-to-gateway")
    monkeypatch.setenv("MARKET_ANALYST_API_KEY", "gateway-fixture")
    model = mocker.patch("market_analyst.llm.ChatOpenAI")
    config = {"configurable": {"model_settings": ModelSettings(provider="compatible", model="team/llama", base_url="http://localhost:4000/v1").model_dump()}}
    get_chat_model(config=config)
    assert model.call_args.kwargs["api_key"] == "gateway-fixture"
    assert model.call_args.kwargs["model"] == "team/llama"
    assert model.call_args.kwargs["use_responses_api"] is False


def test_only_provider_policy_errors_are_reclassified():
    raise_if_intervention(TimeoutError("ordinary timeout"))
    with pytest.raises(ProviderIntervention):
        raise_if_intervention(RuntimeError("misalignment_policy_violation"))
