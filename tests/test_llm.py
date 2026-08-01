import pytest
from langchain_openai import ChatOpenAI

from market_analyst.llm import get_chat_model, get_structured_model


def test_openai_is_used_when_its_key_is_available(mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test", "MARKET_ANALYST_MODEL": "claude-haiku"}, clear=True)
    chat_openai = mocker.patch("market_analyst.llm.ChatOpenAI")

    get_chat_model()

    chat_openai.assert_called_once_with(model="gpt-5.6-luna", timeout=None, use_responses_api=True)


def test_anthropic_can_be_selected_explicitly(mocker):
    mocker.patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "test",
            "MARKET_ANALYST_PROVIDER": "anthropic",
            "MARKET_ANALYST_MODEL": "claude-haiku",
        },
        clear=True,
    )
    chat_anthropic = mocker.patch("market_analyst.llm.ChatAnthropic")

    get_chat_model(temperature=0.3)

    chat_anthropic.assert_called_once_with(
        model_name="claude-haiku",
        temperature=0.3,
        timeout=None,
        stop=None,
    )


def test_unknown_provider_is_rejected(mocker):
    mocker.patch.dict("os.environ", {"MARKET_ANALYST_PROVIDER": "unknown"}, clear=True)

    with pytest.raises(ValueError, match="Unsupported MARKET_ANALYST_PROVIDER"):
        get_chat_model()


def test_openai_structured_output_uses_function_calling(mocker):
    model = ChatOpenAI(model="gpt-5.6-luna", api_key="test")
    structured_output = mocker.patch.object(ChatOpenAI, "with_structured_output", autospec=True)
    mocker.patch("market_analyst.llm.get_chat_model", return_value=model)

    get_structured_model(mocker.MagicMock())

    assert structured_output.call_args.kwargs["method"] == "function_calling"
