"""OpenTelemetry GenAI semantic-convention attribute names.

Constants for the ``gen_ai.*`` attribute keys defined in the GenAI semantic
conventions (development status as of v1.36.0). Keeping them in one place
prevents drift between call sites and makes it obvious which attribute is
"standard" vs project-specific.

Reference: https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/
"""

from typing import Final


class GenAIAttrs:
    """String constants for ``gen_ai.*`` attribute keys.

    Names track the upstream registry exactly. If the spec adds an attribute
    we use, add it here rather than hard-coding the string at the call site.
    """

    # Operation
    OPERATION_NAME: Final = "gen_ai.operation.name"  # "chat" | "execute_tool" | ...

    # Provider / model
    PROVIDER_NAME: Final = "gen_ai.provider.name"  # "anthropic" | "openai" | ...
    REQUEST_MODEL: Final = "gen_ai.request.model"
    RESPONSE_MODEL: Final = "gen_ai.response.model"

    # Conversation / workflow
    CONVERSATION_ID: Final = "gen_ai.conversation.id"
    AGENT_NAME: Final = "gen_ai.agent.name"
    AGENT_ID: Final = "gen_ai.agent.id"
    WORKFLOW_NAME: Final = "gen_ai.workflow.name"

    # Token usage
    INPUT_TOKENS: Final = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS: Final = "gen_ai.usage.output_tokens"
    TOTAL_TOKENS: Final = "gen_ai.usage.total_tokens"

    # Tool calls (gen_ai.execute_tool span)
    TOOL_NAME: Final = "gen_ai.tool.name"
    TOOL_CALL_ID: Final = "gen_ai.tool.call.id"
    TOOL_TYPE: Final = "gen_ai.tool.type"

    # Response
    FINISH_REASONS: Final = "gen_ai.response.finish_reasons"
    RESPONSE_ID: Final = "gen_ai.response.id"


# Span names used across the codebase. Per the spec, the span name is
# "{operation} {model}" for chat spans and "execute_tool {tool_name}" for
# tool spans. We construct these at the call site rather than pre-baking them.
SPAN_NAME_CHAT: Final = "chat"
SPAN_NAME_EXECUTE_TOOL: Final = "execute_tool"
