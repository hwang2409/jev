import json
from pathlib import Path

from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ErrorInfo,
    Message,
    MessageRole,
    RedactedThinkingContent,
    RoutingSchemaContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload


def test_stream_event_serialization_round_trip() -> None:
    event = StreamEvent(
        StreamEventType.MESSAGE_UPDATE,
        message=Message(
            MessageRole.ASSISTANT,
            [ThinkingContent("plan"), TextContent("answer")],
        ),
        content=ToolUseContent(ToolCall("call-1", "read", {"path": "a"})),
        tool_result=ToolResult("call-1", "ok"),
        error=ErrorInfo("sample", "message"),
        data={"turn": 1},
    )

    assert StreamEvent.from_dict(event.to_dict()) == event


def test_signed_and_redacted_thinking_round_trip() -> None:
    message = Message(
        MessageRole.ASSISTANT,
        [ThinkingContent("plan", "sig-1"), RedactedThinkingContent("opaque")],
    )

    assert Message.from_dict(message.to_dict()) == message


def test_routing_schema_round_trip() -> None:
    message = Message(
        MessageRole.USER,
        [TextContent("request"), RoutingSchemaContent("routed schemas")],
    )

    assert Message.from_dict(message.to_dict()) == message


def test_routing_schema_store_replay_preserves_provider_payload_bytes(
    tmp_path: Path,
) -> None:
    schema_text = 'routed tool schemas:\n{"name":"read","parameters":{}}'
    message = Message(
        MessageRole.USER,
        [TextContent("request"), RoutingSchemaContent(schema_text)],
    )
    store = ConversationStore(tmp_path)
    store.append_message(message)
    replayed = Message.from_dict(store.entries[-1].data["message"])
    legacy = Message(
        MessageRole.USER,
        [TextContent("request"), TextContent(schema_text)],
    )

    assert json.dumps(
        build_messages_payload(
            [replayed], [], model="claude", max_tokens=4096, thinking_budget=1024
        ),
        sort_keys=True,
    ).encode() == json.dumps(
        build_messages_payload(
            [legacy], [], model="claude", max_tokens=4096, thinking_budget=1024
        ),
        sort_keys=True,
    ).encode()
    assert json.dumps(
        build_responses_payload([replayed], [], model="codex"),
        sort_keys=True,
    ).encode() == json.dumps(
        build_responses_payload([legacy], [], model="codex"),
        sort_keys=True,
    ).encode()


def test_tool_result_round_trip_preserves_mixed_content_blocks() -> None:
    result = ToolResult(
        "call-1",
        "[image block]\n[resource: file:///tmp/note.txt]",
        content_blocks=[
            {
                "type": "image",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg+M8AAAAEAAEBouDEsAAAAABJRU5ErkJggg==",
                "mimeType": "image/png",
            },
            {
                "type": "resource",
                "resource": {
                    "uri": "file:///tmp/note.txt",
                    "text": "note",
                },
            },
        ],
    )

    assert ToolResult.from_dict(result.to_dict()) == result
