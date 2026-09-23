import json


import time


from pathlib import Path


import pytest


from zeta.agent.background import adopt_agent_children


from zeta.agent.receipt import encode_json


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.store import ConversationStore


from zeta.runtime.loop import AgentLoop


from zeta.skills import SkillCatalog


from zeta.tools.agent import agent_result


from zeta.tui.render import render_event


from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


pytestmark = pytest.mark.usefixtures("stock_router_mode")


async def _collect(events):
    return [event async for event in events]


def _result(store: ConversationStore, call_id: str) -> ToolResult:
    return next(
        message.tool_result
        for message in store.messages()
        if message.tool_result is not None
        and message.tool_result.tool_call_id == call_id
    )


def test_foreground_receipt_shows_lifecycle_stats(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.start_agent_lifecycle(
        handle="parent:1",
        started_at="2026-09-04T10:00:00+00:00",
        tree_budget=25,
        depth=1,
        agent_type="general",
        description="stats",
    )
    child.update_agent_lifecycle(tool_calls=2, turns_used=1)
    child.finish_agent_lifecycle(
        "completed",
        final_result="done",
        turns_used=1,
        finished_at="2026-09-04T10:00:01+00:00",
    )
    call = ToolCall(
        "agent-receipt-stats",
        "agent",
        {"prompt": "inspect", "description": "stats"},
    )
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={
                "child_session_path": str(child.session_dir),
                "turns_used": 1,
            },
        ),
    )

    rendered = render_event(event)
    assert rendered is not None
    assert "2 tool calls" in rendered.plain
