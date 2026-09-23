import asyncio


import json


import threading


import time


from collections.abc import AsyncIterator, Sequence


from dataclasses import replace


from pathlib import Path


import pytest


import zeta.runtime.execution as execution_module


import zeta.tools.agent_send as agent_send_module


from zeta.agent.background import (
    BackgroundAgentOwner,
    adopt_agent_children,
    finish_background_child,
)


from zeta.agent.budget import MAX_AGENT_TURN_CAP, AgentTree


from zeta.core.abort import AbortGenerationRegistry


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.store import ConversationStore, PendingPromptsClosedError


from zeta.runtime.loop import AgentLoop


from zeta.mcp import MCPMount


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.tools.agent import ChildApprovalPolicy, send_to_run


from zeta.agent.presets import (
    AGENT_PRESETS,
    GENERAL_PRESET,
)


from zeta.tui.agent_card import AgentRunCommandMixin


from zeta.tui.render import render_event


from zeta.tui.todo import TodoWidget


from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)


pytestmark = pytest.mark.usefixtures("stock_router_mode")


async def _collect(events):
    return [event async for event in events]


def _agent_call(
    call_id: str = "agent-1", agent_type: str | None = None
) -> ToolCall:
    arguments = {"prompt": "inspect the task", "description": "task research"}
    if agent_type is not None:
        arguments["agent_type"] = agent_type
    return ToolCall(
        call_id,
        "agent",
        arguments,
    )


class ParallelChildrenBackend(CompletionBackend):
    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.parent_calls = list(calls)
        self.call_count = 0
        self.child_count = 0
        self.children_started = asyncio.Event()
        self.release_children = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(call) for call in self.parent_calls]
        else:
            self.child_count += 1
            child_index = self.child_count
            if self.child_count == len(self.parent_calls):
                self.children_started.set()
            await self.release_children.wait()
            blocks = [TextContent(f"child-{child_index}")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ParallelApprovalBackend(CompletionBackend):
    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.parent_calls = list(calls)
        self.call_count = 0
        self.child_count = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(call) for call in self.parent_calls]
        elif self.call_count <= 3:
            self.child_count += 1
            blocks = [
                ToolUseContent(
                    ToolCall(
                        f"child-bash-{self.child_count}",
                        "bash",
                        {"cmd": f"echo child-{self.child_count}"},
                    )
                )
            ]
        else:
            blocks = [TextContent(f"child-final-{self.call_count}")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _parallel_agent_calls() -> list[ToolCall]:
    return [
        ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": f"inspect {index}", "description": f"task {index}"},
        )
        for index in (1, 2)
    ]


class BackgroundBackend(CompletionBackend):
    def __init__(self, calls: Sequence[ToolCall]) -> None:
        self.calls = list(calls)
        self.child_text = "child complete"
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            blocks = [ToolUseContent(call) for call in self.calls]
        elif last_user == "inspect the task":
            self.child_started.set()
            await self.release_child.wait()
            blocks = [TextContent(self.child_text)]
        else:
            blocks = [TextContent("parent continued")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class NestedBlockingBackend(CompletionBackend):
    def __init__(self) -> None:
        self.call_count = 0
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(_agent_call("child"))]
        elif self.call_count == 2:
            blocks = [ToolUseContent(_agent_call("grandchild"))]
        else:
            self.grandchild_started.set()
            await self.release_grandchild.wait()
            blocks = [TextContent("grandchild complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class NestedBackgroundBackend(CompletionBackend):
    def __init__(self) -> None:
        self.call_count = 0
        self.grandchild_started = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        self.call_count += 1
        if self.call_count == 1:
            blocks = [ToolUseContent(_background_agent_call("child"))]
        elif self.call_count == 2:
            nested = _background_agent_call("grandchild")
            nested.arguments["description"] = "grandchild"
            blocks = [ToolUseContent(nested)]
        else:
            self.grandchild_started.set()
            await asyncio.Event().wait()
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ForegroundNestedBackgroundBackend(CompletionBackend):
    def __init__(self) -> None:
        self.child_nested = False
        self.grandchild_started = asyncio.Event()
        self.release_grandchild = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            child = _agent_call("child")
            child.arguments["prompt"] = "child prompt"
            blocks = [ToolUseContent(child)]
        elif last_user == "child prompt" and not self.child_nested:
            self.child_nested = True
            grandchild = _background_agent_call("grandchild")
            grandchild.arguments["prompt"] = "grandchild prompt"
            grandchild.arguments["description"] = "grandchild"
            blocks = [ToolUseContent(grandchild)]
        elif last_user == "grandchild prompt":
            self.grandchild_started.set()
            await self.release_grandchild.wait()
            blocks = [TextContent("grandchild complete")]
        else:
            blocks = [TextContent("child complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class ParallelNestedReadBackend(CompletionBackend):
    def __init__(self, count: int) -> None:
        self.calls = []
        for index in range(count):
            call = _agent_call(f"agent-{index}")
            call.arguments["prompt"] = f"inspect {index}"
            self.calls.append(call)
        self.seen_prompts: set[str] = set()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            blocks = [ToolUseContent(call) for call in self.calls]
        elif last_user not in self.seen_prompts:
            self.seen_prompts.add(last_user)
            blocks = [
                ToolUseContent(
                    ToolCall(
                        f"{last_user}-read",
                        "read",
                        {"path": "missing"},
                    )
                )
            ]
        else:
            blocks = [TextContent("child complete")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _background_agent_call(call_id: str = "background-1") -> ToolCall:
    return ToolCall(
        call_id,
        "agent",
        {
            "prompt": "inspect the task",
            "description": "background research",
            "background": True,
        },
    )


async def _wait_for_notification(
    store: ConversationStore, status: str
) -> object:
    for _ in range(100):
        notifications = store.agent_notifications()
        if notifications and notifications[-1].data["status"] == status:
            return notifications[-1]
        await asyncio.sleep(0.01)
    raise AssertionError(f"missing {status} background notification")


def _persist_background_receipt(
    store: ConversationStore, call: ToolCall, child: ConversationStore
) -> None:
    store.append_message(Message(MessageRole.USER, [TextContent("start")]))
    store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)])
    )
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("background agent started")],
            tool_result=ToolResult(
                call.id,
                "background agent started",
                structured_content={
                    "turns_used": 0,
                    "child_session_path": str(child.session_dir),
                    "status": "running",
                    "child_instance_id": f"{store.session_id}:1",
                    "description": "background research",
                },
            ),
        )
    )


def _model_agent_call(
    model: str = "gpt-5.4",
    call_id: str = "agent-model-1",
    **extra: object,
) -> ToolCall:
    arguments: dict[str, object] = {
        "prompt": "inspect the task",
        "description": "cross provider research",
        "model": model,
    }
    arguments.update(extra)
    return ToolCall(call_id, "agent", arguments)


class _ValidTokens:
    def is_valid(self, *, skew: float = 60) -> bool:
        del skew
        return True


class _FakeCredentialStore:
    """Stand in for an OAuth store without touching the real credential files."""

    def __init__(self, tokens: object | None) -> None:
        self._tokens = tokens

    def read(self) -> object | None:
        return self._tokens


def _stub_backend_factory(
    monkeypatch: pytest.MonkeyPatch,
    child_backend: CompletionBackend,
    *,
    tokens: object | None = None,
) -> list[tuple[str, str | None]]:
    """Record what the runner asks the factory for, and hand back child_backend."""

    requested: list[tuple[str, str | None]] = []

    def build(provider: str, model: str | None, **kwargs: object):
        del kwargs
        requested.append((provider, model))
        return child_backend, model or ""

    monkeypatch.setattr("zeta.agent.runner.build_backend", build)
    monkeypatch.setattr(
        "zeta.agent.runner.credential_store",
        lambda provider, **kwargs: _FakeCredentialStore(
            _ValidTokens() if tokens is None else tokens
        ),
    )
    return requested


class RunBackend(CompletionBackend):
    """Drive a long run whose first turn can be held open mid-flight."""

    def __init__(self) -> None:
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()
        self.child_prompts: list[str] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del tool_schemas
        last_user = next(
            (
                block.text
                for message in reversed(messages)
                if message.role is MessageRole.USER
                for block in message.content
                if isinstance(block, TextContent)
            ),
            "",
        )
        if last_user == "start":
            blocks = [ToolUseContent(_run_agent_call())]
        elif last_user == "work the big task":
            self.child_prompts.append(last_user)
            self.child_started.set()
            await self.release_child.wait()
            blocks = [TextContent("first pass done")]
        else:
            self.child_prompts.append(last_user)
            blocks = [TextContent("follow-up handled")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


def _run_agent_call(call_id: str = "run-1") -> ToolCall:
    return ToolCall(
        call_id,
        "agent",
        {
            "prompt": "work the big task",
            "description": "long horizon run",
            "agent_type": "run",
        },
    )


class _RunCommands(AgentRunCommandMixin):
    """Minimal host for the mixin: it only needs loop.store."""

    def __init__(self, loop: AgentLoop) -> None:
        self.loop = loop


def _run_handle_from_receipt(store: ConversationStore) -> str:
    """Return the child_instance_id a real model would receive for the run."""

    for message in store.messages():
        result = message.tool_result
        if result is None:
            continue
        structured = result.structured_content
        if structured is None:
            continue
        handle = structured.get("child_instance_id")
        if type(handle) is str and handle:
            return handle
    raise AssertionError("no run receipt with a child_instance_id")


@pytest.mark.asyncio
async def test_runs_and_send_commands_drive_a_live_run(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    commands = _RunCommands(loop)

    explore_call = ToolCall(
        "explore-1",
        "agent",
        {"prompt": "look", "description": "explore", "agent_type": "explore"},
    )
    store.register_agent_child(
        explore_call,
        child_session_path=str(tmp_path / "agents" / "explore"),
        description="explore",
        agent_type="explore",
        background=True,
        child_instance_id="sess:explore",
    )

    assert commands.slash_runs("") == "no live runs"

    await _collect(loop.run_turn("start"))
    await asyncio.wait_for(backend.child_started.wait(), timeout=2)

    listing = commands.slash_runs("")
    assert "long horizon run" in listing
    assert "explore" not in listing
    handle = _run_handle_from_receipt(store)
    # /runs shows the model-facing handle, not the opaque provider tool_call.id.
    assert handle in listing
    assert "run-1" not in listing

    assert commands.slash_send("nonsense") == (
        "use /send <run-id> <message>; /runs lists the live ones"
    )
    assert "no live run" in commands.slash_send("bogus-id hello")
    assert f"queued for {handle}" in commands.slash_send(
        f"{handle} also check the tests"
    )

    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    assert backend.child_prompts == ["work the big task", "also check the tests"]
    await loop.close()
