from __future__ import annotations


import asyncio


import hashlib


import inspect


import json


import threading


import time


import uuid


from collections.abc import Callable


from io import StringIO


from pathlib import Path


from types import SimpleNamespace


import pytest


from rich.cells import cell_len


from rich.console import Console


from zeta.cli.main import build_parser, main


from zeta.core.abort import AbortGenerationRegistry


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.context import ContextAssembler


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.loop import AgentLoop


from zeta.core.project_context import ProjectContext


from zeta.core.session import SessionError, SessionManager


from zeta.core.slash import create_slash_registry


from zeta.core.store import ConversationStore


from zeta.skills import SkillCatalog


from zeta.tools.agent import ChildApprovalPolicy


from zeta.tui.app import TUIApp, create_app


from zeta.tui.layout import CONTENT_MARGIN, content_width


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


def _args(*values: str):
    return build_parser().parse_args([*values, "--provider", "fake"])


async def wait_until(check: Callable[[], bool]) -> None:
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_model_swap_is_rejected_with_pending_approval(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(store=store)
    call = ToolCall("pending-model-swap", "echo", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        approval_policy=policy,
    )

    output = create_slash_registry(skill_catalog=SkillCatalog.empty()).dispatch(app, "/model faster")

    assert output == "model unchanged: cannot change model while a turn or approval is active"
    assert app.model == "offline"


@pytest.mark.asyncio
async def test_resumed_pending_approval_is_presented_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "zeta-home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    opened = SessionManager(home).create(provider="fake", model="offline", cwd=tmp_path)
    call = ToolCall("approval-resume", "exec", {"command": "danger"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    app = create_app(
        build_parser().parse_args(
            ["--resume", opened.store.session_id, "--provider", "fake"]
        )
    )
    app.console = Console(file=StringIO(), force_terminal=False)
    app._present_pending_approvals()

    assert call.name in app.console.file.getvalue()  # card shows the tool name
    assert await app._handle_approval_input(f"approve {call.id}")
    assert app.loop.store.pending_approvals() == []


@pytest.mark.asyncio
async def test_tui_resolves_colliding_child_approvals_by_unique_key(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    loop = AgentLoop(FakeBackend([]), parent_store, approval_policy=policy, skill_catalog=SkillCatalog.empty())
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    app.console = Console(file=StringIO(), force_terminal=False)
    child_a = ConversationStore(tmp_path / "children", session_id="a")
    child_b = ConversationStore(tmp_path / "children", session_id="b")
    policy_a = ChildApprovalPolicy(policy, child_a, "child a", "child-a")
    policy_b = ChildApprovalPolicy(policy, child_b, "child b", "child-b")
    call_a = ToolCall("same-request", "bash", {"cmd": "a"})
    call_b = ToolCall("same-request", "bash", {"cmd": "b"})
    signal_a = AbortGenerationRegistry().new_generation()
    signal_b = AbortGenerationRegistry().new_generation()
    task_a = asyncio.create_task(policy_a.authorize(call_a, signal_a))
    task_b = asyncio.create_task(policy_b.authorize(call_b, signal_b))

    await wait_until(lambda: len(app.pending_approvals) == 2)
    app._present_pending_approvals()
    rendered = app.console.file.getvalue()
    assert "child a: bash" in rendered
    assert "child b: bash" in rendered
    # y/n only answer the first card, so the second one names its own key.
    assert "y approve · n deny" in rendered
    assert "approve ('child-b', 'same-request')" in rendered

    assert await app._handle_approval_input(
        "approve ('child-a', 'same-request')"
    )
    assert await app._handle_approval_input("deny ('child-b', 'same-request')")
    assert await task_a == ApprovalDecision.ALLOW
    assert await task_b == ApprovalDecision.DENY
    signal_a.abort()
    signal_b.abort()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["allow", "deny"])
async def test_resume_pending_tool_executes_and_persists_result(
    tmp_path: Path, decision: str
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    executed: list[str] = []

    async def echo(arguments: dict[str, str]) -> str:
        executed.append(arguments["value"])
        return arguments["value"]

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"echo": echo},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-tool", "echo", {"value": "done"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    if decision == "allow":
        assert policy.approve(call.id)
    else:
        assert policy.deny(call.id)

    events = []
    result = await loop.resume_pending_tool(call.id, event_sink=events.append)

    assert result is not None
    assert opened.store.messages()[-1].tool_result == result
    assert executed == (["done"] if decision == "allow" else [])
    assert [event.type for event in events] == (
        [StreamEventType.TOOL_EXECUTION_START, StreamEventType.TOOL_EXECUTION_END]
        if decision == "allow"
        else [StreamEventType.TOOL_EXECUTION_END]
    )


@pytest.mark.asyncio
async def test_resumed_tool_abort_active_persists_canceled_result(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    started = asyncio.Event()

    async def block(arguments: dict[str, str], abort_signal: object) -> str:
        del arguments
        started.set()
        await abort_signal.wait()  # type: ignore[attr-defined]
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"block": block},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-abort", "block", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    approval_task = asyncio.create_task(app._handle_approval_input(f"approve {call.id}"))
    await asyncio.wait_for(started.wait(), timeout=1)
    app.abort_active()
    assert await approval_task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == "tool execution canceled"


@pytest.mark.asyncio
async def test_resumed_tool_direct_cancel_persists_canceled_result(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    started = asyncio.Event()

    async def block(arguments: dict[str, str]) -> str:
        del arguments
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"block": block},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-cancel", "block", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy.approve(call.id)
    events: list[StreamEvent] = []
    task = asyncio.create_task(loop.resume_pending_tool(call.id, event_sink=events.append))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == "tool execution canceled"
    assert [event.type for event in events] == [
        StreamEventType.TOOL_EXECUTION_START,
        StreamEventType.TOOL_EXECUTION_END,
    ]


@pytest.mark.asyncio
async def test_resumed_tool_immediate_abort_persists_canceled_result(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)

    async def never_runs(arguments: dict[str, str]) -> str:
        del arguments
        raise AssertionError("the handler must not run")

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"never": never_runs},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-immediate-abort", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy.approve(call.id)

    assert loop.prepare_resume_pending_tool(call.id)
    loop.abort()
    result = await loop.resume_pending_tool(call.id, prepared=True)

    assert result is not None
    assert result.content == "tool execution canceled"
    assert opened.store.messages()[-1].tool_result == result


def test_finalize_canceled_is_idempotent(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    loop = AgentLoop(FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty())
    call = ToolCall("approval-idempotent-cancel", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    first = loop.finalize_canceled(call.id)
    second = loop.finalize_canceled(call.id)

    results = [
        message.tool_result
        for message in opened.store.messages()
        if message.tool_result is not None
    ]
    assert first is not None
    assert second == first
    assert results == [first]


def test_completion_edge_idempotence_preserves_success(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    loop = AgentLoop(FakeBackend([]), opened.store, skill_catalog=SkillCatalog.empty())
    call = ToolCall("approval-completion-edge", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    success = ToolResult(call.id, "completed")
    opened.store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent(success.content)],
            tool_result=success,
        )
    )

    result = loop.finalize_canceled(call.id)
    results = [
        message.tool_result
        for message in opened.store.messages()
        if message.tool_result is not None
    ]

    assert result == success
    assert results == [success]


@pytest.mark.asyncio
async def test_resume_pending_tool_rejects_existing_result(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    executed: list[str] = []

    async def echo(arguments: dict[str, str]) -> str:
        executed.append(arguments["value"])
        return arguments["value"]

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"echo": echo},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-existing-result", "echo", {"value": "done"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    assert policy.approve(call.id)
    success = ToolResult(call.id, "already completed")
    opened.store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent(success.content)],
            tool_result=success,
        )
    )

    events: list[StreamEvent] = []
    result = await loop.resume_pending_tool(call.id, event_sink=events.append)

    assert result == success
    assert executed == []
    assert events == []


@pytest.mark.asyncio
async def test_strict_pre_start_parent_cancellation_persists_canceled_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)

    async def never_runs(arguments: dict[str, str]) -> str:
        del arguments
        await asyncio.Event().wait()
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"never": never_runs},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-parent-cancel", "never", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    original_prepare = loop.prepare_resume_pending_tool
    parent_task: asyncio.Task[bool]

    def cancel_create(coro: object) -> asyncio.Task[object]:
        close = coro.close
        close()
        current = asyncio.current_task()
        assert current is not None
        current.cancel()
        raise asyncio.CancelledError

    def prepare(request_id: str) -> bool:
        prepared = original_prepare(request_id)
        monkeypatch.setattr(asyncio, "create_task", cancel_create)
        return prepared

    monkeypatch.setattr(loop, "prepare_resume_pending_tool", prepare)
    parent_task = asyncio.create_task(
        app._handle_approval_input(f"approve {call.id}")
    )

    with pytest.raises(asyncio.CancelledError):
        await parent_task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == (
        "tool execution canceled"
    )


@pytest.mark.asyncio
async def test_parent_cancellation_after_child_start_persists_result(
    tmp_path: Path,
) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    policy = ApprovalPolicy(store=opened.store)
    handler_started = asyncio.Event()

    async def blocks(arguments: dict[str, str]) -> str:
        del arguments
        handler_started.set()
        await asyncio.Event().wait()
        return "unreachable"

    loop = AgentLoop(
        FakeBackend([]),
        opened.store,
        tools={"block": blocks},
        approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
    )
    call = ToolCall("approval-parent-after-start", "block", {})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    app = TUIApp(
        loop,
        provider="fake",
        model="offline",
        approval_policy=policy,
    )
    parent_task = asyncio.create_task(
        app._handle_approval_input(f"approve {call.id}")
    )
    await handler_started.wait()
    parent_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await parent_task

    assert opened.store.messages()[-1].tool_result is not None
    assert opened.store.messages()[-1].tool_result.content == (
        "tool execution canceled"
    )


def test_resume_reemits_pending_approval_state(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path / "zeta-home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    call = ToolCall("approval-1", "exec", {"command": "danger"})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )

    resumed = manager.open(opened.store.session_id)

    assert resumed.store.pending_approvals() == [(call.id, call)]
