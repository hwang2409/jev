from __future__ import annotations

from pathlib import Path

import pytest

import zeta.loop as loop_module
import zeta.tools.route as route_module
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.providers.anthropic_payload import build_messages_payload
from zeta.providers.codex_payload import build_responses_payload
from zeta.providers.jev import AutoRouteResult
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.types import Message, MessageRole, TextContent, ToolCall


async def collect(events):
    return [event async for event in events]


def result(
    tool: str,
    *,
    confidence: float = 0.9,
    needs_tool: float = 1.0,
    probabilities: dict[str, float] | None = None,
) -> AutoRouteResult:
    return AutoRouteResult(
        tool,
        probabilities or {tool: confidence},
        confidence,
        needs_tool,
        {},
    )


def async_result(value: AutoRouteResult):
    async def route(*_args):
        return value

    return route


def build_loop(
    tmp_path: Path,
    turns: list[ScriptedTurn],
    *,
    names: tuple[str, ...] = ("read", "write", "bash"),
) -> AgentLoop:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    route_module.register(registry)
    for name in names:
        registry.register(
            name,
            lambda _arguments, *, name=name: name,
            description=f"{name} description",
            parameters={"type": "object"},
            requires_approval=False,
        )
    return AgentLoop(
        FakeBackend(turns),
        store,
        registry=registry,
        approval_policy=ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW),
        router_style="auto",
        skill_catalog=SkillCatalog.empty(),
    )


@pytest.mark.asyncio
async def test_auto_surface_is_static_across_three_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(content=[TextContent("one")]),
            ScriptedTurn(content=[TextContent("two")]),
            ScriptedTurn(content=[TextContent("three")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert all(call[1] == [loop._auto_invoke_schema] for call in loop.backend.calls)
    assert all(call[1][0] is loop.backend.calls[0][1][0] for call in loop.backend.calls)


@pytest.mark.asyncio
async def test_auto_requests_extend_history_for_both_provider_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-1", "invoke", {"tool": "read", "args": {}})
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-2", "invoke", {"tool": "read", "args": {}})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("read twice"))

    anthropic = [
        build_messages_payload(
            messages,
            tools,
            model="test",
            max_tokens=16_384,
            thinking_budget=8_192,
        )["messages"]
        for messages, tools in loop.backend.calls
    ]
    codex = [
        build_responses_payload(messages, tools, model="test")["input"]
        for messages, tools in loop.backend.calls
    ]
    for payloads in (anthropic, codex):
        assert len(payloads[0]) < len(payloads[1]) < len(payloads[2])
        assert payloads[0] == payloads[1][: len(payloads[0])]
        assert payloads[1] == payloads[2][: len(payloads[1])]


@pytest.mark.asyncio
async def test_auto_schema_text_is_appended_to_initial_user_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("done")])])

    await collect(loop.run_turn("read the file"))

    user = next(message for message in loop.backend.calls[0][0] if message.role == "user")
    assert user.content[-1].text.startswith("routed tool schemas:")
    assert '"name": "read"' in user.content[-1].text
    reopened = ConversationStore(tmp_path, session_id=loop.store.session_id)
    assert reopened.messages()[0].content[-1] == user.content[-1]


@pytest.mark.asyncio
async def test_auto_invoke_dispatches_to_real_tool_and_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-1", "invoke", {"tool": "read", "args": {}})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )
    prepared: list[str] = []
    original = loop.tool_registry.prepare_approval

    def capture(call: ToolCall):
        prepared.append(call.name)
        return original(call)

    monkeypatch.setattr(loop.tool_registry, "prepare_approval", capture)

    await collect(loop.run_turn("read it"))

    assert prepared == ["read"]
    assert loop.store.messages()[2].tool_result is not None
    assert loop.store.messages()[2].tool_result.content == "read"
    result_message = next(
        message
        for message in loop.backend.calls[1][0]
        if message.role is MessageRole.TOOL_RESULT
    )
    assert result_message.content[-1].text.startswith("routed tool schemas:")


@pytest.mark.asyncio
async def test_auto_unrouted_invoke_returns_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-1", "invoke", {"tool": "write", "args": {}})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("write it"))

    result_message = loop.store.messages()[2].tool_result
    assert result_message is not None
    assert result_message.structured_content["error_kind"] == "unrouted_tool"
    assert "state what you need in text" in result_message.content


@pytest.mark.asyncio
async def test_auto_route_needs_tool_gate_advertises_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        loop_module, "auto_route", async_result(result("read", needs_tool=0.2))
    )
    loop = build_loop(tmp_path, [ScriptedTurn(content=[TextContent("answer")])])

    await collect(loop.run_turn("what is two plus two?"))

    assert loop.backend.calls[0][1] == [loop._auto_invoke_schema]
    assert loop.backend.calls[0][0][-1].content[-1] == TextContent(
        "no tool is needed this turn — answer directly"
    )


@pytest.mark.asyncio
async def test_auto_route_failure_opens_for_one_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def route(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("jev down")
        return result("read")

    monkeypatch.setattr(loop_module, "auto_route", route)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-1", "invoke", {"tool": "read", "args": {}})
                ]
            ),
            ScriptedTurn(content=[TextContent("second")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert loop.backend.calls[0][1] == [loop._auto_invoke_schema]
    assert '"name": "read"' in loop.backend.calls[0][0][-1].content[-1].text
    assert '"name": "write"' in loop.backend.calls[0][0][-1].content[-1].text
    assert '"name": "bash"' in loop.backend.calls[0][0][-1].content[-1].text
    assert loop.backend.calls[1][1] == [loop._auto_invoke_schema]


@pytest.mark.asyncio
async def test_auto_tools_value_stays_static_across_normal_failure_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def route(*_args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("jev down")
        return result("read")

    monkeypatch.setattr(loop_module, "auto_route", route)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-1", "invoke", {"tool": "read", "args": {}})
                ]
            ),
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-2", "invoke", {"tool": "read", "args": {}})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("keep going"))

    assert all(tools == [loop._auto_invoke_schema] for _, tools in loop.backend.calls)
    assert loop.backend.calls[0][1] == loop.backend.calls[1][1]
    assert loop.backend.calls[1][1] == loop.backend.calls[2][1]


@pytest.mark.asyncio
async def test_auto_schema_persists_at_user_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter((result("read"), result("write")))

    async def route(*_args):
        return next(results)

    monkeypatch.setattr(loop_module, "auto_route", route)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(content=[TextContent("first")]),
            ScriptedTurn(content=[TextContent("second")]),
        ],
    )

    await collect(loop.run_turn("read this"))
    await collect(loop.run_turn("write that"))

    users = [
        message
        for message in loop.backend.calls[1][0]
        if message.role is MessageRole.USER
    ]
    assert '"name": "read"' in users[0].content[-1].text
    assert '"name": "write"' in users[-1].content[-1].text


@pytest.mark.asyncio
async def test_auto_unknown_invoke_is_unrouted_before_registry_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loop_module, "auto_route", async_result(result("read")))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("invoke-1", "invoke", {"tool": "missing", "args": {}})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("use missing"))

    tool_result = next(
        message.tool_result
        for message in reversed(loop.store.messages())
        if message.tool_result is not None
    )
    assert tool_result is not None
    assert tool_result.structured_content["error_kind"] == "unrouted_tool"


def test_tombstoned_tool_result_keeps_durable_schema_block() -> None:
    from zeta.core.context import ContextAssembler

    schema = TextContent('routed tool schemas:\n{"name": "read"}')
    tombstone = ContextAssembler._tombstone(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("large result"), schema],
        ),
        "read",
        20,
    )

    assert tombstone.content[1] == schema
