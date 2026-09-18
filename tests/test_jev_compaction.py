from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Self

import pytest

from zeta.cli import build_parser
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.providers import jev
from zeta.settings import Settings, resolve
from zeta.skills import SkillCatalog
from zeta.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def user(text: str) -> Message:
    return Message(MessageRole.USER, [TextContent(text)])


def tool_messages(call_id: str, tool: str, content: str) -> tuple[Message, Message]:
    call = ToolCall(call_id, tool, {})
    return (
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent(content)],
            tool_result=ToolResult(call_id, content),
        ),
    )


def compact_count(message: Message) -> int:
    if message.role is MessageRole.TOOL_RESULT:
        text = message.tool_result.content if message.tool_result is not None else ""
        return 1 if text.startswith("[dropped by jev-compaction:") else 15
    return 1


def summary_count(message: Message) -> int:
    if message.role is MessageRole.TOOL_RESULT:
        return 12_000
    return 1


def still_over_count(message: Message) -> int:
    if message.role is MessageRole.TOOL_RESULT:
        text = message.tool_result.content if message.tool_result is not None else ""
        return 11_000 if text.startswith("[dropped by jev-compaction:") else 12_000
    return 1


def triage_result(
    probabilities: dict[str, float],
) -> jev.TriageResult:
    return jev.TriageResult(probabilities, {"input_tokens": 7, "output_tokens": 2})


@pytest.mark.asyncio
async def test_budget_recovery_tombstones_and_persists_tool_pairing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old objective"))
    call, result = tool_messages("call-1", "read", "x" * 400)
    store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(user("current objective"))
    backend = FakeBackend([])
    async def fake_triage(_task: str, items: list[dict[str, str]]) -> jev.TriageResult:
        return await _triage(items, {result_entry.id: 0.1})

    monkeypatch.setattr("zeta.core.context.jev.triage", fake_triage)

    assembler = ContextAssembler(
        store,
        token_budget=10,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )
    messages = await assembler.assemble()

    tombstone = messages[2]
    assert tombstone.role is MessageRole.TOOL_RESULT
    assert tombstone.tool_result is not None
    assert tombstone.tool_result.tool_call_id == "call-1"
    assert tombstone.tool_result.content == (
        "[dropped by jev-compaction: read result, ~15 tokens]"
    )
    assert tombstone.tool_result.content_blocks == [
        {
            "type": "text",
            "text": "[dropped by jev-compaction: read result, ~15 tokens]",
            "truncated": False,
            "full_size": 52,
        }
    ]
    assert backend.calls == []

    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["triage_messages"][2] == tombstone.to_dict()
    assert marker.data["jev_triage"] == {
        "candidates": 1,
        "dropped": 1,
        "tokens_recovered": 14,
        "skipped_summarize": True,
        "dropped_items": [{"id": result_entry.id, "tokens": 15}],
        "usage": {"input_tokens": 7, "output_tokens": 2},
    }
    original = next(entry for entry in store.replay() if entry.id == result_entry.id)
    assert original.data["message"] == result.to_dict()
    session_id = store.session_id
    store.close()
    reopened = ConversationStore(tmp_path, session_id=session_id, _must_exist=True)
    persisted = next(entry for entry in reopened.replay() if entry.type == "compaction")
    assert persisted.data["jev_triage"]["dropped_items"] == [
        {"id": result_entry.id, "tokens": 15}
    ]
    reopened.close()


@pytest.mark.asyncio
async def test_calibrated_budget_still_runs_summary_without_recovery(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old objective"))
    call, result = tool_messages("call-1", "read", "x" * 400)
    store.append_message(call)
    store.append_message(result)
    store.append_message(user("current objective"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])

    async def keep_everything(
        _task: str, items: list[dict[str, str]]
    ) -> jev.TriageResult:
        return triage_result({item["id"]: 1.0 for item in items})

    monkeypatch.setattr("zeta.core.context.jev.triage", keep_everything)
    assembler = ContextAssembler(
        store,
        token_budget=20,
        retained_tail=1,
        token_counter=compact_count,
        backend=backend,
    )
    assembler.record_usage({"total_tokens": 100})
    summary_called = False

    async def summarize(*_args: Any, **_kwargs: Any) -> str:
        nonlocal summary_called
        summary_called = True
        return "summary"

    monkeypatch.setattr(assembler.compaction_policy, "summarize", summarize)

    await assembler.assemble()

    assert summary_called is True
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["jev_triage"]["skipped_summarize"] is False


@pytest.mark.asyncio
async def test_calibrated_budget_skips_summary_after_genuine_recovery(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old objective"))
    call, result = tool_messages("call-1", "read", "x" * 400)
    store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(user("current objective"))
    backend = FakeBackend([])

    async def drop_result(
        _task: str, items: list[dict[str, str]]
    ) -> jev.TriageResult:
        return triage_result({items[0]["id"]: 0.1})

    monkeypatch.setattr("zeta.core.context.jev.triage", drop_result)

    def calibrated_count(message: Message) -> int:
        if message.role is MessageRole.TOOL_RESULT:
            return 1 if message.tool_result and message.tool_result.content.startswith(
                "[dropped by jev-compaction:"
            ) else 90
        return 1

    assembler = ContextAssembler(
        store,
        token_budget=20,
        retained_tail=1,
        token_counter=calibrated_count,
        backend=backend,
    )
    assembler.record_usage({"total_tokens": 100})

    await assembler.assemble()

    assert backend.calls == []
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["jev_triage"]["dropped_items"] == [
        {"id": result_entry.id, "tokens": 90}
    ]


@pytest.mark.asyncio
async def test_reloaded_triage_content_can_be_compacted_again(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old objective"))
    call, result = tool_messages("call-1", "read", "x" * 400)
    store.append_message(call)
    store.append_message(result)
    store.append_message(user("current objective"))
    first = ContextAssembler(
        store,
        token_budget=1,
        retained_tail=1,
        token_counter=lambda message: (
            1
            if message.role is MessageRole.TOOL_RESULT
            and message.tool_result is not None
            and message.tool_result.content.startswith("[dropped by jev-compaction:")
            else 1
            if message.content
            and isinstance(message.content[0], TextContent)
            and "new objective" in message.content[0].text
            else 0
        ),
    )

    async def drop_all(
        _task: str, items: list[dict[str, str]]
    ) -> jev.TriageResult:
        return triage_result({item["id"]: 0.1 for item in items})

    monkeypatch.setattr("zeta.core.context.jev.triage", drop_all)
    await first.assemble(force=True)
    session_id = store.session_id
    store.close()

    reopened = ConversationStore(tmp_path, session_id=session_id, _must_exist=True)
    reopened.append_message(user("new objective"))
    backend = FakeBackend([ScriptedTurn([TextContent("second summary")])])
    second = ContextAssembler(
        reopened,
        token_budget=1,
        retained_tail=1,
        token_counter=lambda message: (
            1
            if message.role is MessageRole.TOOL_RESULT
            and message.tool_result is not None
            and message.tool_result.content.startswith("[dropped by jev-compaction:")
            else 1
            if message.content
            and isinstance(message.content[0], TextContent)
            and "new objective" in message.content[0].text
            else 0
        ),
        backend=backend,
    )
    summary_called = False

    async def summarize(*_args: Any, **_kwargs: Any) -> str:
        nonlocal summary_called
        summary_called = True
        return "second summary"

    monkeypatch.setattr(second.compaction_policy, "summarize", summarize)

    await second.assemble()

    assert summary_called is True


@pytest.mark.asyncio
async def test_reloaded_triage_items_keep_distinct_ids(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old objective"))
    call_one, result_one = tool_messages("call-1", "read", "x" * 400)
    call_two, result_two = tool_messages("call-2", "write", "y" * 400)
    store.append_message(call_one)
    result_one_entry = store.append_message(result_one)
    store.append_message(call_two)
    result_two_entry = store.append_message(result_two)
    store.append_message(user("current objective"))
    requests: list[list[dict[str, str]]] = []

    async def triage_items(
        _task: str, items: list[dict[str, str]]
    ) -> jev.TriageResult:
        requests.append(items)
        probabilities = {item["id"]: 1.0 for item in items}
        if len(requests) == 2:
            probabilities[items[0]["id"]] = 0.1
        return triage_result(probabilities)

    monkeypatch.setattr("zeta.core.context.jev.triage", triage_items)
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        token_counter=compact_count,
    )
    await assembler.assemble(force=True)
    session_id = store.session_id
    store.close()

    reopened = ConversationStore(tmp_path, session_id=session_id, _must_exist=True)
    reopened.append_message(user("new objective"))
    fresh = ContextAssembler(
        reopened,
        token_budget=100,
        retained_tail=1,
        token_counter=compact_count,
    )
    messages = await fresh.assemble(force=True)

    assert [item["id"] for item in requests[1]] == [
        result_one_entry.id,
        result_two_entry.id,
    ]
    dropped = [
        message.tool_result.tool_call_id
        for message in messages
        if message.tool_result is not None
        and message.tool_result.content.startswith("[dropped by jev-compaction:")
    ]
    assert dropped == ["call-1"]


@pytest.mark.asyncio
async def test_triage_usage_is_a_service_tagged_event(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old objective"))
    call, result = tool_messages("call-1", "read", "x" * 400)
    store.append_message(call)
    store.append_message(result)
    backend = FakeBackend([ScriptedTurn([TextContent("done")])])

    async def drop_result(
        _task: str, items: list[dict[str, str]]
    ) -> jev.TriageResult:
        return jev.TriageResult(
            {items[0]["id"]: 0.1},
            {"input_tokens": 7, "output_tokens": 2},
        )

    monkeypatch.setattr("zeta.core.context.jev.triage", drop_result)
    assembler = ContextAssembler(
        store,
        token_budget=3,
        retained_tail=1,
        token_counter=lambda message: (
            0
            if message.role is MessageRole.TOOL_RESULT
            and message.tool_result is not None
            and message.tool_result.content.startswith("[dropped by jev-compaction:")
            else 1
        ),
        backend=backend,
    )
    loop = AgentLoop(
        backend,
        store,
        context_assembler=assembler,
        router_mode=False,
        skill_catalog=SkillCatalog.empty(),
    )

    events = [event async for event in loop.run_turn("current objective")]

    usage = next(event for event in events if event.type.value == "usage")
    compaction = next(
        event for event in events if event.type.value == "compaction_end"
    )
    assert usage.data == {
        "service": "jev",
        "usage": {"input_tokens": 7, "output_tokens": 2},
    }
    assert compaction.data["jev_triage"] == {
        "candidates": 1,
        "dropped": 1,
        "tokens_recovered": 1,
        "skipped_summarize": True,
        "dropped_items": [
            {
                "id": next(
                    entry.id
                    for entry in store.replay()
                    if entry.type == "message"
                    and Message.from_dict(entry.data["message"]).role
                    is MessageRole.TOOL_RESULT
                ),
                "tokens": 1,
            }
        ],
    }


@pytest.mark.asyncio
async def test_recent_tool_results_are_not_triaged(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old"))
    old_call, old_result = tool_messages("old-call", "old-tool", "o" * 400)
    store.append_message(old_call)
    old_result_entry = store.append_message(old_result)
    recent_call, recent_result = tool_messages("recent-call", "recent-tool", "r" * 400)
    store.append_message(recent_call)
    recent_result_entry = store.append_message(recent_result)
    store.append_message(user("recent"))
    requests: list[list[dict[str, str]]] = []

    async def fake_triage(_task: str, items: list[dict[str, str]]) -> jev.TriageResult:
        requests.append(items)
        return triage_result({old_result_entry.id: 0.1})

    monkeypatch.setattr("zeta.core.context.jev.triage", fake_triage)
    assembler = ContextAssembler(
        store,
        token_budget=20,
        retained_tail=3,
        token_counter=compact_count,
    )

    messages = await assembler.assemble()

    assert [item["id"] for item in requests[0]] == [old_result_entry.id]
    recent = next(
        message
        for message in messages
        if message.tool_result is not None
        and message.tool_result.tool_call_id == recent_result_entry.data["message"]["tool_result"]["tool_call_id"]
    )
    assert recent.tool_result is not None
    assert recent.tool_result.content == "r" * 400


@pytest.mark.asyncio
async def test_size_floor_skips_jev_and_uses_stock_summary(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old"))
    call, result = tool_messages("call-1", "read", "x" * 200)
    store.append_message(call)
    store.append_message(result)
    store.append_message(user("tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    called = False

    async def fake_triage(_task: str, _items: list[dict[str, str]]) -> jev.TriageResult:
        nonlocal called
        called = True
        return triage_result({})

    monkeypatch.setattr("zeta.core.context.jev.triage", fake_triage)
    assembler = ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=1,
        token_counter=lambda _message: 10,
        backend=backend,
    )

    await assembler.assemble(force=True)

    assert called is False
    assert len(backend.calls) == 1


@pytest.mark.asyncio
async def test_jev_error_keeps_stock_compaction_byte_identical(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make_store(path: Any) -> ConversationStore:
        store = ConversationStore(path)
        store.append_message(user("old"))
        call, result = tool_messages("call-1", "read", "x" * 400)
        store.append_message(call)
        store.append_message(result)
        store.append_message(user("tail"))
        return store

    async def fail_triage(_task: str, _items: list[dict[str, str]]) -> jev.TriageResult:
        raise jev.JevRouterError("JEV_API_KEY is not set")

    monkeypatch.setattr("zeta.core.context.jev.triage", fail_triage)
    on_backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    off_backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    on = ContextAssembler(
        make_store(tmp_path / "on"),
        token_budget=10_000,
        retained_tail=1,
        token_counter=summary_count,
        backend=on_backend,
    )
    off = ContextAssembler(
        make_store(tmp_path / "off"),
        token_budget=10_000,
        retained_tail=1,
        token_counter=summary_count,
        backend=off_backend,
        jev_compaction=False,
    )

    on_context = await on.assemble_context()
    off_context = await off.assemble_context()

    assert [message.to_dict() for message in on_backend.calls[0][0]] == [
        message.to_dict() for message in off_backend.calls[0][0]
    ]
    assert [message.to_dict() for message in on_context.messages] == [
        message.to_dict() for message in off_context.messages
    ]


@pytest.mark.asyncio
async def test_still_over_budget_summarizes_post_tombstone_range(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path)
    store.append_message(user("old"))
    call, result = tool_messages("call-1", "read", "x" * 400)
    store.append_message(call)
    result_entry = store.append_message(result)
    store.append_message(user("tail"))
    backend = FakeBackend([ScriptedTurn([TextContent("summary")])])
    async def fake_triage(_task: str, items: list[dict[str, str]]) -> jev.TriageResult:
        return await _triage(items, {result_entry.id: 0.1})

    monkeypatch.setattr("zeta.core.context.jev.triage", fake_triage)
    assembler = ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=1,
        token_counter=still_over_count,
        backend=backend,
    )

    await assembler.assemble()

    assert len(backend.calls) == 1
    source = backend.calls[0][0][-1].content[0].text
    assert "dropped by jev-compaction" in source
    marker = next(entry for entry in store.replay() if entry.type == "compaction")
    assert marker.data["jev_triage"]["skipped_summarize"] is False


def test_triage_request_shape_and_parser() -> None:
    item = {
        "id": "entry-1",
        "kind": "tool_result",
        "tool": "read",
        "excerpt": "result",
    }
    request = jev.build_triage_request("t" * 600, [item])

    assert request["state"] == {"task": "t" * 500, "items": [item]}
    assert request["questions"]["entry-1"]["type"] == "noul"
    assert "item entry-1" in request["questions"]["entry-1"]["instructions"]
    parsed = jev.parse_triage_response(
        {"answers": {"entry-1": {"noul": 0.2}}}, ["entry-1"]
    )
    assert parsed.keep_probabilities == {"entry-1": 0.2}


@pytest.mark.asyncio
async def test_triage_http_uses_route_auth_and_response_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class Response:
        status_code: int
        body: dict[str, Any]
        text: str = ""

        def json(self) -> dict[str, Any]:
            return self.body

    class Client:
        requests: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> Response:
            self.requests.append({"url": url, **kwargs})
            return Response(
                200,
                {
                    "answers": {"entry-1": {"noul": 0.1}},
                    "usage": {"input_tokens": 3},
                },
            )

    monkeypatch.setattr(jev.httpx, "AsyncClient", Client)
    monkeypatch.setenv("JEV_API_KEY", "test-key")
    result = await jev.triage("task", [{"id": "entry-1"}])

    assert result.keep_probabilities == {"entry-1": 0.1}
    assert Client.requests[0]["headers"] == {"Authorization": "Bearer test-key"}


async def _triage(
    items: list[dict[str, str]], probabilities: dict[str, float]
) -> jev.TriageResult:
    assert items
    return triage_result(probabilities)


def test_config_flag_defaults_on_and_supports_opt_out() -> None:
    parser = build_parser()

    assert parser.parse_args([]).jev_compaction is None
    assert parser.parse_args(["--no-jev-compaction"]).jev_compaction is False
    assert (
        resolve(
            Settings(jev_compaction=False),
            cli_provider=None,
            cli_model=None,
            cli_yolo=None,
            cli_token_budget=None,
        ).jev_compaction
        is False
    )
