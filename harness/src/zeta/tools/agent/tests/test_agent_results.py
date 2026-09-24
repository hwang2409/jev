from zeta.tools.agent.tests._agent_support import (
    AgentLoop,
    ConversationStore,
    FakeBackend,
    Path,
    ScriptedTurn,
    SkillCatalog,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    _agent_call,
    _background_agent_call,
    _collect,
    asyncio,
    json,
    pytest,
    render_event,
)

pytestmark = pytest.mark.usefixtures("stock_router_mode")
@pytest.mark.asyncio
async def test_agent_result_metadata_survives_parent_replay(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    result = next(message.tool_result for message in replayed.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["turns_used"] == 1
    assert result.structured_content["child_session_path"] == str(
        store.session_dir / "agents" / "1"
    )


@pytest.mark.asyncio
async def test_agent_normal_completion_reports_all_child_turns(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(
                [TextContent("first")],
                tool_calls=[ToolCall("child-read", "read", {"path": "missing"})],
            ),
            ScriptedTurn([TextContent("second")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["turns_used"] == 2


@pytest.mark.asyncio
async def test_typed_child_type_survives_completion_and_reopen(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_type() == "explore"
    assert json.loads(child.state_path.read_text())["agent_parent"] == {
        "tool_call_id": "agent-1",
        "agent_type": "explore",
        "status": "finished",
    }
    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"


@pytest.mark.asyncio
async def test_empty_child_final_message_returns_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn()]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.is_error
    assert "empty final assistant message" in result.content


@pytest.mark.asyncio
async def test_child_answer_with_cancellation_prefix_is_success(
    tmp_path: Path,
) -> None:
    answer = "tool execution canceled, but this is the answer"
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent(answer)]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None
    assert result.is_error is False
    assert result.content.startswith(answer)
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_canceled() is None


@pytest.mark.asyncio
async def test_failed_agent_receipt_stats_match_is_error(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn()]
    )
    store = ConversationStore(tmp_path)

    events = await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result is not None
    assert result.is_error is True
    assert "error=true" in result.content
    assert "canceled=false" in result.content
    rendered = render_event(
        next(
            event
            for event in events
            if event.type is StreamEventType.TOOL_EXECUTION_END
        )
    )
    assert rendered is not None
    assert rendered.plain.count("error=true") == 1
    assert rendered.plain.count("canceled=false") == 1


@pytest.mark.asyncio
async def test_failed_background_receipt_matches_error_flag(
    tmp_path: Path,
) -> None:
    call = _background_agent_call()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        FakeBackend([ScriptedTurn(tool_calls=[call]), ScriptedTurn()]),
        store,
        max_turns=1,
skill_catalog=SkillCatalog.empty(),
    )
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
    await loop._background_owner.wait()
    notification = store.agent_notifications(pending_only=False)[0]
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )

    assert notification.data["status"] == "error"
    assert notification.data["text"].count("error=true") == 1
    assert notification.data["text"].count("canceled=false") == 1
    assert terminal.tool_result is not None
    assert terminal.tool_result.is_error is True
    assert terminal.tool_result.is_canceled is False
    rendered = render_event(terminal)
    assert rendered is not None
    assert rendered.plain.count("error=true") == 1
    assert rendered.plain.count("canceled=false") == 1
    await loop.close()


@pytest.mark.asyncio
async def test_multibyte_agent_receipt_stays_within_response_limit(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("😀" * 1_800)]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    persisted = next(message for message in store.messages() if message.tool_result)
    result = persisted.tool_result
    assert result is not None
    assert len(json.dumps(persisted.to_dict(), ensure_ascii=False).encode("utf-8")) <= 10_000
    receipt_row = next(
        row
        for row in store.path.read_bytes().splitlines()
        if b'"tool_result"' in row
    )
    assert len(receipt_row) <= 10_000
    assert result.content.count("error=false") == 1
    assert result.content.count("canceled=false") == 1


@pytest.mark.asyncio
async def test_parent_abort_cancels_child(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.sleep(0.05)
    loop.abort()

    events = await task
    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content.startswith("tool execution canceled")
    assert "error=false" in result.content
    assert "canceled=true" in result.content
    assert result.structured_content == {
        "turns_used": 0,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "child_instance_id": f"{store.session_id}:1",
    }
    rendered = render_event(
        next(
            event
            for event in events
            if event.type is StreamEventType.TOOL_EXECUTION_END
        )
    )
    assert rendered is not None
    assert rendered.plain.count("error=false") == 1
    assert rendered.plain.count("canceled=true") == 1
    child_state = (store.session_dir / "agents" / "1" / "session_state.json").read_text()
    assert '"agent_parent"' not in child_state
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_canceled() == {
        "tool_call_id": "agent-1",
        "content": "tool execution canceled",
    }
    assert not store.agent_children()


@pytest.mark.asyncio
async def test_parent_abort_after_child_turn_reports_completed_turns(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(
                [TextContent("first")],
                tool_calls=[ToolCall("child-read", "read", {"path": "missing"})],
            ),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    async for event in loop.run_turn("start"):
        if (
            event.type is StreamEventType.TOOL_EXECUTION_UPDATE
            and event.delta is not None
            and "turn 2: thinking" in event.delta
        ):
            loop.abort()

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content == {
        "turns_used": 1,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "child_instance_id": f"{store.session_id}:1",
    }


@pytest.mark.asyncio
async def test_typed_child_type_survives_cancellation_and_reopen(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("slow")], delay=5),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.sleep(0.05)
    loop.abort()

    await task

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child.agent_type() == "explore"


@pytest.mark.asyncio
async def test_parent_result_append_precedes_marker_cleanup(tmp_path: Path, monkeypatch) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)
    def fail_cleanup(tool_call_id: str) -> None:
        del tool_call_id
        raise RuntimeError("crash after parent result")

    monkeypatch.setattr(store, "finish_agent_child", fail_cleanup)
    with pytest.raises(RuntimeError, match="crash after parent result"):
        await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert store.agent_children()
    replayed = ConversationStore(tmp_path, session_id=store.session_id)
    AgentLoop(FakeBackend([]), replayed, max_turns=1, skill_catalog=SkillCatalog.empty())
    results = [message.tool_result for message in replayed.messages() if message.tool_result]
    assert len(results) == 1
    assert results[0].content.startswith("done")
    assert not replayed.agent_children()


def test_resume_resolves_dead_child_marker(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _agent_call()
    child = ConversationStore(
        store.session_dir / "agents", session_id="1", cwd=store.cwd
    )
    child.mark_agent_parent(call.id)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="task research",
    )
    store.update_agent_child_turns(call.id, 2)

    AgentLoop(FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty())

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content.startswith("tool execution canceled")
    assert result.structured_content == {
        "turns_used": 2,
        "child_session_path": str(child.session_dir),
        "child_instance_id": "parent:1",
    }
    assert not store.agent_children()
    assert '"agent_parent"' not in (child.state_path).read_text()


def test_resume_preserves_typed_child_receipt(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _agent_call(agent_type="explore")
    child = ConversationStore(
        store.session_dir / "agents", session_id="1", cwd=store.cwd
    )
    child.mark_agent_parent(call.id, agent_type="explore")
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="task research",
        agent_type="explore",
    )
    store.update_agent_child_turns(call.id, 2)

    AgentLoop(FakeBackend([]), store, max_turns=1, skill_catalog=SkillCatalog.empty())

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.structured_content is not None
    assert result.structured_content["agent_type"] == "explore"
    reopened_child = ConversationStore(store.session_dir / "agents", session_id="1")
    assert reopened_child.agent_type() == "explore"
