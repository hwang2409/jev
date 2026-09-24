from zeta.tools.agent.tests._agent_support import (
    AgentLoop,
    ApprovalPolicy,
    BackgroundAgentOwner,
    BackgroundBackend,
    ConversationStore,
    FakeBackend,
    ForegroundNestedBackgroundBackend,
    Message,
    MessageRole,
    NestedBackgroundBackend,
    ParallelApprovalBackend,
    ParallelChildrenBackend,
    ParallelNestedReadBackend,
    Path,
    ScriptedTurn,
    SkillCatalog,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    _agent_call,
    _background_agent_call,
    _collect,
    _parallel_agent_calls,
    _persist_background_receipt,
    _wait_for_notification,
    adopt_agent_children,
    asyncio,
    finish_background_child,
    json,
    pytest,
    render_event,
)

pytestmark = pytest.mark.usefixtures("stock_router_mode")
@pytest.mark.asyncio
async def test_background_owner_waits_for_child_close_before_unregister(
    tmp_path: Path,
) -> None:
    parent_store = ConversationStore(tmp_path / "parent")
    child_store = ConversationStore(tmp_path / "child")
    call = _agent_call()
    parent_store.register_agent_child(
        call,
        child_session_path=str(child_store.session_dir),
        description="task research",
    )
    child_store.mark_agent_parent(call.id)
    owner = BackgroundAgentOwner(parent_store)
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    cleanup_called = asyncio.Event()

    async def close_child() -> None:
        close_started.set()
        await release_close.wait()

    def cleanup() -> None:
        cleanup_called.set()
        owner.unregister("child-1")

    watcher = asyncio.create_task(
        finish_background_child(
            child_task=asyncio.create_task(
                asyncio.sleep(
                    0,
                    result={
                        "content": [{"text": "done"}],
                        "isError": False,
                    },
                )
            ),
            child_store=child_store,
            parent_store=parent_store,
            notification_store=parent_store,
            tool_call=call,
            child_instance_id="child-1",
            child_path=str(child_store.session_dir),
            description="task research",
            child_turns=lambda: 0,
            build_result=lambda text, error, status: {
                "content": [{"text": text}],
                "isError": error,
                "structuredContent": {"status": status},
            },
            validate_result=lambda result, tool_call_id: ToolResult(
                tool_call_id,
                result["content"][0]["text"],
                is_error=result["isError"],
                structured_content=result["structuredContent"],
            ),
            publish_event=lambda event: None,
            cleanup=cleanup,
            close_child=close_child,
            error_message=lambda exc: str(exc),
            background_owner=owner,
        )
    )
    owner.register("child-1", lambda: None, watcher, parent_store)

    await close_started.wait()
    waiting = asyncio.create_task(owner.wait())
    await asyncio.sleep(0)
    assert not waiting.done()
    assert not cleanup_called.is_set()

    release_close.set()
    await watcher
    await waiting
    assert cleanup_called.is_set()


@pytest.mark.asyncio
async def test_background_agent_returns_handle_and_parent_continues(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    assert result.structured_content["description"] == "background research"
    assert backend.child_started.is_set()

    parent_events = await _collect(loop.run_turn("follow up"))
    assert any(
        event.type is StreamEventType.TURN_END for event in parent_events
    )
    assert store.agent_notifications() == []

    backend.release_child.set()
    notification = await _wait_for_notification(store, "completed")
    assert notification.data["text"].startswith("child complete")
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_notification_waits_for_next_turn_boundary(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    events = await _collect(loop.run_turn("follow up"))
    assert events[0].type is StreamEventType.AGENT_NOTIFICATION
    assert events[0].data["text"].startswith("child complete")
    assert events[0].data["text"].count("error=false") == 1
    assert events[0].data["text"].count("canceled=false") == 1
    rendered = render_event(events[0])
    assert rendered is not None
    assert rendered.plain.count("error=false") == 1
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.content.count("error=false") == 1
    assert loop.store.agent_notifications() == []
    await loop.close()


@pytest.mark.asyncio
async def test_background_multibyte_receipt_fits_persisted_limit(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    backend.child_text = "😀" * 1_800
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.content.count("error=false") == 1
    assert terminal.tool_result.content.count("canceled=false") == 1
    notification = store.agent_notifications(pending_only=False)[0]
    notification_row = next(
        row
        for row in store.path.read_bytes().splitlines()
        if b'"type":"notification"' in row
    )
    assert len(notification_row) <= 10_000
    assert notification.data["text"].count("error=false") == 1
    assert notification.data["text"].count("canceled=false") == 1
    persisted = Message(
        MessageRole.TOOL_RESULT,
        [TextContent(terminal.tool_result.content)],
        tool_result=terminal.tool_result,
    )
    assert len(json.dumps(persisted.to_dict(), ensure_ascii=False).encode("utf-8")) <= 10_000
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_during_setup_is_drained_at_turn_start(
    tmp_path: Path,
) -> None:
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skip_mcp_mount=True, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    setup_started = asyncio.Event()

    async def delayed_setup() -> None:
        setup_started.set()
        backend.release_child.set()
        while not store.agent_notifications():
            await asyncio.sleep(0)

    loop._ensure_mcp_servers = delayed_setup
    events = await _collect(loop.run_turn("follow up"))

    assert setup_started.is_set()
    assert events[0].type is StreamEventType.AGENT_NOTIFICATION
    assert events[0].data["status"] == "completed"
    assert events[1].type is StreamEventType.AGENT_START
    assert store.agent_notifications() == []
    await loop.close()


@pytest.mark.asyncio
async def test_background_completion_leaves_no_pending_abort_waiter(
    tmp_path: Path,
) -> None:
    baseline = set(asyncio.all_tasks())
    backend = BackgroundBackend([_background_agent_call()])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    for _ in range(100):
        if not loop._tracked_tasks:
            break
        await asyncio.sleep(0)

    assert not [
        task
        for task in asyncio.all_tasks()
        if task not in baseline and not task.done()
    ]
    await loop.close()


@pytest.mark.asyncio
async def test_parent_abort_cancels_background_agent(tmp_path: Path) -> None:
    call = _background_agent_call()
    backend = BackgroundBackend([call])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    background_events: list[StreamEvent] = []
    loop.set_background_event_sink(background_events.append)

    await _collect(loop.run_turn("start"))
    loop.abort()
    notification = await _wait_for_notification(store, "canceled")
    assert "parent session exited" not in notification.data["text"]
    assert notification.data["text"].count("error=false") == 1
    assert notification.data["text"].count("canceled=true") == 1
    terminal = next(
        event
        for event in background_events
        if event.type is StreamEventType.TOOL_EXECUTION_END
    )
    assert terminal.tool_result is not None
    assert terminal.tool_result.is_error is False
    assert terminal.tool_result.is_canceled is True
    rendered = render_event(terminal)
    assert rendered is not None
    assert rendered.plain.count("error=false") == 1
    assert rendered.plain.count("canceled=true") == 1
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert child_store.agent_canceled() == {
        "tool_call_id": call.id,
        "content": "tool execution canceled",
    }
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_parent_abort_cancels_background_grandchild(
    tmp_path: Path,
) -> None:
    backend = NestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    loop.abort()
    await _wait_for_notification(store, "canceled")

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert child_store.agent_notifications(pending_only=False)[0].data["status"] == (
        "canceled"
    )
    assert grandchild_store.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }
    assert not store.agent_children()
    await loop.close()


@pytest.mark.asyncio
async def test_foreground_child_does_not_wait_for_background_grandchild(
    tmp_path: Path,
) -> None:
    backend = ForegroundNestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    events = await asyncio.wait_for(task, timeout=1)

    assert any(event.type is StreamEventType.TURN_END for event in events)
    assert store.agent_notifications() == []
    marker = next(iter(store.agent_children().values()))
    assert marker["child_session_path"] == str(
        store.session_dir / "agents" / "1" / "agents" / "1"
    )

    backend.release_grandchild.set()
    notification = await _wait_for_notification(store, "completed")
    assert notification.data["text"].startswith("grandchild complete")
    await loop.close()
    assert not store.agent_children()


@pytest.mark.asyncio
async def test_background_and_foreground_tools_mix_in_one_turn(
    tmp_path: Path,
) -> None:
    background = _background_agent_call()
    foreground = ToolCall("read-1", "read", {"path": "missing"})
    backend = BackgroundBackend([background, foreground])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert len(results) == 2
    assert any(
        result is not None
        and result.structured_content is not None
        and result.structured_content.get("status") == "running"
        for result in results
    )
    assert any(result is not None and result.is_error for result in results)
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


def test_resume_cancels_live_background_child(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _background_agent_call()
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.mark_agent_parent(call.id)
    _persist_background_receipt(store, call, child)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="background research",
        background=True,
    )

    resumed = ConversationStore(tmp_path, session_id="parent")
    AgentLoop(BackgroundBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "canceled"
    assert "session exited" in notification.data["text"]
    assert not resumed.agent_children()
    assert ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": call.id,
        "content": "tool execution canceled",
    }


def test_resume_keeps_completed_unnotified_background_notification(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = _background_agent_call()
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.mark_agent_parent(call.id)
    _persist_background_receipt(store, call, child)
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="background research",
        background=True,
    )
    store.append_agent_notification(
        "parent:1",
        child_session_path=str(child.session_dir),
        description="background research",
        status="completed",
        text="child complete",
    )

    resumed = ConversationStore(tmp_path, session_id="parent")
    AgentLoop(BackgroundBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "completed"
    assert notification.data["text"].startswith("child complete")
    assert not resumed.agent_children()
    assert ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).agent_canceled() is None


def test_resume_cancels_adopted_background_grandchild(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(
        child.session_dir / "agents", session_id="1"
    )
    child_call = _background_agent_call("child")
    child.mark_agent_parent(child_call.id)
    child.finish_agent_parent()
    grandchild_call = _background_agent_call("grandchild")
    grandchild.mark_agent_parent(grandchild_call.id)
    root.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        background=True,
        child_instance_id="root:1:1",
    )

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notification = resumed.agent_notifications()[0]
    assert notification.data["status"] == "canceled"
    assert "session exited" in notification.data["text"]
    assert not resumed.agent_children()
    assert ConversationStore(
        child.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": grandchild_call.id,
        "content": "tool execution canceled",
    }


def test_resume_keeps_same_id_adopted_background_grandchildren_separate(
    tmp_path: Path,
) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    children = [
        ConversationStore(root.session_dir / "agents", session_id=str(index))
        for index in (1, 2)
    ]
    grandchild_call = _background_agent_call("same-grandchild")

    for index, child in enumerate(children, start=1):
        grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
        grandchild.mark_agent_parent(grandchild_call.id)
        child.register_agent_child(
            grandchild_call,
            child_session_path=str(grandchild.session_dir),
            description=f"grandchild {index}",
            background=True,
            child_instance_id=f"root:{index}:1",
        )
        adopt_agent_children(child, root)

    assert set(root.agent_children()) == {"root:1:1", "root:2:1"}
    root.finish_agent_child("root:1:1")
    assert set(root.agent_children()) == {"root:2:1"}

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    notifications = resumed.agent_notifications(pending_only=False)
    assert [entry.data["child_instance_id"] for entry in notifications] == [
        "root:2:1"
    ]
    recovered = ConversationStore(
        children[1].session_dir / "agents", session_id="1"
    )
    assert recovered.agent_canceled() == {
        "tool_call_id": "same-grandchild",
        "content": "tool execution canceled",
    }


@pytest.mark.asyncio
async def test_background_persistence_failure_does_not_block_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = _background_agent_call()
    backend = BackgroundBackend([call])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))

    def fail_notification(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("persistence failed")

    monkeypatch.setattr(store, "append_agent_notification", fail_notification)
    backend.release_child.set()
    await asyncio.wait_for(loop.close(), timeout=1)

    assert not loop.background_children_running


@pytest.mark.asyncio
async def test_parallel_agent_calls_overlap_and_keep_child_results(
    tmp_path: Path,
) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelChildrenBackend(calls)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.wait_for(backend.children_started.wait(), timeout=1)
    backend.release_children.set()
    await asyncio.wait_for(task, timeout=1)

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert all(
        result is not None and result.content.startswith(expected)
        for result, expected in zip(results, ("child-1", "child-2"), strict=True)
    )
    assert sorted(path.name for path in (store.session_dir / "agents").iterdir()) == [
        "1",
        "2",
    ]


@pytest.mark.asyncio
async def test_parallel_nested_lifecycle_events_survive_large_batch(
    tmp_path: Path,
) -> None:
    backend = ParallelNestedReadBackend(44)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        max_turns=1,
        agent_turn_budget=100,
        skill_catalog=SkillCatalog.empty(),
    )

    events = await asyncio.wait_for(_collect(loop.run_turn("start")), timeout=30)

    read_lifecycle = [
        event
        for event in events
        if event.tool_call is not None
        and event.tool_call.name == "read"
        and event.type
        in {
            StreamEventType.TOOL_EXECUTION_START,
            StreamEventType.TOOL_EXECUTION_END,
        }
    ]
    assert len(read_lifecycle) == 88
    errors = [
        event.error
        for event in events
        if event.type is StreamEventType.ERROR and event.error is not None
    ]
    assert [error.code for error in errors] == ["max_turns"]
    await loop.close()


@pytest.mark.asyncio
async def test_duplicate_parallel_agent_ids_fail_before_child_dispatch(
    tmp_path: Path,
) -> None:
    calls = [_agent_call("same-id"), _agent_call("same-id")]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=calls),
            ScriptedTurn([TextContent("child one")]),
            ScriptedTurn([TextContent("child two")]),
        ]
    )
    store = ConversationStore(tmp_path)

    with pytest.raises(ValueError, match="duplicate tool call id"):
        await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_parent_abort_cancels_all_parallel_children(tmp_path: Path) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelChildrenBackend(calls)
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await asyncio.wait_for(backend.children_started.wait(), timeout=1)
    loop.abort()
    await asyncio.wait_for(task, timeout=1)

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert len(results) == 2
    assert all(
        result is not None
        and result.content.startswith("tool execution canceled")
        for result in results
    )
    for index, call in enumerate(calls, start=1):
        child_store = ConversationStore(
            store.session_dir / "agents", session_id=str(index)
        )
        assert child_store.agent_canceled() == {
            "tool_call_id": call.id,
            "content": "tool execution canceled",
        }


@pytest.mark.asyncio
async def test_parallel_delegated_approvals_resolve_by_child_instance(
    tmp_path: Path,
) -> None:
    calls = _parallel_agent_calls()
    backend = ParallelApprovalBackend(calls)
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(backend, store, approval_policy=policy, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    for _ in range(100):
        if len(policy.pending_requests()) == 2:
            break
        await asyncio.sleep(0.01)
    pending = policy.pending_requests()
    assert len(pending) == 2
    assert {request.child_instance_id for request in pending} == {
        f"{store.session_id}:1",
        f"{store.session_id}:2",
    }
    allow, deny = pending
    assert policy.approve(allow.key)
    assert policy.deny(deny.key)
    await asyncio.wait_for(task, timeout=1)
    assert policy.pending_requests() == []


@pytest.mark.asyncio
async def test_agent_returns_child_text_and_persists_child_session(tmp_path: Path) -> None:
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content.startswith("done")
    child_dir = store.session_dir / "agents" / "1"
    assert (child_dir / "conversation.jsonl").exists()
    assert [message.role for message in ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()] == [MessageRole.USER, MessageRole.ASSISTANT]
    assert "agent" in {
        schema["name"] for schema in backend.calls[1][1]
    }
    assert {
        schema["name"] for schema in backend.calls[1][1]
    } == {
        schema["name"] for schema in backend.calls[0][1]
    }
