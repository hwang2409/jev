from zeta.tools.agent.tests._agent_support import (
    AgentLoop,
    AgentTree,
    ConversationStore,
    FakeBackend,
    ForegroundNestedBackgroundBackend,
    NestedBlockingBackend,
    Path,
    ScriptedTurn,
    SkillCatalog,
    TextContent,
    ToolCall,
    _agent_call,
    _background_agent_call,
    _collect,
    _persist_background_receipt,
    _wait_for_notification,
    asyncio,
    pytest,
)

pytestmark = pytest.mark.usefixtures("stock_router_mode")
@pytest.mark.asyncio
async def test_agent_turn_cap_returns_loud_error(tmp_path: Path) -> None:
    child_call = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()])]
        + [
            ScriptedTurn(
                [TextContent(f"step-{turn}")],
                tool_calls=[child_call],
            )
            for turn in range(1, 26)
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.is_error
    assert "25-turn cap" in result.content
    assert "partial state is saved" in result.content
    assert "last assistant text: step-25" in result.content
    assert "turns used: 25" in result.content


@pytest.mark.asyncio
async def test_child_agent_call_allows_one_grandchild(tmp_path: Path) -> None:
    nested = _agent_call("nested")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("grandchild complete")]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.content.startswith("child complete")
    child_result = next(
        message.tool_result
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
        if message.tool_result
    )
    assert child_result.content.startswith("grandchild complete")
    grandchild_schemas = {
        schema["name"] for schema in backend.calls[2][1]
    }
    assert "agent" not in grandchild_schemas


@pytest.mark.asyncio
async def test_nested_typed_child_only_tightens_tools(tmp_path: Path) -> None:
    nested = _agent_call("grandchild")
    child = _agent_call("child", "explore")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("grandchild complete")]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    child_tools = {schema["name"] for schema in backend.calls[1][1]}
    grandchild_tools = {schema["name"] for schema in backend.calls[2][1]}
    assert child_tools == {
        "agent",
        "agent_output",
        "agent_status",
        "fetch",
        "read",
        "skill",
        "websearch",
    }
    assert grandchild_tools == {
        "agent_status",
        "agent_output",
        "fetch",
        "read",
        "skill",
        "websearch",
    }


@pytest.mark.asyncio
async def test_shared_turn_budget_covers_generations(tmp_path: Path) -> None:
    nested = _agent_call("grandchild")
    child = _agent_call("child")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[nested]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            agent_turn_budget=1,
skill_catalog=SkillCatalog.empty(),
        ).run_turn("start")
    )

    result = next(message.tool_result for message in store.messages() if message.tool_result)
    assert result.is_error
    assert "shared agent turn budget exhausted" in result.content
    assert result.structured_content is not None
    assert result.structured_content["error_code"] == "agent_turn_budget"
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    child_result = next(
        message.tool_result
        for message in child_store.messages()
        if message.tool_result
    )
    assert child_result.is_error
    assert "shared agent turn budget exhausted" in child_result.content


@pytest.mark.asyncio
async def test_shared_turn_budget_covers_parallel_grandchildren(tmp_path: Path) -> None:
    grandchildren = [_agent_call("grandchild-1"), _agent_call("grandchild-2")]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("child")]),
            ScriptedTurn(tool_calls=grandchildren),
            ScriptedTurn([TextContent("one")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            agent_turn_budget=2,
skill_catalog=SkillCatalog.empty(),
        ).run_turn("start")
    )

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert [result.is_error for result in results if result is not None] == [True]
    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    child_results = [
        message.tool_result
        for message in child_store.messages()
        if message.tool_result is not None
    ]
    assert sorted(result.is_error for result in child_results) == [False, True]
    assert any(
        result.structured_content is not None
        and result.structured_content.get("error_code") == "agent_turn_budget"
        for result in child_results
    )


def test_agent_loop_rejects_conflicting_turn_budget_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pass only one agent turn budget"):
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path),
            agent_turn_budget=1,
            agent_tree=AgentTree(),
skill_catalog=SkillCatalog.empty(),
        )


@pytest.mark.asyncio
async def test_top_level_agent_calls_get_fresh_shared_turn_budgets(
    tmp_path: Path,
) -> None:
    first = _agent_call("first")
    second = _agent_call("second")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[first]),
            ScriptedTurn([TextContent("first complete")]),
            ScriptedTurn(tool_calls=[second]),
            ScriptedTurn([TextContent("second complete")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, agent_turn_budget=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    await _collect(loop.run_turn("follow up"))

    results = [
        message.tool_result for message in store.messages() if message.tool_result
    ]
    assert all(
        result is not None and result.content.startswith(expected)
        for result, expected in zip(
            results, ("first complete", "second complete"), strict=True
        )
    )


@pytest.mark.asyncio
async def test_parallel_top_level_agent_invocations_have_independent_budgets(
    tmp_path: Path,
) -> None:
    async def run_agent(index: int) -> str:
        call = _agent_call(f"agent-{index}")
        backend = FakeBackend(
            [
                ScriptedTurn(tool_calls=[call]),
                ScriptedTurn([TextContent(f"agent {index} complete")]),
            ]
        )
        store = ConversationStore(tmp_path / str(index))
        loop = AgentLoop(backend, store, max_turns=1, agent_turn_budget=1, skill_catalog=SkillCatalog.empty())
        await _collect(loop.run_turn("start"))
        result = next(
            message.tool_result
            for message in store.messages()
            if message.tool_result
        )
        return result.content

    results = await asyncio.gather(run_agent(1), run_agent(2))
    assert results[0].startswith("agent 1 complete")
    assert results[1].startswith("agent 2 complete")


@pytest.mark.asyncio
async def test_adopted_background_child_keeps_origin_tree_budget(
    tmp_path: Path,
) -> None:
    backend = ForegroundNestedBackgroundBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, agent_turn_budget=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    notification = await _wait_for_notification(store, "error")

    assert (
        "shared agent turn budget exhausted for this agent tree"
        in notification.data["text"]
    )


@pytest.mark.asyncio
async def test_background_grandchild_keeps_its_own_notification(tmp_path: Path) -> None:
    nested = _agent_call("grandchild")
    nested.arguments["background"] = True
    child = _background_agent_call("child")
    child.arguments["prompt"] = "inspect the task"
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("child complete")]),
            ScriptedTurn([TextContent("child finished")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))
    await _wait_for_notification(store, "completed")

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    nested_result = next(
        message.tool_result
        for message in child_store.messages()
        if message.tool_result
    )
    assert nested_result.structured_content is not None
    assert nested_result.structured_content["status"] == "running"
    assert [entry.data["status"] for entry in child_store.agent_notifications()] == [
        "completed"
    ]
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert grandchild_store.agent_canceled() is None


@pytest.mark.asyncio
async def test_abort_propagates_through_two_nested_levels(tmp_path: Path) -> None:
    backend = NestedBlockingBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect(loop.run_turn("start")))

    await asyncio.wait_for(backend.grandchild_started.wait(), timeout=1)
    loop.abort()
    await asyncio.wait_for(task, timeout=1)

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    grandchild_store = ConversationStore(
        child_store.session_dir / "agents", session_id="1"
    )
    assert child_store.agent_canceled() == {
        "tool_call_id": "child",
        "content": "tool execution canceled",
    }
    assert grandchild_store.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }
    await loop.close()


def test_resume_cancels_nested_tree_markers(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    child_call = _agent_call("child")
    grandchild_call = _agent_call("grandchild")
    child.mark_agent_parent(child_call.id)
    grandchild.mark_agent_parent(grandchild_call.id)
    child.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        child_instance_id="root:1:1",
    )
    root.register_agent_child(
        child_call,
        child_session_path=str(child.session_dir),
        description="child",
    )

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    assert not resumed.agent_children()
    assert ConversationStore(
        root.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": "child",
        "content": "tool execution canceled",
    }
    assert ConversationStore(
        child.session_dir / "agents", session_id="1"
    ).agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }


def test_resume_cancels_nested_background_tree_markers(tmp_path: Path) -> None:
    root = ConversationStore(tmp_path, session_id="root")
    child = ConversationStore(root.session_dir / "agents", session_id="1")
    grandchild = ConversationStore(child.session_dir / "agents", session_id="1")
    child_call = _background_agent_call("child")
    grandchild_call = _background_agent_call("grandchild")
    child_call.arguments["description"] = "child"
    grandchild_call.arguments["description"] = "grandchild"
    child.mark_agent_parent(child_call.id)
    grandchild.mark_agent_parent(grandchild_call.id)
    _persist_background_receipt(child, grandchild_call, grandchild)
    child.register_agent_child(
        grandchild_call,
        child_session_path=str(grandchild.session_dir),
        description="grandchild",
        background=True,
        child_instance_id="root:1:1",
    )
    _persist_background_receipt(root, child_call, child)
    root.register_agent_child(
        child_call,
        child_session_path=str(child.session_dir),
        description="child",
        background=True,
        child_instance_id="root:1",
    )

    resumed = ConversationStore(tmp_path, session_id="root")
    AgentLoop(FakeBackend([]), resumed, skill_catalog=SkillCatalog.empty())

    assert resumed.agent_notifications()[0].data["status"] == "canceled"
    resumed_child = ConversationStore(
        resumed.session_dir / "agents", session_id="1"
    )
    resumed_grandchild = ConversationStore(
        resumed_child.session_dir / "agents", session_id="1"
    )
    assert resumed_child.agent_notifications()[0].data["status"] == "canceled"
    assert not resumed.agent_children()
    assert not resumed_child.agent_children()
    assert resumed_grandchild.agent_canceled() == {
        "tool_call_id": "grandchild",
        "content": "tool execution canceled",
    }
