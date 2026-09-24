from zeta.tools.agent.tests._agent_support import (
    AGENT_PRESETS,
    GENERAL_PRESET,
    AbortGenerationRegistry,
    AgentLoop,
    ConversationStore,
    FakeBackend,
    MCPMount,
    Path,
    ScriptedTurn,
    SkillCatalog,
    TextContent,
    TodoWidget,
    ToolCall,
    ToolRegistry,
    _agent_call,
    _collect,
    json,
    pytest,
    replace,
)

pytestmark = pytest.mark.usefixtures("stock_router_mode")
def test_agent_schema_uses_preset_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom = replace(
        AGENT_PRESETS["explore"],
        name="custom",  # type: ignore[arg-type]
    )
    monkeypatch.setitem(AGENT_PRESETS, "explore", custom)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(FakeBackend([]), store, registry=registry, skill_catalog=SkillCatalog.empty())

    agent_schema = next(schema for schema in registry.schemas if schema["name"] == "agent")
    agent_type_schema = agent_schema["parameters"]["properties"]["agent_type"]
    assert agent_type_schema["enum"] == [
        preset.name for preset in AGENT_PRESETS.values()
    ]
    expected_description = "Choose one of: " + "; ".join(
        f"{preset.name}: {preset.selection_guidance}"
        for preset in AGENT_PRESETS.values()
    ) + "."
    assert agent_type_schema["description"] == expected_description


@pytest.mark.asyncio
async def test_general_agent_markers_keep_legacy_state_bytes(tmp_path: Path) -> None:
    omitted = ConversationStore(tmp_path / "omitted", cwd=tmp_path)
    explicit = ConversationStore(tmp_path / "explicit", cwd=tmp_path)
    omitted_backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(call_id="omitted")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    explicit_backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    _agent_call(call_id="explicit", agent_type=GENERAL_PRESET.name)
                ]
            ),
            ScriptedTurn([TextContent("done")]),
        ]
    )

    await _collect(AgentLoop(omitted_backend, omitted, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))
    await _collect(AgentLoop(explicit_backend, explicit, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    omitted_child_state = omitted.session_dir / "agents" / "1" / "session_state.json"
    explicit_child_state = explicit.session_dir / "agents" / "1" / "session_state.json"
    assert omitted_child_state.read_bytes() == explicit_child_state.read_bytes()
    assert json.loads(omitted_child_state.read_text()) == {"bash_cwd": str(tmp_path)}
    assert omitted.state_path.read_bytes() == explicit.state_path.read_bytes()


@pytest.mark.asyncio
async def test_explore_child_has_read_only_tools_and_rejects_exec(
    tmp_path: Path,
) -> None:
    child_exec = ToolCall("child-exec", "exec", {"command": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn(tool_calls=[child_exec]),
            ScriptedTurn([TextContent("explore complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    child_schemas = {schema["name"] for schema in backend.calls[1][1]}
    assert child_schemas == {
        "agent",
        "agent_output",
        "agent_status",
        "fetch",
        "read",
        "skill",
        "websearch",
    }
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    child_result = next(
        message.tool_result for message in child_messages if message.tool_result
    )
    assert child_result.is_error
    assert child_result.content == "unknown tool: exec"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_type",
    ["explore", "plan"],
)
async def test_restricted_child_cannot_use_mounted_mcp_write_tool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_type: str,
) -> None:
    mount_calls = 0

    async def mount_write_tool(
        registry: ToolRegistry, config=None, *, notice_sink=None, home=None
    ) -> MCPMount:
        del config, notice_sink, home
        nonlocal mount_calls
        mount_calls += 1

        async def remote_write(
            arguments: dict[str, object], abort_signal
        ) -> dict[str, object]:
            del arguments, abort_signal
            return {
                "content": [{"type": "text", "text": "write executed"}],
                "isError": False,
                "structuredContent": None,
            }

        registry.register(
            "remote:write",
            remote_write,
            parameters={"type": "object"},
            requires_approval=False,
        )
        return MCPMount(())

    monkeypatch.setattr("zeta.runtime.loop.mount_mcp_servers", mount_write_tool)
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type=agent_type)]),
            ScriptedTurn(
                tool_calls=[ToolCall("remote-write", "remote:write", {})]
            ),
            ScriptedTurn([TextContent("restricted complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert mount_calls == 1
    child_result = next(
        message.tool_result
        for message in ConversationStore(
            store.session_dir / "agents", session_id="1"
        ).messages()
        if message.tool_result
    )
    assert child_result.is_error
    assert child_result.content == "unknown tool: remote:write"


@pytest.mark.asyncio
async def test_plan_child_includes_todo_and_only_read_only_tools(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="plan")]),
            ScriptedTurn([TextContent("plan complete")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert {schema["name"] for schema in backend.calls[1][1]} == {
        "agent",
        "agent_output",
        "agent_status",
        "fetch",
        "read",
        "skill",
        "todo",
        "websearch",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_type", "turn_cap"),
    [("explore", 15), ("plan", 20)],
)
async def test_typed_child_turn_cap_is_enforced(
    tmp_path: Path, agent_type: str, turn_cap: int
) -> None:
    child_call = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call(agent_type=agent_type)])]
        + [
            ScriptedTurn(
                [TextContent(f"step-{turn}")],
                tool_calls=[child_call],
            )
            for turn in range(1, turn_cap + 1)
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert f"{turn_cap}-turn cap" in result.content
    assert result.structured_content == {
        "turns_used": turn_cap,
        "child_session_path": str(store.session_dir / "agents" / "1"),
        "agent_type": agent_type,
        "child_instance_id": f"{store.session_id}:1",
    }


@pytest.mark.asyncio
async def test_unknown_agent_type_returns_loud_error(tmp_path: Path) -> None:
    call = _agent_call()
    call = ToolCall(
        call.id,
        call.name,
        {**call.arguments, "agent_type": "unknown"},
    )
    backend = FakeBackend([])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    result = await loop._run_agent_tool(
        call,
        call.arguments,
        AbortGenerationRegistry().new_generation(),
        None,
    )

    assert result["isError"] is True
    assert "unknown agent_type" in result["content"][0]["text"]
    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_typed_preamble_composes_with_child_system_prompt(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call(agent_type="explore")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(
        AgentLoop(
            backend,
            store,
            max_turns=1,
            system_prompt="existing child instructions",
skill_catalog=SkillCatalog.empty(),
        ).run_turn("start")
    )

    child_system = backend.calls[1][0][0]
    system_text = " ".join(
        block.text for block in child_system.content if isinstance(block, TextContent)
    )
    assert "You are an explore sub-agent." in system_text
    assert "existing child instructions" in system_text


@pytest.mark.asyncio
async def test_child_registry_preserves_parent_pre_execution_hook(tmp_path: Path) -> None:
    child_call = ToolCall("child-bash", "bash", {"cmd": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("hook denied")]),
        ]
    )
    observed: list[str] = []

    def deny_bash(name: str, arguments: dict[str, object]) -> str | None:
        del arguments
        observed.append(name)
        return "denied by test hook" if name == "bash" else None

    store = ConversationStore(tmp_path)
    registry = ToolRegistry(store.cwd, pre_execute_hook=deny_bash, skill_catalog=SkillCatalog.empty())
    await _collect(AgentLoop(backend, store, registry=registry, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert observed[-1] == "bash"
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    denied = next(message.tool_result for message in child_messages if message.tool_result)
    assert denied.is_error
    assert "denied by test hook" in denied.content


@pytest.mark.asyncio
async def test_child_registry_shares_todo_store_but_isolates_other_session_tools(
    tmp_path: Path,
) -> None:
    sessions = tmp_path / "sessions"
    parent_store = ConversationStore(sessions, session_id="parent", cwd=tmp_path)
    child_store = ConversationStore(sessions, session_id="child", cwd=tmp_path)
    (tmp_path / "nested").mkdir()
    registry = ToolRegistry(parent_store.cwd, session_store=parent_store, skill_catalog=SkillCatalog.empty())
    child_registry = registry.clone_for_session(child_store)
    child_loop = AgentLoop(FakeBackend([]), child_store, registry=child_registry, skill_catalog=SkillCatalog.empty())
    widget = TodoWidget(parent_store)

    await child_loop.tool_registry.execute(
        ToolCall(
            "child-todo",
            "todo",
            {"items": [{"content": "child work", "status": "pending"}]},
        )
    )
    await child_loop.tool_registry.execute(
        ToolCall("child-bash", "bash", {"cmd": "cd nested && pwd"})
    )

    assert parent_store.todo_items() == [
        {"content": "child work", "status": "pending"}
    ]
    assert child_store.todo_items() == []
    assert widget.visible
    assert parent_store.bash_cwd == str(tmp_path)
    assert child_store.bash_cwd == str(tmp_path / "nested")

    await child_loop.close()
    await registry.close()
