from zeta.tools.agent.tests._agent_support import (
    MAX_AGENT_TURN_CAP,
    AgentLoop,
    BackgroundBackend,
    ConversationStore,
    FakeBackend,
    Message,
    MessageRole,
    Path,
    PendingPromptsClosedError,
    RunBackend,
    ScriptedTurn,
    SkillCatalog,
    TextContent,
    ToolCall,
    ToolRegistry,
    _agent_call,
    _background_agent_call,
    _collect,
    _FakeCredentialStore,
    _model_agent_call,
    _run_agent_call,
    _stub_backend_factory,
    _wait_for_notification,
    pytest,
)

pytestmark = pytest.mark.usefixtures("stock_router_mode")
def test_provider_for_model_maps_each_catalog_entry() -> None:
    from zeta.models.catalog import PROVIDER_MODELS, provider_for_model

    for provider, models in PROVIDER_MODELS.items():
        for model in models:
            assert provider_for_model(model) == provider


def test_provider_for_model_rejects_an_unknown_name() -> None:
    from zeta.models.catalog import provider_for_model

    with pytest.raises(ValueError) as excinfo:
        provider_for_model("gpt-nonexistent")
    message = str(excinfo.value)
    assert "gpt-nonexistent" in message
    assert "claude-opus-5" in message


def test_agent_schema_offers_every_known_model(tmp_path: Path) -> None:
    from zeta.models.catalog import known_model_names

    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(FakeBackend([]), store, registry=registry, skill_catalog=SkillCatalog.empty())

    agent_schema = next(
        schema for schema in registry.schemas if schema["name"] == "agent"
    )
    model_schema = agent_schema["parameters"]["properties"]["model"]
    assert model_schema["enum"] == known_model_names()
    assert "claude-opus-5" in model_schema["enum"]
    assert "gpt-5.4" in model_schema["enum"]


@pytest.mark.asyncio
async def test_agent_without_a_model_still_inherits_the_parent_backend(
    tmp_path: Path,
) -> None:
    """The pre-existing spawn path must keep using loop.backend."""

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_agent_call()]), ScriptedTurn([TextContent("done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("done")
    # Parent and child both ran on the one backend, so it saw both turns.
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_agent_with_a_model_runs_the_child_on_that_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("codex done")])])
    requested = _stub_backend_factory(monkeypatch, child_backend)
    parent_backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_model_agent_call(background=False)])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(parent_backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    assert requested == [("codex", "gpt-5.4")]
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.content.startswith("codex done")
    # The child talked to the substitute, never to the parent's backend.
    assert len(child_backend.calls) == 1
    assert len(parent_backend.calls) == 1


@pytest.mark.asyncio
async def test_agent_model_implies_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("codex done")])])
    _stub_backend_factory(monkeypatch, child_backend)
    parent_backend = FakeBackend([ScriptedTurn(tool_calls=[_model_agent_call()])])
    store = ConversationStore(tmp_path)
    loop = AgentLoop(parent_backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    await loop.close()


@pytest.mark.asyncio
async def test_agent_rejects_an_unknown_model_before_spawning(tmp_path: Path) -> None:
    """The schema enum catches a bad model name before the runner is reached."""

    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[_model_agent_call(model="gpt-nonexistent")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert "model is not an allowed value" in result.content
    assert not (store.session_dir / "agents").exists()


def test_resolve_child_backend_guards_bad_models(tmp_path: Path) -> None:
    """Second line of defence, for any caller that skips schema validation."""

    from zeta.agent.runner import resolve_child_backend

    parent_backend = FakeBackend([])
    loop = AgentLoop(parent_backend, ConversationStore(tmp_path), skill_catalog=SkillCatalog.empty())

    # No model at all keeps the parent's backend.
    assert resolve_child_backend(loop, None) == (parent_backend, None)

    backend, error = resolve_child_backend(loop, "gpt-nonexistent")
    assert backend is None
    assert error is not None and "unknown model" in error

    backend, error = resolve_child_backend(loop, "")
    assert backend is None
    assert error is not None and "nonempty string" in error

    backend, error = resolve_child_backend(loop, 7)
    assert backend is None
    assert error is not None and "nonempty string" in error


@pytest.mark.asyncio
async def test_agent_reports_a_missing_provider_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_backend = FakeBackend([ScriptedTurn([TextContent("unreachable")])])
    _stub_backend_factory(monkeypatch, child_backend, tokens=None)
    monkeypatch.setattr(
        "zeta.agent.runner.credential_store",
        lambda provider, **kwargs: _FakeCredentialStore(None),
    )
    backend = FakeBackend([ScriptedTurn(tool_calls=[_model_agent_call()])])
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error
    assert "not logged in to codex" in result.content
    assert "zeta login --provider codex" in result.content
    assert not child_backend.calls


def test_agent_loop_turn_cap_allows_long_runs(tmp_path: Path) -> None:
    """50 turns was too few for real work; the default has to clear it."""

    import inspect

    default = inspect.signature(AgentLoop.__init__).parameters["max_turns"].default
    assert default == 150
    assert AgentLoop(FakeBackend([]), ConversationStore(tmp_path), skill_catalog=SkillCatalog.empty()).max_turns == 150


@pytest.mark.asyncio
async def test_background_start_text_names_handle_and_polling_tools(
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
    handle = result.structured_content["child_instance_id"]
    assert type(handle) is str and handle
    assert result.structured_content["status"] == "running"
    assert f"handle={handle}" in result.content
    assert "agent_status" in result.content
    assert "agent_output" in result.content
    assert "task_output" not in result.content
    assert result.content.startswith("background agent started:")

    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_max_turns_raises_shared_tree_budget(tmp_path: Path) -> None:
    """max_turns lifts the shared tree budget without changing the outer loop cap."""

    call = _agent_call()
    call.arguments["max_turns"] = 60
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn([TextContent(f"step-{turn}")], tool_calls=[child_read])
            for turn in range(1, 41)
        ]
        + [ScriptedTurn([TextContent("child done")])]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle["tree_budget"] == 60
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is False
    assert result.content.startswith("child done")


@pytest.mark.asyncio
async def test_max_turns_hard_cap_rejects_oversized_request(tmp_path: Path) -> None:
    call = _agent_call()
    call.arguments["max_turns"] = MAX_AGENT_TURN_CAP + 1
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is True
    assert f"hard cap of {MAX_AGENT_TURN_CAP}" in result.content
    assert not (store.session_dir / "agents").exists()


@pytest.mark.asyncio
async def test_max_turns_rejected_by_schema_for_non_positive_input(
    tmp_path: Path,
) -> None:
    """Schema-level minimum:1 catches zero/negative before the runner sees them."""

    call = _agent_call()
    call.arguments["max_turns"] = 0
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    store = ConversationStore(tmp_path)
    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))
    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is True
    assert "invalid arguments" in result.content
    assert "max_turns" in result.content


@pytest.mark.asyncio
async def test_max_turns_rejected_from_nested_agent_calls(tmp_path: Path) -> None:
    """Children inherit the tree budget; they can't override it mid-tree."""

    child = _agent_call("child")
    grandchild = _agent_call("grandchild")
    grandchild.arguments["max_turns"] = 5
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[child]),
            ScriptedTurn(tool_calls=[grandchild]),
            ScriptedTurn([TextContent("recover")]),
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    child_store = ConversationStore(store.session_dir / "agents", session_id="1")
    nested = next(
        message.tool_result
        for message in child_store.messages()
        if message.tool_result and message.tool_result.tool_call_id == grandchild.id
    )
    assert nested.is_error is True
    assert "max_turns is only accepted at the top-level" in nested.content


@pytest.mark.asyncio
async def test_budget_exhaustion_error_reports_used_and_allocated(
    tmp_path: Path,
) -> None:
    call = _agent_call()
    call.arguments["max_turns"] = 2
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn([TextContent(f"step-{turn}")], tool_calls=[child_read])
            for turn in range(1, 4)
        ]
    )
    store = ConversationStore(tmp_path)

    await _collect(AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.is_error is True
    assert result.structured_content is not None
    assert result.structured_content["error_code"] == "agent_turn_budget"
    assert "shared agent turn budget exhausted" in result.content
    assert "2 of 2 turns used" in result.content
    assert "agent_output" in result.content
    assert str(store.session_dir / "agents" / "1") in result.content


@pytest.mark.asyncio
async def test_child_transcript_survives_budget_exhaustion(tmp_path: Path) -> None:
    """After budget death the child work must still be readable via agent_output."""

    call = _agent_call()
    call.arguments["max_turns"] = 2
    child_read = ToolCall("child-read", "read", {"path": "missing"})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call])]
        + [
            ScriptedTurn(
                [TextContent(f"work step {turn}")], tool_calls=[child_read]
            )
            for turn in range(1, 4)
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())
    await _collect(loop.run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    handle = result.structured_content["child_instance_id"]

    output = await loop.tool_registry.execute(
        ToolCall("output-after-death", "agent_output", {"handle": handle})
    )

    assert output["isError"] is False
    text = output["content"][0]["text"]
    assert "assistant: work step 1" in text
    assert "assistant: work step 2" in text


@pytest.mark.asyncio
async def test_max_turns_bounded_by_hard_cap_constant() -> None:
    """The hard cap constant must be documented, positive, and above defaults."""

    assert type(MAX_AGENT_TURN_CAP) is int
    assert MAX_AGENT_TURN_CAP >= 100
    from zeta.agent.presets import AGENT_PRESETS

    assert MAX_AGENT_TURN_CAP >= max(
        preset.turn_cap for preset in AGENT_PRESETS.values()
    )


def test_run_preset_is_registered_with_a_long_cap(tmp_path: Path) -> None:
    from zeta.agent.presets import AGENT_PRESETS, RUN_PRESET

    assert RUN_PRESET.turn_cap == 150
    assert RUN_PRESET.tool_names is None
    assert AGENT_PRESETS["run"] is RUN_PRESET

    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    AgentLoop(FakeBackend([]), store, registry=registry, skill_catalog=SkillCatalog.empty())
    agent_schema = next(
        schema for schema in registry.schemas if schema["name"] == "agent"
    )
    assert "run" in agent_schema["parameters"]["properties"]["agent_type"]["enum"]


@pytest.mark.asyncio
async def test_a_run_does_not_draw_on_the_shared_sibling_budget(
    tmp_path: Path,
) -> None:
    """A tree budget is fresh per top-level call (ZETA-62), so a run started
    after a smaller-preset sibling on the same loop must not inherit its cap.
    """

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("explore-1", agent_type="explore")]),
            ScriptedTurn([TextContent("explore done")]),
            ScriptedTurn(tool_calls=[_run_agent_call()]),
            ScriptedTurn([TextContent("run done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    # The explore child sized its own tree at its 15-turn preset cap.
    explore_store = ConversationStore(store.session_dir / "agents", session_id="1")
    assert explore_store.agent_lifecycle()["tree_budget"] == 15

    await _collect(loop.run_turn("now start the run"))
    # The run gets a fresh tree, not the explore sibling's smaller cap.
    run_store = ConversationStore(store.session_dir / "agents", session_id="2")
    assert run_store.agent_lifecycle()["tree_budget"] == 150

    await _wait_for_notification(store, "completed")
    await loop.close()


@pytest.mark.asyncio
async def test_a_run_goes_to_the_background_without_being_asked(
    tmp_path: Path,
) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))

    result = next(
        message.tool_result for message in store.messages() if message.tool_result
    )
    assert result.structured_content is not None
    assert result.structured_content["status"] == "running"
    backend.release_child.set()
    await _wait_for_notification(store, "completed")
    await loop.close()


def test_pending_prompt_queue_round_trips(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="run")
    first = store.append_pending_prompt("also update the docs")
    store.append_pending_prompt("and run the linter")
    assert [entry.data["text"] for entry in store.pending_prompts()] == [
        "also update the docs",
        "and run the linter",
    ]

    store.acknowledge_pending_prompt(first.id)
    assert [entry.data["text"] for entry in store.pending_prompts()] == [
        "and run the linter"
    ]
    # Acknowledging twice is a no-op rather than an integrity error.
    store.acknowledge_pending_prompt(first.id)

    # A separate handle sees the queue, and the queue survives a reload.
    assert [
        entry.data["text"]
        for entry in ConversationStore(tmp_path, session_id="run").pending_prompts()
    ] == ["and run the linter"]

    with pytest.raises(ValueError):
        store.append_pending_prompt("   ")
    with pytest.raises(ValueError):
        store.acknowledge_pending_prompt("nonexistent")


def test_close_pending_queue_if_empty_locks_out_new_prompts(tmp_path: Path) -> None:
    """Once a run declares itself done, further follow-ups must be rejected."""

    store = ConversationStore(tmp_path, session_id="run")

    # With prompts pending the queue stays open and callers still see them.
    store.append_pending_prompt("keep working")
    pending = store.close_pending_queue_if_empty()
    assert [entry.data["text"] for entry in pending] == ["keep working"]
    store.acknowledge_pending_prompt(pending[0].id)

    # Second call finds nothing pending and closes the door.
    assert store.close_pending_queue_if_empty() == []
    with pytest.raises(PendingPromptsClosedError):
        store.append_pending_prompt("too late")
    # A separate handle sees the closed door too, not just this instance.
    other = ConversationStore(tmp_path, session_id="run")
    with pytest.raises(PendingPromptsClosedError):
        other.append_pending_prompt("also too late")


@pytest.mark.asyncio
async def test_consume_run_keeps_pending_entry_when_delivery_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed follow-up delivery must leave the queue intact for recovery."""

    from zeta.agent import runner as agent_runner

    child_store = ConversationStore(tmp_path, session_id="run")
    child_store.append_pending_prompt("follow up")

    call_prompts: list[str] = []

    async def fake_consume_child(child_loop, prompt, **kwargs):
        del child_loop, kwargs
        call_prompts.append(prompt)
        if len(call_prompts) == 1:
            return {"content": [], "isError": False, "structuredContent": None}
        return {"content": [], "isError": True, "structuredContent": None}

    monkeypatch.setattr(agent_runner, "consume_child", fake_consume_child)

    result = await agent_runner.consume_run(
        object(),  # child_loop is unused by the fake
        "initial",
        child_store=child_store,
    )
    assert result["isError"] is True
    assert call_prompts == ["initial", "follow up"]
    # Ack must not have run since the follow-up delivery errored.
    assert [entry.data["text"] for entry in child_store.pending_prompts()] == [
        "follow up"
    ]
    with pytest.raises(PendingPromptsClosedError):
        child_store.append_pending_prompt("after failure")


@pytest.mark.asyncio
async def test_restart_keeps_run_lifecycle_open_for_an_in_flight_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zeta.agent import runner as agent_runner

    child_store = ConversationStore(tmp_path, session_id="run")
    child_store.start_agent_lifecycle(
        handle="parent:run",
        started_at="2026-09-08T00:00:00+00:00",
        tree_budget=150,
        depth=1,
        agent_type="run",
        description="long horizon run",
    )
    child_store.append_pending_prompt("follow up")
    prompts: list[str] = []
    finish_calls = 0

    async def fake_consume_child(child_loop, prompt, **kwargs):
        del child_loop, kwargs
        prompts.append(prompt)
        return {
            "content": [
                {
                    "text": (
                        f"handled {prompt} · 2 turns · 1.2s · 3 tool calls "
                        "· error=false · canceled=false"
                    )
                }
            ],
            "isError": False,
            "structuredContent": None,
        }

    def finish_lifecycle(state: str, text: str) -> dict[str, object]:
        nonlocal finish_calls
        finish_calls += 1
        child_store.finish_agent_lifecycle(state, final_result=text)
        return {}

    original_close = child_store.close_pending_queue_if_empty
    checked_restart = False

    def close_after_restart_check():
        nonlocal checked_restart
        if not checked_restart:
            checked_restart = True
            restarted = ConversationStore(tmp_path, session_id="run")
            lifecycle = restarted.agent_lifecycle()
            assert lifecycle is not None
            assert lifecycle["finished_at"] is None
            assert restarted.pending_prompts()[0].data["text"] == "follow up"
        return original_close()

    monkeypatch.setattr(agent_runner, "consume_child", fake_consume_child)
    monkeypatch.setattr(
        child_store,
        "close_pending_queue_if_empty",
        close_after_restart_check,
    )

    result = await agent_runner.consume_run(
        object(),
        "initial",
        child_store=child_store,
        child_turns=lambda: len(prompts),
        finish_lifecycle=finish_lifecycle,
    )

    assert result["isError"] is False
    assert prompts == ["initial", "follow up"]
    assert finish_calls == 1
    lifecycle = child_store.agent_lifecycle()
    assert lifecycle is not None
    assert lifecycle["state"] == "completed"
    assert lifecycle["finished_at"] is not None
    assert lifecycle["final_result"] == "handled follow up"


@pytest.mark.asyncio
async def test_a_queued_prompt_stays_out_of_the_run_context(tmp_path: Path) -> None:
    """Only messages reach the model; a queued follow-up must not leak in early."""

    store = ConversationStore(tmp_path, session_id="run")
    store.append_message(Message(MessageRole.USER, [TextContent("work the big task")]))
    store.append_pending_prompt("secret follow-up")
    loop = AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty())

    assembled = await loop.context_assembler.assemble()

    rendered = "\n".join(
        block.text
        for message in assembled
        for block in message.content
        if isinstance(block, TextContent)
    )
    assert "work the big task" in rendered
    assert "secret follow-up" not in rendered


@pytest.mark.asyncio
async def test_a_run_with_an_empty_queue_finishes_normally(tmp_path: Path) -> None:
    backend = RunBackend()
    store = ConversationStore(tmp_path)
    loop = AgentLoop(backend, store, max_turns=1, skill_catalog=SkillCatalog.empty())

    await _collect(loop.run_turn("start"))
    backend.release_child.set()
    await _wait_for_notification(store, "completed")

    assert backend.child_prompts == ["work the big task"]
    assert not store.agent_children()
    await loop.close()
