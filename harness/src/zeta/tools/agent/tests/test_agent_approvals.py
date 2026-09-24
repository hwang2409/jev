from zeta.tools.agent.tests._agent_support import (
    AbortGenerationRegistry,
    AgentLoop,
    ApprovalDecision,
    ApprovalPolicy,
    ChildApprovalPolicy,
    ConversationStore,
    FakeBackend,
    MessageRole,
    Path,
    ScriptedTurn,
    SkillCatalog,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolRegistry,
    ToolUseContent,
    _agent_call,
    _collect,
    asyncio,
    pytest,
)

pytestmark = pytest.mark.usefixtures("stock_router_mode")
@pytest.mark.asyncio
async def test_delegated_approvals_use_child_instance_keys(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    call_a = ToolCall("same-request", "bash", {"cmd": "a"})
    call_b = ToolCall("same-request", "bash", {"cmd": "b"})
    child_a = ConversationStore(tmp_path / "children", session_id="a")
    child_b = ConversationStore(tmp_path / "children", session_id="b")
    policy_a = ChildApprovalPolicy(policy, child_a, "a", "child-a")
    policy_b = ChildApprovalPolicy(policy, child_b, "b", "child-b")

    child_a_signal = AbortGenerationRegistry().new_generation()
    child_b_signal = AbortGenerationRegistry().new_generation()
    task_a = asyncio.create_task(policy_a.authorize(call_a, child_a_signal))
    task_b = asyncio.create_task(policy_b.authorize(call_b, child_b_signal))
    while len(policy.pending_requests()) < 2:
        await asyncio.sleep(0)

    pending = {request.key for request in policy.pending_requests()}
    assert pending == {("child-a", "same-request"), ("child-b", "same-request")}
    assert policy.approve(("child-a", "same-request"))
    assert policy.deny(("child-b", "same-request"))
    assert await task_a == ApprovalDecision.ALLOW
    assert await task_b == ApprovalDecision.DENY
    child_a_signal.abort()
    child_b_signal.abort()


@pytest.mark.asyncio
async def test_child_approval_cleanup_removes_pending_requests(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")
    signal = AbortGenerationRegistry().new_generation()
    task = asyncio.create_task(
        child_policy.authorize(ToolCall("pending", "bash", {"cmd": "wait"}), signal)
    )
    while not policy.pending_requests():
        await asyncio.sleep(0)

    child_policy.cleanup()
    assert await task == ApprovalDecision.DENY
    assert policy.pending_requests() == []
    signal.abort()


@pytest.mark.asyncio
async def test_child_approval_force_ask_overrides_allow_rule(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store, always_allow={"bash"})
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")
    signal = AbortGenerationRegistry().new_generation()
    task = asyncio.create_task(
        child_policy.authorize(
            ToolCall("forced", "bash", {"cmd": "echo forced"}),
            signal,
            force_ask=True,
        )
    )
    await asyncio.sleep(0)
    try:
        assert not task.done()
        pending = policy.pending_requests()
        assert len(pending) == 1
        assert pending[0].tool_call.id == "forced"
        assert policy.deny(pending[0].key)
        assert await task is ApprovalDecision.DENY
    finally:
        if not task.done():
            signal.abort()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_child_approval_preserves_explicit_label(tmp_path: Path) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")
    signal = AbortGenerationRegistry().new_generation()
    task = asyncio.create_task(
        child_policy.authorize(
            ToolCall("labeled", "bash", {"cmd": "echo labeled"}),
            signal,
            label="keep this label",
        )
    )
    await asyncio.sleep(0)
    try:
        assert not task.done()
        pending = policy.pending_requests()
        assert len(pending) == 1
        assert pending[0].label == "keep this label"
        assert policy.approve(pending[0].key)
        assert await task is ApprovalDecision.ALLOW
    finally:
        if not task.done():
            signal.abort()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("approve", ApprovalDecision.ALLOW),
        ("deny", ApprovalDecision.DENY),
        ("abort", None),
    ],
)
async def test_child_ephemeral_approval_resolves_without_persistence(
    tmp_path: Path, action: str, expected: ApprovalDecision | None
) -> None:
    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(store=parent_store)
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")
    signal = AbortGenerationRegistry().new_generation()
    task = asyncio.create_task(
        child_policy.authorize(
            ToolCall("ephemeral", "bash", {"cmd": "echo ephemeral"}),
            signal,
            persist_request=False,
        )
    )
    await asyncio.sleep(0)
    try:
        assert not task.done()
        pending = policy.pending_requests()
        assert len(pending) == 1
        request = pending[0]
        assert request.key == ("child-1", "ephemeral")
        assert child_store.approval_states() == {}
        assert getattr(policy, action)(request.key)
        assert await task is expected
        assert child_store.approval_states() == {}
        assert child_policy.approval_states() == {}
        assert policy.pending_requests() == []
    finally:
        if not task.done():
            signal.abort()
        await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_child_policy_inherits_scoped_rules_and_delegates_prompts(
    tmp_path: Path,
) -> None:
    """ZETA-86: scoped rules apply to children; unmatched calls still surface."""

    parent_store = ConversationStore(tmp_path / "sessions", session_id="parent")
    policy = ApprovalPolicy(
        store=parent_store,
        always_allow={"bash(git status*)"},
        always_deny={"bash(rm *)"},
    )
    policy.declare_subjects({"bash": "command"})
    child_store = ConversationStore(tmp_path / "children", session_id="child")
    child_policy = ChildApprovalPolicy(policy, child_store, "child", "child-1")

    # A child registry declares its (subset of) tools through the child policy
    # without erasing what the parent already knows.
    assert child_policy.declare_subjects({"read": "path"}) == ()
    assert child_policy.decide("bash", {"command": "git status"}) is ApprovalDecision.ALLOW
    assert child_policy.decide("bash", {"command": "rm -rf /"}) is ApprovalDecision.DENY
    assert child_policy.prepare(ToolCall("ok", "bash", {"command": "git status"})) is None
    assert child_policy.notices == ()

    signal = AbortGenerationRegistry().new_generation()
    call = ToolCall("pending", "bash", {"command": "git push"})
    task = asyncio.create_task(child_policy.authorize(call, signal))
    while not policy.pending_requests():
        await asyncio.sleep(0)

    assert [request.key for request in policy.pending_requests()] == [
        ("child-1", "pending")
    ]
    assert policy.approve(("child-1", "pending"))
    assert await task is ApprovalDecision.ALLOW
    signal.abort()


@pytest.mark.asyncio
async def test_child_loop_inherits_argument_scoped_approval_rules(
    tmp_path: Path,
) -> None:
    """ZETA-86: the agent tool's child loop is gated by the parent's scoped rules."""

    allowed_call = ToolCall("child-echo", "exec", {"command": "echo scoped-ok"})
    denied_call = ToolCall("child-rm", "exec", {"command": "rm -rf nothing-here"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[allowed_call, denied_call]),
            ScriptedTurn([TextContent("child done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(
        store=store,
        always_allow={"exec(echo *)"},
        default=ApprovalDecision.DENY,
    )

    await _collect(
        AgentLoop(backend, store, approval_policy=policy, max_turns=1, skill_catalog=SkillCatalog.empty()).run_turn("start")
    )

    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    results = {
        message.tool_result.tool_call_id: message.tool_result
        for message in child_messages
        if message.tool_result
    }
    assert not results["child-echo"].is_error
    assert "scoped-ok" in results["child-echo"].content
    assert results["child-rm"].is_error
    assert results["child-rm"].content == "tool execution denied"


@pytest.mark.asyncio
async def test_child_abort_closes_all_loop_tasks(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def wait_forever(arguments: dict[str, object], abort_signal) -> str:
        del arguments
        started.set()
        await abort_signal.wait()
        return "stopped"

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[ToolCall("child-wait", "wait", {})]),
        ]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(store.cwd, skill_catalog=SkillCatalog.empty())
    registry.register("wait", wait_forever)
    loop = AgentLoop(backend, store, registry=registry, max_turns=1, skill_catalog=SkillCatalog.empty())
    baseline = set(asyncio.all_tasks())
    task = asyncio.create_task(_collect(loop.run_turn("start")))
    await started.wait()
    loop.abort()

    await task
    await loop.close()

    assert not [
        child_task
        for child_task in asyncio.all_tasks()
        if child_task not in baseline and not child_task.done()
    ]

@pytest.mark.asyncio
async def test_child_approval_uses_parent_policy(tmp_path: Path) -> None:
    child_call = ToolCall("child-bash", "bash", {"cmd": "echo no"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call()]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("approval handled")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    events = []
    async for event in AgentLoop(
        backend, store, approval_policy=policy, max_turns=1,
        skill_catalog=SkillCatalog.empty(),
    ).run_turn("start"):
        events.append(event)
        if event.type is StreamEventType.TOOL_APPROVAL_START:
            assert [request.tool_call.id for request in policy.pending_requests()] == [
                child_call.id
            ]
            assert policy.pending_requests()[0].label == "task research: bash"
            assert all(
                message.tool_result is None or message.tool_result.tool_call_id != child_call.id
                for message in store.messages()
            )
            policy.deny(child_call.id)

    assert any(event.type is StreamEventType.TOOL_APPROVAL_END for event in events)
    assert not policy.pending_requests()
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL_RESULT,
    ]
    assert child_call.id not in {
        block.tool_call.id
        for message in store.messages()
        for block in message.content
        if isinstance(block, ToolUseContent)
    }
    child_messages = ConversationStore(
        store.session_dir / "agents", session_id="1"
    ).messages()
    denied = next(message.tool_result for message in child_messages if message.tool_result)
    assert denied.is_error
    assert denied.content == "tool execution denied"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("approve", "nested"),
        ("deny", "tool execution denied"),
        ("abort", "tool execution canceled"),
    ],
)
async def test_grandchild_approval_composes_with_parent_policy(
    tmp_path: Path, action: str, expected: str
) -> None:
    grandchild_bash = ToolCall("grandchild-bash", "bash", {"cmd": "echo nested"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[_agent_call("child")]),
            ScriptedTurn(tool_calls=[_agent_call("grandchild")]),
            ScriptedTurn(tool_calls=[grandchild_bash]),
            ScriptedTurn([TextContent("grandchild finished")]),
            ScriptedTurn([TextContent("child finished")]),
        ]
    )
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(store=store)
    loop = AgentLoop(backend, store, approval_policy=policy, max_turns=1, skill_catalog=SkillCatalog.empty())

    task = asyncio.create_task(_collect(loop.run_turn("start")))
    for _ in range(100):
        pending = policy.pending_requests()
        if pending:
            break
        await asyncio.sleep(0.01)
    assert len(pending) == 1
    request = pending[0]
    assert request.child_instance_id == f"{store.session_id}:1:1"
    assert getattr(policy, action)(request.key)
    await asyncio.wait_for(task, timeout=1)

    grandchild_store = ConversationStore(
        store.session_dir / "agents" / "1" / "agents", session_id="1"
    )
    bash_result = next(
        message.tool_result
        for message in grandchild_store.messages()
        if message.tool_result is not None
    )
    assert expected in bash_result.content
    await loop.close()
