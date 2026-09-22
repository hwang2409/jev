import asyncio


import shlex


import subprocess


from io import StringIO


from pathlib import Path


import pytest


from prompt_toolkit import PromptSession


from prompt_toolkit.buffer import Buffer


from prompt_toolkit.completion import CompleteEvent


from prompt_toolkit.data_structures import Size


from prompt_toolkit.document import Document


from prompt_toolkit.input import create_pipe_input


from prompt_toolkit.output.vt100 import Vt100_Output


from rich.console import Console


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.slash import (
    COMMAND_FILE_SIZE_LIMIT,
    INIT_PROMPT,
    CustomCommand,
    SlashModelInput,
    create_slash_registry,
    load_custom_commands,
)


from zeta.core.store import ConversationStore


from zeta.loop import AgentLoop


from zeta.mcp import MCPPrompt, MCPPromptArgument


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.tools.exec import (
    INLINE_SHELL_BATCH_TIMEOUT_MESSAGE,
    INLINE_SHELL_OUTPUT_LIMIT_MESSAGE,
    INLINE_SHELL_SPAN_LIMIT_MESSAGE,
    MacroDisplay,
    run_exec_macro,
    run_inline_shell_batch,
)


from zeta.tui.app import TUIApp


from zeta.tui.composer import (
    FullScreenPromptSession,
    SlashCompleter,
    build_key_bindings,
)


from zeta.tui.render import render_approval_card


from zeta.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent


def _write_command(directory: Path, name: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(content, encoding="utf-8")


def test_loads_home_and_project_commands_with_project_precedence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_command(home / "commands", "shared", "home $1 $ARGUMENTS")
    _write_command(home / "commands", "home-only", "home command")
    _write_command(
        project / ".zeta" / "commands",
        "shared",
        "---\ndescription: project command\nunknown: ignored\n---\nproject $1",
    )
    _write_command(project / ".zeta" / "commands", "project-only", "project command")

    registry = create_slash_registry(zeta_home=home, project_dir=project, skill_catalog=SkillCatalog.empty())

    assert registry.input_for_model("/shared first second") == "project first"
    assert registry.input_for_model("/home-only") == "home command"
    assert {command.name for command in registry.custom_commands} == {
        "home-only",
        "project-only",
        "shared",
    }
    assert str(project / ".zeta" / "commands" / "shared.md") in registry.help_text()


def test_substitution_uses_empty_missing_positions_and_keeps_raw_tail(
    tmp_path: Path,
) -> None:
    _write_command(
        tmp_path / "commands",
        "args",
        "one=$1 two=$2 nine=$9 raw=[$ARGUMENTS]",
    )

    registry = create_slash_registry(zeta_home=tmp_path, project_dir=tmp_path / "empty", skill_catalog=SkillCatalog.empty())

    assert registry.input_for_model("/args alpha  beta") == (
        "one=alpha two=beta nine= raw=[alpha  beta]"
    )
    assert registry.input_for_model("/args") == "one= two= nine= raw=[]"


def test_builtin_shadow_is_ignored_and_notices_include_bad_files(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    _write_command(home / "commands", "status", "do not replace status")
    _write_command(project / ".zeta" / "commands", "status", "also ignored")
    _write_command(project / ".zeta" / "commands", "broken", "---\nnot: [yaml")

    registry = create_slash_registry(zeta_home=home, project_dir=project, skill_catalog=SkillCatalog.empty())

    assert registry.input_for_model("/status") == "/status"
    assert any("status.md" in notice for notice in registry.notices)
    assert any("broken.md" in notice for notice in registry.notices)
    assert any("shadows built-in" in notice for notice in registry.warning_notices)


def test_completer_shows_description_and_source_badge(tmp_path: Path) -> None:
    _write_command(
        tmp_path / ".zeta" / "commands",
        "review",
        "---\ndescription: review the change\n---\nreview",
    )
    registry = create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path, skill_catalog=SkillCatalog.empty())
    completions = list(
        SlashCompleter(registry).get_completions(
            Document("/rev"), CompleteEvent(completion_requested=True)
        )
    )

    assert len(completions) == 1
    assert completions[0].text == "review"
    assert completions[0].display_meta[0][1] == "[project] review the change"


def test_init_is_in_help_and_completion(tmp_path: Path) -> None:
    registry = create_slash_registry(
        zeta_home=tmp_path / "home",
        project_dir=tmp_path,
        skill_catalog=SkillCatalog.empty(),
    )

    assert "/init — generate or improve project instructions" in registry.help_text()
    completions = list(
        SlashCompleter(registry).get_completions(
            Document("/ini"), CompleteEvent(completion_requested=True)
        )
    )
    assert len(completions) == 1
    assert completions[0].text == "init"
    assert completions[0].display_meta[0][1] == (
        "generate or improve project instructions"
    )


@pytest.mark.asyncio
async def test_init_rejects_non_project_without_model_input(tmp_path: Path) -> None:
    output = StringIO()
    backend = FakeBackend([])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=tmp_path / "home",
        console=Console(file=output, force_terminal=False),
    )

    await app._handle_prompt_value("/init")

    assert backend.calls == []
    assert "init error: not inside a project" in output.getvalue()
    await app.close()


def test_init_returns_canned_prompt_inside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    result = create_slash_registry(
        zeta_home=tmp_path / "home",
        project_dir=project,
        skill_catalog=SkillCatalog.empty(),
    ).dispatch(object(), "/init")

    assert isinstance(result, SlashModelInput)
    assert result.text == INIT_PROMPT
    assert "improve or extend" in result.text
    assert "Do not overwrite" in result.text
    assert "nested AGENTS.md" in result.text


def test_completion_applies_to_a_real_buffer(tmp_path: Path) -> None:
    _write_command(tmp_path / ".zeta" / "commands", "review", "review")
    registry = create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path, skill_catalog=SkillCatalog.empty())
    buffer = Buffer(
        completer=SlashCompleter(registry),
        document=Document("/rev"),
    )
    completion = next(
        SlashCompleter(registry).get_completions(
            buffer.document, CompleteEvent(completion_requested=True)
        )
    )

    buffer.apply_completion(completion)

    assert buffer.text == "/review"


@pytest.mark.asyncio
async def test_mcp_prompt_completion_and_resolution() -> None:
    registry = create_slash_registry(
        zeta_home=Path("/does/not/exist"),
        project_dir=Path("/does/not/exist"),
        skill_catalog=SkillCatalog.empty(),
    )
    registry.set_mcp_prompts(
        [
            (
                "server:review",
                "server",
                MCPPrompt(
                    "review",
                    "review code",
                    (MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )

    class Session:
        async def slash_mcp_prompt(
            self, name: str, arguments: dict[str, str]
        ) -> str:
            assert name == "server:review"
            return f"resolved {arguments['topic']}"

    completions = list(
        SlashCompleter(registry).get_completions(
            Document("/server:rev"), CompleteEvent(completion_requested=True)
        )
    )
    result = await registry.dispatch_async(Session(), "/server:review tests")

    assert completions[0].display_meta[0][1] == "[server] review code"
    assert isinstance(result, SlashModelInput)
    assert result.text == "resolved tests"


@pytest.mark.asyncio
async def test_mcp_prompt_missing_required_argument_does_not_call_session() -> None:
    registry = create_slash_registry(
        zeta_home=Path("/does/not/exist"),
        project_dir=Path("/does/not/exist"),
            skill_catalog=SkillCatalog.empty(),
    )
    registry.set_mcp_prompts(
        [
            (
                "server:review",
                "server",
                MCPPrompt(
                    "review",
                    "review code",
                    (MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )

    class Session:
        async def slash_mcp_prompt(
            self, name: str, arguments: dict[str, str]
        ) -> str:
            raise AssertionError("missing argument made a remote call")

    result = await registry.dispatch_async(Session(), "/server:review")

    assert isinstance(result, str)
    assert "missing required argument" in result


def test_custom_colon_name_is_rejected_for_mcp_namespace(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "user:prompt", "body")

    registry = create_slash_registry(
        zeta_home=tmp_path,
        project_dir=tmp_path / "project",
        skill_catalog=SkillCatalog.empty(),
    )

    assert all(command.name != "user:prompt" for command in registry.custom_commands)
    assert any("colon names" in notice for notice in registry.notices)


async def test_completion_menu_arrows_do_not_navigate_history(tmp_path: Path) -> None:
    _write_command(tmp_path / ".zeta" / "commands", "rebase", "rebase")
    _write_command(tmp_path / ".zeta" / "commands", "review", "review")
    registry = create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path, skill_catalog=SkillCatalog.empty())
    output_text = StringIO()

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=Vt100_Output(
                output_text,
                lambda: Size(rows=24, columns=80),
            ),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
            ),
            completer=SlashCompleter(registry),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0)
        pipe.send_text("/r\t")
        for _ in range(100):
            if session.app.current_buffer.complete_state is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("completion menu did not open")

        pipe.send_text("\x1b[A")
        for _ in range(100):
            state = session.app.current_buffer.complete_state
            if state is not None and state.complete_index is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("up arrow did not select a completion")

        assert session.app.current_buffer.text == "/review"
        session.app.exit()
        await task


def test_malformed_files_fail_open(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "empty", "   ")
    _write_command(tmp_path / "commands", "unterminated", "---\ndescription: bad")

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert len(result.notices) == 2


def test_oversized_command_file_is_ignored(tmp_path: Path) -> None:
    _write_command(
        tmp_path / "commands",
        "large",
        "x" * (COMMAND_FILE_SIZE_LIMIT + 1),
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert "byte limit" in result.notices[0]


def test_deep_yaml_command_fails_open(tmp_path: Path) -> None:
    nested_sequence = "[" * 500 + "]" * 500
    _write_command(
        tmp_path / "commands",
        "deep",
        f"---\nvalue: {nested_sequence}\n---\nbody",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert result.commands == ()
    assert any("deep.md" in notice for notice in result.notices)


def test_tui_command_loading_uses_configured_home(
    tmp_path: Path, monkeypatch
) -> None:
    live_home = tmp_path / "live-home"
    _write_command(live_home / "commands", "leak", "must not load")
    fake_home = tmp_path / "fake-home"
    _write_command(fake_home / "commands", "configured", "must load")
    monkeypatch.setattr(Path, "home", lambda: live_home)

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=fake_home,
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert all(command.name != "leak" for command in app._slash_commands.custom_commands)
    assert [command.name for command in app._slash_commands.custom_commands] == [
        "configured"
    ]


def test_tui_command_loading_skips_host_home_without_configured_home(
    tmp_path: Path, monkeypatch
) -> None:
    sentinel_home = tmp_path / "sentinel-home"
    sentinel_path = sentinel_home / ".zeta" / "commands" / "sentinel.md"
    _write_command(sentinel_home / ".zeta" / "commands", "sentinel", "must not load")
    reads: list[Path] = []
    original_read_text = Path.read_text

    def read_text(path: Path, *args, **kwargs) -> str:
        if path == sentinel_path:
            reads.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.delenv("ZETA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(sentinel_home))
    monkeypatch.setattr(Path, "home", lambda: sentinel_home)
    monkeypatch.setattr(Path, "read_text", read_text)

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert all(
        command.name != "sentinel" for command in app._slash_commands.custom_commands
    )
    assert reads == []


def test_tui_command_loading_uses_repository_root_from_nested_cwd(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    nested = repo / "src" / "nested"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    _write_command(repo / ".zeta" / "commands", "review", "review the change")

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions", cwd=nested),
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert [command.name for command in app._slash_commands.custom_commands] == [
        "review"
    ]


async def test_transcript_search_cancels_completion_before_up_navigation(
    tmp_path: Path,
) -> None:
    _write_command(tmp_path / ".zeta" / "commands", "review", "review")
    output_text = StringIO()
    search_active = False

    def start_search() -> None:
        nonlocal search_active
        search_active = True

    def end_search() -> None:
        nonlocal search_active
        search_active = False

    with create_pipe_input() as pipe:
        session = FullScreenPromptSession(
            input=pipe,
            output=Vt100_Output(
                output_text,
                lambda: Size(rows=24, columns=80),
            ),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_search_start=start_search,
                search_active=lambda: search_active,
                on_search_input=lambda _value: None,
                on_search_next=lambda: None,
                on_search_end=end_search,
            ),
            completer=SlashCompleter(
                create_slash_registry(zeta_home=tmp_path / "home", project_dir=tmp_path, skill_catalog=SkillCatalog.empty())
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0.05)
        pipe.send_text("/r\t")
        for _ in range(100):
            if session.app.current_buffer.complete_state is not None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("completion menu did not open")

        pipe.send_text("\x06")
        for _ in range(100):
            if search_active and session.app.current_buffer.complete_state is None:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("transcript search did not cancel completion")

        pipe.send_text("\r")
        await asyncio.sleep(0.05)
        pipe.send_text("\x1b")
        for _ in range(100):
            if not search_active:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("transcript search did not end")

        pipe.send_text("\x1b[A")
        await asyncio.sleep(0.05)
        assert session.app.current_buffer.text == "/r"
        session.app.exit()
        await task


async def test_custom_command_becomes_the_model_user_message(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "review", "Review $1")
    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("done")])]
    )
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions", cwd=tmp_path), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._handle_prompt_value("/review changes")
    assert app._active_task is not None
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "Review changes"


@pytest.mark.asyncio
async def test_init_becomes_the_model_user_message(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    await asyncio.to_thread(
        subprocess.run, ["git", "init", "-q"], cwd=project, check=True
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=project)
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=tmp_path / "zeta-home",
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._handle_prompt_value("/init")
    assert app._active_task is not None
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == INIT_PROMPT
    await app.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/checkpoint test", "checkpoint 'test'"),
        ("/compact", "compact: nothing to compact"),
        ("/fork", "no user messages to fork from"),
        ("/model offline", "model: offline"),
        ("/plan on", "plan mode: on"),
    ],
)
async def test_input_loop_control_commands_exclude_their_own_submission(
    tmp_path: Path, value: str, expected: str
) -> None:
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    app._input_loop_active = True

    await app._handle_prompt_value(value)
    for _ in range(100):
        if expected in output.getvalue():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError(f"{value} did not dispatch")
    assert "unavailable while a turn is running" not in output.getvalue()
    await app.loop.close()


@pytest.mark.asyncio
async def test_handler_failure_acknowledges_entry_and_advances_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    original_dispatch = app._slash_commands.dispatch_async

    async def dispatch(session: object, value: str) -> str | None:
        if value == "/fail":
            raise RuntimeError("forced handler failure")
        return await original_dispatch(session, value)

    monkeypatch.setattr(app._slash_commands, "dispatch_async", dispatch)
    app._input_loop_active = True
    await asyncio.gather(
        app._handle_prompt_value("/fail"),
        app._handle_prompt_value("later"),
    )

    for _ in range(100):
        if len(backend.calls) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("pipeline did not advance after handler failure")
    assert "submission failed: forced handler failure" in app.console.file.getvalue()
    user_message = next(
        message for message in backend.calls[0][0] if message.role.value == "user"
    )
    assert user_message.content[0].text == "later"
    await app.loop.close()


@pytest.mark.asyncio
async def test_provider_start_failure_rolls_back_and_advances_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=tmp_path / "home",
        history_path=tmp_path / "history",
        console=Console(file=StringIO(), force_terminal=False),
    )
    original_start_turn = app._start_turn
    attempts = 0

    def start_turn(*args: object, **kwargs: object) -> asyncio.Task[None]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("forced provider failure")
        return original_start_turn(*args, **kwargs)

    monkeypatch.setattr(app, "_start_turn", start_turn)
    app._input_loop_active = True
    await app._handle_prompt_value("first")
    for _ in range(100):
        if "submission failed: forced provider failure" in app.console.file.getvalue():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("provider failure was not handled")
    await app._handle_prompt_value("later")

    for _ in range(100):
        if len(backend.calls) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("pipeline did not advance after provider failure")
    assert attempts == 2
    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role is MessageRole.USER
    )
    assert user_message.content[0].text == "later"
    await app.loop.close()


@pytest.mark.asyncio
async def test_same_tick_provider_failure_dispatches_the_next_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(content=[TextContent("done")]),
            ScriptedTurn(content=[TextContent("done again")]),
        ]
    )
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    original_start_turn = app._start_turn
    attempts = 0

    def start_turn(*args: object, **kwargs: object) -> asyncio.Task[None]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("forced provider failure")
        return original_start_turn(*args, **kwargs)

    monkeypatch.setattr(app, "_start_turn", start_turn)
    app._input_loop_active = True
    await asyncio.gather(
        app._handle_prompt_value("first"),
        app._handle_prompt_value("second"),
    )

    for _ in range(100):
        if len(backend.calls) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("second submission was not dispatched")
    assert attempts == 2
    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role is MessageRole.USER
    )
    assert user_message.content[0].text == "second"
    await app.loop.close()


@pytest.mark.asyncio
async def test_failed_approval_action_acknowledges_waiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    call = ToolCall("approval-write-failure", "danger", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    policy = ApprovalPolicy(store=store)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=tmp_path / "home",
        history_path=tmp_path / "history",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )

    def fail_approval(request_id: str | tuple[str, str]) -> bool:
        del request_id
        raise OSError("forced approval write failure")

    monkeypatch.setattr(policy, "approve", fail_approval)
    await asyncio.wait_for(
        app._submissions.approval_action_wait(ApprovalDecision.ALLOW, call.id),
        timeout=1,
    )
    assert "submission failed: forced approval write failure" in app.console.file.getvalue()
    await app.loop.close()


@pytest.mark.asyncio
async def test_preprocessing_timing_cannot_reorder_provider_submissions(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "first", "first !`sleep 0.15; printf first`")
    _write_command(home / "commands", "second", "second !`printf second`")
    _write_command(home / "commands", "third", "third !`sleep 0.03; printf third`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    backend = FakeBackend(
        [
            ScriptedTurn(content=[TextContent("first reply")]),
            ScriptedTurn(content=[TextContent("second reply")]),
            ScriptedTurn(content=[TextContent("third reply")]),
        ]
    )
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        history_path=tmp_path / "history",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    await asyncio.gather(
        *(app._handle_prompt_value(f"/{name}") for name in ("first", "second", "third"))
    )
    for _ in range(100):
        if len(backend.calls) == 3:
            break
        await asyncio.sleep(0.01)

    user_texts = [
        next(
            block.text
            for message in reversed(messages)
            if message.role.value == "user"
            for block in message.content
            if isinstance(block, TextContent) and block.path is None
        )
        for messages, _schemas in backend.calls
    ]
    assert user_texts == [
        "first first",
        "second second",
        "third third",
    ]
    await app._submissions.close()
    await app.loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", [ApprovalDecision.ALLOW, ApprovalDecision.DENY])
async def test_unmapped_durable_approval_is_finalized(
    tmp_path: Path, decision: ApprovalDecision
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "inspect", "value !`printf ready`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    old_call = ToolCall("old-request", "danger", {})
    store.append_approval_request(old_call.id, old_call)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])

    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        history_path=home / "history",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=True),
    )
    app._input_loop_active = True

    submission_task = asyncio.create_task(app._handle_prompt_value("/inspect"))
    for _ in range(100):
        if len(app.pending_approvals) == 2:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("old and inline approvals did not appear")

    inline_key = next(
        request.key
        for request in app.pending_approvals
        if request.key != old_call.id
    )
    verb = "approve" if decision is ApprovalDecision.ALLOW else "deny"
    await app._handle_prompt_value(f"{verb} {old_call.id}")
    for _ in range(100):
        old_result = next(
            (
                message.tool_result
                for message in store.messages()
                if message.tool_result is not None
                and message.tool_result.tool_call_id == old_call.id
            ),
            None,
        )
        if old_result is not None:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("unmapped durable approval was not finalized")
    assert old_result is not None
    assert {request.key for request in app.pending_approvals} == {inline_key}

    await app._handle_prompt_value(f"approve {inline_key}")
    await submission_task
    await app._preprocessing_task
    app._preprocessing_task = None
    await app._active_task
    assert backend.calls
    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role is MessageRole.USER
    )
    assert isinstance(user_message.content[0], TextContent)
    assert user_message.content[0].text == "value ready"
    await app.loop.close()


@pytest.mark.asyncio
async def test_resumed_durable_tool_abort_is_processed_by_submission_consumer(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "inspect", "value !`printf ready`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    old_call = ToolCall("slow-request", "block", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(old_call)]),
        [(old_call.id, old_call)],
    )
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    started = asyncio.Event()

    async def block(arguments: dict[str, object], abort_signal: object) -> str:
        del arguments
        started.set()
        await abort_signal.wait()  # type: ignore[attr-defined]
        return "unreachable"

    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            store,
            tools={"block": block},
            approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )
    await app._submissions.approval_action_wait(
        ApprovalDecision.ALLOW, old_call.id
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    assert app._submissions.active
    app.abort_active()

    for _ in range(100):
        result = next(
            (
                message.tool_result
                for message in store.messages()
                if message.tool_result is not None
                and message.tool_result.tool_call_id == old_call.id
            ),
            None,
        )
        if result is not None:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("durable tool abort was not processed")
    assert result is not None
    assert result.content == "tool execution canceled"

    await app._submissions.close()
    await app.loop.close()


@pytest.mark.asyncio
async def test_submission_waits_for_resumed_durable_tool_result(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    old_call = ToolCall("slow-request", "block", {})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(old_call)]),
        [(old_call.id, old_call)],
    )
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    started = asyncio.Event()
    release = asyncio.Event()

    async def block(arguments: dict[str, object], abort_signal: object) -> str:
        del arguments, abort_signal
        started.set()
        await release.wait()
        return "done"

    backend = FakeBackend([ScriptedTurn(content=[TextContent("later reply")])])
    app = TUIApp(
        AgentLoop(
            backend,
            store,
            tools={"block": block},
            approval_policy=policy,
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._submissions.approval_action_wait(
        ApprovalDecision.ALLOW, old_call.id
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    await app._submissions.submit_text("later")
    await asyncio.sleep(0)
    assert backend.calls == []
    assert [message.role for message in store.messages()] == [MessageRole.ASSISTANT]

    release.set()
    for _ in range(100):
        if len(backend.calls) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("submission was not dispatched after durable result")

    messages = store.messages()
    assert [message.role for message in messages] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL_RESULT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert messages[1].tool_result is not None
    await app._submissions.close()
    await app.loop.close()


@pytest.mark.asyncio
async def test_close_resolves_pending_submission_ack_and_rejects_new_sends(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "inspect", "value !`sleep 10`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=True),
    )

    submission = asyncio.create_task(app._submissions.submit_text("/inspect"))
    await asyncio.sleep(0.05)
    assert not submission.done()

    await app._submissions.close()
    await asyncio.wait_for(submission, timeout=1)
    with pytest.raises(RuntimeError, match="submission pipeline is closed"):
        await app._submissions.submit_text("after close")
    await app.loop.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "deny"])
async def test_inline_approval_queues_unrelated_submission(
    tmp_path: Path, decision: str
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "inspect", "first !`printf first`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    backend = FakeBackend(
        [ScriptedTurn(content=[TextContent("first reply")]),
         ScriptedTurn(content=[TextContent("second reply")])]
    )
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=True),
    )
    app._input_loop_active = True

    first_task = asyncio.create_task(app._handle_prompt_value("/inspect"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("inline shell approval did not appear")

    await app._handle_prompt_value("second")
    assert len(app._approval_queue) == 1
    key = app.pending_approvals[0].key
    await app._handle_prompt_value(f"{decision} {key}")
    await first_task

    expected_call_count = 2 if decision == "approve" else 1
    for _ in range(100):
        if len(backend.calls) == expected_call_count:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("queued provider turn did not start")

    user_texts = [
        next(
            block.text
            for message in reversed(messages)
            if message.role.value == "user"
            for block in message.content
            if isinstance(block, TextContent) and block.path is None
        )
        for messages, _schemas in backend.calls
    ]
    expected_user_texts = (
        ["first first", "second"] if decision == "approve" else ["second"]
    )
    assert user_texts == expected_user_texts
    await app.loop.close()


@pytest.mark.asyncio
async def test_undo_second_inline_submission_keeps_first_alive(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "first", "first !`printf first`")
    _write_command(home / "commands", "second", "second !`printf second`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=True),
    )
    app._input_loop_active = True

    first_task = asyncio.create_task(app._handle_prompt_value("/first"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("first inline shell approval did not appear")
    second_task = asyncio.create_task(app._handle_prompt_value("/second"))
    for _ in range(100):
        if len(app.pending_approvals) == 2:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("second inline shell approval did not appear")

    first_id, second_id = app._inline_abort_signals
    first_signal = app._inline_abort_signals[first_id]
    second_signal = app._inline_abort_signals[second_id]
    app.undo_sent_turn()
    for _ in range(100):
        if second_signal.is_set():
            break
        await asyncio.sleep(0.01)
    assert not first_signal.is_set()
    assert second_signal.is_set()
    for _ in range(100):
        if len(app.pending_approvals) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("undo did not close the second approval")

    await app._handle_prompt_value(
        f"approve {app.pending_approvals[0].key}"
    )
    await asyncio.gather(first_task, second_task)
    await asyncio.gather(
        *app._preprocessing_tasks.values(), return_exceptions=True
    )
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "first first"
    await app.loop.close()


@pytest.mark.asyncio
async def test_scoped_inline_abort_keeps_other_submission_alive(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "first", "first !`printf first`")
    _write_command(home / "commands", "second", "second !`printf second`")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=True),
    )
    app._input_loop_active = True

    first_task = asyncio.create_task(app._handle_prompt_value("/first"))
    second_task = asyncio.create_task(app._handle_prompt_value("/second"))
    for _ in range(100):
        if len(app.pending_approvals) == 2:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("both inline shell approvals did not appear")

    first_id, second_id = app._inline_abort_signals
    first_signal = app._inline_abort_signals[first_id]
    second_signal = app._inline_abort_signals[second_id]
    app.abort_active(first_id)
    for _ in range(100):
        if first_signal.is_set():
            break
        await asyncio.sleep(0.01)
    assert first_signal.is_set()
    assert not second_signal.is_set()
    for _ in range(100):
        if len(app.pending_approvals) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("scoped abort did not close the first approval")

    await app._handle_prompt_value(
        f"approve {app.pending_approvals[0].key}"
    )
    await asyncio.gather(first_task, second_task)
    await asyncio.gather(
        *app._preprocessing_tasks.values(), return_exceptions=True
    )
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "second second"
    await app.loop.close()


def test_render_approval_card_ignores_provider_supplied_display_fields() -> None:
    """Provider-injected display fields must never override the real command."""

    arguments = {
        "command": "rm -rf /tmp/real-target",
        "display_command": "printf safe",
        "display_argv": ["harmless"],
    }

    output = StringIO()
    Console(file=output, force_terminal=False, width=120).print(
        render_approval_card("exec", arguments)
    )

    card = output.getvalue()
    assert "command=rm -rf /tmp/real-target" in card
    assert "printf safe" not in card
    assert "harmless" not in card
    assert "argv:" not in card
