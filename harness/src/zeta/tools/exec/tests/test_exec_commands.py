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


from zeta.runtime.loop import AgentLoop


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


from zeta.protocol.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent


def _write_command(directory: Path, name: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.md").write_text(content, encoding="utf-8")


async def test_prompt_macro_resolves_inline_shell_and_template_attachments(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    notes = tmp_path / "notes.txt"
    image = tmp_path / "image.png"
    notes.write_text("template note", encoding="utf-8")
    image.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c6360606000000004000000a6f645"
            "0000000049454e44ae426082"
        )
    )
    _write_command(
        home / "commands",
        "inspect",
        "read !`printf shell-value` @./notes.txt @./image.png",
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )

    await app._handle_prompt_value("/inspect")
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "read shell-value @./notes.txt @./image.png"
    assert any(
        isinstance(block, TextContent) and block.text.endswith("template note")
        for block in user_message.content
    )
    assert any(block.type.value == "image" for block in user_message.content)


async def test_inline_shell_approval_covers_the_complete_batch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "inspect",
        "values !`printf first` and !`printf second`",
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=True, width=120),
    )

    task = asyncio.create_task(app._handle_prompt_value("/inspect"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("inline shell approval did not appear")
    assert len(app.pending_approvals) == 1
    assert "printf first" in output.getvalue()
    assert "printf second" in output.getvalue()
    assert policy.approve(app.pending_approvals[0].key)
    await task
    await app._active_task

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "values first and second"


async def test_inline_shell_approval_input_is_consumed_during_preprocessing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "inspect", "value !`printf ready`")
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

    submission_task = asyncio.create_task(app._handle_prompt_value("/inspect"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("inline shell approval did not appear")

    await app._handle_prompt_value(f"approve {app.pending_approvals[0].key}")
    await submission_task
    assert app._preprocessing_task is not None
    await app._preprocessing_task
    app._preprocessing_task = None
    assert app._active_task is not None
    await app._active_task
    assert backend.calls


async def test_inline_shell_abort_stops_before_provider_dispatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "inspect", "value !`sleep 30`")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("must not run")])])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )
    app._input_loop_active = True

    submission_task = asyncio.create_task(app._handle_prompt_value("/inspect"))
    for _ in range(100):
        if app._inline_abort_signals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("inline shell did not start")
    app.abort_active()
    await submission_task
    assert app._preprocessing_task is not None
    await app._preprocessing_task
    app._preprocessing_task = None
    assert backend.calls == []


async def test_inline_shell_denial_prevents_later_commands(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    registry = ToolRegistry(tmp_path, approval_policy=policy, skill_catalog=SkillCatalog.empty())
    marker = tmp_path / "must-not-execute"
    lifecycle_calls: list[str] = []
    task = asyncio.create_task(
        run_inline_shell_batch(
            registry,
            ("printf first", f"touch {shlex.quote(str(marker))}"),
            lifecycle_sink=lambda _kind, call: lifecycle_calls.append(call.id),
        )
    )
    for _ in range(100):
        if policy.pending_requests():
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("inline shell approval did not appear")
    assert policy.deny(policy.pending_requests()[0].key)
    output = await task

    assert output == ("[inline shell failed: denied]", "[inline shell failed: denied]")
    assert not marker.exists()
    assert lifecycle_calls


async def test_capture_output_does_not_create_file_named_none(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    await registry.execute(
        ToolCall("capture", "exec", {"command": "printf output"}),
        _capture_output=True,
    )

    assert not (tmp_path / "None").exists()


async def test_inline_shell_caps_spans_output_and_batch_time(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    marker = tmp_path / "span-cap-marker"
    span_output = await run_inline_shell_batch(
        registry,
        ("printf first", f"touch {shlex.quote(str(marker))}"),
        lifecycle_sink=lambda _kind, _call: None,
        max_spans=1,
    )
    output_cap = await run_inline_shell_batch(
        registry,
        ("printf 123", "printf never"),
        lifecycle_sink=lambda _kind, _call: None,
        total_output_limit=3,
    )
    time_cap = await run_inline_shell_batch(
        registry,
        ("sleep 1", "printf never"),
        lifecycle_sink=lambda _kind, _call: None,
        timeout=1.0,
        batch_timeout=0.01,
    )

    assert span_output == ("first", INLINE_SHELL_SPAN_LIMIT_MESSAGE)
    assert not marker.exists()
    assert output_cap == ("123", INLINE_SHELL_OUTPUT_LIMIT_MESSAGE)
    assert time_cap == (INLINE_SHELL_BATCH_TIMEOUT_MESSAGE,) * 2


async def test_inline_shell_failure_and_output_are_bounded(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "inspect",
        "bad !`sh -c 'printf bad >&2; exit 4'` long !`printf '%020000d' 1`",
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions", cwd=tmp_path),
skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=True),
    )

    await app._handle_prompt_value("/inspect")
    await app._active_task
    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert "[inline shell failed: exit 4]" in user_message.content[0].text
    assert len(user_message.content[0].text) < 10_000
    assert await run_inline_shell_batch(
        ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()),
        ("sleep 1",),
        lifecycle_sink=lambda _kind, _call: None,
        timeout=0.01,
    ) == ("[inline shell failed: timed out]",)


async def test_background_exec_macro_notifies_on_next_turn_and_cancels_on_exit(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "background",
        "---\nkind: exec\nbackground: true\n---\nprintf complete\n",
    )
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=True),
    )

    await app._handle_prompt_value("/background")
    assert "/background · running" in output.getvalue()
    await app.loop._background_owner.wait()
    assert store.agent_notifications()[0].data["status"] == "completed"

    await app._handle_prompt_value("continue")
    await app._active_task
    assert "background · /background · completed" in output.getvalue()

    slow_home = tmp_path / "slow-home"
    _write_command(
        slow_home / "commands",
        "slow",
        "---\nkind: exec\nbackground: true\n---\nsleep 30\n",
    )
    slow_store = ConversationStore(tmp_path / "slow-sessions", cwd=tmp_path)
    slow_app = TUIApp(
        AgentLoop(FakeBackend([]), slow_store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=slow_home,
        console=Console(file=StringIO(), force_terminal=True),
    )
    await slow_app._handle_prompt_value("/slow")
    await slow_app.loop.close()
    assert slow_store.agent_notifications()[0].data["status"] == "canceled"


def test_exec_kind_loads_and_unknown_kind_fails_open(tmp_path: Path) -> None:
    _write_command(
        tmp_path / "commands",
        "rebuild",
        "---\nkind: exec\n---\nprintf rebuild\n",
    )
    _write_command(
        tmp_path / "commands",
        "unknown",
        "---\nkind: mystery\n---\nprintf unknown\n",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert [command.kind for command in result.commands] == ["exec"]
    assert any("unknown command kind" in notice for notice in result.notices)


def test_exec_macro_timeout_defaults_and_reads_frontmatter(tmp_path: Path) -> None:
    _write_command(tmp_path / "commands", "default", "---\nkind: exec\n---\necho default")
    _write_command(
        tmp_path / "commands",
        "custom",
        "---\nkind: exec\ntimeout: 12.5\n---\necho custom",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert [(command.name, command.timeout) for command in result.commands] == [
        ("custom", 12.5),
        ("default", 300.0),
    ]


def test_prompt_timeout_metadata_is_ignored(tmp_path: Path) -> None:
    _write_command(
        tmp_path / "commands",
        "prompt",
        "---\ntimeout: not-a-number\n---\nuse the prompt",
    )
    _write_command(
        tmp_path / "commands",
        "exec",
        "---\nkind: exec\ntimeout: not-a-number\n---\necho exec",
    )

    result = load_custom_commands(home=tmp_path, project_dir=tmp_path / "project")

    assert [command.name for command in result.commands] == ["prompt"]
    assert result.commands[0].timeout == 300.0
    assert any("exec.md" in notice for notice in result.notices)


async def test_exec_macro_streams_receipt_writes_log_and_skips_provider(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "rebuild",
        "---\nkind: exec\n---\nprintf 'arg=%s all=%s\\n' $1 \"$ARGUMENTS\"\nprintf failure >&2\nexit 3\n",
    )
    backend = FakeBackend([])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=False),
    )

    await app._handle_prompt_value("/rebuild first second")

    log = next(store.session_dir.glob("macro-*.log"))
    assert log.read_text(encoding="utf-8") == "arg=first all=first second\nfailure"
    assert backend.calls == []
    assert "/rebuild · exit 3" in output.getvalue()
    assert "failure" not in output.getvalue()


async def test_exec_macro_approval_is_ephemeral_and_uses_substituted_script(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "deploy",
        "---\nkind: exec\n---\nprintf approved-$1\n",
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=False),
    )

    task = asyncio.create_task(app._handle_prompt_value("/deploy now"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")
    assert "command=printf approved-now" in output.getvalue()
    assert store.messages() == []
    assert policy.approve(app.pending_approvals[0].key)
    await task
    assert store.messages() == []


def test_exec_macro_approval_card_keeps_script_and_every_argv_value() -> None:
    trusted = MacroDisplay(
        "printf '%s\\n' \"$@\"",
        ("first-target", "second target", "$HOME"),
    )

    output = StringIO()
    Console(file=output, force_terminal=False, width=80).print(
        render_approval_card(
            "exec",
            {"command": "sh -c 'printf ...'"},
            trusted_display=trusted,
        )
    )

    card = output.getvalue()
    assert 'command=printf \'%s\\n\' "$@"' in card
    assert "[1] first-target" in card
    assert "[2] second target" in card
    assert "[3] $HOME" in card


async def test_exec_macro_approval_card_shows_argv_via_trusted_display(
    tmp_path: Path,
) -> None:
    """The rendered card must show argv for the actual macro tool call."""

    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "deploy",
        "---\nkind: exec\n---\nprintf '%s\\n' \"$@\"\n",
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=False, width=120),
    )

    task = asyncio.create_task(
        app._handle_prompt_value("/deploy staging release-42 $HOME")
    )
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")

    card = output.getvalue()
    assert "argv:" in card
    assert "[1] staging" in card
    assert "[2] release-42" in card
    assert "[3] $HOME" in card

    assert policy.approve(app.pending_approvals[0].key)
    await task


async def test_exec_macro_deny_renders_denied_receipt(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "deploy", "---\nkind: exec\n---\nprintf deploy")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=output, force_terminal=False),
    )

    task = asyncio.create_task(app._handle_prompt_value("/deploy"))
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")

    assert policy.deny(app.pending_approvals[0].key)
    await task

    assert "/deploy · denied" in output.getvalue()
    assert "/deploy · failed" not in output.getvalue()
    assert store.messages() == []


async def test_exec_macro_abort_kills_process_and_renders_canceled_receipt(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "wait", "---\nkind: exec\n---\nsleep 30\n")
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            FakeBackend([]), ConversationStore(tmp_path / "sessions", cwd=tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=False),
    )

    task = asyncio.create_task(app._handle_prompt_value("/wait"))
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)
    app.abort_active()
    await asyncio.wait_for(task, timeout=3)

    assert "/wait · canceled" in output.getvalue()
    log = next(app.loop.store.session_dir.glob("macro-*.log"))
    assert log.exists()
    assert "canceled · log " in output.getvalue()


async def test_exec_macro_passes_special_arguments_as_shell_argv(tmp_path: Path) -> None:
    command = CustomCommand(
        "args",
        "",
        "printf 'one=<%s>\\n' \"$1\"; printf 'all=<%s>\\n' \"$@\"; "
        "printf 'raw=<%s>\\n' \"$ARGUMENTS\"; "
        "printf 'ten=<%s> ten0=<%s>\\n' \"${10}\" \"$10\"",
        tmp_path / "args.md",
        "home",
        "exec",
    )
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    call = ToolCall(
        "macro-args",
        "exec",
        {
            "command": command.render_exec("one; '$HOME'\nline two"),
            "timeout": command.timeout,
        },
    )

    result = await run_exec_macro(
        registry,
        call,
        tmp_path / "args.log",
        stream_sink=lambda _event: None,
        lifecycle_sink=lambda _kind: None,
    )

    assert result.is_error is False
    assert "one=<one;>" in result.content
    assert "all=<'$HOME'>" in result.content
    assert "raw=<one; '$HOME'\nline two>" in result.content
    assert "ten=<> ten0=<one;0>" in result.content


async def test_macro_input_loop_keeps_processing_approval_input(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "deploy", "---\nkind: exec\n---\nsleep 0.1")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    policy = ApprovalPolicy(store=store, default=ApprovalDecision.ASK)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, approval_policy=policy, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    await app._handle_prompt_value("/deploy")

    assert app.active
    for _ in range(100):
        if app.pending_approvals:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro approval did not appear")
    key = app.pending_approvals[0].key
    await app._handle_prompt_value(f"approve {key}")
    await asyncio.wait_for(app._active_task, timeout=2)


async def test_macro_receipts_queue_until_the_next_provider_turn(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "first", "---\nkind: exec\n---\nprintf first")
    _write_command(home / "commands", "second", "---\nkind: exec\n---\nprintf second")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._handle_prompt_value("/first")
    await app._handle_prompt_value("/second")
    await app._handle_prompt_value("continue")
    await asyncio.wait_for(app._active_task, timeout=2)

    user_message = next(message for message in backend.calls[0][0] if message.role.value == "user")
    assert user_message.content[0].text.startswith(
        "ran /first, exit 0\nran /second, exit 0\n\ncontinue"
    )


async def test_macro_receipts_commit_in_submission_order_when_completion_reverses(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "first", "---\nkind: exec\n---\nsleep 0.1")
    _write_command(home / "commands", "second", "---\nkind: exec\n---\nprintf second")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    await asyncio.gather(
        app._handle_prompt_value("/first"),
        app._handle_prompt_value("/second"),
    )
    for _ in range(200):
        if list(app._macro_receipts) == [
            "ran /first, exit 0",
            "ran /second, exit 0",
        ]:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("macro receipts were committed out of order")

    await app._handle_prompt_value("continue")
    for _ in range(100):
        if len(backend.calls) == 1:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("provider turn did not start")
    await app._active_task
    user_message = next(
        message for message in backend.calls[0][0] if message.role.value == "user"
    )
    assert user_message.content[0].text.startswith(
        "ran /first, exit 0\nran /second, exit 0\n\ncontinue"
    )


async def test_queued_prompt_consumes_macro_receipt_at_provider_start(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "slow", "---\nkind: exec\n---\nsleep 0.05")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    macro_task = asyncio.create_task(app._handle_prompt_value("/slow"))
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("slow macro did not start")

    await app._handle_prompt_value("continue")
    assert len(app._queued) == 1
    macro_child = app._active_task
    assert macro_child is not None
    await asyncio.wait_for(macro_child, timeout=2)
    await macro_task
    provider_task = app._active_task
    assert provider_task is not None
    await asyncio.wait_for(provider_task, timeout=2)

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == "ran /slow, exit 0\n\ncontinue"


async def test_two_macros_queued_prompt_and_abort_keep_receipt_order(
    tmp_path: Path,
) -> None:
    """The provider must see receipts in insertion order: first, second, then abort receipt."""

    home = tmp_path / "home"
    _write_command(home / "commands", "first", "---\nkind: exec\n---\nprintf first")
    _write_command(home / "commands", "second", "---\nkind: exec\n---\nsleep 30\n")
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._input_loop_active = True

    await app._handle_prompt_value("/first")
    if app._active_task is not None:
        await asyncio.wait_for(app._active_task, timeout=3)
        app._active_task = None
    assert list(app._macro_receipts) == ["ran /first, exit 0"]

    await app._handle_prompt_value("/second")
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("second macro did not start")

    await app._handle_prompt_value("continue")
    assert len(app._queued) == 1

    macro_task = app._active_task
    assert macro_task is not None
    app.abort_active()
    await asyncio.wait_for(macro_task, timeout=3)
    provider_task = app._active_task
    assert provider_task is not None
    await asyncio.wait_for(provider_task, timeout=3)

    user_message = next(
        message
        for message in backend.calls[0][0]
        if message.role.value == "user"
    )
    assert user_message.content[0].text == (
        "ran /first, exit 0\nran /second, canceled\n\ncontinue"
    )


async def test_macro_abort_does_not_cancel_background_agent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(home / "commands", "wait", "---\nkind: exec\n---\nsleep 30")
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions", cwd=tmp_path), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=StringIO(), force_terminal=False),
    )
    canceled = asyncio.Event()

    async def background() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            canceled.set()
            raise

    watcher = asyncio.create_task(background())
    app.loop._background_owner.register("background", watcher.cancel, watcher)
    macro_task = asyncio.create_task(app._handle_prompt_value("/wait"))
    for _ in range(100):
        if app.active:
            break
        await asyncio.sleep(0.01)

    app.abort_active()
    await asyncio.wait_for(macro_task, timeout=3)
    assert not canceled.is_set()
    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)
    app.loop._background_owner.unregister("background")


async def test_exec_macro_timeout_has_a_distinct_receipt_status(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_command(
        home / "commands",
        "short",
        "---\nkind: exec\ntimeout: 0.02\n---\nsleep 1",
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            FakeBackend([]), ConversationStore(tmp_path / "sessions", cwd=tmp_path),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        zeta_home=home,
        console=Console(file=output, force_terminal=False),
    )

    await app._handle_prompt_value("/short")

    assert "/short · timeout" in output.getvalue()
    assert "/short · exit" not in output.getvalue()
