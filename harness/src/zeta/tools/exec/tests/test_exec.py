import asyncio


import errno


import fcntl


import hashlib


import math


import os


import shlex


import shutil


import sys


import threading


from pathlib import Path


import pytest


import zeta.tools._shared.sandbox as sandbox_module


import zeta.tools.exec as exec_module


import zeta.tools.read as read_module


import zeta.tools.write as write_module


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.loop import AgentLoop


from zeta.core.process_env import CREDENTIAL_ENV_NAMES, subprocess_env


from zeta.core.store import ConversationStore


from zeta.skills import SkillCatalog


from zeta.tools import ToolAbortSignal, ToolRegistry


from zeta.types import MessageRole, StreamEventType, TextContent, ToolCall, ToolResult


pytestmark = pytest.mark.usefixtures("stock_router_mode")


def _python_command(source: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(source)}"


def _descendant_command(marker: Path, delay: float = 0.3) -> str:
    child = (
        "import pathlib,time; "
        f"time.sleep({delay}); pathlib.Path({str(marker)!r}).write_text('alive')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(5)"
    )
    return _python_command(parent)


async def _collect_loop(loop: AgentLoop) -> list[object]:
    return [event async for event in loop.run_turn("go")]


_CREDENTIAL_ENV_FIXTURES: dict[str, str] = {
    name: f"{name.lower()}-should-not-leak" for name in CREDENTIAL_ENV_NAMES
}


_UNRELATED_ENV_FIXTURES: dict[str, str] = {
    "ZETA_CANARY_UNRELATED": "survives",
    "HOSTNAME_HINT": "kept",
    "TOKENIZERS_PARALLELISM": "true",
    "SECRETARY_MODE": "briefing",
    "COOKIECUTTER_REPLAY": "enabled",
}


def _seed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in _CREDENTIAL_ENV_FIXTURES.items():
        monkeypatch.setenv(name, value)
    for name, value in _UNRELATED_ENV_FIXTURES.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("ZETA_HOME", "/tmp/zeta-private-home")


def _assert_env_dump_is_scrubbed(dump: str) -> None:
    names = {
        line.split("=", 1)[0]
        for line in dump.splitlines()
        if "=" in line
    }
    for credential in _CREDENTIAL_ENV_FIXTURES:
        assert credential not in names, (
            f"{credential} leaked into tool subprocess env"
        )
    for keeper in _UNRELATED_ENV_FIXTURES:
        assert keeper in names, f"{keeper} was stripped by the credential filter"
    assert "PATH" in names, "PATH must survive so shell commands still resolve"
    assert "HOME" in names, "HOME must survive for ordinary child behavior"
    assert "ZETA_HOME" not in names, "ZETA_HOME must not expose the credential store"


@pytest.mark.asyncio
async def test_exec_retains_only_bounded_output_from_large_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = []
    real_capture = exec_module._BoundedOutput

    class TrackingCapture(real_capture):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)
            captures.append(self)

    monkeypatch.setattr(exec_module, "_BoundedOutput", TrackingCapture)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    result = await registry.execute(
        ToolCall(
            "exec-large",
            "exec",
            {
                "command": _python_command(
                    "import sys; sys.stdout.write('x' * 2000000)"
                ),
                "max_output": 64,
            },
        )
    )

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 64
    assert result["content"][0]["truncated"] is True
    assert len(captures) == 2
    assert all(capture.retained_bytes <= 64 for capture in captures)
    assert sum(capture.retained_bytes for capture in captures) <= 128


@pytest.mark.asyncio
async def test_exec_full_size_is_stable_for_capped_utf8_output(tmp_path: Path) -> None:
    command = _python_command("import sys; sys.stdout.write('é')")
    uncapped = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("exec-utf8-full", "exec", {"command": command})
    )
    capped = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall(
            "exec-utf8-capped",
            "exec",
            {"command": command, "max_output": 1},
        )
    )

    uncapped_block = uncapped["content"][0]
    capped_block = capped["content"][0]
    assert uncapped_block["full_size"] == capped_block["full_size"]
    assert uncapped_block["full_size"] == len(uncapped_block["text"].encode("utf-8"))


@pytest.mark.asyncio
async def test_exec_timeout_kills_and_reaps_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "timeout-child-alive"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "exec-timeout",
            "exec",
            {"command": _descendant_command(marker), "timeout": 0.05},
        )
    )
    await asyncio.sleep(0.4)

    assert result["isError"] is True
    assert "timed out after" in result["content"][0]["text"]
    assert result["structuredContent"]["timed_out"] is True
    assert result["structuredContent"]["error"]["tool"] == "exec"
    assert result["structuredContent"]["error"]["kind"] == "timeout"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_cancellation_kills_and_reaps_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "cancel-child-alive"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "exec-cancel",
                "exec",
                {"command": _descendant_command(marker), "timeout": 5},
            )
        )
    )
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.4)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_abort_kills_process_group_and_returns_canceled_result(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "abort-child-alive"
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(
        registry.execute(
            ToolCall(
                "exec-abort",
                "exec",
                {"command": _descendant_command(marker), "timeout": 5},
            )
        )
    )
    await asyncio.sleep(0.05)
    abort_signal.abort()

    result = await asyncio.wait_for(task, timeout=0.5)
    await asyncio.sleep(0.4)

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_exec_output_cap_includes_final_content_boundary(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("exec-cap", "exec", {"command": "printf 1234567890", "max_output": 5})
    )

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 5
    assert result["content"][0]["truncated"] is True


@pytest.mark.asyncio
async def test_exec_abort_wins_when_completion_and_abort_are_ready_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal, skill_catalog=SkillCatalog.empty())
    real_wait = exec_module.asyncio.wait

    async def forced_tie(tasks, *, return_when):
        await asyncio.sleep(0.1)
        abort_signal.abort()
        await asyncio.sleep(0)
        task_set = set(tasks)
        done = {task for task in task_set if task.done()}
        if len(done) < 2:
            return await real_wait(task_set, return_when=return_when)
        return done, task_set - done

    monkeypatch.setattr(exec_module.asyncio, "wait", forced_tie)
    result = await registry.execute(
        ToolCall(
            "exec-race",
            "exec",
            {"command": _python_command("import time; time.sleep(0.01)")},
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"


@pytest.mark.asyncio
async def test_argument_finiteness_covers_undeclared_and_default_fields(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register(
        "permissive",
        lambda arguments: "ran",
        parameters={"type": "object"},
    )
    registry.register("default", lambda arguments: "ran")

    undeclared = await registry.execute(
        ToolCall("undeclared", "permissive", {"extra": math.nan})
    )
    default = await registry.execute(
        ToolCall("default", "default", {"extra": math.inf})
    )

    assert undeclared["isError"] is True
    assert default["isError"] is True


@pytest.mark.asyncio
async def test_registered_schema_copies_cannot_disable_validation(tmp_path: Path) -> None:
    called = False

    def handler(arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "ran"

    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    definition = registry.register(
        "typed",
        handler,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )
    definition.parameters.clear()
    registry.definitions_by_name["typed"].parameters["properties"].clear()
    registry.schemas[0]["parameters"]["properties"].clear()

    result = await registry.execute(ToolCall("typed", "typed", {}))

    assert result["isError"] is True
    assert "required" in result["content"][0]["text"]
    assert not called


@pytest.mark.asyncio
async def test_numeric_validation_rejects_nonfinite_and_bool_enum_values(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register(
        "number",
        lambda arguments: "number",
        parameters={
            "type": "object",
            "properties": {
                "value": {"type": "number", "minimum": 0, "maximum": 10}
            },
        },
    )
    registry.register(
        "enum",
        lambda arguments: "enum",
        parameters={
            "type": "object",
            "properties": {"value": {"enum": [0, 1]}},
        },
    )

    nan_result = await registry.execute(
        ToolCall("nan", "number", {"value": math.nan})
    )
    bool_result = await registry.execute(
        ToolCall("bool", "enum", {"value": True})
    )
    int_result = await registry.execute(
        ToolCall("int", "enum", {"value": 1})
    )
    assert nan_result["isError"] is True
    assert bool_result["isError"] is True
    assert int_result["content"][0]["text"] == "enum"


@pytest.mark.asyncio
async def test_abort_cancels_calls_after_the_signal_is_set(tmp_path: Path) -> None:
    abort_signal = ToolAbortSignal()
    called: list[str] = []
    registry = ToolRegistry(
        tmp_path,
        abort_signal=abort_signal,
        register_builtin=False,
skill_catalog=SkillCatalog.empty(),
    )

    async def handler(
        arguments: dict[str, str],
        signal: ToolAbortSignal,
    ) -> str:
        called.append(arguments["value"])
        if arguments["value"] == "first":
            signal.abort()
        await asyncio.sleep(0)
        return arguments["value"]

    registry.register("step", handler)
    results = await registry.execute_many(
        [
            ToolCall("call-1", "step", {"value": "first"}),
            ToolCall("call-2", "step", {"value": "second"}),
        ]
    )

    assert [result["content"][0]["text"] for result in results] == [
        "first",
        "tool execution canceled",
    ]
    assert called == ["first"]
    assert results[1]["isError"] is True


@pytest.mark.asyncio
async def test_parallel_safe_calls_overlap_and_keep_call_order(tmp_path: Path) -> None:
    finished: list[str] = []
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())

    async def worker(arguments: dict[str, str]) -> str:
        if arguments["value"] == "slow":
            await asyncio.sleep(0.03)
        finished.append(arguments["value"])
        return arguments["value"]

    registry.register(
        "work",
        worker,
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        parallel_safe=True,
    )
    results = await registry.execute_many(
        [
            ToolCall("call-1", "work", {"value": "slow"}),
            ToolCall("call-2", "work", {"value": "fast"}),
        ]
    )

    assert finished == ["fast", "slow"]
    assert [result["content"][0]["text"] for result in results] == ["slow", "fast"]


@pytest.mark.asyncio
async def test_execute_many_rejects_duplicate_ids_before_dispatch(tmp_path: Path) -> None:
    called = False
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())

    async def handler(arguments: dict[str, object]) -> str:
        nonlocal called
        del arguments
        called = True
        return "ran"

    registry.register("work", handler, parallel_safe=True)

    with pytest.raises(ValueError, match="duplicate tool call id"):
        await registry.execute_many(
            [
                ToolCall("same-id", "work", {}),
                ToolCall("same-id", "work", {}),
            ]
        )

    assert not called


@pytest.mark.asyncio
async def test_execute_many_abort_cancels_every_parallel_handler(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    calls = [ToolCall("parallel-a", "wait", {}), ToolCall("parallel-b", "wait", {})]
    started = asyncio.Event()
    started_count = 0
    generations: list[int] = []

    async def handler(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        nonlocal started_count
        del arguments
        started_count += 1
        generations.append(abort_signal.generation)
        if started_count == len(calls):
            started.set()
        await abort_signal.wait()
        return "canceled"

    registry.register("wait", handler, parallel_safe=True)
    task = asyncio.create_task(registry.execute_many(calls))
    await asyncio.wait_for(started.wait(), timeout=1)

    registry.abort()

    assert await asyncio.wait_for(task, timeout=1) == [
        {
            "content": [
                {
                    "type": "text",
                    "text": "canceled",
                    "truncated": False,
                    "full_size": 8,
                }
            ],
            "isError": False,
            "structuredContent": None,
        }
        for _ in calls
    ]
    assert generations == [generations[0], generations[0]]


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_calls_through_registry(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("from registry", encoding="utf-8")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "read", {"path": "note.txt"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    events = [
        event
        async for event in AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty()).run_turn("read it")
    ]

    result_message = store.messages()[2]
    assert result_message.role is MessageRole.TOOL_RESULT
    assert result_message.tool_result is not None
    assert result_message.tool_result.content == "from registry"
    assert events[-1].type.value == "agent_end"
    assert any(schema["name"] == "read" for schema in backend.calls[0][1])


@pytest.mark.asyncio
async def test_agent_loop_mapping_tools_still_validate_through_registry(
    tmp_path: Path,
) -> None:
    called = False

    def typed(arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "ran"

    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "typed", {"count": "bad"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    schema = {
        "name": "typed",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        },
    }

    [
        event
        async for event in AgentLoop(
            backend,
            store,
            tools={"typed": typed},
            tool_schemas=[schema],
skill_catalog=SkillCatalog.empty(),
        ).run_turn("go")
    ]

    result = store.messages()[2].tool_result
    assert result is not None and result.is_error
    assert "invalid arguments" in result.content
    assert not called


@pytest.mark.asyncio
async def test_agent_loop_mapping_tools_do_not_expose_builtins(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", "exec", {"command": "printf unsafe"})
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)

    await _collect_loop(
        AgentLoop(
            backend,
            store,
            tools={"safe_only": lambda arguments: "safe"},
            tool_schemas=[{"name": "safe_only"}],
skill_catalog=SkillCatalog.empty(),
        )
    )

    result = store.messages()[2].tool_result
    assert result is not None
    assert result.content == "unknown tool: exec"
    assert result.is_error


@pytest.mark.asyncio
async def test_agent_loop_keeps_boundary_abort_for_pending_tools(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("call-1", "step", {})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register("step", lambda arguments: "ran")
    aborted = False

    async for event in AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty()).run_turn("go"):
        if event.type is StreamEventType.MESSAGE_END and not aborted:
            registry.abort()
            aborted = True

    result = store.messages()[2].tool_result
    assert result is not None
    assert result == ToolResult("call-1", "tool execution canceled", True)


@pytest.mark.asyncio
async def test_agent_loop_refreshes_abort_signal_each_turn(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", "step", {"value": "abort"}),
                    ToolCall("call-2", "step", {"value": "canceled"}),
                ]
            ),
            ScriptedTurn(tool_calls=[ToolCall("call-3", "step", {"value": "next"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)

    async def step(
        arguments: dict[str, str],
        abort_signal: ToolAbortSignal,
    ) -> str:
        if arguments["value"] == "abort":
            abort_signal.abort()
        return arguments["value"]

    await _collect_loop(
        AgentLoop(backend, store, tools={"step": step}, skill_catalog=SkillCatalog.empty())
    )

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert [result.content for result in results] == [
        "abort",
        "tool execution canceled",
        "next",
    ]


@pytest.mark.asyncio
async def test_registry_abort_cancels_loop_batch_and_next_tool(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall("call-1", "exec", {"command": "sleep 5"}),
                    ToolCall("call-2", "read", {"path": "missing.txt"}),
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ]
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    started = asyncio.Event()

    def hook(name: str, arguments: dict[str, object]) -> bool:
        if name == "exec":
            started.set()
        return True

    registry = ToolRegistry(tmp_path, pre_execute_hook=hook, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(_collect_loop(AgentLoop(backend, store, registry=registry, skill_catalog=SkillCatalog.empty())))
    await started.wait()
    await asyncio.sleep(0.05)
    registry.abort()
    await task

    results = [message.tool_result for message in store.messages() if message.tool_result]
    assert [result.content for result in results] == [
        "tool execution canceled",
        "tool execution canceled",
    ]


@pytest.mark.asyncio
async def test_exec_scrubs_credentials_from_child_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_env(monkeypatch)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("exec-env-dump", "exec", {"command": "/usr/bin/env"})
    )

    assert result["isError"] is False
    _assert_env_dump_is_scrubbed(result["content"][0]["text"])
