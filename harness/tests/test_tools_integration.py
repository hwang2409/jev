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
async def test_registry_validates_arguments_before_running_handler(tmp_path: Path) -> None:
    called = False

    def handler(arguments: dict[str, object]) -> str:
        nonlocal called
        called = True
        return "ran"

    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register(
        "typed",
        handler,
        parameters={
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
            "additionalProperties": False,
        },
    )

    result = await registry.execute(ToolCall("call-1", "typed", {"count": "one"}))

    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]
    assert not called


def test_registry_rejects_unsupported_schema_keywords_and_non_json_data(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())

    with pytest.raises(ValueError, match="unsupported schema keywords"):
        registry.register(
            "alternative",
            lambda arguments: "ran",
            parameters={
                "type": "string",
                "anyOf": [{"minLength": 2}],
            },
        )
    with pytest.raises(ValueError, match="schema must contain JSON data"):
        registry.register(
            "non-json",
            lambda arguments: "ran",
            parameters={"type": "object", "const": object()},
        )
    with pytest.raises(ValueError, match="schema must contain JSON data"):
        registry.register(
            "tuple",
            lambda arguments: "ran",
            parameters={"type": "object", "properties": {"value": {"enum": [("x",)]}}},
        )


@pytest.mark.asyncio
async def test_registry_validates_union_schema_types(tmp_path: Path) -> None:
    called: list[dict[str, object]] = []

    def handler(arguments: dict[str, object]) -> str:
        called.append(arguments)
        return "ran"

    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register(
        "union",
        handler,
        parameters={
            "type": "object",
            "properties": {"value": {"type": ["string", "null"]}},
            "required": ["value"],
            "additionalProperties": False,
        },
    )

    string_result = await registry.execute(ToolCall("union-string", "union", {"value": "text"}))
    null_result = await registry.execute(ToolCall("union-null", "union", {"value": None}))
    invalid_result = await registry.execute(ToolCall("union-invalid", "union", {"value": 1}))

    assert string_result["isError"] is False
    assert null_result["isError"] is False
    assert invalid_result["isError"] is True
    assert "invalid arguments" in invalid_result["content"][0]["text"]
    assert called == [{"value": "text"}, {"value": None}]


@pytest.mark.asyncio
async def test_paths_outside_session_cwd_are_allowed(tmp_path: Path) -> None:
    outside = tmp_path.parent / "zeta-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    outside_dir = tmp_path.parent / "zeta-outside-dir"
    outside_dir.mkdir()
    (outside_dir / "nested.txt").write_text("nested", encoding="utf-8")
    link = tmp_path / "outside-link"
    link.symlink_to(outside)
    dir_link = tmp_path / "outside-dir-link"
    dir_link.symlink_to(outside_dir, target_is_directory=True)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    absolute_result = await registry.execute(
        ToolCall("read-1", "read", {"path": str(outside)})
    )
    symlink_result = await registry.execute(
        ToolCall("read-2", "read", {"path": "outside-link"})
    )

    assert absolute_result["isError"] is False
    assert absolute_result["content"][0]["text"] == "outside"
    assert symlink_result["isError"] is False
    assert symlink_result["content"][0]["text"] == "outside"


@pytest.mark.asyncio
async def test_builtin_tools_read_and_exec_use_session_cwd(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "note.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    read_result = await registry.execute(
        ToolCall("read-1", "read", {"path": "nested/note.txt", "offset": 1, "limit": 1})
    )
    exec_result = await registry.execute(
        ToolCall("exec-1", "exec", {"command": "pwd"})
    )

    assert read_result["isError"] is False
    assert read_result["content"][0]["text"] == "two"
    assert exec_result["isError"] is False
    assert str(tmp_path) in exec_result["content"][0]["text"]
    assert "not a sandbox" in registry.definitions_by_name["exec"].description


def test_list_is_not_registered(tmp_path: Path) -> None:
    assert "list" not in ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).definitions_by_name


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "note.txt", "old_string": "old"},
        {"path": "note.txt", "new_string": "new"},
        {
            "path": "note.txt",
            "old_string": "old",
            "new_string": "new",
            "extra": True,
        },
        {"path": "note.txt", "old_string": 1, "new_string": "new"},
    ],
)
async def test_registry_rejects_malformed_edit_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(ToolCall("edit-invalid", "edit", arguments))

    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]


def test_deleted_marker_stripped_only_for_unlinked_descriptors() -> None:
    """procfs marks unlinked fds; a real file may still be named that way."""

    assert (
        sandbox_module._without_deleted_marker("/s/target (deleted)", unlinked=True)
        == "/s/target"
    )
    assert (
        sandbox_module._without_deleted_marker("/s/target", unlinked=True)
        == "/s/target"
    )
    # A file genuinely named "report (deleted)" keeps its name while linked.
    assert (
        sandbox_module._without_deleted_marker("/s/report (deleted)", unlinked=False)
        == "/s/report (deleted)"
    )
    assert (
        sandbox_module._without_deleted_marker("/s/report", unlinked=False)
        == "/s/report"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "note.txt"},
        {"path": "note.txt", "content": "x", "create_parents": "yes"},
        {"path": "note.txt", "content": "x", "extra": True},
    ],
)
async def test_registry_rejects_malformed_write_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(ToolCall("write-invalid", "write", arguments))

    assert result["isError"] is True
    assert "invalid arguments" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_abort_signal_stays_set_for_an_active_handler(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, register_builtin=False, skill_catalog=SkillCatalog.empty())
    started = asyncio.Event()
    observed: list[bool] = []

    async def handler(
        arguments: dict[str, object],
        abort_signal: ToolAbortSignal,
    ) -> str:
        del arguments
        started.set()
        await abort_signal.wait()
        observed.append(abort_signal.is_set())
        await asyncio.sleep(0)
        observed.append(abort_signal.is_set())
        return "canceled"

    registry.register("wait", handler)
    task = asyncio.create_task(registry.execute(ToolCall("active", "wait", {})))
    await asyncio.wait_for(started.wait(), timeout=1)

    registry.abort()

    assert await asyncio.wait_for(task, timeout=1) == {
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
    assert observed == [True, True]


@pytest.mark.asyncio
async def test_pre_execution_hook_can_allow_and_deny(tmp_path: Path) -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def hook(name: str, arguments: dict[str, object]) -> bool:
        seen.append((name, arguments))
        return arguments.get("allow") is True

    registry = ToolRegistry(tmp_path, pre_execute_hook=hook, register_builtin=False, skill_catalog=SkillCatalog.empty())
    registry.register(
        "gated",
        lambda arguments: "allowed",
        parameters={
            "type": "object",
            "properties": {"allow": {"type": "boolean"}},
            "required": ["allow"],
        },
    )

    allowed = await registry.execute(ToolCall("call-1", "gated", {"allow": True}))
    denied = await registry.execute(ToolCall("call-2", "gated", {"allow": False}))

    assert allowed["isError"] is False
    assert allowed["content"][0]["text"] == "allowed"
    assert denied["isError"] is True
    assert denied["content"][0]["text"] == "tool execution denied by hook"
    assert seen == [("gated", {"allow": True}), ("gated", {"allow": False})]


@pytest.mark.asyncio
async def test_run_background_scrubs_credentials_from_child_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_env(monkeypatch)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    started = await registry.execute(
        ToolCall(
            "bg-env-dump",
            "run_background",
            {"command": "/usr/bin/env"},
        )
    )
    task_id = started["structuredContent"]["task_id"]
    await asyncio.wait_for(registry.background_tasks.wait(task_id), timeout=15)
    output = await registry.execute(
        ToolCall("bg-env-read", "task_output", {"task_id": task_id})
    )

    _assert_env_dump_is_scrubbed(output["structuredContent"]["output"])
    await registry.close()
