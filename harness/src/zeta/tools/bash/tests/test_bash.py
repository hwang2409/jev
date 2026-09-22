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
async def test_bash_captures_stdout_stderr_and_exit_code(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "bash-output",
            "bash",
            {"cmd": "printf out; printf err >&2; exit 7"},
        )
    )

    assert result["isError"] is True
    structured = result["structuredContent"]
    assert structured["stdout"] == "out"
    assert structured["stderr"] == "err"
    assert structured["exit_code"] == 7
    assert structured["cwd_after"] == str(tmp_path)
    assert structured["error"]["tool"] == "bash"
    assert structured["error"]["kind"] == "exit_nonzero"
    assert structured["error"]["hint"]
    assert "stdout:\nout\nstderr:\nerr" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_bash_persists_cwd_across_calls(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    changed = await registry.execute(
        ToolCall("bash-cd", "bash", {"cmd": "cd /tmp"})
    )
    current = await registry.execute(ToolCall("bash-pwd", "bash", {"cmd": "pwd"}))

    assert changed["structuredContent"]["cwd_after"] == "/tmp"
    assert current["structuredContent"]["stdout"].strip() == "/tmp"
    assert registry.bash_cwd == "/tmp"


@pytest.mark.asyncio
async def test_bash_persistent_cwd_isolated_between_sessions(tmp_path: Path) -> None:
    first_store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    first = ToolRegistry(tmp_path, session_store=first_store, skill_catalog=SkillCatalog.empty())
    await first.execute(ToolCall("first-cd", "bash", {"cmd": "cd /tmp"}))

    second_store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    second = ToolRegistry(tmp_path, session_store=second_store, skill_catalog=SkillCatalog.empty())
    result = await second.execute(ToolCall("second-pwd", "bash", {"cmd": "pwd"}))

    assert result["structuredContent"]["stdout"].strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_cwd_channel_rejects_user_forgery(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "bash-channel-attack",
            "bash",
            {"cmd": 'printf "/tmp\\n" > "$3"; trap - EXIT; exit 0'},
        )
    )

    assert result["isError"] or result["structuredContent"]["cwd_after"] == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_failed_persistence_keeps_registry_state_on_replace_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    registry = ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())
    before_state = store.state_path.read_bytes()

    def fail_replace(source: object, destination: object) -> None:
        del source, destination
        raise OSError("injected state replace failure")

    monkeypatch.setattr("zeta.core.store.os.replace", fail_replace)
    result = await registry.execute(
        ToolCall("bash-state-failure", "bash", {"cmd": "cd /tmp"})
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["tool"] == "bash"
    assert registry.bash_cwd == str(tmp_path)
    assert store.bash_cwd == str(tmp_path)
    assert store.state_path.read_bytes() == before_state

    monkeypatch.undo()
    current = await registry.execute(ToolCall("bash-state-old", "bash", {"cmd": "pwd"}))
    assert current["structuredContent"]["stdout"].strip() == str(tmp_path)


@pytest.mark.asyncio
async def test_bash_accepts_per_call_cwd_outside_sandbox(tmp_path: Path) -> None:
    outside = tmp_path.parent
    marker = outside / f"zeta-bash-outside-{tmp_path.name}"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    try:
        result = await registry.execute(
            ToolCall(
                "bash-outside-cwd",
                "bash",
                {"cmd": f"touch {shlex.quote(str(marker))}", "cwd": str(outside)},
            )
        )

        assert result["isError"] is False
        assert result["structuredContent"]["cwd_after"] == str(outside)
        assert marker.exists()
    finally:
        if marker.exists():
            marker.unlink()


@pytest.mark.asyncio
async def test_bash_persists_cwd_after_failed_command(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    failed = await registry.execute(
        ToolCall("bash-failed-cd", "bash", {"cmd": "cd /tmp; exit 3"})
    )
    current = await registry.execute(ToolCall("bash-failed-pwd", "bash", {"cmd": "pwd"}))

    assert failed["isError"] is True
    failed_structured = failed["structuredContent"]
    assert failed_structured["stdout"] == ""
    assert failed_structured["stderr"] == ""
    assert failed_structured["exit_code"] == 3
    assert failed_structured["cwd_after"] == "/tmp"
    assert failed_structured["error"]["tool"] == "bash"
    assert failed_structured["error"]["kind"] == "exit_nonzero"
    assert current["structuredContent"]["stdout"].strip() == "/tmp"


@pytest.mark.asyncio
async def test_bash_persists_explicit_cd_from_override(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("bash-explicit-cd", "bash", {"cmd": "cd /tmp", "cwd": str(nested)})
    )
    current = await registry.execute(ToolCall("bash-explicit-pwd", "bash", {"cmd": "pwd"}))

    assert result["structuredContent"]["cwd_after"] == "/tmp"
    assert current["structuredContent"]["stdout"].strip() == "/tmp"


@pytest.mark.asyncio
async def test_bash_abort_kills_process_group_and_reaps_descendants(tmp_path: Path) -> None:
    marker = tmp_path / "child-alive"
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(tmp_path, abort_signal=abort_signal, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(
        registry.execute(
            ToolCall("bash-abort", "bash", {"cmd": _descendant_command(marker)})
        )
    )

    await asyncio.sleep(0.05)
    abort_signal.abort()
    result = await task
    await asyncio.sleep(0.5)

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_bash_task_cancellation_kills_process_group(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "child-canceled"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    task = asyncio.create_task(
        registry.execute(
            ToolCall("bash-cancel", "bash", {"cmd": _descendant_command(marker)})
        )
    )

    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.5)

    assert not marker.exists()


@pytest.mark.asyncio
async def test_bash_invalid_start_cwd_falls_back_without_persisting(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    registry.bash_cwd = str(missing)

    result = await registry.execute(ToolCall("bash-missing-cwd", "bash", {"cmd": "pwd"}))

    assert result["isError"] is True
    assert result["structuredContent"]["cwd_after"] == str(missing)
    assert registry.bash_cwd == str(missing)


@pytest.mark.asyncio
async def test_bash_cwd_override_does_not_persist_without_cd(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("bash-override", "bash", {"cmd": "pwd", "cwd": str(nested)})
    )
    current = await registry.execute(ToolCall("bash-default", "bash", {"cmd": "pwd"}))

    assert result["structuredContent"]["cwd_after"] == str(nested)
    assert current["structuredContent"]["stdout"].strip() == str(tmp_path)
    assert registry.bash_cwd == str(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"cmd": ""},
        {"cmd": 1},
        {"cmd": "pwd", "cwd": 1},
        {"cmd": "pwd", "extra": True},
    ],
)
async def test_bash_rejects_malformed_arguments(
    tmp_path: Path,
    arguments: dict[str, object],
) -> None:
    result = await ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty()).execute(
        ToolCall("bash-invalid", "bash", arguments)
    )

    assert result["isError"] is True
    assert result["structuredContent"]["error"]["tool"] == "bash"
    assert result["structuredContent"]["error"]["kind"] in {
        "invalid_arguments",
        "error",
    }


def test_tool_subprocess_env_blocks_credentials_and_preserves_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zeta.tools._shared.process import tool_subprocess_env

    _seed_env(monkeypatch)

    env = tool_subprocess_env()

    for credential in _CREDENTIAL_ENV_FIXTURES:
        assert credential not in env
    for keeper, expected in _UNRELATED_ENV_FIXTURES.items():
        assert env[keeper] == expected
    assert env["PATH"] == os.environ["PATH"]
    assert env["HOME"] == os.environ["HOME"]
    assert "ZETA_HOME" not in env


def test_subprocess_env_applies_explicit_overrides_after_scrubbing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-secret")
    monkeypatch.setenv("ZETA_HOME", "/tmp/parent-zeta-home")

    env = subprocess_env(
        {
            "ANTHROPIC_API_KEY": "explicit-secret",
            "ZETA_HOME": "/tmp/explicit-zeta-home",
        }
    )

    assert env["ANTHROPIC_API_KEY"] == "explicit-secret"
    assert env["ZETA_HOME"] == "/tmp/explicit-zeta-home"


def test_subprocess_env_uses_exact_normalized_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SEC-WEBSOCKET-KEY", "hyphen-secret")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    monkeypatch.setenv("SECRETARY_MODE", "briefing")
    monkeypatch.setenv("COOKIECUTTER_REPLAY", "enabled")

    env = subprocess_env()

    assert "SEC-WEBSOCKET-KEY" not in env
    assert env["TOKENIZERS_PARALLELISM"] == "true"
    assert env["SECRETARY_MODE"] == "briefing"
    assert env["COOKIECUTTER_REPLAY"] == "enabled"


@pytest.mark.asyncio
async def test_bash_scrubs_credentials_from_child_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_env(monkeypatch)
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("bash-env-dump", "bash", {"cmd": "/usr/bin/env"})
    )

    assert result["isError"] is False
    _assert_env_dump_is_scrubbed(result["structuredContent"]["stdout"])
