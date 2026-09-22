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


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="the procfs (deleted) marker is Linux-only",
)
def test_path_from_fd_reports_bare_path_for_unlinked_file(tmp_path: Path) -> None:
    """Linux procfs must report the same bare path macOS F_GETPATH reports."""

    target = tmp_path / "target"
    target.write_bytes(b"inside")
    file_descriptor = os.open(target, os.O_RDONLY)
    try:
        assert sandbox_module._path_from_fd(file_descriptor) == str(target)
        target.unlink()
        assert os.fstat(file_descriptor).st_nlink == 0
        assert sandbox_module._path_from_fd(file_descriptor) == str(target)
    finally:
        os.close(file_descriptor)


@pytest.mark.asyncio
async def test_read_retains_only_bounded_output_from_large_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures = []
    real_capture = read_module._BoundedText

    class TrackingCapture(real_capture):
        def __init__(self, limit: int) -> None:
            super().__init__(limit)
            captures.append(self)

    monkeypatch.setattr(read_module, "_BoundedText", TrackingCapture)
    (tmp_path / "large.txt").write_text("x\n" * 1_000_000, encoding="utf-8")
    real_fdopen = read_module.os.fdopen
    open_count = 0

    def tracking_fdopen(
        file_descriptor: int,
        mode: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        nonlocal open_count
        open_count += 1
        return real_fdopen(file_descriptor, mode, *args, **kwargs)

    monkeypatch.setattr(read_module.os, "fdopen", tracking_fdopen)
    registry = ToolRegistry(tmp_path, max_output_chars=64, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("read-large", "read", {"path": "large.txt"})
    )

    assert result["isError"] is False
    assert len(result["content"][0]["text"]) == 64
    assert result["content"][0]["truncated"] is True
    assert len(captures) == 1
    assert captures[0].retained_chars <= 64
    assert open_count == 1


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_read_abort_returns_canceled_result_during_scan(tmp_path: Path) -> None:
    (tmp_path / "large.txt").write_text("x\n" * 1_000_000, encoding="utf-8")
    abort_signal = ToolAbortSignal()
    registry = ToolRegistry(
        tmp_path,
        abort_signal=abort_signal,
        max_output_chars=3_000_000,
skill_catalog=SkillCatalog.empty(),
    )

    async def abort_soon() -> None:
        await asyncio.sleep(0)
        abort_signal.abort()

    abort_task = asyncio.create_task(abort_soon())
    result = await registry.execute(ToolCall("read-abort", "read", {"path": "large.txt"}))
    await abort_task

    assert result["isError"] is True
    assert result["content"][0]["text"] == "tool execution canceled"
