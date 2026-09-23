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


from zeta.protocol.types import MessageRole, StreamEventType, TextContent, ToolCall, ToolResult


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
async def test_write_creates_file_with_structured_result(tmp_path: Path) -> None:
    content = "héllo\n"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("write-new", "write", {"path": "note.txt", "content": content})
    )

    file_path = tmp_path / "note.txt"
    assert result["isError"] is False
    assert result["content"][0]["text"] == f"wrote 7 bytes to {file_path}"
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_written": 7,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "was_created": True,
        "was_overwritten": False,
    }
    assert file_path.read_bytes() == content.encode("utf-8")


@pytest.mark.asyncio
async def test_write_reports_overwrite(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("old", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("write-overwrite", "write", {"path": "note.txt", "content": "new"})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_written": 3,
        "sha256": hashlib.sha256(b"new").hexdigest(),
        "was_created": False,
        "was_overwritten": True,
    }
    assert file_path.read_text(encoding="utf-8") == "new"


@pytest.mark.asyncio
async def test_write_getpath_failure_does_not_truncate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "getpath-failure.txt"
    target.write_text("original", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    def fail_getpath(file_descriptor: int) -> str:
        raise OSError("injected F_GETPATH failure")

    monkeypatch.setattr(write_module, "_path_from_fd", fail_getpath)
    result = await registry.execute(
        ToolCall("write-getpath-failure", "write", {"path": target.name, "content": "new"})
    )

    assert result["isError"] is True
    assert target.read_text(encoding="utf-8") == "original"


@pytest.mark.asyncio
async def test_write_fdopen_failure_closes_raw_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "fdopen-failure.txt"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    raw_fds: list[int] = []

    def fail_fdopen(file_descriptor: int, mode: str) -> object:
        raw_fds.append(file_descriptor)
        raise OSError("injected fdopen failure")

    monkeypatch.setattr(write_module.os, "fdopen", fail_fdopen)
    result = await registry.execute(
        ToolCall("write-fdopen-failure", "write", {"path": target.name, "content": "x"})
    )

    assert result["isError"] is True
    assert len(raw_fds) == 1
    with pytest.raises(OSError) as error:
        fcntl.fcntl(raw_fds[0], fcntl.F_GETFD)
    assert error.value.errno == errno.EBADF


@pytest.mark.asyncio
async def test_write_rejects_missing_parent_by_default(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "note.txt"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("write-missing-parent", "write", {"path": str(target), "content": "x"})
    )

    assert result["isError"] is True
    assert str(target.parent) in result["content"][0]["text"]
    assert not target.parent.exists()
    assert not target.exists()


@pytest.mark.asyncio
async def test_write_can_create_missing_parents(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "note.txt"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "write-create-parent",
            "write",
            {"path": str(target), "content": "x", "create_parents": True},
        )
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(target),
        "bytes_written": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
        "was_created": True,
        "was_overwritten": False,
    }
    assert target.read_text(encoding="utf-8") == "x"
    assert target.parent.is_dir()


@pytest.mark.asyncio
async def test_write_race_reports_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "raced.txt"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    original_open = write_module.os.open

    def racing_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "raced.txt" and dir_fd is not None and flags & write_module.os.O_EXCL:
            creator = threading.Thread(
                target=lambda: target.write_bytes(b"concurrent")
            )
            creator.start()
            creator.join()
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(write_module.os, "open", racing_open)
    result = await registry.execute(
        ToolCall("write-race", "write", {"path": "raced.txt", "content": "x"})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(target),
        "bytes_written": 1,
        "sha256": hashlib.sha256(b"x").hexdigest(),
        "was_created": False,
        "was_overwritten": True,
    }
    assert target.read_bytes() == b"x"


@pytest.mark.asyncio
async def test_write_overwrite_symlink_race_stays_in_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    target = sandbox / "target"
    target.write_bytes(b"inside")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    stop = threading.Event()

    def swap_target() -> None:
        while not stop.is_set():
            try:
                target.unlink()
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                pass
            try:
                target.symlink_to(outside)
            except FileExistsError:
                pass

    swapper = threading.Thread(target=swap_target)
    swapper.start()
    try:
        registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
        for index in range(50):
            result = await registry.execute(
                ToolCall(
                    f"write-overwrite-race-{index}",
                    "write",
                    {"path": "sandbox/target", "content": "x"},
                )
            )
            if not result["isError"]:
                assert result["structuredContent"]["path"] == str(target)
            assert outside.read_bytes() == b"outside"
    finally:
        stop.set()
        swapper.join()
        if target.is_symlink():
            target.unlink()
        if not target.exists():
            target.write_bytes(b"inside")


@pytest.mark.asyncio
async def test_write_create_parents_symlink_race_stays_in_sandbox(
    tmp_path: Path,
) -> None:
    sandbox = tmp_path / "sandbox"
    intermediate = sandbox / "a" / "b"
    intermediate.mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "c").mkdir(parents=True)
    outside_target = outside / "c" / "target"
    outside_target.write_bytes(b"outside")
    stop = threading.Event()

    def swap_intermediate() -> None:
        while not stop.is_set():
            shutil.rmtree(intermediate, ignore_errors=True)
            try:
                intermediate.symlink_to(outside, target_is_directory=True)
            except FileExistsError:
                pass
            try:
                intermediate.unlink()
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                pass
            try:
                intermediate.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    swapper = threading.Thread(target=swap_intermediate)
    swapper.start()
    try:
        registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
        for index in range(50):
            result = await registry.execute(
                ToolCall(
                    f"write-create-parents-race-{index}",
                    "write",
                    {
                        "path": "sandbox/a/b/c/target",
                        "content": "x",
                        "create_parents": True,
                    },
                )
            )
            if not result["isError"]:
                assert Path(result["structuredContent"]["path"]).is_relative_to(
                    sandbox
                )
            assert outside_target.read_bytes() == b"outside"
    finally:
        stop.set()
        swapper.join()


@pytest.mark.asyncio
async def test_write_allows_path_outside_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside = tmp_path / "outside.txt"
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall("write-outside", "write", {"path": str(outside), "content": "x"})
    )

    assert result["isError"] is False
    assert outside.read_text(encoding="utf-8") == "x"


@pytest.mark.asyncio
async def test_write_rejects_replaced_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    attack = tmp_path / "attack"
    attack.mkdir()
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    os.rename(session_cwd, tmp_path / "session-original")
    session_cwd.symlink_to(attack, target_is_directory=True)
    result = await registry.execute(
        ToolCall(
            "write-replaced-cwd",
            "write",
            {
                "path": "sandbox/file.txt",
                "content": "x",
                "create_parents": True,
            },
        )
    )

    assert result["isError"] is True
    assert "session cwd was replaced" in result["content"][0]["text"]
    assert not (attack / "sandbox" / "file.txt").exists()


@pytest.mark.asyncio
async def test_write_rejects_replaced_session_cwd_identity(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    os.rename(session_cwd, tmp_path / "session-original")
    session_cwd.mkdir()
    result = await registry.execute(
        ToolCall(
            "write-replaced-cwd-identity",
            "write",
            {
                "path": "sandbox/file.txt",
                "content": "x",
                "create_parents": True,
            },
        )
    )

    assert result["isError"] is True
    assert "session cwd was replaced" in result["content"][0]["text"]
    assert not (session_cwd / "sandbox" / "file.txt").exists()


@pytest.mark.asyncio
async def test_write_rejects_invalid_utf8_content_before_writing(
    tmp_path: Path,
) -> None:
    target = tmp_path / "invalid.txt"
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "write-invalid-utf8", "write", {"path": str(target), "content": "\ud800"}
        )
    )

    assert result["isError"] is True
    assert "valid UTF-8" in result["content"][0]["text"]
    assert not target.exists()
