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
async def test_edit_replaces_unique_string_with_structured_result(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("before: old\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "edit-unique",
            "edit",
            {"path": "note.txt", "old_string": "old", "new_string": "new"},
        )
    )

    updated = b"before: new\n"
    assert result["isError"] is False
    assert result["content"][0]["text"] == (
        f"edited {file_path}: 12 bytes → 12 bytes"
    )
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_before": 12,
        "bytes_after": 12,
        "sha256_after": hashlib.sha256(updated).hexdigest(),
    }
    assert file_path.read_bytes() == updated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "old_string", "message"),
    [
        ("one", "missing", "old_string not found in note.txt"),
        (
            "old and old",
            "old",
            "old_string found 2 times in note.txt; must be unique",
        ),
    ],
)
async def test_edit_requires_one_match(
    tmp_path: Path,
    content: str,
    old_string: str,
    message: str,
) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text(content, encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "edit-match-count",
            "edit",
            {"path": "note.txt", "old_string": old_string, "new_string": "new"},
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == message
    assert result["structuredContent"]["error"]["tool"] == "edit"
    assert file_path.read_text(encoding="utf-8") == content


@pytest.mark.asyncio
async def test_edit_rejects_overlapping_matches(tmp_path: Path) -> None:
    file_path = tmp_path / "note.txt"
    file_path.write_text("aaa", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "edit-overlap",
            "edit",
            {"path": "note.txt", "old_string": "aa", "new_string": "X"},
        )
    )

    assert result["isError"] is True
    assert result["content"][0]["text"] == (
        "old_string found 2 times in note.txt; must be unique"
    )
    assert file_path.read_text(encoding="utf-8") == "aaa"


@pytest.mark.asyncio
async def test_edit_preserves_utf8_and_reports_byte_lengths(tmp_path: Path) -> None:
    file_path = tmp_path / "unicode.txt"
    file_path.write_text("café: 世界\n", encoding="utf-8")
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "edit-unicode",
            "edit",
            {"path": "unicode.txt", "old_string": "世界", "new_string": "мир"},
        )
    )

    updated = "café: мир\n".encode()
    assert result["isError"] is False
    assert result["structuredContent"] == {
        "path": str(file_path),
        "bytes_before": len("café: 世界\n".encode()),
        "bytes_after": len(updated),
        "sha256_after": hashlib.sha256(updated).hexdigest(),
    }
    assert file_path.read_bytes() == updated


@pytest.mark.asyncio
async def test_edit_allows_path_outside_session_cwd(tmp_path: Path) -> None:
    session_cwd = tmp_path / "session"
    session_cwd.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("old", encoding="utf-8")
    registry = ToolRegistry(session_cwd, skill_catalog=SkillCatalog.empty())

    result = await registry.execute(
        ToolCall(
            "edit-outside",
            "edit",
            {"path": str(outside), "old_string": "old", "new_string": "new"},
        )
    )

    assert result["isError"] is False
    assert outside.read_text(encoding="utf-8") == "new"
