"""Tests for session-scoped background process tools."""


from __future__ import annotations


import asyncio


import shlex


import signal


import sys


from pathlib import Path


import pytest


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.store import ConversationStore


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.tools._shared.process import BackgroundTaskRegistry, _group_exists


from zeta.tui.render import format_status


from zeta.types import ToolCall


def _python(*parts: str) -> str:
    return shlex.join((sys.executable, "-c", *parts))


async def _wait_for_exit(registry: BackgroundTaskRegistry, task_id: str) -> None:
    await asyncio.wait_for(registry.wait(task_id), timeout=30)


@pytest.mark.asyncio
async def test_background_approval_cap_and_session_cleanup(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "denied-session")
    denied = ToolRegistry(
        tmp_path,
        session_store=store,
        approval_store=store,
        approval_policy=ApprovalPolicy(
            always_deny={"run_background"},
            default=ApprovalDecision.ALLOW,
        ),
skill_catalog=SkillCatalog.empty(),
    )
    result = await denied.execute(
        ToolCall("deny", "run_background", {"command": "sleep 30"})
    )
    assert result["isError"] is True
    await denied.close()

    tasks = BackgroundTaskRegistry(max_tasks=1)
    first, _ = await tasks.start("sleep 30", tmp_path)
    with pytest.raises(ValueError, match="limit reached"):
        await tasks.start("sleep 30", tmp_path)
    await tasks.close()
    assert (await tasks.output(first))["running"] is False


def test_background_footer_segment_degrades_as_a_whole() -> None:
    assert "bg 2" in format_status("fake", "offline", "idle", background_count=2).plain
    narrow = format_status(
        "fake",
        "offline",
        "idle",
        background_count=2,
        vim_state="NORMAL",
        width=30,
    ).plain
    assert "bg 2" not in narrow


@pytest.mark.parametrize(
    "payload",
    [b"[" * 65 + b"]" * 65, b"[" * 10_000 + b"]" * 10_000, b"\xff"],
    ids=["depth65", "depth10000", "binary"],
)
async def test_registry_setup_ignores_corrupt_background_state(
    tmp_path: Path, payload: bytes
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    path = store.session_dir / "background_tasks.json"
    path.write_bytes(payload)
    registry = ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())
    assert registry.background_tasks.running_count == 0
    assert path.read_bytes() == payload
    await registry.close()
