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
async def test_background_kill_escalates_for_term_ignoring_process(tmp_path: Path) -> None:
    tasks = BackgroundTaskRegistry(term_grace=0.03)
    task_id, _ = await tasks.start(
        _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"),
        tmp_path,
    )
    result = await tasks.kill(task_id)
    assert result["running"] is False
    assert result["exit_code"] in {-signal.SIGTERM, -signal.SIGKILL}
    assert not _group_exists(tasks.records[0].pid)
    await tasks.close()
