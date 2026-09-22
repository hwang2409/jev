"""Filesystem regressions for session rename, deletion, and store leases."""


from __future__ import annotations


import fcntl


import os


import subprocess


import sys


from concurrent.futures import ThreadPoolExecutor


from threading import Event


import pytest


from zeta.cli import main


from zeta.core import session as session_module


from zeta.core.session import SessionError, SessionInUseError, SessionManager


from zeta.core.store import ConversationStore


from zeta.types import Message, MessageRole, TextContent


def closed_session(tmp_path):
    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    opened.store.close()
    return manager, opened.metadata.session_id


def test_child_store_rejects_symlink_in_nested_sessions_root(tmp_path):
    manager, sid = closed_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (manager.sessions_dir / sid / "agents").symlink_to(outside, target_is_directory=True)
    with pytest.raises((SessionError, OSError)):
        ConversationStore(manager.sessions_dir / sid / "agents", session_id="1")
    with manager.open(sid).store as store, pytest.raises((SessionError, OSError)):
        store.allocate_agent_index()
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("replacement", ["directory-swap", "temporary-symlink"])
async def test_background_shutdown_stays_in_pinned_directory(tmp_path, replacement):
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path / "home", cwd=tmp_path, provider="fake")
    await runtime.create_session()
    store = runtime.opened.store
    tasks = runtime.loop.tool_registry.background_tasks
    directory = store.session_dir
    outside = tmp_path / "outside"
    outside.mkdir()
    pinned = tmp_path / "pinned"
    if replacement == "directory-swap":
        directory.rename(pinned)
        directory.symlink_to(outside, target_is_directory=True)
    else:
        pinned = directory
        (directory / "background_tasks.tmp").symlink_to(outside / "must-not-create")
    try:
        task_id, _ = await tasks.start("printf contained", tmp_path, log_path=directory / "macro.log")
        await tasks.wait(task_id)
    finally:
        await runtime.close()
    assert list(outside.iterdir()) == []
    assert (pinned / "macro.log").read_text() == "contained"
    from zeta.core.checkpoints import load_session_json

    rows = load_session_json((pinned / "background_tasks.json").read_bytes())
    assert rows[0]["task_id"] == task_id
    assert rows[0]["running"] is False
    assert tasks._directory_fd is None


async def test_background_descriptor_keeps_lease_until_registry_close(tmp_path):
    from zeta.tools._shared.process import BackgroundTaskRegistry

    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    store = opened.store
    tasks = BackgroundTaskRegistry(session_dir=store.session_dir, directory_fd=store.directory_fd)
    store.close()
    try:
        with pytest.raises(SessionInUseError):
            manager.delete(store.session_id)
    finally:
        await tasks.close()
    manager.delete(store.session_id)


def test_child_transcript_reads_use_parent_descriptor(tmp_path):
    from zeta.tools.agent import _read_child_file
    from zeta.tui.agent_card import AgentCard

    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.allocate_agent_index()
    child = ConversationStore(store.session_dir / "agents", session_id="1", cwd=tmp_path)
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("pinned child")]))
    child.close()
    pinned = tmp_path / "pinned"
    outside = tmp_path / "outside"
    outside.mkdir()
    store.session_dir.rename(pinned)
    store.session_dir.symlink_to(outside, target_is_directory=True)
    try:
        assert b"pinned child" in _read_child_file(store, child.session_dir, "conversation.jsonl")
        assert AgentCard._tail_lines(str(child.session_dir), 5) == []
        assert list(outside.iterdir()) == []
    finally:
        store.close()
