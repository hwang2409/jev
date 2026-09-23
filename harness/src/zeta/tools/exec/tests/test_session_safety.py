"""Filesystem regressions for session rename, deletion, and store leases."""


from __future__ import annotations


import fcntl


import os


import subprocess


import sys


from concurrent.futures import ThreadPoolExecutor


from threading import Event


import pytest


from zeta.cli.main import main


from zeta.core import session as session_module


from zeta.core.session import SessionError, SessionInUseError, SessionManager


from zeta.core.store import ConversationStore


from zeta.protocol.types import Message, MessageRole, TextContent


def closed_session(tmp_path):
    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    opened.store.close()
    return manager, opened.metadata.session_id


async def test_session_lifecycle_has_no_absolute_session_file_operations(tmp_path, monkeypatch):
    """Audit the real runtime, including tool and TUI persistence boundaries."""
    from pathlib import Path

    from zeta.tui.persistence import DraftPersistence
    from zeta.server.runtime import ServerRuntime
    from zeta.tools.exec import run_exec_macro
    from zeta.tui.composer import build_user_message
    from zeta.protocol.types import StreamEventType, ToolCall

    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    sessions = home / "sessions"
    events = []
    recording = True
    # session_root pins the trusted home, then opens "sessions" relative to it.
    # No absolute open of a session child belongs in this allowlist.
    root_pinning_opens = {(str(home), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)}
    path_arguments = {
        "open": (0,), "os.rename": (0, 1), "os.remove": (0,), "os.mkdir": (0,),
        "os.rmdir": (0,), "os.scandir": (0,), "os.listdir": (0,),
        "os.stat": (0,), "os.lstat": (0,),
    }

    def audit(event, args):
        if recording and event in path_arguments:
            events.append((event, args))

    # CPython emits no stat/lstat audit events. Wrap their Python entry points
    # as well, covering Path.stat/lstat/exists/is_file and descriptor-safe calls.
    def checked_stat(function):
        def call(path, *args, **kwargs):
            audit("os." + function.__name__, (path,))
            return function(path, *args, **kwargs)
        return call

    for name in ("stat", "lstat"):
        monkeypatch.setattr(os, name, checked_stat(getattr(os, name)))

    def violations():
        found = []
        for event, args in events:
            if event == "open" and (args[0], args[2]) in root_pinning_opens:
                continue
            for index in path_arguments[event]:
                path = args[index]
                if isinstance(path, (str, bytes, os.PathLike)):
                    path = Path(os.fsdecode(path))
                    if path.is_absolute() and path.is_relative_to(sessions):
                        found.append((event, path))
        return found

    sys.addaudithook(audit)
    runtime = None
    try:
        runtime = ServerRuntime(home, cwd=tmp_path, provider="fake")
        await runtime.create_session()
        store = runtime.opened.store
        sid = store.session_id
        turn = [event async for event in runtime.loop.run_turn("hello")]
        assert any(event.type == StreamEventType.MESSAGE_UPDATE for event in turn)
        tasks = runtime.loop.tool_registry.background_tasks
        task_id, _ = await tasks.start("printf background", tmp_path, log_path=store.session_dir / "background.log")
        await tasks.wait(task_id)
        runtime.policy.always_allow = ("exec(printf foreground)",)
        result = await run_exec_macro(
            runtime.loop.tool_registry,
            ToolCall("macro-audit", "exec", {"command": "printf foreground"}),
            store.session_dir / "foreground.log",
            stream_sink=lambda event: None,
            lifecycle_sink=lambda kind: None,
        )
        assert not result.is_error
        draft = DraftPersistence(store.session_dir / "draft", directory_fd=store.directory_fd)
        draft.schedule("draft text")
        draft.flush()
        assert draft.load() == "draft text"
        draft.clear()
        message = build_user_message("log", tmp_path, (store.session_dir / "background.log",), session_store=store)
        assert "background" in message.content[1].text
        assert runtime.manager.rename(sid, "renamed").name == "renamed"
        store.append_checkpoint("audit")
        await runtime.close()
        await runtime.resume_session(sid)
        assert runtime.loop.tool_registry.background_tasks.records[0].task_id == task_id
        await runtime.close()
        runtime.manager.delete(sid)
    finally:
        if runtime is not None:
            await runtime.close()
        recording = False
    assert {"open", "os.rename", "os.remove", "os.mkdir"} <= {event for event, _ in events}
    assert any(event == "open" and (args[0], args[2]) in root_pinning_opens for event, args in events)
    assert not violations()

    # Each negative control uses the same listener and predicate as the runtime.
    # Prepare the fixture while recording is off, then isolate each operation.
    probe = sessions / "audit-probe"
    probe.mkdir()
    try:
        # Path.lstat() raises os.stat on Linux and os.lstat on macOS. Which
        # name CPython uses is a platform detail; the contract under test is
        # that the listener catches the operation at all, so accept either.
        for accepted, operation in (
            (("os.scandir",), lambda: os.scandir(probe).close()),
            (("os.listdir",), lambda: os.listdir(probe)),
            (("os.stat",), probe.stat),
            (("os.stat", "os.lstat"), probe.lstat),
            (("os.lstat",), lambda: os.lstat(probe)),
            (("os.rmdir",), probe.rmdir),
        ):
            events.clear()
            recording = True
            operation()
            recording = False
            found = violations()
            assert any((event, probe) in found for event in accepted), (
                f"expected one of {accepted} for {probe}, saw {sorted(found)}"
            )
    finally:
        recording = False
