import ast


import inspect


import json


import re


from io import StringIO


from pathlib import Path


import pytest


from prompt_toolkit.application.current import set_app


from prompt_toolkit.data_structures import Size


from prompt_toolkit.output.vt100 import Vt100_Output


from rich.console import Console


from zeta.core.fake import FakeBackend


from zeta.core.slash import create_slash_registry


from zeta.core.store import ConversationStore


from zeta.core.todo import TODO_STATUSES


from zeta.loop import AgentLoop


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.tools import todo as todo_tool


from zeta.tui.app import TUIApp


from zeta.tui.todo import TodoWidget


from zeta.types import ToolCall


def _registry(tmp_path: Path) -> tuple[ConversationStore, ToolRegistry]:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    return store, ToolRegistry(tmp_path, session_store=store, skill_catalog=SkillCatalog.empty())


def _todo_handler_argument_keys() -> set[str]:
    tree = ast.parse(inspect.getsource(todo_tool._todo))
    keys: set[str] = set()

    def string_constant(node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id == "arguments":
                key = string_constant(node.slice)
                if key is not None:
                    keys.add(key)
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "arguments"
                and node.func.attr == "get"
                and node.args
            ):
                key = string_constant(node.args[0])
                if key is not None:
                    keys.add(key)
        elif isinstance(node, ast.Compare):
            if len(node.comparators) != 1 or not isinstance(
                node.ops[0], (ast.In, ast.NotIn)
            ):
                continue
            argument_name = node.comparators[0]
            if not (
                isinstance(argument_name, ast.Name) and argument_name.id == "arguments"
            ):
                continue
            key = string_constant(node.left)
            if key is not None:
                keys.add(key)

    return keys


@pytest.mark.asyncio
async def test_todo_writes_and_reads_the_full_list(tmp_path: Path) -> None:
    store, registry = _registry(tmp_path)
    items = [
        {"content": "inspect code", "status": "in_progress"},
        {"content": "run tests", "status": "pending"},
    ]

    written = await registry.execute(ToolCall("write", "todo", {"items": items}))
    read = await registry.execute(ToolCall("read", "todo", {}))
    action_read = await registry.execute(ToolCall("action-read", "todo", {}))

    expected = {
        "items": items,
        "counts": {
            "pending": 1,
            "in_progress": 1,
            "completed": 0,
            "canceled": 0,
        },
    }
    assert written["isError"] is False
    assert written["structuredContent"] == expected
    assert read["structuredContent"] == expected
    assert action_read["structuredContent"] == expected
    assert store.todo_items() == items


@pytest.mark.asyncio
async def test_todo_text_result_includes_canceled_count(tmp_path: Path) -> None:
    _, registry = _registry(tmp_path)

    result = await registry.execute(
        ToolCall(
            "canceled",
            "todo",
            {"items": [{"content": "stopped", "status": "canceled"}]},
        )
    )

    assert result["content"][0]["text"] == (
        "todo list: 0 pending, 0 in progress, 0 completed, 1 canceled"
    )


def test_todo_schema_matches_handler_contract(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
    schema = next(schema for schema in registry.schemas if schema["name"] == "todo")
    parameters = schema["parameters"]
    properties = parameters["properties"]
    item_schema = properties["items"]["items"]

    assert "Read the current todo list when items is omitted." in schema["description"]
    assert "Write the full todo list by providing items." in schema["description"]
    assert set(properties) == _todo_handler_argument_keys()
    assert set(item_schema["properties"]) == {"content", "status"}
    assert item_schema["required"] == ["content", "status"]
    assert item_schema["additionalProperties"] is False
    assert item_schema["properties"]["status"]["enum"] == list(TODO_STATUSES)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {},
        *(
            {"items": [{"content": status, "status": status}]}
            for status in TODO_STATUSES
        ),
    ],
)
async def test_every_todo_handler_branch_is_schema_representable(
    tmp_path: Path, arguments: dict[str, object]
) -> None:
    _, registry = _registry(tmp_path)

    result = await registry.execute(ToolCall("branch", "todo", arguments))

    assert result["isError"] is False


@pytest.mark.asyncio
async def test_todo_rejects_removed_action_argument(tmp_path: Path) -> None:
    _, registry = _registry(tmp_path)

    result = await registry.execute(ToolCall("action", "todo", {"action": "read"}))

    assert result["isError"] is True
    assert (
        "unexpected properties: action"
        in result["structuredContent"]["error"]["message"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("items", "reason"),
    [
        ([{"content": " ", "status": "pending"}], "content must be nonempty"),
        ([{"content": "bad status", "status": "paused"}], "status must be one of"),
    ],
)
async def test_todo_rejects_invalid_lists_without_mutating_state(
    tmp_path: Path, items: list[dict[str, str]], reason: str
) -> None:
    store, registry = _registry(tmp_path)
    await registry.execute(
        ToolCall(
            "initial",
            "todo",
            {"items": [{"content": "keep", "status": "pending"}]},
        )
    )
    state_before = store.state_path.read_bytes()

    result = await registry.execute(ToolCall("invalid", "todo", {"items": items}))

    assert result["isError"] is True
    assert reason in result["structuredContent"]["error"]["message"]
    assert store.todo_items() == [{"content": "keep", "status": "pending"}]
    assert store.state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_todo_accepts_multiple_in_progress_items(tmp_path: Path) -> None:
    """ZETA-70: multiple in_progress items are allowed (was a live error path)."""

    store, registry = _registry(tmp_path)
    items = [
        {"content": "first", "status": "in_progress"},
        {"content": "second", "status": "in_progress"},
        {"content": "third", "status": "pending"},
    ]

    result = await registry.execute(ToolCall("multi", "todo", {"items": items}))

    assert result["isError"] is False
    assert result["structuredContent"]["counts"]["in_progress"] == 2
    assert store.todo_items() == items


@pytest.mark.asyncio
async def test_todo_accepts_fifty_items(tmp_path: Path) -> None:
    store, registry = _registry(tmp_path)
    items = [{"content": f"task {index}", "status": "pending"} for index in range(50)]

    result = await registry.execute(ToolCall("fifty", "todo", {"items": items}))

    assert result["isError"] is False
    assert store.todo_items() == items


@pytest.mark.asyncio
async def test_todo_rejects_more_than_fifty_items_without_mutating_state(
    tmp_path: Path,
) -> None:
    store, registry = _registry(tmp_path)
    initial = [{"content": "keep", "status": "pending"}]
    await registry.execute(ToolCall("initial", "todo", {"items": initial}))
    state_before = store.state_path.read_bytes()
    items = [{"content": f"task {index}", "status": "pending"} for index in range(51)]

    result = await registry.execute(ToolCall("fifty-one", "todo", {"items": items}))

    assert result["isError"] is True
    assert (
        "more than 50 items"
        in result["structuredContent"]["error"]["message"]
    )
    assert store.todo_items() == initial
    assert store.state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_todo_rejects_overlong_content_without_mutating_state(
    tmp_path: Path,
) -> None:
    store, registry = _registry(tmp_path)
    initial = [{"content": "keep", "status": "pending"}]
    await registry.execute(ToolCall("initial", "todo", {"items": initial}))
    state_before = store.state_path.read_bytes()

    result = await registry.execute(
        ToolCall(
            "overlong",
            "todo",
            {"items": [{"content": "x" * 501, "status": "pending"}]},
        )
    )

    assert result["isError"] is True
    assert (
        "cannot exceed 500 characters"
        in result["structuredContent"]["error"]["message"]
    )
    assert store.todo_items() == initial
    assert store.state_path.read_bytes() == state_before


@pytest.mark.asyncio
async def test_todo_empty_list_clears_state_and_does_not_pollute_transcript(
    tmp_path: Path,
) -> None:
    store, registry = _registry(tmp_path)
    await registry.execute(
        ToolCall(
            "write",
            "todo",
            {"items": [{"content": "remove", "status": "completed"}]},
        )
    )

    result = await registry.execute(ToolCall("clear", "todo", {"items": []}))
    reopened = ConversationStore(tmp_path / "sessions", session_id=store.session_id)

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "items": [],
        "counts": {
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "canceled": 0,
        },
    }
    assert reopened.todo_items() == []
    assert "todo_items" not in json.loads(reopened.state_path.read_text())
    assert reopened.messages() == []


def test_todo_canceled_items_validate_count_and_render(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items(
        [
            {"content": "stopped", "status": "canceled"},
            {"content": "next", "status": "pending"},
        ]
    )
    widget = TodoWidget(store)

    assert widget.create_content(80, 20).line_count == 2
    rendered = "".join(
        fragment[1] for fragment in widget.create_content(80, 20).get_line(0)
    )
    assert "[-] stopped" in rendered


def test_todo_dismissal_persists_across_resume_and_fork(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items([{"content": "done", "status": "canceled"}])
    store.append_checkpoint("before dismissal")
    widget = TodoWidget(store)
    widget.turn_boundary()

    resumed = ConversationStore(tmp_path / "sessions", session_id=store.session_id)
    assert not TodoWidget(resumed).visible

    store.append_fork("before dismissal")
    assert TodoWidget(store).visible
