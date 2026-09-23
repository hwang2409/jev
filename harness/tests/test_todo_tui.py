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


def test_todo_widget_hides_empty_lists_and_bounds_visible_rows(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    assert widget.create_content(80, 20).line_count == 0

    store.set_todo_items(
        [{"content": f"task {index}", "status": "pending"} for index in range(8)]
    )
    content = widget.create_content(80, 20)
    rendered = [
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    ]

    assert content.line_count == 7
    assert rendered[:2] == ["[ ] task 0", "[ ] task 1"]
    assert rendered[-1] == "+2 more"
    assert all(len(line) <= 80 for line in rendered)


def test_todo_widget_collapses_completed_list_and_dismisses_at_boundary(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    store.set_todo_items(
        [
            {"content": "first", "status": "completed"},
            {"content": "second", "status": "completed"},
        ]
    )

    content = widget.create_content(80, 20)
    assert content.line_count == 1
    assert "todos done (2)" in "".join(fragment[1] for fragment in content.get_line(0))
    assert widget.visible

    widget.turn_boundary()

    assert not widget.visible
    assert widget.create_content(80, 20).line_count == 0


def test_todo_widget_repins_after_a_new_write(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    store.set_todo_items([{"content": "done", "status": "completed"}])
    widget.create_content(80, 20)
    widget.turn_boundary()
    assert not widget.visible

    store.set_todo_items([{"content": "new", "status": "pending"}])

    assert widget.visible
    assert widget.create_content(80, 20).line_count == 1


def test_todo_widget_keeps_mixed_lists_pinned(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    widget = TodoWidget(store)
    store.set_todo_items(
        [
            {"content": "done", "status": "completed"},
            {"content": "work", "status": "in_progress"},
        ]
    )

    widget.turn_boundary()

    assert widget.visible
    assert widget.create_content(80, 20).line_count == 2


@pytest.mark.asyncio
async def test_todo_widget_keeps_overflow_summary_in_an_80_by_24_terminal(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items(
        [{"content": f"task {index}", "status": "pending"} for index in range(8)]
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._install_full_screen_layout(session)
    output = StringIO()
    terminal_output = Vt100_Output(output, lambda: Size(rows=24, columns=80))
    session.app.output = terminal_output
    session.app.renderer.output = terminal_output

    with set_app(session.app):
        session.app.renderer.render(session.app, session.app.layout)

    rendered = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output.getvalue())
    assert rendered.count("[ ] task ") == 6
    assert "+2 more" in rendered


def test_todo_widget_uses_plain_status_glyphs_and_truncates_content(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items(
        [
            {"content": "pending", "status": "pending"},
            {"content": "active", "status": "in_progress"},
            {"content": "done", "status": "completed"},
        ]
    )
    widget = TodoWidget(store)
    content = widget.create_content(10, 10)
    rendered = [
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    ]

    assert rendered == ["[ ] pendi…", "[>] active", "[x] done"]
    assert all("✱" not in line for line in rendered)


def test_status_includes_todo_counts_only_when_nonempty(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    registry = create_slash_registry(skill_catalog=SkillCatalog.empty())

    assert "todo:" not in registry.dispatch(app, "/status")
    store.set_todo_items([{"content": "one", "status": "completed"}])

    output = registry.dispatch(app, "/status")
    assert output is not None
    assert "todo: pending=0, in_progress=0, completed=1, canceled=0" in output


def test_full_screen_layout_places_todo_between_transcript_and_composer(
    tmp_path: Path,
) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions"), skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._install_full_screen_layout(session)

    # The command-menu float container wraps the padded content.
    content = session.layout.container.children[0].content.children[1]
    bottom = content.children[1].content
    todo_panel = bottom.children[0]

    assert todo_panel.__class__.__name__ == "ConditionalContainer"
    assert todo_panel.content.content is app._todo_widget
