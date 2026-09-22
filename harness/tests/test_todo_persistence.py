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


def test_todo_items_persist_across_store_resume(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.set_todo_items([{"content": "resume me", "status": "pending"}])

    resumed = ConversationStore(tmp_path / "sessions", session_id=store.session_id)

    assert resumed.todo_items() == [{"content": "resume me", "status": "pending"}]
    assert json.loads(resumed.state_path.read_text())["todo_items"] == resumed.todo_items()
