from __future__ import annotations

from pathlib import Path

import pytest

import zeta.tools.route as route_module
from zeta.cli import build_parser
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.providers.jev import JevRouterError, RouteResult
from zeta.settings import load_settings, resolve
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry
from zeta.types import TextContent, ToolCall, ToolResult


async def collect(events):
    return [event async for event in events]


def routed(tool: str, confidence: float = 0.9, probabilities=None) -> RouteResult:
    return RouteResult(
        tool=tool,
        probabilities=probabilities or {tool: confidence},
        confidence=confidence,
        needs_tool=1.0,
        step_clarity=0.8,
        usage={},
    )


def build_loop(
    tmp_path: Path,
    turns: list[ScriptedTurn],
    *,
    router_mode: bool = True,
    names: tuple[str, ...] = ("read", "write", "bash"),
) -> AgentLoop:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    route_module.register(registry)
    for name in names:
        registry.register(
            name,
            lambda _arguments, name=name: name,
            description=f"{name} first line\nsecond line",
            parameters={"type": "object"},
            requires_approval=False,
        )
    return AgentLoop(
        FakeBackend(turns),
        store,
        registry=registry,
        approval_policy=ApprovalPolicy(
            store=store, default=ApprovalDecision.ALLOW
        ),
        router_mode=router_mode,
        skill_catalog=SkillCatalog.empty(),
    )


@pytest.mark.asyncio
async def test_router_advertises_route_then_the_selected_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(route_module, "route_step", lambda *_args: _route("read"))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "read it"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert {schema["name"] for schema in loop.backend.calls[0][1]} == {"route"}
    assert {schema["name"] for schema in loop.backend.calls[1][1]} == {"route", "read"}


@pytest.mark.asyncio
async def test_router_replaces_then_clears_routed_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(route_module, "route_step", lambda *_args: _route("read"))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "read"})]),
            ScriptedTurn(tool_calls=[ToolCall("read-1", "read", {})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert {schema["name"] for schema in loop.backend.calls[2][1]} == {"route"}
    assert loop._routed_tools == []


@pytest.mark.asyncio
async def test_router_resets_state_at_the_start_of_each_user_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(route_module, "route_step", lambda *_args: _route("read"))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "read"})]),
            ScriptedTurn(content=[TextContent("first done")]),
            ScriptedTurn(content=[TextContent("second done")]),
        ],
    )

    await collect(loop.run_turn("first"))
    await collect(loop.run_turn("second"))

    assert {schema["name"] for schema in loop.backend.calls[2][1]} == {"route"}
    assert loop._routed_tools == []


@pytest.mark.asyncio
async def test_router_mixed_batch_keeps_new_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = iter([routed("read"), routed("write")])

    async def next_route(*_args):
        return next(results)

    monkeypatch.setattr(route_module, "route_step", next_route)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "read"})]),
            ScriptedTurn(
                tool_calls=[
                    ToolCall("route-2", "route", {"step": "write"}),
                    ToolCall("read-1", "read", {}),
                ]
            ),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert loop._routed_tools == ["write"]
    assert {schema["name"] for schema in loop.backend.calls[2][1]} == {"route", "write"}


@pytest.mark.asyncio
async def test_unrouted_registered_tool_is_rejected(
    tmp_path: Path,
) -> None:
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("write-1", "write", {})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    result = loop.store.messages()[2].tool_result
    assert result is not None
    assert result.is_error is True
    assert "not available this turn" in result.content
    assert result.structured_content == {
        "error": {
            "tool": "write",
            "kind": "error",
            "hint": "",
            "message": result.content,
        }
    }
    assert loop.unrouted_attempts == 1


@pytest.mark.asyncio
async def test_low_confidence_route_advertises_top_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        route_module,
        "route_step",
        lambda *_args: _route(
            "read", 0.7, {"read": 0.7, "write": 0.2, "bash": 0.1, "route": 0.0}
        ),
    )
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "do it"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert {schema["name"] for schema in loop.backend.calls[1][1]} == {
        "route",
        "read",
        "write",
        "bash",
    }


@pytest.mark.asyncio
async def test_router_failure_fails_open_and_excludes_route_from_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalogs: list[dict[str, str]] = []

    async def fail(
        step: str, catalog: dict[str, str], history: list[str]
    ) -> RouteResult:
        del step, history
        catalogs.append(catalog)
        raise JevRouterError("backend unavailable")

    monkeypatch.setattr(route_module, "route_step", fail)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "do it"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert "route" not in catalogs[0]
    assert {schema["name"] for schema in loop.backend.calls[1][1]} == {
        "route",
        "read",
        "write",
        "bash",
    }


@pytest.mark.asyncio
async def test_router_failure_fail_open_is_consumed_by_one_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(*_args):
        raise JevRouterError("backend unavailable")

    monkeypatch.setattr(route_module, "route_step", fail)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "do it"})]),
            ScriptedTurn(tool_calls=[ToolCall("read-1", "read", {})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("first"))

    assert {schema["name"] for schema in loop.backend.calls[1][1]} == {
        "route",
        "read",
        "write",
        "bash",
    }
    assert {schema["name"] for schema in loop.backend.calls[2][1]} == {"route"}
    assert loop._router_fail_open is False


@pytest.mark.asyncio
async def test_route_passes_recent_steps_to_jev(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    histories: list[list[str]] = []
    results = iter([routed("read"), routed("write")])

    async def capture_history(
        _step: str, _catalog: dict[str, str], history: list[str]
    ) -> RouteResult:
        histories.append(history)
        return next(results)

    monkeypatch.setattr(route_module, "route_step", capture_history)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "first"})]),
            ScriptedTurn(tool_calls=[ToolCall("route-2", "route", {"step": "second"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )

    await collect(loop.run_turn("start"))

    assert histories == [[], ["first"]]


@pytest.mark.asyncio
async def test_unrouted_rejection_uses_registry_governance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("write-1", "write", {})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )
    governed = ToolResult(
        "write-1",
        "governed rejection",
        is_error=True,
        structured_content={"error": {"source": "registry"}},
    )
    calls: list[str] = []

    def govern(tool_call: ToolCall, result: ToolResult) -> ToolResult:
        calls.append(tool_call.name)
        assert result.is_error is True
        return governed

    monkeypatch.setattr(loop.tool_registry, "govern_tool_result", govern)

    await collect(loop.run_turn("start"))

    result = loop.store.messages()[2].tool_result
    assert result is not None
    assert result.structured_content == {"error": {"source": "registry"}}
    assert calls == ["write"]


def test_no_router_flag_and_setting_restore_full_toolset(
    tmp_path: Path,
) -> None:
    parser = build_parser()
    assert parser.parse_args(["--no-router"]).router is False
    (tmp_path / "settings.toml").write_text("router = false\n", encoding="utf-8")
    settings = load_settings(home=tmp_path).settings
    assert resolve(
        settings,
        cli_provider=None,
        cli_model=None,
        cli_yolo=None,
        cli_token_budget=None,
    ).router is False


@pytest.mark.asyncio
async def test_plan_filter_runs_after_router_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(route_module, "route_step", lambda *_args: _route("read"))
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "read"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
        names=("read", "bash"),
    )
    loop.set_plan_mode(True)

    await collect(loop.run_turn("start"))

    assert {schema["name"] for schema in loop.backend.calls[0][1]} == {"route"}
    assert {schema["name"] for schema in loop.backend.calls[1][1]} == {
        "route",
        "read",
    }


@pytest.mark.asyncio
async def test_child_loop_inherits_router_mode(
    tmp_path: Path,
) -> None:
    call = ToolCall(
        "agent-1",
        "agent",
        {
            "prompt": "inspect",
            "description": "child",
            "background": False,
        },
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[call]),
            ScriptedTurn(content=[TextContent("child done")]),
        ]
    )
    store = ConversationStore(tmp_path)
    loop = AgentLoop(
        backend,
        store,
        approval_policy=ApprovalPolicy(store=store, default=ApprovalDecision.ALLOW),
        max_turns=1,
        router_mode=False,
        skill_catalog=SkillCatalog.empty(),
    )

    await collect(loop.run_turn("start"))

    child_schemas = {schema["name"] for schema in backend.calls[1][1]}
    assert "route" not in child_schemas
    assert "read" in child_schemas


async def _route(tool: str) -> RouteResult:
    return routed(tool)
