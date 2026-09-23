from __future__ import annotations

from pathlib import Path

import pytest

import zeta.tools.route as route_module
from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import TextContent, ToolCall, ToolResult
from zeta.providers.jev import JevRouterError, RouteResult
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools.registry import ToolRegistry


async def collect(events):
    return [event async for event in events]


def routed(
    tool: str, confidence: float = 0.9, probabilities=None, usage=None
) -> RouteResult:
    return RouteResult(
        tool=tool,
        probabilities=probabilities or {tool: confidence},
        confidence=confidence,
        needs_tool=1.0,
        step_clarity=0.8,
        usage=usage or {},
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
            lambda _arguments, *, name=name: name,
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
        router_style="tool",
        skill_catalog=SkillCatalog.empty(),
    )


async def _route(tool: str) -> RouteResult:
    return routed(tool)


def test_catalog_uses_structured_criteria_for_confusable_tools() -> None:
    names = (
        "read",
        "write",
        "edit",
        "fetch",
        "websearch",
        "bash",
        "exec",
        "automation",
        "skill",
        "todo",
        "run_background",
        "task_output",
        "task_kill",
        "agent",
        "agent_status",
        "agent_output",
        "agent_send",
    )
    catalog = route_module.build_catalog(
        [{"name": name, "description": f"{name} description"} for name in names]
    )

    assert catalog["bash"] == {
        "what": "bash description",
        "not_for": "Editing a file in place; use edit. Use exec for fixed-cwd commands with timeout or output limits, and run_background for long-running commands.",
        "examples": [
            "Use bash to cd into the reports directory, then list its files in the next shell call.",
        ],
    }
    assert "timeout" in catalog["exec"]["examples"][0]
    assert "task_id" in catalog["run_background"]["examples"][0]
    assert "cursor" in catalog["task_output"]["examples"][0]
    assert "transcript" in catalog["agent_output"]["examples"][0]
    examples = [
        example
        for entry in catalog.values()
        for example in entry["examples"]
    ]
    assert len(examples) == len(set(examples))
    assert catalog["task_output"]["examples"] != catalog["agent_output"]["examples"]
    assert all(
        set(entry) == {"what", "not_for", "examples"}
        and 1 <= len(entry["examples"]) <= 2
        for entry in catalog.values()
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
async def test_routed_tool_executes_before_route_state_is_cleared(
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

    result = loop.store.messages()[4].tool_result
    assert result is not None
    assert result.is_error is False
    assert result.content == "read"
    assert {schema["name"] for schema in loop.backend.calls[2][1]} == {"route"}


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
        "error_kind": "unrouted_tool",
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
async def test_fail_open_tool_executes_before_fail_open_state_is_consumed(
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

    await collect(loop.run_turn("start"))

    result = loop.store.messages()[4].tool_result
    assert result is not None
    assert result.is_error is False
    assert result.content == "read"
    assert {schema["name"] for schema in loop.backend.calls[2][1]} == {"route"}


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
