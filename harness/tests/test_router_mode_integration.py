from __future__ import annotations


import io


import json


from pathlib import Path


import pytest


import zeta.tools.route as route_module


from zeta.cli import build_parser


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.store import ConversationStore


from zeta.loop import AgentLoop


from zeta.providers.jev import JevRouterError, RouteResult


from zeta.runtime.driver import drive_turn


from zeta.settings import load_settings, resolve


from zeta.skills import SkillCatalog


from zeta.tools.registry import ToolRegistry


from zeta.types import TextContent, ToolCall, ToolResult


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


@pytest.mark.asyncio
async def test_headless_route_event_preserves_jev_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def route_with_usage(*_args) -> RouteResult:
        return routed("read", usage={"input_tokens": 12, "output_tokens": 3})

    monkeypatch.setattr(route_module, "route_step", route_with_usage)
    loop = build_loop(
        tmp_path,
        [
            ScriptedTurn(tool_calls=[ToolCall("route-1", "route", {"step": "read it"})]),
            ScriptedTurn(content=[TextContent("done")]),
        ],
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    assert (
        await drive_turn(
            loop,
            "start",
            format="json",
            stdout=stdout,
            stderr=stderr,
        )
        == 0
    )
    events = [json.loads(line) for line in stdout.getvalue().splitlines()]
    route_result = next(
        event
        for event in events
        if event.get("type") == "tool_result" and event.get("name") == "route"
    )

    assert route_result["structured_content"] == {
        "service": "jev",
        "usage": {"input_tokens": 12, "output_tokens": 3},
        "confidence": 0.9,
        "needs_tool": 1.0,
        "step_clarity": 0.8,
    }


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


@pytest.mark.parametrize("jev_compaction", [False, True])
@pytest.mark.asyncio
async def test_child_loop_inherits_jev_compaction_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    jev_compaction: bool,
) -> None:
    call = ToolCall(
        "agent-1",
        "agent",
        {"prompt": "inspect", "description": "child", "background": False},
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
        jev_compaction=jev_compaction,
        skill_catalog=SkillCatalog.empty(),
    )
    import zeta.loop as loop_module

    captured: list[bool] = []
    captured_styles: list[str] = []
    real_agent_loop = loop_module.AgentLoop

    class SpyAgentLoop(real_agent_loop):
        def __init__(self, *args, **kwargs):
            captured.append(kwargs["jev_compaction"])
            captured_styles.append(kwargs["router_style"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(loop_module, "AgentLoop", SpyAgentLoop)

    await collect(loop.run_turn("start"))

    assert captured == [jev_compaction]
    assert captured_styles == ["auto"]
