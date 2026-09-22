from __future__ import annotations


import io


import json


from pathlib import Path


import pytest


import evals.run_evals as eval_runner


from evals.run_evals import (
    _print_report,
    build_command,
    contains_forbidden_call_shapes,
    contains_forbidden_tool,
    contains_ordered_subsequence,
    load_tasks,
    main,
    parse_events,
    qualified_tool_calls,
    run_evals,
    run_subprocess,
    verify_checks,
)


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.store import ConversationStore


from zeta.loop import AgentLoop


from zeta.runtime.driver import drive_turn


from zeta.skills import SkillCatalog


from zeta.tools import route as route_module


from zeta.tools.registry import ToolRegistry


from zeta.types import TextContent, ToolCall


async def _run_tool_event(
    tmp_path: Path, *, router_mode: bool, handler
) -> list[dict[str, object]]:
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        skill_catalog=SkillCatalog.empty(),
    )
    route_module.register(registry)
    registry.register(
        "write",
        handler,
        description="write first line",
        parameters={"type": "object"},
        requires_approval=False,
    )
    loop = AgentLoop(
        FakeBackend(
            [
                ScriptedTurn(tool_calls=[ToolCall("write-1", "write", {})]),
                ScriptedTurn(content=[TextContent("done")]),
            ]
        ),
        store,
        registry=registry,
        approval_policy=ApprovalPolicy(
            store=store, default=ApprovalDecision.ALLOW
        ),
        router_mode=router_mode,
        router_style="tool",
        skill_catalog=SkillCatalog.empty(),
    )
    stdout = io.StringIO()
    assert (
        await drive_turn(
            loop,
            "start",
            format="json",
            stdout=stdout,
            stderr=io.StringIO(),
        )
        == 0
    )
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def _sequence_bypass_events() -> list[dict[str, object]]:
    return [
        {
            "type": "tool_call",
            "id": "compound-bash",
            "name": "bash",
            "arguments": {
                "cmd": "create incoming/manifest.txt stage/total.txt summary.md"
            },
        },
        {
            "type": "tool_result",
            "id": "compound-bash",
            "name": "bash",
            "is_error": False,
        },
        {
            "type": "tool_call",
            "id": "failed-padding",
            "name": "read",
            "arguments": {"path": "incoming/manifest.txt"},
        },
        {
            "type": "tool_result",
            "id": "failed-padding",
            "name": "read",
            "is_error": True,
        },
        {
            "type": "tool_call",
            "id": "wrong-padding",
            "name": "read",
            "arguments": {"path": "unrelated.txt"},
        },
        {
            "type": "tool_result",
            "id": "wrong-padding",
            "name": "read",
            "is_error": False,
        },
        {
            "type": "tool_call",
            "id": "write-total",
            "name": "write",
            "arguments": {"path": "stage/total.txt", "content": "31\n"},
        },
        {
            "type": "tool_result",
            "id": "write-total",
            "name": "write",
            "is_error": False,
        },
        {
            "type": "tool_call",
            "id": "read-summary",
            "name": "read",
            "arguments": {"path": "summary.md"},
        },
        {
            "type": "tool_result",
            "id": "read-summary",
            "name": "read",
            "is_error": False,
        },
    ]


def test_parse_events_sums_usage_and_route_stats() -> None:
    events = [
        {"type": "turn_start", "prompt": "do it"},
        {"type": "usage", "usage": {"input_tokens": 10, "output_tokens": 4}},
        {"type": "tool_call", "name": "route"},
        {
            "type": "tool_result",
            "name": "route",
            "is_error": False,
            "content": "routed tools: read (0.79), bash (0.17), exec (0.04)",
            "structured_content": {
                "service": "jev",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 7,
                },
            },
        },
        {
            "type": "usage",
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "cache_read_input_tokens": 3,
            },
        },
        {"type": "tool_call", "name": "read"},
        {
            "type": "tool_result",
            "name": "read",
            "is_error": True,
            "content": "not available this turn — describe your step to route first: read",
            "structured_content": {
                "error_kind": "unrouted_tool",
                "error": {
                    "tool": "read",
                    "kind": "error",
                    "hint": "",
                    "message": "not available this turn — describe your step to route first: read",
                }
            },
        },
        {"type": "turn_end", "tool_calls": 2},
        {"type": "message", "role": "assistant", "text": "done"},
    ]

    result = parse_events(events)

    assert result["tool_calls"] == ["route", "read"]
    assert result["api_calls"] == 2
    assert result["input_tokens"] == 37
    assert result["output_tokens"] == 11
    assert result["cache_read_tokens"] == 10
    assert result["claude_input_tokens"] == 17
    assert result["claude_output_tokens"] == 6
    assert result["claude_cache_read_tokens"] == 3
    assert result["jev_input_tokens"] == 20
    assert result["jev_output_tokens"] == 5
    assert result["jev_cache_read_tokens"] == 7
    assert result["claude_tokens"] == 26
    assert result["jev_tokens"] == 32
    assert result["combined_tokens"] == 58
    assert result["route_calls"] == 1
    assert result["route_expansions"] == 1
    assert result["unrouted_attempts"] == 1
    assert result["router_errors"] == 0
    assert result["final_message_present"] is True


@pytest.mark.asyncio
async def test_parse_events_counts_only_real_unrouted_rejections(
    tmp_path: Path,
) -> None:
    def ordinary_failure(_arguments: dict[str, object]) -> str:
        raise RuntimeError("ordinary failure")

    ordinary_events = await _run_tool_event(
        tmp_path / "ordinary", router_mode=False, handler=ordinary_failure
    )
    rejection_events = await _run_tool_event(
        tmp_path / "rejection", router_mode=True, handler=lambda _arguments: "ok"
    )

    rejection_result = next(
        event
        for event in rejection_events
        if event.get("type") == "tool_result"
    )
    assert rejection_result["structured_content"]["error_kind"] == "unrouted_tool"
    assert parse_events(ordinary_events)["unrouted_attempts"] == 0
    assert parse_events(rejection_events)["unrouted_attempts"] == 1


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [
        (
            [
                {"tool": "read", "arguments": '{"path":"manifest.txt"}'},
                {"tool": "write", "arguments": '{"path":"total.txt"}'},
                {"tool": "read", "arguments": '{"path":"summary.md"}'},
            ],
            True,
        ),
        (
            [
                {"tool": "read", "arguments": '{"path":"manifest.txt"}'},
                {"tool": "write", "arguments": '{"path":"total.txt"}'},
                {"tool": "bash", "arguments": '{"cmd":"run"}'},
            ],
            False,
        ),
    ],
)
def test_required_call_sequence_checks_ordered_non_route_calls(
    tool_calls: list[dict[str, str]], expected: bool
) -> None:
    assert (
        contains_ordered_subsequence(
            tool_calls,
            [
                {"tool": "read", "args_contains": "manifest.txt"},
                {"tool": "write", "args_contains": "total.txt"},
                {"tool": "read", "args_contains": "summary.md"},
            ],
        )
        is expected
    )


def test_parse_events_counts_direct_route_and_ignores_non_route_results() -> None:
    result = parse_events(
        [
            {
                "type": "tool_result",
                "name": "route",
                "is_error": False,
                "content": "routed to write (confidence 1.00)",
            },
            {
                "type": "tool_result",
                "name": "other",
                "is_error": True,
                "content": "ordinary failure",
            },
            {"type": "message", "role": "assistant", "text": "ok"},
        ]
    )

    assert result["route_calls"] == 0
    assert result["route_expansions"] == 0
    assert result["unrouted_attempts"] == 0


def test_parse_events_separates_router_failures_and_counts_route_call_without_result() -> None:
    result = parse_events(
        [
            {"type": "tool_call", "name": "route"},
            {
                "type": "tool_result",
                "name": "route",
                "is_error": True,
                "content": "router failed: backend unavailable",
                "structured_content": {
                    "error": {
                        "kind": "error",
                        "hint": "",
                        "message": "router failed: backend unavailable",
                    }
                },
            },
        ]
    )

    assert result["route_calls"] == 1
    assert result["router_errors"] == 1
    assert result["unrouted_attempts"] == 0


def test_parse_events_counts_route_call_when_timeout_has_no_result() -> None:
    assert parse_events([{"type": "tool_call", "name": "route"}])["route_calls"] == 1


def test_build_command_selects_router_mode() -> None:
    router = build_command("count-and-write", "prompt", 8, "router")
    auto = build_command("count-and-write", "prompt", 8, "auto")
    stock = build_command("count-and-write", "prompt", 8, "stock")

    assert router == [
        "uv",
        "run",
        "--frozen",
        "zeta",
        "--provider",
        "claude",
        "--no-session",
        "--yolo",
        "--max-turns",
        "8",
        "--format",
        "json",
        "--router-style",
        "tool",
        "--no-memory-injection",
        "-p",
        "prompt",
    ]
    assert auto[-5:] == [
        "--router-style",
        "auto",
        "--no-memory-injection",
        "-p",
        "prompt",
    ]
    assert "--no-router" not in router
    assert stock[-4:] == [
        "--no-router",
        "--no-memory-injection",
        "-p",
        "prompt",
    ]

    isolated = build_command(
        "task", "prompt", 1, "stock", memory_config=Path("/tmp/eval.toml")
    )
    assert isolated[-6:] == [
        "--no-router",
        "--memory-config",
        "/tmp/eval.toml",
        "--no-memory-injection",
        "-p",
        "prompt",
    ]
    injected = build_command(
        "task", "prompt", 1, "stock", memory_injection=True
    )
    assert injected[-4:] == [
        "--no-router",
        "--memory-injection",
        "-p",
        "prompt",
    ]
