from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import evals.run_evals as eval_runner
from evals.run_evals import (
    _print_report,
    build_command,
    contains_ordered_subsequence,
    load_tasks,
    parse_events,
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


def test_tasks_have_required_shape_and_safe_prompts() -> None:
    tasks = load_tasks()

    assert len(tasks) == 6
    assert {task["id"] for task in tasks} == {
        "count-and-write",
        "merge-and-sort",
        "find-patterns",
        "in-place-edit",
        "run-and-record",
        "chain-and-verify",
    }
    forbidden = {"read", "write", "edit", "bash", "exec", "grep", "route"}
    for task in tasks:
        assert set(task) in (
            {"id", "prompt", "setup", "checks", "max_turns"},
            {
                "id",
                "prompt",
                "setup",
                "checks",
                "max_turns",
                "required_call_sequence",
            },
        )
        assert isinstance(task["id"], str)
        assert isinstance(task["prompt"], str)
        assert task["checks"]
        assert isinstance(task["setup"], dict)
        assert isinstance(task["max_turns"], int)
        if task["id"] == "chain-and-verify":
            assert task["required_call_sequence"] == ["read", "write", "read"]
        else:
            assert "required_call_sequence" not in task
        assert all(
            set(check)
            in ({"path", "equals"}, {"path", "normalized_equals"}, {"path", "contains"})
            for check in task["checks"]
        )
        words = set(task["prompt"].lower().replace(".", "").split())
        assert not words & forbidden, task["id"]


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
        (["read", "write", "read"], True),
        (["route", "read", "write", "bash"], False),
    ],
)
def test_required_call_sequence_checks_ordered_non_route_calls(
    tool_calls: list[str], expected: bool
) -> None:
    non_route_calls = [name for name in tool_calls if name != "route"]
    assert (
        contains_ordered_subsequence(non_route_calls, ["read", "write", "read"])
        is expected
    )


@pytest.mark.parametrize(
    "required_call_sequence",
    ["read", ["read", 3]],
)
def test_load_tasks_rejects_invalid_required_call_sequence(
    tmp_path: Path, required_call_sequence: object
) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "task",
                "prompt": "prompt",
                "setup": {},
                "checks": [],
                "required_call_sequence": required_call_sequence,
                "max_turns": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="required_call_sequence"):
        load_tasks(path)


def test_run_one_requires_the_call_sequence(tmp_path: Path, monkeypatch) -> None:
    def fake_run_subprocess(*args: object, **kwargs: object) -> dict[str, object]:
        del kwargs
        events_path = Path(args[2])
        events_path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "tool_call", "name": "bash"}),
                    json.dumps(
                        {"type": "message", "role": "assistant", "text": "done"}
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {"returncode": 0, "stderr": "", "timed_out": False}

    monkeypatch.setattr(eval_runner, "run_subprocess", fake_run_subprocess)
    record = eval_runner._run_one(
        {
            "id": "chain-and-verify",
            "prompt": "prompt",
            "setup": {},
            "checks": [],
            "required_call_sequence": ["read", "write", "read"],
            "max_turns": 1,
        },
        "stock",
        tmp_path / "runs",
    )

    assert record["checks_passed"] == [False]


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


def test_verify_checks_supports_equals_contains_and_missing(tmp_path: Path) -> None:
    (tmp_path / "done.txt").write_text("alpha\nbeta\n", encoding="utf-8")

    assert verify_checks(
        tmp_path,
        [
            {"path": "done.txt", "equals": "alpha\nbeta\n"},
            {"path": "done.txt", "contains": "beta"},
            {"path": "done.txt", "normalized_equals": "alpha beta"},
            {"path": "missing.txt", "contains": "nope"},
        ],
    ) == [True, True, True, False]


def test_verify_normalized_equals_rejects_extra_output(tmp_path: Path) -> None:
    (tmp_path / "word-count.txt").write_text("      4 input.txt\n", encoding="utf-8")

    assert verify_checks(
        tmp_path,
        [{"path": "word-count.txt", "normalized_equals": "4 input.txt"}],
    ) == [True]
    assert verify_checks(
        tmp_path,
        [{"path": "word-count.txt", "normalized_equals": "4 input.txt extra"}],
    ) == [False]


def test_build_command_selects_router_mode() -> None:
    router = build_command("count-and-write", "prompt", 8, "router")
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
        "--router",
        "-p",
        "prompt",
    ]
    assert "--no-router" not in router
    assert stock[-3:] == ["--no-router", "-p", "prompt"]


def test_run_subprocess_records_timeout_and_partial_stream(tmp_path: Path) -> None:
    events_path = tmp_path / "events.jsonl"

    class TimeoutRunner:
        def __call__(self, *args: object, **kwargs: object) -> object:
            raise __import__("subprocess").TimeoutExpired(
                kwargs["timeout"], args[0], output=b'{"type":"tool_call"}\n'
            )

    result = run_subprocess(
        ["fake"],
        tmp_path,
        events_path,
        timeout=3,
        runner=TimeoutRunner(),
    )

    assert result["timed_out"] is True
    assert result["returncode"] is None
    assert events_path.read_text(encoding="utf-8") == '{"type":"tool_call"}\n'


def test_report_shows_separate_and_combined_token_totals(capsys) -> None:
    _print_report(
        [
            {
                "task_id": "task",
                "mode": "router",
                "completed": True,
                "checks_passed": [True],
                "tool_calls": [],
                "claude_tokens": 26,
                "jev_tokens": 32,
                "cache_read_tokens": 10,
                "router_errors": 1,
                "combined_tokens": 58,
            }
        ]
    )

    output = capsys.readouterr().out
    assert "claude=26 jev=32 cache_read=10 router_errors=1 combined=58" in output
    assert (
        "claude_tokens=26 jev_tokens=32 cache_read=10 router_errors=1 "
        "combined_tokens=58"
    ) in output


def test_run_evals_continues_after_timed_out_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(eval_runner, "SCRATCH_ROOT", tmp_path / "scratch")
    results = iter(
        [
            {
                "returncode": None,
                "stderr": "timeout",
                "timed_out": True,
            },
            {
                "returncode": 0,
                "stderr": "",
                "timed_out": False,
            },
        ]
    )

    def fake_run_subprocess(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return next(results)

    monkeypatch.setattr(eval_runner, "run_subprocess", fake_run_subprocess)
    tasks = [
        {"id": "first", "prompt": "one", "setup": {}, "checks": [], "max_turns": 1},
        {"id": "second", "prompt": "two", "setup": {}, "checks": [], "max_turns": 1},
    ]

    records = run_evals(tasks, ["stock"], tmp_path / "results.json")

    assert [record["task_id"] for record in records] == ["first", "second"]
    assert records[0]["timed_out"] is True
    assert records[1]["timed_out"] is False
