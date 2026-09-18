from __future__ import annotations

from pathlib import Path

import evals.run_evals as eval_runner
from evals.run_evals import (
    _print_report,
    build_command,
    load_tasks,
    parse_events,
    run_evals,
    run_subprocess,
    verify_checks,
)


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
        assert set(task) == {"id", "prompt", "setup", "checks", "max_turns"}
        assert isinstance(task["id"], str)
        assert isinstance(task["prompt"], str)
        assert task["checks"]
        assert isinstance(task["setup"], dict)
        assert isinstance(task["max_turns"], int)
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
