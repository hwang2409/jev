from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

import evals.run_evals as eval_runner
from evals.run_evals import (
    _print_report,
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
from zeta.protocol.types import TextContent, ToolCall
from zeta.runtime.driver import drive_turn
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tools import route as route_module
from zeta.tools.registry import ToolRegistry


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
                "forbidden_tools",
            },
        )
        assert isinstance(task["id"], str)
        assert isinstance(task["prompt"], str)
        assert task["checks"]
        assert isinstance(task["setup"], dict)
        assert isinstance(task["max_turns"], int)
        if task["id"] == "chain-and-verify":
            assert task["required_call_sequence"] == [
                {"tool": "read", "args_contains": "incoming/manifest.txt"},
                {"tool": "write", "args_contains": "stage/total.txt"},
                {"tool": "read", "args_contains": "summary.md"},
            ]
            assert task["forbidden_tools"] == ["bash", "exec"]
        else:
            assert "required_call_sequence" not in task
            assert "forbidden_tools" not in task
        assert all(
            set(check)
            in ({"path", "equals"}, {"path", "normalized_equals"}, {"path", "contains"})
            for check in task["checks"]
        )
        words = set(task["prompt"].lower().replace(".", "").split())
        assert not words & forbidden, task["id"]


def test_tools_tasks_define_eight_realistic_offline_tasks() -> None:
    tasks = load_tasks(Path(__file__).parents[1] / "evals" / "tasks_tools.jsonl")

    assert len(tasks) == 8
    assert {task["id"] for task in tasks} == {
        "store-then-recall",
        "seeded-recall",
        "two-note-synthesis",
        "append-then-latest",
        "calendar-window",
        "free-slot-reasoning",
        "create-then-verify",
        "cross-surface",
    }
    assert tasks[0]["required_call_sequence"] == [
        {"tool": "memory_store", "args_contains": "launch-review"},
        {"tool": "memory_search"},
    ]
    assert tasks[0]["checks"][0] == {
        "corpus_path": "launch-review.md",
        "contains": "Tuesday at 15:00 in Room Cedar",
    }
    assert tasks[6]["checks_calendar_created"] == [
        {
            "title": "Project kickoff",
            "start": "2026-09-19T16:00:00",
            "end": "2026-09-19T17:00:00",
            "calendar": "work",
        }
    ]
    assert tasks[5]["required_call_sequence"] == [
        {
            "tool": "calendar_events",
            "args": {
                "start": {
                    "covers": {
                        "start": "2026-09-19T09:00:00",
                        "end": "2026-09-19T12:00:00",
                    }
                }
            },
        }
    ]
    assert tasks[6]["required_call_sequence"] == [
        {
            "tool": "calendar_create",
            "args": {
                "title": {"equals": "Project kickoff"},
                "start": {"equals": "2026-09-19T16:00:00"},
                "end": {"equals": "2026-09-19T17:00:00"},
                "calendar": {"equals": "work", "casefold": True},
            },
        },
        {
            "tool": "calendar_events",
            "args": {
                "start": {
                    "covers": {
                        "start": "2026-09-19T16:00:00",
                        "end": "2026-09-19T17:00:00",
                    }
                }
            },
        },
    ]
    assert tasks[6]["forbidden_call_shapes"] == [
        {
            "tool": "calendar_create",
            "args_not": {
                "title": {"equals": "Project kickoff"},
                "start": {"equals": "2026-09-19T16:00:00"},
                "end": {"equals": "2026-09-19T17:00:00"},
                "calendar": {"equals": "work", "casefold": True},
            },
        }
    ]
    assert "write exactly: <ISO start> to <ISO end>" in tasks[5]["prompt"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("memory_seed", {"../outside.md": "no"}),
        ("memory_seed", {"note.txt": "wrong suffix"}),
        ("calendar_seed", [{"title": "missing fields"}]),
        ("checks_calendar_created", [{"title": "missing fields"}]),
    ],
)
def test_load_tasks_rejects_invalid_tool_fixture_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "task",
                "prompt": "prompt",
                "setup": {},
                "checks": [],
                "max_turns": 1,
                field: value,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        load_tasks(path)


def test_memory_seed_roundtrip_reindexes_and_cleans_up(tmp_path: Path) -> None:
    with eval_runner._prepare_task_environment(
        {"memory_seed": {"notes/seed.md": "# Seed\n\nA durable fact.\n"}},
        tmp_path / "scratch",
    ) as (environment, memory_root, config):
        assert environment["ZETA_CALENDAR_ADAPTER"].startswith("fake:")
        assert (memory_root / "notes/seed.md").read_text(encoding="utf-8").endswith(
            "A durable fact.\n"
        )
        assert config.parent.exists()
    assert not config.parent.exists()


def test_sequential_memory_task_environments_do_not_share_corpus(
    tmp_path: Path,
) -> None:
    task = {"memory_seed": {}}
    with eval_runner._prepare_task_environment(task, tmp_path / "first") as (
        _environment,
        first_root,
        _first_config,
    ):
        (first_root / "created-by-agent.md").write_text("private\n", encoding="utf-8")

    with eval_runner._prepare_task_environment(task, tmp_path / "second") as (
        _environment,
        second_root,
        _second_config,
    ):
        assert not (second_root / "created-by-agent.md").exists()


def test_memory_root_probe_fails_closed_for_non_eval_config(tmp_path: Path) -> None:
    _expected_root, _expected_config = eval_runner._write_memory_config(
        tmp_path / "expected"
    )
    _live_root, live_config = eval_runner._write_memory_config(tmp_path / "live")

    with pytest.raises(ValueError, match="isolated eval corpus"):
        eval_runner._verify_memory_root(
            ["zeta", "--memory-config", str(live_config)], _expected_root
        )


def test_verify_calendar_created_checks_exact_event_fields(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "calendar-seed.json.out").write_text(
        json.dumps(
            [
                {
                    "title": "Project kickoff",
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                    "calendar": "work",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert eval_runner._verify_calendar_created(
        scratch,
        [
            {
                "title": "Project kickoff",
                "start": "2026-09-19T16:00:00",
                "end": "2026-09-19T17:00:00",
                "calendar": "work",
            }
        ],
    ) == [True]
    (scratch / "calendar-seed.json.out").write_text(
        json.dumps(
            [
                {
                    "title": "Project kickoff",
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                    "calendar": "Work",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert eval_runner._verify_calendar_created(
        scratch,
        [
            {
                "title": "Project kickoff",
                "start": "2026-09-19T16:00:00",
                "end": "2026-09-19T17:00:00",
                "calendar": "work",
            }
        ],
    ) == [True]
    (scratch / "calendar-seed.json.out").write_text(
        json.dumps(
            [
                {
                    "title": "Project kickoff",
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                    "calendar": "personal",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert eval_runner._verify_calendar_created(
        scratch,
        [
            {
                "title": "Project kickoff",
                "start": "2026-09-19T16:00:00",
                "end": "2026-09-19T17:00:00",
                "calendar": "work",
            }
        ],
    ) == [False]
    (scratch / "calendar-seed.json.out").write_text(
        json.dumps(
            [
                {
                    "title": "Project kickoff",
                    "start": "2026-09-19T17:00:00",
                    "end": "2026-09-19T18:00:00",
                    "calendar": "work",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert eval_runner._verify_calendar_created(
        scratch,
        [
            {
                "title": "Project kickoff",
                "start": "2026-09-19T16:00:00",
                "end": "2026-09-19T17:00:00",
                "calendar": "work",
            }
        ],
    ) == [False]


def test_parse_events_sums_service_tagged_jev_usage() -> None:
    result = parse_events(
        [
            {
                "type": "usage",
                "service": "jev",
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
            {
                "type": "compaction_end",
                "jev_triage": {"candidates": 1, "dropped": 0},
            }
        ]
    )

    assert result["jev_input_tokens"] == 5
    assert result["jev_output_tokens"] == 1


def test_parse_events_counts_auto_routing_decisions() -> None:
    result = parse_events(
        [
            {
                "type": "usage",
                "service": "jev",
                "routing_decision": {
                    "advertised": ["read", "write", "bash"],
                },
            },
            {
                "type": "usage",
                "service": "jev",
                "routing_decision": {
                    "advertised": ["read"],
                },
            },
        ]
    )

    assert result["route_calls"] == 2
    assert result["route_expansions"] == 1


def test_parse_events_surfaces_memory_injection_stats() -> None:
    result = parse_events(
        [
            {
                "type": "usage",
                "service": "jev",
                "routing_decision": {
                    "memory_injection": {
                        "candidate_scores": [{"id": "candidate-0", "score": 0.8}],
                        "injected_count": 2,
                        "chars": 1200,
                    }
                },
            }
        ]
    )

    assert result["memory_injection"] == {
        "decisions": [
            {
                "candidate_scores": [{"id": "candidate-0", "score": 0.8}],
                "injected_count": 2,
                "chars": 1200,
            }
        ],
        "injected_count": 2,
        "chars": 1200,
    }


def test_required_call_sequence_matches_exact_fields_and_windows() -> None:
    required = [
        {
            "tool": "calendar_events",
            "args": {
                "start": {
                    "covers": {
                        "start": "2026-09-19T09:00:00",
                        "end": "2026-09-19T12:00:00",
                    }
                }
            },
        }
    ]
    bypass = [
        {
            "tool": "calendar_events",
            "arguments": json.dumps(
                {
                    "start": "2026-09-19T09:00:00",
                    "end": "2026-09-19T09:00:01",
                    "calendar": None,
                }
            ),
        }
    ]
    genuine = [
        {
            "tool": "calendar_events",
            "arguments": json.dumps(
                {
                    "start": "2026-09-19T08:00:00",
                    "end": "2026-09-19T13:00:00",
                    "calendar": None,
                }
            ),
        }
    ]

    assert contains_ordered_subsequence(bypass, required) is False
    assert contains_ordered_subsequence(genuine, required) is True


def test_create_sequence_rejects_nonmatching_creates_and_accepts_genuine_run() -> None:
    create_args = {
        "title": {"equals": "Project kickoff"},
        "start": {"equals": "2026-09-19T16:00:00"},
        "end": {"equals": "2026-09-19T17:00:00"},
        "calendar": {"equals": "work"},
    }
    create = {"tool": "calendar_create", "args": create_args}
    events = {
        "tool": "calendar_events",
        "args": {
            "start": {
                "covers": {
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                }
            }
        },
    }
    forbidden = [{"tool": "calendar_create", "args_not": create_args}]
    wrong_create = {
        "type": "tool_call",
        "name": "calendar_create",
        "arguments": {
            "title": "Wrong meeting",
            "start": "2026-09-19T15:00:00",
            "end": "2026-09-19T15:01:00",
            "calendar": "work",
            "notes": None,
        },
    }
    matching_create = {
        "type": "tool_call",
        "name": "calendar_create",
        "arguments": {
            "title": "Project kickoff",
            "start": "2026-09-19T16:00:00",
            "end": "2026-09-19T17:00:00",
            "calendar": "work",
            "notes": None,
        },
    }
    bypass_calls = [
        wrong_create,
        matching_create,
    ]
    qualified_bypass_calls = [
        {
            "tool": "calendar_create",
            "arguments": json.dumps(wrong_create["arguments"]),
        },
        {
            "tool": "calendar_create",
            "arguments": json.dumps(matching_create["arguments"]),
        },
    ]
    genuine_calls = [matching_create]
    assert contains_ordered_subsequence(qualified_bypass_calls, [create]) is True
    assert contains_forbidden_call_shapes(bypass_calls, forbidden) is True
    assert contains_forbidden_call_shapes(genuine_calls, forbidden) is False

    events_call = {
        "tool": "calendar_events",
        "arguments": json.dumps(
            {
                "start": "2026-09-19T00:00:00",
                "end": "2026-09-20T00:00:00",
                "calendar": None,
            }
        ),
    }
    assert contains_ordered_subsequence(
        [
            {
                "tool": "calendar_create",
                "arguments": json.dumps(matching_create["arguments"]),
            },
            events_call,
        ],
        [create, events],
    ) is True


@pytest.mark.parametrize(
    "required_call_sequence",
    [
        "read",
        ["read", 3],
        [{"tool": 3}],
        [{"tool": "read", "args_contains": 3}],
        [{"tool": "read", "args": {"path": {"contains": "manifest.txt"}}}],
        [
            {
                "tool": "calendar_events",
                "args": {
                    "start": {
                        "covers": {
                            "start": "not-a-date",
                            "end": "2026-09-19T12:00:00",
                        }
                    }
                },
            }
        ],
    ],
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


@pytest.mark.parametrize("forbidden_tools", ["bash", ["bash", 3]])
def test_load_tasks_rejects_invalid_forbidden_tools(
    tmp_path: Path, forbidden_tools: object
) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "task",
                "prompt": "prompt",
                "setup": {},
                "checks": [],
                "forbidden_tools": forbidden_tools,
                "max_turns": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="forbidden_tools"):
        load_tasks(path)


def test_qualified_tool_calls_require_successful_result_pair() -> None:
    events = [
        {
            "type": "tool_call",
            "id": "missing-result",
            "name": "read",
            "arguments": {"path": "manifest.txt"},
        },
        {
            "type": "tool_call",
            "id": "failed",
            "name": "read",
            "arguments": {"path": "manifest.txt"},
        },
        {
            "type": "tool_result",
            "id": "failed",
            "name": "read",
            "is_error": True,
        },
        {
            "type": "tool_call",
            "id": "success",
            "name": "read",
            "arguments": {"path": "manifest.txt"},
        },
        {
            "type": "tool_result",
            "id": "success",
            "name": "read",
            "is_error": False,
        },
    ]

    assert qualified_tool_calls(events) == [
        {"tool": "read", "arguments": '{"path": "manifest.txt"}'},
    ]


def test_required_call_sequence_rejects_padding_without_forbidden_rule() -> None:
    events = _sequence_bypass_events()

    assert contains_ordered_subsequence(
        qualified_tool_calls(events),
        [
            {"tool": "read", "args_contains": "incoming/manifest.txt"},
            {"tool": "write", "args_contains": "stage/total.txt"},
            {"tool": "read", "args_contains": "summary.md"},
        ],
    ) is False


def test_required_call_sequence_rejects_bash_and_padding() -> None:
    events = _sequence_bypass_events()
    required = [
        {"tool": "read", "args_contains": "incoming/manifest.txt"},
        {"tool": "write", "args_contains": "stage/total.txt"},
        {"tool": "read", "args_contains": "summary.md"},
    ]

    assert contains_ordered_subsequence(qualified_tool_calls(events), required) is False
    assert contains_forbidden_tool(events, ["bash", "exec"]) is True


def test_required_call_sequence_accepts_genuine_staged_run() -> None:
    events = [
        {
            "type": "tool_call",
            "id": "read-manifest",
            "name": "read",
            "arguments": {"path": "incoming/manifest.txt"},
        },
        {
            "type": "tool_result",
            "id": "read-manifest",
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

    assert contains_ordered_subsequence(
        qualified_tool_calls(events),
        [
            {"tool": "read", "args_contains": "incoming/manifest.txt"},
            {"tool": "write", "args_contains": "stage/total.txt"},
            {"tool": "read", "args_contains": "summary.md"},
        ],
    ) is True
    assert contains_forbidden_tool(events, ["bash", "exec"]) is False


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
            "required_call_sequence": [
                {"tool": "read", "args_contains": "manifest.txt"},
                {"tool": "write", "args_contains": "total.txt"},
                {"tool": "read", "args_contains": "summary.md"},
            ],
            "max_turns": 1,
        },
        "stock",
        tmp_path / "runs",
    )

    assert record["checks_passed"] == [False]


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


def test_verify_checks_resolves_corpus_paths_and_rejects_wrong_content(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "isolated-corpus"
    corpus.mkdir()
    (corpus / "launch-review.md").write_text(
        "The launch review is on Tuesday at 15:00 in Room Cedar.\n",
        encoding="utf-8",
    )

    assert verify_checks(
        tmp_path,
        [
            {
                "corpus_path": "launch-review.md",
                "contains": "Tuesday at 15:00 in Room Cedar",
            }
        ],
        corpus,
    ) == [True]
    assert verify_checks(
        tmp_path,
        [{"corpus_path": "launch-review.md", "contains": "Wednesday"}],
        corpus,
    ) == [False]


def test_verify_checks_keeps_normalized_slot_check_exact(tmp_path: Path) -> None:
    slot = tmp_path / "free-slot.txt"
    slot.write_text("2026-09-19T10:00:00 to 2026-09-19T11:00:00\n", encoding="utf-8")

    assert verify_checks(
        tmp_path,
        [
            {
                "path": "free-slot.txt",
                "normalized_equals": "2026-09-19T10:00:00 to 2026-09-19T11:00:00",
            }
        ],
    ) == [True]
    slot.write_text("2026-09-19T11:00:00 to 2026-09-19T12:00:00\n", encoding="utf-8")
    assert verify_checks(
        tmp_path,
        [
            {
                "path": "free-slot.txt",
                "normalized_equals": "2026-09-19T10:00:00 to 2026-09-19T11:00:00",
            }
        ],
    ) == [False]


def test_calendar_casefold_only_normalizes_calendar_field() -> None:
    required = [
        {
            "tool": "calendar_create",
            "args": {
                "title": {"equals": "Project kickoff"},
                "start": {"equals": "2026-09-19T16:00:00"},
                "end": {"equals": "2026-09-19T17:00:00"},
                "calendar": {"equals": "work", "casefold": True},
            },
        }
    ]
    matching = [
        {
            "tool": "calendar_create",
            "arguments": json.dumps(
                {
                    "title": "Project kickoff",
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                    "calendar": "Work",
                }
            ),
        }
    ]
    wrong_calendar = [
        {
            "tool": "calendar_create",
            "arguments": json.dumps(
                {
                    "title": "Project kickoff",
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                    "calendar": "personal",
                }
            ),
        }
    ]
    wrong_title = [
        {
            "tool": "calendar_create",
            "arguments": json.dumps(
                {
                    "title": "Project Kickoff",
                    "start": "2026-09-19T16:00:00",
                    "end": "2026-09-19T17:00:00",
                    "calendar": "Work",
                }
            ),
        }
    ]

    assert contains_ordered_subsequence(matching, required) is True
    assert contains_ordered_subsequence(wrong_calendar, required) is False
    assert contains_ordered_subsequence(wrong_title, required) is False


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


def test_memory_injection_eval_does_not_preflight_gateway_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = [{"id": "task", "prompt": "prompt", "setup": {}, "checks": []}]
    captured: dict[str, object] = {}
    monkeypatch.setattr(eval_runner, "load_tasks", lambda _path: tasks)
    monkeypatch.setattr(
        eval_runner,
        "run_evals",
        lambda loaded, modes, out, *, memory_injection: captured.update(
            loaded=loaded,
            modes=modes,
            out=out,
            memory_injection=memory_injection,
        ),
    )

    assert main(
        [
            "--mode",
            "stock",
            "--memory-injection",
            "--tasks-file",
            str(tmp_path / "tasks.jsonl"),
        ]
    ) == 0
    assert captured["loaded"] == tasks
    assert captured["modes"] == ("stock",)
    assert isinstance(captured["out"], Path)
    assert captured["memory_injection"] is True


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
