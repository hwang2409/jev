"""Run offline router and stock zeta evaluations and compare their streams."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pausanias.config import ConfigError, load_config

TASKS_PATH = Path(__file__).with_name("tasks.jsonl")
SCRATCH_ROOT = Path("/tmp/jev-zeta-evals")
RUN_TIMEOUT_SECONDS = 300


def _parse_datetime(value: object) -> datetime | None:
    if type(value) is not str or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _valid_argument_constraints(value: object) -> bool:
    if type(value) is not dict or not value:
        return False
    for field, constraint in value.items():
        if type(field) is not str or not field or type(constraint) is not dict:
            return False
        if set(constraint) == {"equals"}:
            continue
        if set(constraint) == {"equals", "casefold"} and (
            type(constraint["equals"]) is str
            and type(constraint["casefold"]) is bool
        ):
            continue
        if set(constraint) == {"covers"}:
            window = constraint["covers"]
            start = _parse_datetime(window.get("start")) if type(window) is dict else None
            end = _parse_datetime(window.get("end")) if type(window) is dict else None
            if (
                type(window) is not dict
                or set(window) != {"start", "end"}
                or start is None
                or end is None
                or field not in {"start", "end"}
            ):
                return False
            try:
                if start > end:
                    return False
            except TypeError:
                return False
            continue
        return False
    return True


def _valid_relative_path(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def _valid_check(value: object) -> bool:
    if type(value) is not dict:
        return False
    if set(value) in (
        {"path", "equals"},
        {"path", "normalized_equals"},
        {"path", "contains"},
        {"corpus_path", "equals"},
        {"corpus_path", "normalized_equals"},
        {"corpus_path", "contains"},
    ):
        path_key = "corpus_path" if "corpus_path" in value else "path"
        value_key = next(key for key in value if key != path_key)
        return _valid_relative_path(value[path_key]) and type(value[value_key]) is str
    return False


def _valid_required_call(value: object) -> bool:
    if type(value) is not dict or type(value.get("tool")) is not str:
        return False
    if set(value) == {"tool"}:
        return bool(value["tool"])
    if set(value) == {"tool", "args_contains"}:
        return bool(value["tool"]) and type(value["args_contains"]) is str and bool(
            value["args_contains"]
        )
    if set(value) == {"tool", "args"}:
        return bool(value["tool"]) and _valid_argument_constraints(value["args"])
    return False


def _valid_call_shape(value: object, *, allow_args_not: bool = False) -> bool:
    if (
        type(value) is not dict
        or type(value.get("tool")) is not str
        or not value["tool"]
    ):
        return False
    valid_keys = ({"tool", "args"}, {"tool", "args_not"})
    if set(value) not in valid_keys or (
        not allow_args_not and set(value) == {"tool", "args_not"}
    ):
        return False
    constraint_key = "args_not" if "args_not" in value else "args"
    return _valid_argument_constraints(value[constraint_key])


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    """Load task definitions from JSONL."""
    tasks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            if not isinstance(task, dict):
                raise ValueError("each task must be a JSON object")
            checks = task.get("checks")
            if type(checks) is not list or any(
                not _valid_check(check) for check in checks
            ):
                raise ValueError("checks must contain valid file checks")
            required_call_sequence = task.get("required_call_sequence")
            if required_call_sequence is not None and (
                type(required_call_sequence) is not list
                or any(not _valid_required_call(entry) for entry in required_call_sequence)
            ):
                raise ValueError(
                    "required_call_sequence must contain valid call constraints"
                )
            forbidden_tools = task.get("forbidden_tools")
            if forbidden_tools is not None and (
                type(forbidden_tools) is not list
                or any(
                    type(tool_name) is not str or not tool_name
                    for tool_name in forbidden_tools
                )
            ):
                raise ValueError(
                    "forbidden_tools must be a list of nonempty tool names"
                )
            forbidden_call_shapes = task.get("forbidden_call_shapes")
            if forbidden_call_shapes is not None and (
                type(forbidden_call_shapes) is not list
                or any(
                    not _valid_call_shape(shape, allow_args_not=True)
                    for shape in forbidden_call_shapes
                )
            ):
                raise ValueError(
                    "forbidden_call_shapes must contain valid call constraints"
                )
            memory_seed = task.get("memory_seed")
            if memory_seed is not None and (
                type(memory_seed) is not dict
                or any(
                    type(seed_path) is not str
                    or not seed_path
                    or Path(seed_path).is_absolute()
                    or ".." in Path(seed_path).parts
                    or not seed_path.endswith(".md")
                    or type(content) is not str
                    for seed_path, content in memory_seed.items()
                )
            ):
                raise ValueError("memory_seed must map safe markdown paths to strings")
            calendar_seed = task.get("calendar_seed")
            raw_events = (
                calendar_seed.get("events")
                if isinstance(calendar_seed, dict)
                else calendar_seed
            )
            if calendar_seed is not None and (
                not isinstance(raw_events, list)
                or any(
                    type(event) is not dict
                    or set(event) != {"title", "start", "end", "calendar", "all_day"}
                    or any(
                        type(event[key]) is not str
                        for key in ("title", "start", "end", "calendar")
                    )
                    or type(event["all_day"]) is not bool
                    for event in raw_events
                )
            ):
                raise ValueError(
                    "calendar_seed must be an events list with valid event objects"
                )
            checks_calendar_created = task.get("checks_calendar_created")
            if checks_calendar_created is not None and (
                type(checks_calendar_created) is not list
                or any(
                    type(check) is not dict
                    or set(check) != {"title", "start", "end", "calendar"}
                    or any(
                        type(check[key]) is not str or not check[key]
                        for key in ("title", "start", "end", "calendar")
                    )
                    for check in checks_calendar_created
                )
            ):
                raise ValueError(
                    "checks_calendar_created must contain title, start, end, and calendar"
                )
            tasks.append(task)
    return tasks


def build_command(
    task_id: str,
    prompt: str,
    max_turns: int,
    mode: str,
    *,
    memory_config: Path | None = None,
    memory_injection: bool = False,
) -> list[str]:
    """Build the exact headless command for one task and mode."""
    del task_id
    command = [
        "uv",
        "run",
        "--frozen",
        "zeta",
        "--provider",
        "claude",
        "--no-session",
        "--yolo",
        "--max-turns",
        str(max_turns),
        "--format",
        "json",
    ]
    if mode == "router":
        command.extend(["--router-style", "tool"])
    elif mode == "auto":
        command.extend(["--router-style", "auto"])
    else:
        command.append("--no-router")
    if memory_config is not None:
        command.extend(["--memory-config", str(memory_config)])
    if memory_injection:
        command.append("--memory-injection")
    command.extend(["-p", prompt])
    return command


def verify_checks(
    scratch_dir: Path,
    checks: Sequence[Mapping[str, str]],
    memory_root: Path | None = None,
) -> list[bool]:
    """Verify task checks against files in a scratch directory."""
    results: list[bool] = []
    for check in checks:
        if "corpus_path" in check:
            if memory_root is None:
                results.append(False)
                continue
            path = _safe_check_path(memory_root, check["corpus_path"])
        else:
            path = _safe_check_path(scratch_dir, check["path"])
        if path is None:
            results.append(False)
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
            results.append(False)
            continue
        if "equals" in check:
            results.append(content == check["equals"])
        elif "normalized_equals" in check:
            results.append(
                " ".join(content.split())
                == " ".join(check["normalized_equals"].split())
            )
        elif "contains" in check:
            results.append(check["contains"] in content)
        else:
            results.append(False)
    return results


def _safe_check_path(root: Path, relative: str) -> Path | None:
    if type(relative) is not str:
        return None
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        return None
    return target


def qualified_tool_calls(
    events: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return successful, paired, non-route tool calls with serialized arguments."""
    events = list(events)
    successful_result_ids = {
        event.get("id")
        for event in events
        if event.get("type") == "tool_result"
        and type(event.get("id")) is str
        and event.get("is_error") is False
    }
    qualified: list[dict[str, str]] = []
    for event in events:
        if event.get("type") != "tool_call":
            continue
        name = event.get("name")
        call_id = event.get("id")
        if (
            type(name) is not str
            or name == "route"
            or type(call_id) is not str
            or call_id not in successful_result_ids
        ):
            continue
        arguments = json.dumps(
            event.get("arguments"), ensure_ascii=False, sort_keys=True
        )
        qualified.append({"tool": name, "arguments": arguments})
    return qualified


def contains_ordered_subsequence(
    tool_calls: Sequence[Mapping[str, str]],
    required_call_sequence: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether required successful calls occur in order."""

    def matches(tool_call: Mapping[str, str], required: Mapping[str, Any]) -> bool:
        if tool_call["tool"] != required["tool"]:
            return False
        if "args_contains" in required:
            return required["args_contains"] in tool_call["arguments"]
        if set(required) == {"tool"}:
            return True
        try:
            arguments = json.loads(tool_call["arguments"])
        except json.JSONDecodeError:
            return False
        return _matches_argument_constraints(arguments, required["args"])

    required_index = 0
    for tool_call in tool_calls:
        if (
            required_index < len(required_call_sequence)
            and matches(tool_call, required_call_sequence[required_index])
        ):
            required_index += 1
    return required_index == len(required_call_sequence)


def _matches_argument_constraints(
    arguments: object, constraints: Mapping[str, Any]
) -> bool:
    if type(arguments) is not dict:
        return False
    for field, constraint in constraints.items():
        if set(constraint) in ({"equals"}, {"equals", "casefold"}):
            actual = arguments.get(field)
            expected = constraint["equals"]
            if constraint.get("casefold"):
                matches = (
                    type(actual) is str
                    and actual.casefold() == expected.casefold()
                )
            else:
                matches = actual == expected
            if field not in arguments or not matches:
                return False
            continue
        window = constraint["covers"]
        actual_start = _parse_datetime(arguments.get("start"))
        actual_end = _parse_datetime(arguments.get("end"))
        expected_start = _parse_datetime(window["start"])
        expected_end = _parse_datetime(window["end"])
        if None in (actual_start, actual_end, expected_start, expected_end):
            return False
        try:
            if actual_start > expected_start or actual_end < expected_end:
                return False
        except TypeError:
            return False
    return True


def contains_forbidden_call_shapes(
    events: Iterable[Mapping[str, Any]],
    forbidden_call_shapes: Sequence[Mapping[str, Any]],
) -> bool:
    """Return whether any tool call matches a forbidden argument shape."""
    for event in events:
        if event.get("type") != "tool_call" or type(event.get("name")) is not str:
            continue
        tool_call = {
            "tool": event["name"],
            "arguments": json.dumps(
                event.get("arguments"), ensure_ascii=False, sort_keys=True
            ),
        }
        for shape in forbidden_call_shapes:
            if tool_call["tool"] != shape["tool"]:
                continue
            try:
                arguments = json.loads(tool_call["arguments"])
            except json.JSONDecodeError:
                continue
            if "args" in shape and _matches_argument_constraints(
                arguments, shape["args"]
            ):
                return True
            if "args_not" in shape and not _matches_argument_constraints(
                arguments, shape["args_not"]
            ):
                return True
    return False


def contains_forbidden_tool(
    events: Iterable[Mapping[str, Any]], forbidden_tools: Sequence[str]
) -> bool:
    """Return whether any tool call uses a forbidden tool name."""
    return any(
        event.get("type") == "tool_call"
        and event.get("name") in forbidden_tools
        for event in events
    )


def _number(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def parse_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize the JSONL lifecycle events emitted by headless zeta."""
    events = list(events)
    tool_calls = [
        event["name"]
        for event in events
        if event.get("type") == "tool_call" and isinstance(event.get("name"), str)
    ]
    api_calls = 0
    usage_totals = {
        "claude": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
        "jev": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
    }
    route_calls = 0
    route_expansions = 0
    unrouted_attempts = 0
    router_errors = 0
    memory_decisions: list[dict[str, Any]] = []
    final_message_present = False

    def add_usage(service: str, usage: Mapping[str, Any]) -> None:
        totals = usage_totals.get(service)
        if totals is None:
            return
        totals["input_tokens"] += _number(usage.get("input_tokens"))
        totals["output_tokens"] += _number(usage.get("output_tokens"))
        totals["cache_read_tokens"] += _number(
            usage.get("cache_read_input_tokens")
        )

    for event in events:
        event_type = event.get("type")
        if event_type == "tool_call":
            if event.get("name") == "route":
                route_calls += 1
        elif event_type == "usage":
            api_calls += 1
            usage = event.get("usage")
            if isinstance(usage, Mapping):
                service = event.get("service", "claude")
                add_usage(service if service == "jev" else "claude", usage)
            decision = event.get("routing_decision")
            if isinstance(decision, Mapping):
                route_calls += 1
                advertised = decision.get("advertised")
                if type(advertised) is list and len(advertised) == 3:
                    route_expansions += 1
                if isinstance(decision.get("error"), str):
                    router_errors += 1
            memory = event.get("memory_injection")
            if memory is None and isinstance(decision, Mapping):
                memory = decision.get("memory_injection")
            if isinstance(memory, Mapping):
                memory_decisions.append(dict(memory))
        elif event_type == "tool_result":
            name = event.get("name")
            content = event.get("content")
            if name == "route":
                if isinstance(content, str) and content.startswith("routed tools:"):
                    choices = content.removeprefix("routed tools:").strip().split(", ")
                    if len(choices) == 3:
                        route_expansions += 1
                if (
                    event.get("is_error") is True
                    and isinstance(content, str)
                    and content.startswith("router failed: ")
                ):
                    router_errors += 1
            structured = event.get("structured_content")
            if isinstance(structured, Mapping):
                service = structured.get("service")
                usage = structured.get("usage")
                if service == "jev" and isinstance(usage, Mapping):
                    add_usage("jev", usage)
                if structured.get("error_kind") == "unrouted_tool":
                    unrouted_attempts += 1
        elif (
            event_type == "message"
            and event.get("role") == "assistant"
            and isinstance(event.get("text"), str)
        ):
            final_message_present = True

    claude = usage_totals["claude"]
    jev = usage_totals["jev"]
    combined = {
        key: claude[key] + jev[key]
        for key in ("input_tokens", "output_tokens", "cache_read_tokens")
    }
    claude_tokens = sum(claude.values())
    jev_tokens = sum(jev.values())
    combined_tokens = sum(combined.values())
    return {
        "tool_calls": tool_calls,
        "api_calls": api_calls,
        "claude_input_tokens": claude["input_tokens"],
        "claude_output_tokens": claude["output_tokens"],
        "claude_cache_read_tokens": claude["cache_read_tokens"],
        "jev_input_tokens": jev["input_tokens"],
        "jev_output_tokens": jev["output_tokens"],
        "jev_cache_read_tokens": jev["cache_read_tokens"],
        "input_tokens": combined["input_tokens"],
        "output_tokens": combined["output_tokens"],
        "cache_read_tokens": combined["cache_read_tokens"],
        "claude_tokens": claude_tokens,
        "jev_tokens": jev_tokens,
        "combined_tokens": combined_tokens,
        "route_calls": route_calls,
        "route_expansions": route_expansions,
        "unrouted_attempts": unrouted_attempts,
        "router_errors": router_errors,
        "memory_injection": {
            "decisions": memory_decisions,
            "injected_count": sum(
                _number(decision.get("injected_count"))
                for decision in memory_decisions
            ),
            "chars": sum(
                _number(decision.get("chars")) for decision in memory_decisions
            ),
        },
        "final_message_present": final_message_present,
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )


def run_subprocess(
    command: Sequence[str],
    cwd: Path,
    events_path: Path,
    *,
    timeout: int = RUN_TIMEOUT_SECONDS,
    runner: Callable[..., Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a command and tee its stdout into the event stream file."""
    if runner is not None:
        try:
            runner_args: dict[str, Any] = {
                "cwd": str(cwd),
                "capture_output": True,
                "text": True,
                "timeout": timeout,
                "check": False,
            }
            if env is not None:
                runner_args["env"] = dict(env)
            completed = runner(list(command), **runner_args)
        except subprocess.TimeoutExpired as exc:
            events_path.write_text(_output_text(exc.output), encoding="utf-8")
            return {
                "returncode": None,
                "stderr": _output_text(exc.stderr),
                "timed_out": True,
            }
        events_path.write_text(
            _output_text(getattr(completed, "stdout", "")), encoding="utf-8"
        )
        return {
            "returncode": completed.returncode,
            "stderr": _output_text(getattr(completed, "stderr", "")),
            "timed_out": False,
        }

    with events_path.open("w", encoding="utf-8") as events_file:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            stdout=events_file,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(env) if env is not None else None,
        )
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
            return {
                "returncode": None,
                "stderr": _output_text(stderr),
                "timed_out": True,
            }
    return {
        "returncode": process.returncode,
        "stderr": _output_text(stderr),
        "timed_out": False,
    }


def _write_setup(scratch_dir: Path, setup: Mapping[str, str]) -> None:
    for relative, content in setup.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"setup path escapes scratch directory: {relative}")
        path = scratch_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _safe_memory_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise ValueError(f"memory seed path escapes memory root: {relative}")
    return target


def _reindex_memory(config: Path) -> None:
    subprocess.run(
        [sys.executable, "-m", "pausanias", "--config", str(config), "index"],
        check=True,
        capture_output=True,
        text=True,
    )


def _write_memory_config(workspace: Path) -> tuple[Path, Path]:
    root = workspace / "corpus"
    root.mkdir(parents=True)
    config = workspace / "pausanias.toml"
    config.write_text(
        "database = "
        + json.dumps(str(workspace / "index.sqlite3"))
        + "\n\n[[roots]]\n"
        + 'id = "eval"\n'
        + "path = "
        + json.dumps(str(root))
        + '\nproject = "eval"\n',
        encoding="utf-8",
    )
    return root, config


def _verify_memory_root(command: Sequence[str], expected_root: Path) -> None:
    try:
        config_index = command.index("--memory-config")
        config_path = Path(command[config_index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError("eval child memory-config override is missing") from exc
    try:
        config = load_config(config_path)
    except (ConfigError, OSError) as exc:
        raise ValueError(f"eval memory config could not be loaded: {config_path}") from exc
    roots = tuple(root.path.resolve() for root in config.roots)
    if roots != (expected_root.resolve(),):
        raise ValueError(
            "eval memory config does not point to the isolated eval corpus"
        )


@contextlib.contextmanager
def _prepare_task_environment(
    task: Mapping[str, Any], scratch_dir: Path
) -> Iterator[tuple[dict[str, str], Path, Path]]:
    """Seed task-only external fixtures and return the child environment."""

    environment = dict(os.environ)
    memory_workspace = scratch_dir / "memory-eval"
    memory_root, config = _write_memory_config(memory_workspace)
    environment.pop("JEV_EVAL_MEMORY_CONFIG", None)
    environment.pop("PAUSANIAS_CONFIG", None)
    environment["JEV_EVAL_MEMORY_ROOT"] = str(memory_root)
    try:
        memory_seed = task.get("memory_seed")
        if memory_seed:
            for relative, content in memory_seed.items():
                target = _safe_memory_path(memory_root, relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        _reindex_memory(config)

        calendar_seed = task.get("calendar_seed")
        if calendar_seed is None:
            calendar_seed = []
        seed_path = scratch_dir / "calendar-seed.json"
        seed_path.write_text(
            json.dumps(calendar_seed, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        environment["ZETA_CALENDAR_ADAPTER"] = f"fake:{seed_path}"
        yield environment, memory_root, config
    finally:
        shutil.rmtree(memory_workspace, ignore_errors=True)


def _verify_calendar_created(
    scratch_dir: Path,
    checks: Sequence[Mapping[str, str]],
) -> list[bool]:
    seed_path = scratch_dir / "calendar-seed.json"
    output_path = Path(f"{seed_path}.out")
    try:
        created = json.loads(output_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, json.JSONDecodeError):
        return [False for _ in checks]
    if not isinstance(created, list):
        return [False for _ in checks]
    return [
        any(
            isinstance(event, dict)
            and event.get("title") == check["title"]
            and event.get("start") == check["start"]
            and event.get("end") == check["end"]
            and isinstance(event.get("calendar"), str)
            and event["calendar"].casefold() == check["calendar"].casefold()
            for event in created
        )
        for check in checks
    ]


def _run_one(
    task: Mapping[str, Any],
    mode: str,
    run_root: Path,
    *,
    memory_injection: bool = False,
) -> dict[str, Any]:
    task_id = task["id"]
    scratch_dir = run_root / f"{task_id}-{mode}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    _write_setup(scratch_dir, task.get("setup", {}))
    events_path = scratch_dir / "events.jsonl"
    with _prepare_task_environment(task, scratch_dir) as (
        environment,
        memory_root,
        memory_config,
    ):
        command = build_command(
            task_id,
            task["prompt"],
            task["max_turns"],
            mode,
            memory_config=memory_config,
            memory_injection=memory_injection,
        )
        _verify_memory_root(command, memory_root)
        started = time.monotonic()
        process = run_subprocess(
            command, scratch_dir, events_path, env=environment
        )
        wall_seconds = time.monotonic() - started
        events = read_events(events_path)
        summary = parse_events(events)
        checks_passed = verify_checks(scratch_dir, task["checks"], memory_root)
        checks_calendar_created = task.get("checks_calendar_created")
        if checks_calendar_created is not None:
            checks_passed.extend(
                _verify_calendar_created(scratch_dir, checks_calendar_created)
            )
        required_call_sequence = task.get("required_call_sequence")
        forbidden_tools = task.get("forbidden_tools")
        forbidden_call_shapes = task.get("forbidden_call_shapes")
        if (
            required_call_sequence is not None
            or forbidden_tools is not None
            or forbidden_call_shapes is not None
        ):
            sequence_passed = required_call_sequence is None or contains_ordered_subsequence(
                qualified_tool_calls(events), required_call_sequence
            )
            forbidden_passed = forbidden_tools is None or not contains_forbidden_tool(
                events, forbidden_tools
            )
            shapes_passed = forbidden_call_shapes is None or not contains_forbidden_call_shapes(
                events, forbidden_call_shapes
            )
            checks_passed.append(sequence_passed and forbidden_passed and shapes_passed)
    completed = process["returncode"] == 0 and summary["final_message_present"]
    return {
        "task_id": task_id,
        "mode": mode,
        "command": command,
        "scratch_dir": str(scratch_dir),
        "completed": completed,
        "checks_passed": checks_passed,
        "tool_calls": summary["tool_calls"],
        "api_calls": summary["api_calls"],
        "claude_input_tokens": summary["claude_input_tokens"],
        "claude_output_tokens": summary["claude_output_tokens"],
        "claude_cache_read_tokens": summary["claude_cache_read_tokens"],
        "jev_input_tokens": summary["jev_input_tokens"],
        "jev_output_tokens": summary["jev_output_tokens"],
        "jev_cache_read_tokens": summary["jev_cache_read_tokens"],
        "input_tokens": summary["input_tokens"],
        "output_tokens": summary["output_tokens"],
        "cache_read_tokens": summary["cache_read_tokens"],
        "claude_tokens": summary["claude_tokens"],
        "jev_tokens": summary["jev_tokens"],
        "combined_tokens": summary["combined_tokens"],
        "route_calls": summary["route_calls"],
        "route_expansions": summary["route_expansions"],
        "unrouted_attempts": summary["unrouted_attempts"],
        "router_errors": summary["router_errors"],
        "memory_injection": summary["memory_injection"],
        "wall_seconds": round(wall_seconds, 3),
        "timed_out": process["timed_out"],
        "returncode": process["returncode"],
        "stderr": process["stderr"],
    }


def _print_report(records: Sequence[Mapping[str, Any]]) -> None:
    comparison_modes = ("router", "auto", "stock")
    print("task                 router                    auto                      stock")
    print("-" * 72)
    by_task: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], {})[record["mode"]] = record
    for task_id, task_modes in by_task.items():
        cells = []
        for mode in comparison_modes:
            record = task_modes.get(mode)
            if record is None:
                cells.append("-")
                continue
            checks = all(record["checks_passed"])
            cells.append(
                f"{'ok' if record['completed'] and checks else 'fail'} "
                f"tools={len(record['tool_calls'])} "
                f"claude={record['claude_tokens']} "
                f"jev={record['jev_tokens']} "
                f"cache_read={record['cache_read_tokens']} "
                f"router_errors={record['router_errors']} "
                f"combined={record['combined_tokens']}"
            )
        print(f"{task_id:<20} {cells[0]:<25} {cells[1]:<25} {cells[2]}")
    print("\nmode totals")
    for mode in comparison_modes:
        mode_records = [record for record in records if record["mode"] == mode]
        completed = sum(record["completed"] for record in mode_records)
        checks = sum(all(record["checks_passed"]) for record in mode_records)
        claude_tokens = sum(record["claude_tokens"] for record in mode_records)
        jev_tokens = sum(record["jev_tokens"] for record in mode_records)
        cache_read_tokens = sum(
            record["cache_read_tokens"] for record in mode_records
        )
        router_errors = sum(record["router_errors"] for record in mode_records)
        combined_tokens = sum(record["combined_tokens"] for record in mode_records)
        print(
            f"{mode}: runs={len(mode_records)} completed={completed} checks={checks} "
            f"claude_tokens={claude_tokens} jev_tokens={jev_tokens} "
            f"cache_read={cache_read_tokens} router_errors={router_errors} "
            f"combined_tokens={combined_tokens}"
        )


def run_evals(
    tasks: Sequence[Mapping[str, Any]],
    modes: Sequence[str],
    out: Path,
    *,
    memory_injection: bool = False,
) -> list[dict[str, Any]]:
    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = SCRATCH_ROOT / run_timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    records = [
        _run_one(task, mode, run_root, memory_injection=memory_injection)
        for task in tasks
        for mode in modes
    ]
    _print_report(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"run_timestamp": run_timestamp, "records": records}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"\nresults saved to {out}")
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("router", "auto", "stock", "both"), default="both"
    )
    parser.add_argument(
        "--memory-injection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable Jev-gated memory injection for every sweep command",
    )
    parser.add_argument(
        "--tasks", nargs="+", help="task ids, separated by spaces or commas"
    )
    parser.add_argument(
        "--tasks-file",
        type=Path,
        default=TASKS_PATH,
        help="task JSONL file",
    )
    parser.add_argument("--out", help="results JSON path")
    args = parser.parse_args(argv)
    modes = ("router", "auto", "stock") if args.mode == "both" else (args.mode,)
    if args.memory_injection and not os.environ.get("JEV_API_KEY"):
        print(
            "error: JEV_API_KEY is required when memory injection is requested",
            file=sys.stderr,
        )
        return 2
    if {"router", "auto"} & set(modes) and not os.environ.get("JEV_API_KEY"):
        print(
            "error: JEV_API_KEY is required when routed mode is requested",
            file=sys.stderr,
        )
        return 2
    tasks = load_tasks(args.tasks_file)
    if args.tasks:
        requested = {
            task_id.strip()
            for value in args.tasks
            for task_id in value.split(",")
            if task_id.strip()
        }
        tasks = [task for task in tasks if task["id"] in requested]
        found = {task["id"] for task in tasks}
        missing = requested - found
        if missing:
            print(
                f"error: unknown task id(s): {', '.join(sorted(missing))}",
                file=sys.stderr,
            )
            return 2
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(args.out) if args.out else Path("results") / f"{timestamp}.json"
    run_evals(tasks, modes, out, memory_injection=args.memory_injection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
