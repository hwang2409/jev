"""Run offline router and stock zeta evaluations and compare their streams."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TASKS_PATH = Path(__file__).with_name("tasks.jsonl")
SCRATCH_ROOT = Path("/tmp/jev-zeta-evals")
RUN_TIMEOUT_SECONDS = 300


def load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    """Load task definitions from JSONL."""
    tasks: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            if not isinstance(task, dict):
                raise ValueError("each task must be a JSON object")
            required_call_sequence = task.get("required_call_sequence")
            if required_call_sequence is not None and (
                type(required_call_sequence) is not list
                or any(
                    type(tool_name) is not str or not tool_name
                    for tool_name in required_call_sequence
                )
            ):
                raise ValueError(
                    "required_call_sequence must be a list of nonempty tool names"
                )
            tasks.append(task)
    return tasks


def build_command(task_id: str, prompt: str, max_turns: int, mode: str) -> list[str]:
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
    command.append("--router" if mode == "router" else "--no-router")
    command.extend(["-p", prompt])
    return command


def verify_checks(scratch_dir: Path, checks: Sequence[Mapping[str, str]]) -> list[bool]:
    """Verify task checks against files in a scratch directory."""
    results: list[bool] = []
    for check in checks:
        path = scratch_dir / check["path"]
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


def contains_ordered_subsequence(
    tool_calls: Sequence[str], required_call_sequence: Sequence[str]
) -> bool:
    """Return whether required tool names occur in order."""
    required_index = 0
    for tool_name in tool_calls:
        if (
            required_index < len(required_call_sequence)
            and tool_name == required_call_sequence[required_index]
        ):
            required_index += 1
    return required_index == len(required_call_sequence)


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
) -> dict[str, Any]:
    """Run a command and tee its stdout into the event stream file."""
    if runner is not None:
        try:
            completed = runner(
                list(command),
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
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


def _run_one(task: Mapping[str, Any], mode: str, run_root: Path) -> dict[str, Any]:
    task_id = task["id"]
    scratch_dir = run_root / f"{task_id}-{mode}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    _write_setup(scratch_dir, task["setup"])
    command = build_command(task_id, task["prompt"], task["max_turns"], mode)
    events_path = scratch_dir / "events.jsonl"
    started = time.monotonic()
    process = run_subprocess(command, scratch_dir, events_path)
    wall_seconds = time.monotonic() - started
    events = read_events(events_path)
    summary = parse_events(events)
    checks_passed = verify_checks(scratch_dir, task["checks"])
    required_call_sequence = task.get("required_call_sequence")
    if required_call_sequence is not None:
        non_route_tool_calls = [
            name for name in summary["tool_calls"] if name != "route"
        ]
        checks_passed.append(
            contains_ordered_subsequence(
                non_route_tool_calls, required_call_sequence
            )
        )
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
        "wall_seconds": round(wall_seconds, 3),
        "timed_out": process["timed_out"],
        "returncode": process["returncode"],
        "stderr": process["stderr"],
    }


def _print_report(records: Sequence[Mapping[str, Any]]) -> None:
    print("task                 router                    stock")
    print("-" * 72)
    by_task: dict[str, dict[str, Mapping[str, Any]]] = {}
    for record in records:
        by_task.setdefault(record["task_id"], {})[record["mode"]] = record
    for task_id, modes in by_task.items():
        cells = []
        for mode in ("router", "stock"):
            record = modes.get(mode)
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
        print(f"{task_id:<20} {cells[0]:<25} {cells[1]}")
    print("\nmode totals")
    for mode in ("router", "stock"):
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
    tasks: Sequence[Mapping[str, Any]], modes: Sequence[str], out: Path
) -> list[dict[str, Any]]:
    run_timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_root = SCRATCH_ROOT / run_timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    records = [_run_one(task, mode, run_root) for task in tasks for mode in modes]
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
    parser.add_argument("--mode", choices=("router", "stock", "both"), default="both")
    parser.add_argument(
        "--tasks", nargs="+", help="task ids, separated by spaces or commas"
    )
    parser.add_argument("--out", help="results JSON path")
    args = parser.parse_args(argv)
    modes = ("router", "stock") if args.mode == "both" else (args.mode,)
    if "router" in modes and not os.environ.get("JEV_API_KEY"):
        print(
            "error: JEV_API_KEY is required when router mode is requested",
            file=sys.stderr,
        )
        return 2
    tasks = load_tasks()
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
    run_evals(tasks, modes, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
