"""The Jev-backed tool router."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypedDict

from ..execution import ToolExecutionContext
from ..providers.jev import route_step
from ..types import StructuredToolResult
from .registry import ToolRegistry, _error_result, _success_result, text_block


class RouteArguments(TypedDict):
    step: str


ROUTE_TOPK_CONFIDENCE = 0.8
"""Choice cutoff; thresholds do not transfer (Jev jaggedness section 8)."""
# TODO: calibrate this Choice threshold with route confidence data.

LOW_CLARITY_NUDGE = 0.3
"""Noul cutoff; thresholds do not transfer (Jev jaggedness section 8)."""
# TODO: calibrate this Noul threshold with step-clarity data.

_BOUNDARIES: dict[str, tuple[str, list[str]]] = {
    "read": (
        "Searching file contents by pattern; use grep or the matching search tool.",
        ["Read src/zeta/loop.py to inspect the routing logic."],
    ),
    "write": (
        "Changing part of an existing file; use edit.",
        ["Write a new notes.md file with the requested content."],
    ),
    "edit": (
        "Creating or replacing a whole file; use write.",
        ["Replace one old threshold in src/zeta/loop.py with a new constant."],
    ),
    "fetch": (
        "Searching the web for relevant pages; use websearch.",
        ["Fetch https://docs.typesafe.ai/primitives/advanced.md."],
    ),
    "websearch": (
        "Opening a known URL; use fetch.",
        ["Search the web for the latest TypeSafe routing documentation."],
    ),
    "memory_search": (
        "Searching file contents in the working repo with grep, reading a known file, fetching a URL, or searching the web; use grep, read, fetch, or websearch.",
        [
            "Recall the decision behind the current memory retention policy from Henry's notes.",
        ],
    ),
    "memory_read": (
        "Searching memory by topic, searching working-repo files, fetching a URL, or searching the web; use memory_search, grep, fetch, or websearch.",
        [
            "Read the recalled retention decision note at a path returned by memory_search.",
        ],
    ),
    "bash": (
        "Editing a file in place; use edit. Use exec for fixed-cwd commands with timeout or output limits, and run_background for long-running commands.",
        [
            "Use bash to cd into the reports directory, then list its files in the next shell call.",
        ],
    ),
    "exec": (
        "Use bash when the shell cwd must persist or a per-call cwd is needed, and run_background for long-running commands.",
        [
            "Use exec to run pytest with a 10 second timeout and an 8192 character output limit.",
        ],
    ),
    "automation": (
        "Start an immediate shell task with run_background, or inspect or stop an existing task with task_output or task_kill.",
        [
            "Draft a weekday 09:00 Slack digest job named daily-report.",
        ],
    ),
    "skill": (
        "Read a file with read, or update the session checklist with todo.",
        [
            "Load the webapp-testing skill prompt before testing the local web app.",
        ],
    ),
    "todo": (
        "Load instructions with skill, or delegate work with agent.",
        [
            "Mark write regression tests in_progress in the session todo list.",
        ],
    ),
    "run_background": (
        "Run a command and wait for its result with bash or exec. Read or stop an existing task with task_output or task_kill.",
        [
            "Start npm run dev as a background task and return its task_id.",
        ],
    ),
    "task_output": (
        "Start or stop a task with run_background or task_kill. Read a child agent transcript with agent_output.",
        [
            "Read background task bg-42 output since cursor 1200 and check whether it is still running.",
        ],
    ),
    "task_kill": (
        "Read incremental output with task_output. Start a task with run_background; child-agent controls use agent_status, agent_output, or agent_send.",
        [
            "Terminate background task bg-42 after its server check is complete.",
        ],
    ),
    "agent": (
        "Run a shell command with bash or exec. Inspect an existing child with agent_status or agent_output, or send it a follow-up with agent_send.",
        [
            "Delegate repository research on the authentication flow and return its child handle.",
        ],
    ),
    "agent_status": (
        "Read child transcript text with agent_output. Start a child or send it a prompt with agent or agent_send.",
        [
            "List all child agents and their current steps without reading their transcripts.",
        ],
    ),
    "agent_output": (
        "Read lifecycle or progress only with agent_status. Read background shell output with task_output.",
        [
            "Read child agent child-7 transcript characters 4000 through 5000.",
        ],
    ),
    "agent_send": (
        "Start a child with agent. Inspect it with agent_status or agent_output. This tool only sends to a live agent_type=run child.",
        [
            "Send child child-7 a follow-up to inspect the failing test after its current turn.",
        ],
    ),
}


def build_catalog(
    schemas: Iterable[Mapping[str, object]],
    *,
    excluded_names: set[str] | frozenset[str] = frozenset(),
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for schema in schemas:
        name = schema.get("name")
        if not isinstance(name, str) or name in excluded_names:
            continue
        description = schema.get("description", "")
        if not isinstance(description, str):
            description = ""
        lines = description.splitlines()
        what = (lines[0] if lines else "")[:150]
        not_for, examples = _BOUNDARIES.get(
            name,
            (
                "Actions outside this tool's description; use the matching tool.",
                [f"Use {name} for the action described by its tool description."],
            ),
        )
        catalog[name] = {
            "what": what,
            "not_for": not_for,
            "examples": examples,
        }
    return catalog


def _catalog(registry: ToolRegistry) -> dict[str, dict[str, Any]]:
    return build_catalog(registry.schemas, excluded_names={"route"})


def _route_error(message: str) -> StructuredToolResult:
    return _error_result(message, kind="error")


def _route_text(tool: str, probabilities: dict[str, float], confidence: float) -> str:
    if confidence >= ROUTE_TOPK_CONFIDENCE:
        return f"routed to {tool} (confidence {confidence:.2f})"
    top_tools = sorted(
        probabilities.items(), key=lambda item: item[1], reverse=True
    )[:3]
    choices = ", ".join(f"{name} ({probability:.2f})" for name, probability in top_tools)
    return f"routed tools: {choices}"


async def _route(
    registry: ToolRegistry,
    arguments: RouteArguments,
    *,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    recent_steps = (
        execution_context.router_recent_steps
        if execution_context is not None
        else None
    )
    try:
        result = await route_step(
            arguments["step"],
            _catalog(registry),
            list(recent_steps or []),
        )
        routed_tools = (
            [result.tool]
            if result.confidence >= ROUTE_TOPK_CONFIDENCE
            else [
                name
                for name, _probability in sorted(
                    result.probabilities.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:3]
            ]
        )
        available = registry.registered_names - {"route"}
        routed_tools = [name for name in routed_tools if name in available]
        if not routed_tools:
            raise ValueError(f"Jev selected no registered tool for {result.tool!r}")
        if execution_context is not None:
            sink = execution_context.router_tools_sink
            if sink is not None:
                sink(routed_tools)
        message = _route_text(result.tool, result.probabilities, result.confidence)
        if result.step_clarity < LOW_CLARITY_NUDGE:
            message += "; restate the step more concretely"
        telemetry: dict[str, Any] = {
            "service": "jev",
            "usage": dict(result.usage),
            "confidence": result.confidence,
            "needs_tool": result.needs_tool,
            "step_clarity": result.step_clarity,
        }
        if result.call_confidence is not None:
            telemetry["call_confidence"] = result.call_confidence
        return _success_result(text_block(message), structured_content=telemetry)
    except Exception as exc:  # noqa: BLE001 - routing must fail open
        if execution_context is not None:
            sink = execution_context.router_tools_sink
            if sink is not None:
                sink(None)
        return _route_error(f"router failed: {exc}")
    finally:
        if recent_steps is not None:
            recent_steps.append(arguments["step"])
            del recent_steps[:-5]


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "route",
        _route,
        description="Describe the next concrete action so Jev can choose a tool.",
        parameters={
            "type": "object",
            "properties": {"step": {"type": "string"}},
            "required": ["step"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
